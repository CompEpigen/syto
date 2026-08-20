"""
Pseudo-Bulk Generation Pipeline

End-to-end pipeline using the new PseudobulkGenerator class with:
    - Crash-resilient batch processing with HDF5 checkpointing
    - Dask-based parallel generation
    - Full reproducibility metadata storage

This module orchestrates:
    1. Loading train/valid/test splits
    2. Read preparation (renaming and deriving columns)
    3. Classifier prediction (if needed)
    4. Pseudobulk generation with the new HDF5-based generator
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from syto.data.read_preparation import prepare_splits_for_pseudobulk
from syto.data.pseudobulk_input import load_raw_splits
from syto.data.pseudobulk_generator import PseudobulkGenerator
from syto.data.pseudobulk_hdf5_utils import (
    GenerationMetadata,
    GenerationParameters,
)
from syto.classification.classifiers.lazy_classifier_factory import (
    read_classifier_factory,
)
from syto.data.dataset import resolve_column


class PseudoBulkPipeline:
    """Orchestrate pseudo-bulk mixture generation using PseudobulkGenerator.

    This pipeline replaces the legacy pickle-based generation with a new
    HDF5-based approach that provides:
    - Crash-resilient checkpointing at the batch level
    - Full reproducibility metadata (seeds, samples per GRG, etc.)
    - Dask-based parallel generation
    - Pure profile generation and uniform prior computation

    Parameters
    ----------
    config : dict
        Parsed YAML configuration. See config/pseudobulk_v2_template.yaml
        for available options.
    logger : logging.Logger
        Logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Read-level classifier, built once on first use (predictions + the
        # required-column projection for raw columnar splits).
        self._classifier = None

        # Labels
        labels_dict_path = config["labels_dict_path"]
        with open(labels_dict_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}

        # Number of classifier prediction classes (how many prediction_{i}_wavg
        # feature columns to produce). Distinct from the number of cell types /
        # deconvolution targets (len(labels_dict)): the classifier may emit extra
        # classes (e.g. a cancer class) with no cell-type entry. Lives under
        # classifier_config since it must match the classifier's num_labels.
        self.num_prediction_classes: int = config["classifier_config"][
            "num_prediction_classes"
        ]

        # Output paths
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generator parameters
        self.batch_size = config["batch_size"]
        self.n_workers = config["n_workers"]
        self.n_reads_to_sample = config["n_reads_to_sample"]

        # columns
        self.class_label_column = config["class_label_column"]
        self.grg_label_column = config["grg_label_column"]
        self.columns_to_keep = config.get("columns_to_keep")

        # [for Debug] parameter to generate n pseudobulks by sampling first n from the target proportions
        self.n_pseudobulks_to_sample = config.get("n_pseudobulks_to_sample", None)

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Path:
        """Execute the full pseudo-bulk generation pipeline.

        Returns
        -------
        Path
            Path to the final consolidated HDF5 file.
        """
        # ── Stage 1: Load splits ──────────────────────────────────
        self.logger.info("Stage 1: Loading data splits ...")
        splits_data = self._load_splits()
        sizes = ", ".join(f"{name}={len(df)}" for name, df in splits_data.items())
        self.logger.info(f"  {sizes} reads")

        # ── Stage 2: Prepare reads ────────────────────────────────
        input_type = self.config["input_type"]
        trim = self.config.get("trim_reads_to_grg_regions", False)
        if input_type == "raw_splits":
            self.logger.info("Stage 2: Preparing reads ...")
            splits_data = self._prepare_reads(splits_data, trim)
        elif input_type == "pre_predicted":
            self.logger.info("Stage 2: Skipped (input already has predictions)")
        else:
            raise ValueError(f"Unknown input_type: '{input_type}'")

        # ── Stage 3: Classifier predictions (if needed) ───────────
        if input_type in ["raw_splits"]:
            splits_data = self._run_classifier_predictions(splits_data)
            if self.config.get("generate_predictions_only", False):
                self.logger.info(
                    "generate_predictions_only was set to True: Skipping the rest of the pseudobulk pipeline"
                )
                return
        else:
            self.logger.info("Stage 3: Skipped (input already has predictions)")

        # ── Stage 4: Generate pseudobulks ─────────────────────────
        self.logger.info("Stage 4: Generating pseudobulk samples ...")

        # Filter columns to reduce memory usage
        self._filter_split_columns(splits_data)

        # Load target proportions
        target_proportions_per_split = self._load_target_proportions()

        # Build metadata and parameters
        metadata = self._build_metadata(splits_data)
        parameters = self._build_parameters()

        # Create and run generator
        generator = PseudobulkGenerator(
            splits_df=splits_data,
            output_directory=self.output_dir / "pseudobulk_generation",
            target_proportions_per_split=target_proportions_per_split,
            batch_size=self.batch_size,
            n_workers=self.n_workers,
            metadata=metadata,
            parameters=parameters,
            n_reads_to_sample=self.n_reads_to_sample,
            class_label_column=self.class_label_column,
            grg_label_column=self.grg_label_column,
            columns_to_keep=self.columns_to_keep,
            num_prediction_classes=self.num_prediction_classes,
            logger=self.logger,
        )

        output_path = generator.run()
        self.logger.info(f"Generation complete: {output_path}")

        return output_path

    # ═══════════════════════════════════════════════════════════════
    #  Stage helpers
    # ═══════════════════════════════════════════════════════════════

    def _prepare_reads(
        self, splits_data: Dict[str, pd.DataFrame], trim=False
    ) -> Dict[str, pd.DataFrame]:
        """Prepare reads for pseudobulk generation."""
        from syto.data.atlases.uxm_atlases import UXMMethylationAtlas

        atlas_path = self.config["atlas_path"]
        atlas_name = self.config["atlas_name"]
        atlas = UXMMethylationAtlas(
            atlas_name=atlas_name,
            reference_genome="hg38" if "hg38" in atlas_path else "hg19",
            atlas_path=atlas_path,
            sep="\t",
        )

        splits_data = prepare_splits_for_pseudobulk(
            splits_data, num_labels=self.num_prediction_classes, atlas=atlas, trim=trim
        )
        sizes = ", ".join(f"{name}={len(df)}" for name, df in splits_data.items())
        self.logger.info(f"  After preparation: {sizes}")
        return splits_data

    def _run_classifier_predictions(
        self, splits_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Run classifier predictions on all splits."""
        self.logger.info("Stage 3: Running classifier predictions ...")
        classifier_config = self.config["classifier_config"]
        read_level_classifier = self._get_classifier()

        for name, df in splits_data.items():
            self.logger.info(f"  Predicting {name} split ({len(df)} reads) ...")
            predicted = read_level_classifier.predict_split(df, **classifier_config)

            # Save intermediate predicted split
            intermediate_path = self.output_dir / f"{name}_predicted.pkl"
            with open(intermediate_path, "wb") as f:
                pickle.dump(predicted, f)
            self.logger.info(f"  Saved predicted {name} to {intermediate_path}")

            splits_data[name] = predicted

        return splits_data

    def _get_classifier(self):
        """Build (once) and cache the read-level classifier."""
        if self._classifier is None:
            classifier_config = self.config["classifier_config"]
            self._classifier = read_classifier_factory(
                name=self.config["classifier_type"],
                path=self.config["classifier_checkpoint"],
                labels_dict=self.labels_dict,
                num_labels=classifier_config.get("num_prediction_classes"),
                seq_length=classifier_config.get("seq_length"),
                foundation_model_path=classifier_config.get("foundation_model"),
                classifier_head_implementation=classifier_config.get(
                    "classifier_head_implementation"
                ),
                grg_label_column=classifier_config.get("grg_label_column"),
                soft_labels=classifier_config.get("soft_labels", True),
                dismir_flavor=classifier_config.get("dismir_flavor", "lstm"),
                batch_size=classifier_config.get("batch_size"),
            )
        return self._classifier

    def _filter_split_columns(self, splits_data: Dict[str, pd.DataFrame]) -> None:
        """Filter dataframes to retain only necessary columns."""
        self.logger.info("  Filtering unused columns to optimize RAM usage ...")

        # Detect prediction columns
        sample_df = next(iter(splits_data.values()))
        pred_cols = [
            c
            for c in sample_df.columns
            if c.startswith("prediction_") and c[11:].isdigit()
        ]
        dna_col = resolve_column(sample_df.columns, "input_ids")
        meth_col = resolve_column(sample_df.columns, "methylation_ids")

        # Required columns for pseudobulk generation
        base_cols = [
            dna_col,
            meth_col,
            "original_label",
            "dmr_ctype_label",
            "dmr_ctype",
            "NCPGS",
            "total_marked_cpgs",
            "chr",
            "chromosome",
            "label",
        ]
        if (
            "trimmed_start" in sample_df.columns
            and "read_start" not in sample_df.columns
        ):
            base_cols.extend(["trimmed_start", "trimmed_end"])
        elif "read_start" in sample_df.columns:
            base_cols.extend(["read_start", "read_end"])
        if "file" in sample_df.columns:
            base_cols.append("file")
        if "name" in sample_df.columns:
            base_cols.append("name")

        for split_name, df in splits_data.items():
            keep_cols = [c for c in base_cols + pred_cols if c in df.columns]
            splits_data[split_name] = df[keep_cols]

    def _build_metadata(
        self, splits_data: Dict[str, pd.DataFrame]
    ) -> GenerationMetadata:
        """Build generation metadata from configuration and data."""
        # Compute data watermark (hash of first few rows of each split)
        watermark_parts = []
        for split_name, df in sorted(splits_data.items()):
            sample = df.head(100).to_json()
            watermark_parts.append(f"{split_name}:{hash(sample)}")
        data_watermark = ":".join(watermark_parts)

        # Compute data stats
        data_stats = {}
        for split_name, df in splits_data.items():
            data_stats[split_name] = {
                "n_reads": len(df),
                "n_unique_labels": (
                    df["original_label"].nunique()
                    if "original_label" in df.columns
                    else 0
                ),
                "n_unique_grg": (
                    df["dmr_ctype_label"].nunique()
                    if "dmr_ctype_label" in df.columns
                    else 0
                ),
            }

        return GenerationMetadata(
            grg_id_column=self.grg_label_column,
            labeling_scheme=self.config.get("labeling_scheme", "Not Set | Irrelevant"),
            classifier=self.config["classifier_type"],
            data_watermark=data_watermark,
            data_stats=data_stats,
        )

    def _build_parameters(self) -> GenerationParameters:
        """Build generation parameters from configuration and data."""
        # Build cell types mapping from labels_dict
        cell_types_mapping = {v: k for k, v in self.labels_dict.items()}

        # Specific to Loyfer dataset: we build one GR group per ctype
        gr_groups_mapping = {
            f"{ctype}_grg": i for i, ctype in enumerate(cell_types_mapping.keys())
        }

        # Get substitution method
        substitution_method = self.config["substitution_method"]

        # Get GR sampling method
        grg_sampling_method = self.config["grg_sampling_method"]

        return GenerationParameters(
            cell_types_mapping=cell_types_mapping,
            gr_groups_mapping=gr_groups_mapping,
            substitution_method=substitution_method,
            grg_sampling_method=grg_sampling_method,
        )

    # ═══════════════════════════════════════════════════════════════
    #  Data loading helpers
    # ═══════════════════════════════════════════════════════════════

    def _load_splits(self) -> Dict[str, pd.DataFrame]:
        """Load configured splits based on ``input_type``.

        ``raw_splits`` reads the shared columnar/legacy dataset directory;
        ``pre_predicted`` loads this pipeline's own per-split output files.
        """
        if self.config["input_type"] == "raw_splits":
            return self._load_raw_splits()
        return self._load_pre_predicted_splits()

    def _load_raw_splits(self) -> Dict[str, pd.DataFrame]:
        """Load raw training splits from the shared columnar/legacy dataset dir."""
        classifier = self._get_classifier()
        classifier_cfg = self.config.get("classifier_config", {})
        # required_fit_columns reads grg_label_column / classifier_head_implementation
        # from a fit-config's model/training sections; the pseudobulk config keeps
        # those under classifier_config, so expose it under both keys.
        fit_shim = {"model": classifier_cfg, "training": classifier_cfg}
        return load_raw_splits(
            self.config["data_path"],
            list(self.config["split_information"].keys()),
            classifier_required_columns=classifier.required_fit_columns(fit_shim),
            data_format=self.config.get("data_format", "auto"),
            split_column=self.config.get("split_column", "split"),
            min_pattern_length=self.config.get("min_pattern_length"),
        )

    def _load_pre_predicted_splits(self) -> Dict[str, pd.DataFrame]:
        """Load pre-predicted splits from per-split parquet or pickle files."""
        splits_configs = self.config["split_information"]
        splits_data = {}

        for split_name, split_cfg in splits_configs.items():
            path = Path(split_cfg["data_path"])
            file_extension = path.suffix.lower()
            if file_extension == ".parquet":
                splits_data[split_name] = pd.read_parquet(path)
            elif file_extension in [".pkl", ".pickle"]:
                with open(path, "rb") as f:
                    splits_data[split_name] = pickle.load(f)
            else:
                raise ValueError(
                    f"Unsupported file extension '{file_extension}' for split '{split_name}'. "
                    "Expected .parquet or .pkl/.pickle."
                )
            self.logger.debug("  Loaded %s from %s", split_name, path)

            splits_data[split_name] = splits_data[split_name]

        return splits_data

    def _load_target_proportions(self) -> Dict[str, np.ndarray]:
        """Load target proportions for each split."""
        target_proportions_per_split = {}

        splits_configs = self.config["split_information"]

        # Per-split target proportions
        for split_name, split_cfg in splits_configs.items():

            tp_path = split_cfg["target_proportions_path"]
            tp_data = np.load(tp_path)
            proportions = tp_data["proportions"]
            if self.n_pseudobulks_to_sample is not None:
                self.logger.info(
                    "  n_pseudobulks_to_sample was set in config. Sampling first %d pseudobulks",
                    self.n_pseudobulks_to_sample,
                )
                proportions = proportions[: self.n_pseudobulks_to_sample]

            self.logger.info(
                "  Loaded %d proportions for %s", len(proportions), split_name
            )

            target_proportions_per_split[split_name] = proportions

        return target_proportions_per_split
