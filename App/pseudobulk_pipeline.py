"""
Pseudo-Bulk Generation Pipeline

End-to-end pipeline orchestrating:
    1. Loading / rebalancing train/valid/test splits
    2. Read preparation (derived columns, optional UXM alignment)
    3. Classifier prediction (MethylBERT / Dismir / CancerDetector)
    4. Parallel IO-example generation (random or target-proportion mode)
    5. Consolidation of partial pickles into a single .npz
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from methyldl.data import LOYFER_CELL_TYPE_MATCH_DICT
from methyldl.data.pseudo_bulk_generation import (
    consolidate_ios_pickles,
    run_ios_generation_parallel,
)
from methyldl.data.read_preparation import prepare_splits_for_pseudobulk
from methyldl.data.split_rebalancing import rebalance_splits
from methyldl.modelling.classifier_adapter import ClassifierAdapter


class PseudoBulkPipeline:
    """Orchestrate pseudo-bulk mixture generation end-to-end.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    logger : logging.Logger
        Logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Labels
        labels_dict_path = config["labels_dict_path"]
        with open(labels_dict_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.num_labels = config.get("num_labels", len(self.labels_dict))

        # Cell-type matching dict
        self.cell_type_match_dict = config.get(
            "cell_type_match_dict", LOYFER_CELL_TYPE_MATCH_DICT
        )

        # Output paths
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.ios_dir = os.path.join(self.output_dir, "ios_deconvolution")
        os.makedirs(self.ios_dir, exist_ok=True)

        self.ios_base_path = os.path.join(self.ios_dir, "ios_deconvolution_data.pkl")
        self.consolidated_path = os.path.join(self.output_dir, "ios_full_matrices.npz")

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """Execute the full pseudo-bulk generation pipeline.

        Returns
        -------
        dict
            Consolidated output arrays keyed by ``proportions``,
            ``features_train``, ``features_valid``, ``features_test``.
        """
        if self.config["input_type"] == "raw_splits":
            # ── Stage 1: Load splits ──────────────────────────────────
            self.logger.info("Stage 1: Loading data splits ...")
            splits_data = self._load_splits()
            sizes = ", ".join(f"{name}={len(df)}" for name, df in splits_data.items())
            self.logger.info(f"  {sizes} reads")

            # ── Stage 1b: Optional rebalancing ────────────────────────
            if self.config.get("rebalance_splits", False):
                self.logger.info("Stage 1b: Rebalancing splits ...")
                if (
                    "train" in splits_data
                    and "valid" in splits_data
                    and "test" in splits_data
                    and len(splits_data) == 3
                ):
                    manual_labels = self.config.get("manual_common_labels", [28, 35])
                    train, valid, test = rebalance_splits(
                        splits_data["train"],
                        splits_data["valid"],
                        splits_data["test"],
                        manual_common_labels=manual_labels,
                    )
                    splits_data["train"] = train
                    splits_data["valid"] = valid
                    splits_data["test"] = test
                    sizes = ", ".join(
                        f"{name}={len(df)}" for name, df in splits_data.items()
                    )
                    self.logger.info(f"  After rebalance: {sizes}")
                else:
                    self.logger.warning(
                        "Rebalancing is only supported when exactly train, valid, and test splits are present. Skipping."
                    )

            # ── Stage 2: Prepare reads ────────────────────────────────
            self.logger.info("Stage 2: Preparing reads ...")
            generate_uxm = self.config.get("generate_uxm_inputs", True)
            atlas_path = self.config.get("atlas_path")

            splits_data = prepare_splits_for_pseudobulk(
                splits_data,
                labels_dict=self.labels_dict,
                num_labels=self.num_labels,
                generate_uxm_inputs=generate_uxm,
                atlas_path=atlas_path,
                cell_type_match_dict=self.cell_type_match_dict,
            )
            sizes = ", ".join(f"{name}={len(df)}" for name, df in splits_data.items())
            self.logger.info(f"  After preparation: {sizes}")
            with open(
                os.path.join(self.output_dir, "uxm_prepared_reads.pkl"), "wb"
            ) as f:
                pickle.dump(splits_data, f)
        elif self.config["input_type"] == "uxm_prepared":
            self.logger.info("Skipping Stage 1 and 2: Loading uxm prepared reads ...")
            uxm_path = self.config.get(
                "uxm_prepared_reads_path",
                os.path.join(self.output_dir, "uxm_prepared_reads.pkl"),
            )
            with open(uxm_path, "rb") as f:
                splits_data = pickle.load(f)
            sizes = ", ".join(f"{name}={len(df)}" for name, df in splits_data.items())
            self.logger.info(f"  After loading: {sizes}")
        elif self.config["input_type"] == "pre_predicted":
            self.logger.info("Skipping Stage 1,2,3: Loading reads with predictions ..")
            splits_data = self._load_splits()
            sizes = ", ".join(f"{name}={len(df)}" for name, df in splits_data.items())
            # for name, df in splits_data.items():
            #     df.drop(["cpg_sig", "prediction_source"], axis=1, inplace=True)
            self.logger.info(f"  {sizes} reads")

        # ── Stage 3: Classifier predictions (if needed) ───────────
        if self.config["input_type"] in ["raw_splits", "uxm_prepared"]:
            self.logger.info("Stage 3: Running classifier predictions ...")
            adapter = self._build_classifier_adapter()

            for name, df in splits_data.items():
                self.logger.info(f"  Predicting {name} split ({len(df)} reads) ...")
                predicted = adapter.predict_split(df)

                # Save intermediate predicted split
                intermediate_path = os.path.join(
                    self.output_dir, f"{name}_predicted.pkl"
                )
                with open(intermediate_path, "wb") as f:
                    pickle.dump(predicted, f)
                self.logger.info(f"  Saved predicted {name} to {intermediate_path}")

                splits_data[name] = predicted

            with open(os.path.join(self.output_dir, "predicted_reads.pkl"), "wb") as f:
                pickle.dump(splits_data, f)
        else:
            self.logger.info("Stage 3: Skipped (input already has predictions)")

        # ── Stage 4: Generate IO examples ─────────────────────────
        self.logger.info("Stage 4: Generating pseudo-bulk IO examples ...")
        generate_uxm_in_ios = self.config.get("generate_uxm_inputs_in_ios", True)

        self.logger.info("  Filtering unused columns to optimize RAM usage ...")
        self._filter_split_columns(splits_data, generate_uxm_in_ios)

        split_generation = self.config.get("split_generation")

        if split_generation is not None:
            # ── Per-split generation mode ──────────────────────────
            self._run_per_split_generation(
                splits_data, split_generation, generate_uxm_in_ios
            )
        else:
            # ── Legacy shared generation mode ─────────────────────
            self._run_shared_generation(splits_data, generate_uxm_in_ios)

        # ── Stage 5: Consolidate ──────────────────────────────────
        self.logger.info("Stage 5: Consolidating partial pickles ...")

        if split_generation is not None:
            # Per-split generation writes to subdirectories; consolidate
            # each subdirectory, then merge all results into one .npz.
            merged_result = {}
            for split_name in split_generation:
                split_ios_dir = os.path.join(self.ios_dir, split_name)
                if not os.path.isdir(split_ios_dir):
                    continue
                pkl_files = [f for f in os.listdir(split_ios_dir) if f.endswith(".pkl")]
                if not pkl_files:
                    continue

                split_output = os.path.join(
                    self.output_dir, f"ios_full_matrices_{split_name}.npz"
                )
                part = consolidate_ios_pickles(
                    ios_dir=split_ios_dir,
                    output_path=split_output,
                    num_labels=self.num_labels,
                    labels_dict=self.labels_dict,
                )
                merged_result.update(part)

            # Write combined .npz
            np.savez_compressed(self.consolidated_path, **merged_result)
            result = merged_result
        else:
            result = consolidate_ios_pickles(
                ios_dir=self.ios_dir,
                output_path=self.consolidated_path,
                num_labels=self.num_labels,
                labels_dict=self.labels_dict,
            )

        # Log summary — handle both legacy and variant-aware key schemes
        if "proportions" in result:
            n_examples = result["proportions"].shape[0]
        else:
            # Pick the first proportions_{split} key
            prop_keys = [k for k in result if k.startswith("proportions_")]
            if prop_keys:
                n_examples = result[prop_keys[0]].shape[0]
            else:
                n_examples = 0
        self.logger.info(
            f"  Consolidated {n_examples} examples to " f"{self.consolidated_path}"
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    #  Stage-4 generation helpers
    # ═══════════════════════════════════════════════════════════════

    def _filter_split_columns(
        self,
        splits_data: Dict[str, pd.DataFrame],
        generate_uxm_in_ios: bool,
    ) -> None:
        """Filter dataframes to retain only the necessary columns for pseudo-bulk logic."""
        base_cols = [
            "original_label",
            "dmr_ctype_label",
            "dmr_ctype",
            "NCPGS",
            "total_marked_cpgs",
            "M_rate",
            "methylation_level",
            "chr",
            "chromosome",
            "label",
        ]

        if generate_uxm_in_ios:
            base_cols.extend(["name", "record_M", "record_U", "record_X"])

        for split_name, df in splits_data.items():
            pred_cols = [
                c
                for c in df.columns
                if c.startswith("prediction_") and c[11:].isdigit()
            ]
            # Keep only columns that exist in the dataframe to avoid KeyErrors
            keep_cols = [c for c in base_cols + pred_cols if c in df.columns]

            # Select the columns in-place conceptually
            # (assigning a sub-slice reference back to the dict)
            splits_data[split_name] = df[keep_cols]

    def _build_shared_kwargs(
        self,
        splits_data: Dict[str, pd.DataFrame],
        generate_uxm_in_ios: bool,
    ) -> dict:
        """Build the keyword arguments shared across generation calls."""
        return dict(
            splits=splits_data,
            file_name=self.ios_base_path,
            n_workers=self.config.get("n_workers"),
            batch_size=self.config.get("batch_size", 100),
            checkpoint_interval=self.config.get("checkpoint_interval", 1000),
            start_checkpoint_idx=self.config.get("start_checkpoint_idx", 0),
            n_read_per_split=self.config.get("n_read_per_split"),
            num_labels=self.num_labels,
            generate_uxm_inputs=generate_uxm_in_ios,
        )

    def _run_shared_generation(
        self,
        splits_data: Dict[str, pd.DataFrame],
        generate_uxm_in_ios: bool,
    ) -> None:
        """Run legacy shared-proportions generation (all splits share config).

        Uses top-level ``generation_mode``, ``target_proportions_path``,
        and a single ``dmr_sampling`` list that applies to all splits.
        """
        generation_mode = self.config["generation_mode"]
        shared_kwargs = self._build_shared_kwargs(splits_data, generate_uxm_in_ios)

        # Resolve dmr_sampling variants from config (default: ["uniform"])
        dmr_sampling = self.config.get("dmr_sampling", ["uniform"])
        if isinstance(dmr_sampling, str):
            dmr_sampling = [dmr_sampling]

        if generation_mode == "random":
            allowed_labels = self.config.get("allowed_labels")
            if allowed_labels is None:
                allowed_labels = list(range(self.num_labels))

            ios, exceptions = run_ios_generation_parallel(
                n_io_examples=self.config.get("n_io_examples", 30000),
                allowed_labels=allowed_labels,
                n_cells_max=self.config.get("n_cells_max", 10),
                dmr_sampling_variants=dmr_sampling,
                **shared_kwargs,
            )
        elif generation_mode == "target_proportions":
            self.logger.info(
                f"Using provided target proportions for generation "
                f"with start checkpoint index of "
                f"{shared_kwargs['start_checkpoint_idx']}"
            )
            target_proportions_path = self.config["target_proportions_path"]
            target_proportions = np.load(target_proportions_path)
            target_proportions = target_proportions["proportions"]
            ios, exceptions = run_ios_generation_parallel(
                target_proportions=target_proportions,
                dmr_sampling_variants=dmr_sampling,
                **shared_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown generation_mode: '{generation_mode}'. "
                "Must be 'random' or 'target_proportions'."
            )

        self._log_generation_exceptions(exceptions)

    def _run_per_split_generation(
        self,
        splits_data: Dict[str, pd.DataFrame],
        split_generation: Dict[str, Any],
        generate_uxm_in_ios: bool,
    ) -> None:
        """Run per-split generation with independent configs per split.

        Each split entry in ``split_generation`` may specify:

        - ``target_proportions_path`` — path to a ``.npz`` file with a
          ``proportions`` key.
        - ``dmr_sampling`` — list of strategies (``["uniform"]``,
          ``["random"]``, or ``["uniform", "random"]``).
          Defaults to ``["uniform"]``.
        """
        for split_name, split_cfg in split_generation.items():
            if split_name not in splits_data:
                self.logger.warning(
                    f"  Split '{split_name}' in split_generation but not "
                    f"in loaded splits — skipping."
                )
                continue

            self.logger.info(f"  Generating IO examples for split '{split_name}' ...")

            # Per-split DMR sampling variants
            dmr_sampling = split_cfg.get("dmr_sampling", ["uniform"])
            if isinstance(dmr_sampling, str):
                dmr_sampling = [dmr_sampling]
            self.logger.info(f"    DMR sampling variants: {dmr_sampling}")

            # Load per-split target proportions
            tp_path = split_cfg.get("target_proportions_path")
            if tp_path is not None:
                tp_data = np.load(tp_path)
                target_proportions = tp_data["proportions"]
                self.logger.info(
                    f"    Target proportions: {len(target_proportions)} "
                    f"vectors from {tp_path}"
                )
            else:
                target_proportions = None

            # Build per-split kwargs — pass only this split's data
            split_ios_dir = os.path.join(self.ios_dir, split_name)
            os.makedirs(split_ios_dir, exist_ok=True)
            ios_base_path = os.path.join(split_ios_dir, "ios_deconvolution_data.pkl")

            shared_kwargs = dict(
                splits={split_name: splits_data[split_name]},
                file_name=ios_base_path,
                n_workers=self.config.get("n_workers"),
                batch_size=self.config.get("batch_size", 100),
                checkpoint_interval=self.config.get("checkpoint_interval", 1000),
                start_checkpoint_idx=split_cfg.get(
                    "start_checkpoint_idx",
                    self.config.get("start_checkpoint_idx", 0),
                ),
                n_read_per_split=self.config.get("n_read_per_split"),
                num_labels=self.num_labels,
                generate_uxm_inputs=generate_uxm_in_ios,
            )

            if target_proportions is not None:
                ios, exceptions = run_ios_generation_parallel(
                    target_proportions=target_proportions,
                    dmr_sampling_variants=dmr_sampling,
                    **shared_kwargs,
                )
            else:
                # Random mode for this split
                allowed_labels = split_cfg.get(
                    "allowed_labels",
                    self.config.get("allowed_labels"),
                )
                if allowed_labels is None:
                    allowed_labels = list(range(self.num_labels))

                ios, exceptions = run_ios_generation_parallel(
                    n_io_examples=split_cfg.get(
                        "n_io_examples",
                        self.config.get("n_io_examples", 30000),
                    ),
                    allowed_labels=allowed_labels,
                    n_cells_max=split_cfg.get(
                        "n_cells_max",
                        self.config.get("n_cells_max", 10),
                    ),
                    dmr_sampling_variants=dmr_sampling,
                    **shared_kwargs,
                )

            self._log_generation_exceptions(exceptions)

    def _log_generation_exceptions(self, exceptions: List[Any]) -> None:
        """Log a summary of generation exceptions."""
        if exceptions:
            self.logger.warning(
                f"  {len(exceptions)} example(s) failed during generation."
            )
            for lbl, prop, err in exceptions[:5]:
                self.logger.warning(f"    labels={lbl}, proportions={prop}: {err}")

    # ═══════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _load_splits(
        self,
    ) -> Dict[str, pd.DataFrame]:
        """Load configured splits from parquet or pickle."""
        input_type = self.config["input_type"]
        splits_cfg = self.config.get("splits", ["train", "valid", "test"])
        splits_data = {}

        if input_type == "raw_splits":
            data_path = self.config["data_path"]
            for split_name in splits_cfg:
                splits_data[split_name] = pd.read_parquet(
                    os.path.join(data_path, f"{split_name}.parquet")
                )
        elif input_type == "pre_predicted":
            pickle_paths = self.config["pickle_paths"]
            for split_name in splits_cfg:
                with open(pickle_paths[split_name], "rb") as f:
                    splits_data[split_name] = pickle.load(f)

        else:
            raise ValueError(
                f"Unknown input_type: '{input_type}'. " "Must be 'parquet' or 'pickle'."
            )

        return splits_data

    def _build_classifier_adapter(self) -> ClassifierAdapter:
        """Build a ClassifierAdapter from config."""
        classifier_cfg = self.config.get("classifier_config", {})

        return ClassifierAdapter(
            classifier_type=self.config["classifier_type"],
            checkpoint_path=self.config["classifier_checkpoint"],
            labels_dict=self.labels_dict,
            num_labels=self.num_labels,
            seq_length=classifier_cfg.get("seq_length", 150),
            foundation_model_path=classifier_cfg.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            classifier_head_implementation=classifier_cfg.get(
                "classifier_head_implementation", "dmr_attention_based"
            ),
            dmr_label_column=classifier_cfg.get("dmr_label_column", "dmr_ctype_label"),
            soft_labels=classifier_cfg.get("soft_labels", True),
            dismir_flavor=classifier_cfg.get("dismir_flavor", "lstm"),
            batch_size=classifier_cfg.get("batch_size", 2200),
        )
