"""
Inference Pipeline

End-to-end pipeline for processing BAM files (or pre-processed reads),
running classifier predictions (MethylBERT, Dismir, CancerDetector,
or LookupClassifier), and performing deconvolution using multiple
methods simultaneously.
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch

import numpy as np
import pandas as pd
from baselines.deconvolution.uxm import mark_records_methyl_state

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


from syto.data import LOYFER_CELL_TYPE_MATCH_DICT
from syto.data.sequencing.bam_processing import process_bam_with_chunking
from syto.classification.prediction_aggregation import (
    aggregate_predictions_by_grg,
)
from syto.classification.classifiers.lazy_classifier_factory import (
    read_classifier_factory,
)
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.deconvolution.feature_selection import apply_feature_mask
from syto.deconvolution.least_squares_deconvolvers import (
    PSLSDeconvolver,
    NNLSDeconvolver,
)
from syto.deconvolution.deep_deconvolvers.mlp import MLPDeconvolver
from syto.deconvolution.deep_deconvolvers.swn import SWNDeconvolver
from syto.deconvolution.xgbdeconvolver import (
    XGBoostDeconvolver,
)
from syto.calibration.linear_calibrator import LinearCalibrator
from syto.calibration.vector_scaling_calibrator import VectorScalingCalibrator

from syto.data.dataset import resolve_column

LINEAR_NORM_METHODS = ["clip-normalize", "simplex-projection"]


class InferencePipeline:
    """
    Orchestrates the full inference pipeline.

    Stages:
        1. BAM → processed reads (or load pre-processed)
        2. Reads x atlas → region-overlapped, annotated reads
        3. Annotated reads → classifier predictions
           (MethylBERT / Dismir / CancerDetector / LookupClassifier)
        4. Read-level predictions → DMR-level aggregation
        5. DMR-level predictions → deconvolution proportions

    Parameters
    ----------
    config : dict
        Parsed inference configuration (from inference.yaml).
    logger : logging.Logger
        Logger instance for status messages.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # ── Load labels dictionary ──────────────────────────────────────
        labels_dict_path = config["labels_dict_path"]
        with open(labels_dict_path, "r", encoding="utf-8") as f:
            # JSON keys are strings; convert to {int: str}
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.labels_dict_reversed: Dict[str, int] = {
            v: k for k, v in self.labels_dict.items()
        }
        self.logger.info(
            f"Loaded labels dictionary with {len(self.labels_dict)} cell types"
        )

        self.cell_type_match_dict = LOYFER_CELL_TYPE_MATCH_DICT

        # ── Load atlas ──────────────────────────────────────────────────
        atlas_path = config["atlas_path"]
        atlas_name = config.get("atlas_name", Path(atlas_path).stem)
        self.atlas = UXMMethylationAtlas(
            atlas_name=atlas_name,
            reference_genome="hg38" if "hg38" in atlas_path else "hg19",
            atlas_path=atlas_path,
            sep="\t",
        )
        self.logger.info(
            f"Loaded atlas with {len(self.atlas.atlas)} regions from {atlas_path}"
        )

        # ── Load baseline atlas states ───────────────────────────────
        self._baseline_states = self._load_baseline_states()

        # ── Placeholder attributes populated during run() ───────────────
        self.processed_reads: Optional[pd.DataFrame] = None
        self.prepared_reads: Optional[pd.DataFrame] = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.dmr_aggregated: Optional[pd.DataFrame] = None
        self.deconvolution_results: Dict[str, Any] = {}

        # By default the algorithm assumes that we have at least some data for each DMR group.
        self.fill_in_missing_labels = self.config.get("fill_in_missing_labels", False)

        # ── Missing-label substitution strategy ──────────────────────────
        self.missing_label_strategy = self.config.get(
            "missing_label_strategy", "zeroes"
        )
        self.prior_weight = float(self.config.get("prior_weight", 1.0))
        self.uniform_prior: Optional[pd.DataFrame] = None

        if self.missing_label_strategy != "zeroes":
            self.uniform_prior = self._load_or_compute_uniform_prior()

        # Resolve classifier type early so it's available even when
        # classification is skipped (e.g. predicted_reads input).
        classifier_cfg = config.get("classifier", config.get("model", {}))
        self.classifier_type = classifier_cfg.get("classifier_type", "methylbert")

        if self.config.get("num_labels", None) is None:
            self.num_labels = len(self.labels_dict)
        else:
            self.num_labels = self.config["num_labels"]
        deconv_cfg = self.config.get("deconvolution", {})
        self.syto_methods_enabled = deconv_cfg.get("syto", False)
        if self.syto_methods_enabled:
            self.features_mask = np.load(config["features_mask_path"])["features_mask"]
            self.input_length = int(np.sum(self.features_mask))


    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def run(self) -> List[Tuple[str, str, np.ndarray]]:
        """Execute the full inference pipeline end-to-end."""
        # pylint: disable=attribute-defined-outside-init
        self.skip_classification = False
        self.skip_reads_processing = False
        self.skip_aggregation_for_syto = False
        self.logger.info(f"The config doesn't feature syto methods --> related classification and feature extraction methods will be skipped")

        if not self.syto_methods_enabled:
            self.skip_classification = True
            self.skip_aggregation_for_syto = True

        # ── Stage 1: obtain processed reads ─────────────────────────────
        input_cfg = self.config["input"]
        if input_cfg["type"] == "bam":
            self.file_name = Path(input_cfg["bam_path"]).name
            self.processed_reads = self._process_bam()
        elif input_cfg["type"] == "parsed_reads":
            self.file_name = Path(input_cfg["parsed_reads_path"]).name
            self.processed_reads = self._load_parsed_reads()
        elif input_cfg["type"] == "predicted_reads":
            self.file_name = Path(input_cfg["predicted_reads_path"]).name
            self.predictions_df = self._load_reads_with_predictions()
            self.prepared_reads = self.predictions_df  # For UXM to work
            self.logger.info(
                f"Classified reads were provided: {len(self.predictions_df)} processed reads. Proceeding with deconvolution."
            )
            self.skip_classification = True
            self.skip_reads_processing = True
        else:
            raise ValueError(
                f"Unknown input type: {input_cfg['type']}. "
                "Must be 'bam' or 'parsed_reads' or 'predicted_reads'."
            )
        if not self.processed_reads is None:
            if not self.skip_reads_processing:
                self.logger.info(
                    f"Stage 1 complete: {len(self.processed_reads)} processed reads"
                )

                # ── Stage 2: overlap reads with atlas regions ───────────────────
                self.prepared_reads = self._prepare_reads()
                self.logger.info(
                    f"Stage 2 complete: {len(self.prepared_reads)} atlas-overlapped reads"
                )

                # ── Stage 3: classifier predictions ─────────────────────────────
                if not self.skip_classification:
                    self.predictions_df = self._predict_classifier()
                    self.logger.info(
                        f"Stage 3 complete: predictions for {len(self.predictions_df)} reads"
                    )

        # ── Stage 4: aggregate to DMR level ────────────────────────────
        if not self.skip_aggregation_for_syto:
            self.dmr_aggregated = self._aggregate_to_dmr()
            self.logger.info(
                f"Stage 4 complete: {len(self.dmr_aggregated)} DMR-level aggregations"
            )
        else: 
             self.logger.info(
                f"Stage 4 is skipped as no prediction aggregation is required for baseline methods"
            )
        # ── Stage 5: deconvolution ─────────────────────────────────────
        self.deconvolution_results = self._run_deconvolution()
        self.logger.info(
            f"Stage 5 complete: ran {len(self.deconvolution_results)} deconvolution methods"
        )
        # ── Save results ────────────────────────────────────────────────
        self._save_results()
        return self.deconvolution_results

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 1: BAM processing / loading pre-processed reads
    # ═══════════════════════════════════════════════════════════════════

    def _process_bam(self) -> pd.DataFrame:
        """Process a BAM file into a read-level DataFrame."""
        input_cfg = self.config["input"]
        bam_cfg = self.config.get("bam_processing", {})

        bam_path = input_cfg["bam_path"]
        reference_path = input_cfg.get("reference_path")
        data_type = input_cfg.get("data_type")

        # Resolve chromosomes
        chromosomes = input_cfg.get("chromosomes", "all")
        if chromosomes == "all":
            chromosomes = [f"chr{i}" for i in range(1, 23)]
            chromosomes += ["chrX"]
            chromosomes += ["chrY"]

        self.logger.info(f"Processing BAM: {bam_path} ({len(chromosomes)} chromosomes)")

        if data_type == "wgbs":
            require_flags = bam_cfg.get("require_flags", 3)
        elif data_type == "ont":
            self.logger.info(
                f"ont is selected as data_type. The exclude_flags will not be used and require_flags is ovveriden to 0."
            )
            require_flags = 0
        else:
            raise ValueError(
                "The pipeline only supports BAMS originating from WGBS or ONT"
            )

        df = process_bam_with_chunking(
            bam_path=bam_path,
            chromosomes=chromosomes,
            methyl_tr=bam_cfg.get("ont_methyl_tr", 122),
            n_jobs=bam_cfg.get("n_jobs", 4),
            reference_path=reference_path,
            data_type=data_type,
            min_mapq=bam_cfg.get("min_mapq", 10),
            require_flags=require_flags,
            exclude_flags=bam_cfg.get("exclude_flags", 1796),
            min_cpgs=bam_cfg.get("min_cpgs", 1),
            merge_pairs=bam_cfg.get("merge_pairs", True),
        )
        if not len(df):
            self.logger.warning(
                "The dataset has 0 reads after applying all samtools filters. "
                "The attempt will be made to reparse .bam without applying flag filters"
            )
            df = process_bam_with_chunking(
                bam_path=bam_path,
                chromosomes=chromosomes,
                methyl_tr=bam_cfg.get("ont_methyl_tr", 122),
                n_jobs=bam_cfg.get("n_jobs", 4),
                reference_path=reference_path,
                data_type=data_type,
                min_mapq=bam_cfg.get("min_mapq", 10),
                require_flags=None,
                exclude_flags=None,
                min_cpgs=bam_cfg.get("min_cpgs", 1),
                merge_pairs=bam_cfg.get("merge_pairs", True),
            )
        if not len(df):
            self.logger.warning(
                "Setting exclude_flags=None and require_flags=None didn't help. "
                "The processing of this file will be terminated."
            )
            return None

        return df

    def _load_parsed_reads(self) -> pd.DataFrame:
        """Load pre-parsed reads from a pickle file."""
        path = self.config["input"]["parsed_reads_path"]
        self.logger.info(f"Loading pre-parsed reads from {path}")
        if ".csv" in path:
            df = pd.read_csv(path, sep="\t")
            df.rename(columns={"ref_name": "chromosome"}, inplace=True)
            df.rename(columns={"ref_pos": "read_start"}, inplace=True)
            df.rename(columns={"methyl_seq": "methylation_encoding"}, inplace=True)
            df.rename(columns={"original_seq": "seq"}, inplace=True)
            df["read_end"] = df["read_start"] + df["seq"].apply(len)
            df["read_name"] = range(len(df))
            # TODO: temporary set here to avoid eval loop crashing the predict.
            # MUST FIX IN THE FUTURE IN THE EVAL LOOP!!!
            df["label"] = 0
        else:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Expected a pandas DataFrame in {path}, got {type(df)}"
                )
        return df

    def _load_reads_with_predictions(self) -> pd.DataFrame:
        """Load reads augmented with read-level predictions from a pickle file."""
        path = self.config["input"]["predicted_reads_path"]
        self.logger.info(f"Loading reads with predictions from {path}")
        with open(path, "rb") as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in {path}, got {type(df)}")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 2: overlap reads with atlas regions
    # ═══════════════════════════════════════════════════════════════════

    def _prepare_reads(self) -> pd.DataFrame:
        """Overlap processed reads with atlas regions and resolve DMR labels.

        Calls ``atlas.prepare_reads`` which:
        - overlaps reads with atlas regions (adds ``name``, region coords)
        - trims reads to region boundaries (coordinates, seq, pattern)
        - resolves ``dmr_ctype_label`` via labels_dict and cell_type_match_dict
        """
        df = self.processed_reads.copy()
        df = df.sort_values(["chromosome", "read_start"]).reset_index(drop=True)

        self.logger.info("Overlapping reads with atlas regions ...")
        prepared = self.atlas.prepare_reads(
            df,
            trim=True,
            labels_dict=self.labels_dict,
            cell_type_match_dict=self.cell_type_match_dict,
        )

        prepared = mark_records_methyl_state(prepared)

        if len(prepared) == 0:
            raise RuntimeError(
                "No reads overlapped with atlas regions. "
                "Check that chromosome naming is consistent between BAM and atlas."
            )

        return prepared

    def _predict_classifier(self) -> pd.DataFrame:
        """
        Run the configured classifier on the prepared reads.

        Returns
        -------
        pd.DataFrame
            The prepared_reads DataFrame augmented with ``prediction_*``
            columns (one per cell type).
        """
        classifier_cfg = self.config.get("classifier", self.config.get("model", {}))
        self.classifier_type = classifier_cfg.get("classifier_type")
        read_classifier = read_classifier_factory(
            name=self.classifier_type,
            path=self.config["checkpoint_path"],
            labels_dict=self.labels_dict,
            num_labels=self.num_labels,
            seq_length=self.config.get("max_sequence_length", 150),
            foundation_model_path=classifier_cfg.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            classifier_head_implementation=classifier_cfg.get(
                "classifier_head_implementation", "grg_attention_based"
            ),
            grg_label_column=classifier_cfg.get("grg_label_column", "dmr_ctype_label"),
            dismir_flavor=classifier_cfg.get("dismir_flavor", "lstm"),
            cancer_detector_prior_type=classifier_cfg.get(
                "cancer_detector_prior_type", "uniform"
            ),
            soft_labels=classifier_cfg.get("soft_labels", False),
            batch_size=self.config.get("prediction_batch_size", 2200),
        )

        self.logger.info("Running classifier predictions ...")

        result_df = read_classifier.predict_split(self.prepared_reads, **classifier_cfg)
        result_df = result_df.dropna(
            subset=result_df.columns.difference(["soft_label", "ctype"])
        )
        if "M_rate" in result_df.columns:
            result_df.rename(columns={"M_rate": "methylation_level"}, inplace=True)

        return result_df

    def _load_or_compute_uniform_prior(self) -> pd.DataFrame:
        """Load the uniform prior matrix from the pseudobulk HDF5 file.

        Reads ``outputs/{split}/pure_profiles/uniform_prior`` from the HDF5
        produced by the pseudobulk generation pipeline and returns it as a
        DataFrame with a ``dmr_ctype_label`` column, ready for
        :func:`aggregate_predictions_by_grg`.

        Config keys
        -----------
        pseudobulk_h5_path : str
            Path to the pseudobulk HDF5 file containing pure profiles.
        pure_profiles_split : str, optional
            Split name from which to read the prior (default ``"train"``).

        Raises
        ------
        ValueError
            If ``pseudobulk_h5_path`` is not configured.
        KeyError
            If the HDF5 file has no pure profiles for the requested split.
        """
        from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader

        h5_path = self.config.get("pseudobulk_h5_path")
        if not h5_path:
            raise ValueError(
                f"missing_label_strategy='{self.missing_label_strategy}' requires "
                "'pseudobulk_h5_path' in config pointing to a pseudobulk HDF5 file "
                "with pre-computed pure profiles."
            )

        split = self.config.get("pure_profiles_split", "train")
        self.logger.info(
            "Loading uniform prior from pseudobulk HDF5 (split=%r): %s", split, h5_path
        )
        reader = PseudobulkHDF5Reader(h5_path, logger=self.logger)
        return reader.read_uniform_prior(split)

    # ═════════════════════════════════════════════════════════════════
    #  Stage 4: aggregate predictions by DMR
    # ═════════════════════════════════════════════════════════════════

    def _aggregate_to_dmr(self) -> pd.DataFrame:
        """
        Aggregate read-level predictions to DMR level using
        ``aggregate_predictions_by_grg`` from edautils.
        """
        self.logger.info("Aggregating predictions by DMR ...")

        aggregated = aggregate_predictions_by_grg(
            df=self.predictions_df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            weight_col="NCPGS",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=self.fill_in_missing_labels,
            labels_dict=self.labels_dict,
            substitution_strategy=self.missing_label_strategy,
            uniform_prior=self.uniform_prior,
            prior_weight=self.prior_weight,
        )

        return aggregated

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 5: deconvolution
    # ═══════════════════════════════════════════════════════════════════

    def _run_deconvolution(self) -> List[Tuple[str, str, np.ndarray]]:
        """
        Execute all enabled deconvolution methods and return results.

        For each deconvolution method, if a ``calibrators_dir`` is specified,
        all calibrator files found in that directory are automatically loaded
        and applied to produce additional calibrated result entries.

        Returns
        -------
        list of (deconvolver, calibrator, proportions)
            Each element is a tuple of (deconvolver name, calibrator name
            or "None", proportions array).
        """
        deconv_cfg = self.config.get("deconvolution", {})
        results: List[Tuple[str, str, np.ndarray]] = []

        # ── Syto feature-based methods ──────────────────────────────────
        for method_cfg in deconv_cfg.get("syto", []):
            if not method_cfg.get("enabled", False):
                continue

            name = method_cfg["name"]
            self.logger.info(f"Running deconvolution method: {name}")

            try:
                base_name = name
                if name == "ls":
                    base_name = method_cfg["flavor"]

                if name == "xgboost":
                    proportions = self._run_xgboost_deconvolution(method_cfg)
                elif name in ["3Layer_MLP", "Shallow_Wide_Network"]:
                    proportions = self._run_nn_deconvolution(method_cfg)
                elif name == "ls":
                    proportions = self._run_ls_deconvolution(method_cfg)
                else:
                    self.logger.warning(
                        f"Unknown deconvolution method: {name}, skipping"
                    )
                    continue

                results.append((base_name, "None", proportions))

                calibrators_dir = method_cfg.get("calibrators_dir", None)
                if calibrators_dir is None and method_cfg.get("use_callibration", False):
                    calibrators_dir = str(Path(method_cfg["callibrator_path"]).parent)
                if calibrators_dir is not None:
                    results.extend(
                        self._apply_all_calibrators(proportions, calibrators_dir, base_name)
                    )

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error(
                    f"Deconvolution method '{name}' failed: {e}", exc_info=True
                )

        # ── Read-based baseline methods ─────────────────────────────────
        for baseline_state in self._baseline_states:
            model = baseline_state["name"]
            self.logger.info(f"Running baseline deconvolution: {model}")

            try:
                if model == "uxm":
                    proportions = self._run_uxm_deconvolution(baseline_state)
                    if proportions is not None:
                        results.append(("uxm", "None", proportions))

                elif model == "celfieish":
                    results.extend(self._run_celfieish_baseline(baseline_state))

                elif model == "celfie":
                    results.extend(self._run_celfie_baseline(baseline_state))

                else:
                    self.logger.warning(f"Unknown baseline model: {model}, skipping")

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error(
                    f"Baseline method '{model}' failed: {e}", exc_info=True
                )

        return results

    def _apply_all_calibrators(
        self,
        proportions: np.ndarray,
        calibrators_dir: str,
        base_name: str,
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Discover and apply all calibrator files in *calibrators_dir*.

        Parameters
        ----------
        proportions : np.ndarray
            Raw (uncalibrated) deconvolution proportions.  May be 1-D
            (single sample) or 2-D.
        calibrators_dir : str
            Path to the directory that contains calibrator ``.npz`` files
            produced by :class:`CalibratorFittingPipeline`.
        base_name : str
            Deconvolver name, e.g. ``"nnls"``.

        Returns
        -------
        list of (deconvolver, calibrator, proportions)
            Each element is a tuple of (deconvolver name, calibrator
            description, calibrated proportions array).
        """
        results: List[Tuple[str, str, np.ndarray]] = []
        calibrators_dir = Path(calibrators_dir)

        if not calibrators_dir.is_dir():
            self.logger.warning(
                f"Calibrators directory does not exist: {calibrators_dir}"
            )
            return results

        # Ensure 2-D input for the calibrators
        props_2d = proportions
        if props_2d.ndim == 1:
            props_2d = np.expand_dims(props_2d, 0)

        # ── Linear calibrator ──────────────────────────────────────
        linear_path = calibrators_dir / "linear_calibrator.npz"
        if linear_path.exists():
            self.logger.info(f"  Loading linear calibrator from {linear_path}")
            linear_cal = LinearCalibrator.load(linear_path)

            for norm_method in LINEAR_NORM_METHODS:
                short_name = norm_method.replace("-", "_")
                calibrator_label = f"linear_{short_name}"
                try:
                    calib, _ = linear_cal.predict(props_2d, norm_method=norm_method)
                    results.append((base_name, calibrator_label, np.round(calib, 4)))
                    self.logger.info(f"    ✓ {base_name} + {calibrator_label}")
                except Exception as e:
                    self.logger.error(
                        f"    ✗ {base_name} + {calibrator_label} failed: {e}",
                        exc_info=True,
                    )

        # ── Vector-scaling calibrator ──────────────────────────────
        vs_path = calibrators_dir / "vector_scaling_calibrator.joblib"
        if vs_path.exists():
            calibrator_label = "vector_scaling"
            self.logger.info(f"  Loading vector-scaling calibrator from {vs_path}")
            try:
                vs_cal = VectorScalingCalibrator.load(vs_path)
                calib = vs_cal.predict(props_2d)
                results.append((base_name, calibrator_label, np.round(calib, 4)))
                self.logger.info(f"    ✓ {base_name} + {calibrator_label}")
            except Exception as e:
                self.logger.error(
                    f"    ✗ {base_name} + {calibrator_label} failed: {e}",
                    exc_info=True,
                )

        if not results:
            self.logger.warning(f"  No calibrator files found in {calibrators_dir}")

        return results

    def _run_ls_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run LS based deconvolution on the aggregated predictions selected feature matrices.

        Note: calibration is now handled centrally by ``_apply_all_calibrators``
        via the ``calibrators_dir`` config key.  The legacy inline calibration
        block has been removed.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        flavor = method_cfg["flavor"]
        self.logger.info(f"Loading {flavor} from {checkpoint_path}")
        X = apply_feature_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )
        if "nnls" in flavor:
            deconvolver = NNLSDeconvolver.load(checkpoint_path)
            proportions, _, _ = deconvolver.predict_single_sample(X)
        elif "psls" in flavor:
            deconvolver = PSLSDeconvolver.load(checkpoint_path)
            proportions = deconvolver.predict_single_sample(X)
        else:
            raise ValueError(
                "LS family of deconvolvers supports only two flavors: nnls and psls"
            )

        proportions = np.round(proportions, 4)
        self.logger.debug(f"{flavor} proportions: {proportions}")

        return proportions

    def _run_xgboost_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run XGBoost-based deconvolution.

        Loads a pre-trained ``XGBoostDeconvolver`` from checkpoint and
        runs prediction on the GR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        self.logger.info(f"Loading XGBoostDeconvolver from {checkpoint_path}")

        deconvolver = XGBoostDeconvolver.load(checkpoint_path)

        # Build the prediction matrix from GR-aggregated data
        X = apply_feature_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )[np.newaxis]
        deconv_preds = deconvolver.predict(X)
        proportions = np.round(deconv_preds, 4)
        self.logger.debug(f"XGBoost proportions: {proportions}")

        return proportions

    def _run_nn_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run torch-nn-based deconvolution.

        Loads a pre-trained model of specified architectur from checkpoint and
        runs prediction on the GR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        architecture = method_cfg["name"]
        self.logger.info(f"Loading {architecture} from {checkpoint_path}")

        if architecture == "Shallow_Wide_Network":
            deconvolver = SWNDeconvolver.load(checkpoint_path)
        elif architecture == "3Layer_MLP":
            deconvolver = MLPDeconvolver.load(checkpoint_path)
        else:
            raise ValueError(
                "Architecture for NN method should be either Shallow_Wide_Network or 3Layer_MLP"
            )
        X = torch.FloatTensor(
            apply_feature_mask(
                np.array(
                    self.dmr_aggregated[
                        [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                    ]
                ),
                self.features_mask,
            )[np.newaxis]
        ).to("cuda")

        deconv_preds = deconvolver.predict(X)
        proportions = np.round(deconv_preds.to("cpu").detach().numpy(), 4)
        self.logger.debug(f"{architecture} proportions: {proportions}")

        return proportions

    def _load_baseline_states(self) -> List[Dict[str, Any]]:
        """Load atlas objects for all baselines listed under deconvolution.baselines."""
        baselines_cfg = self.config.get("deconvolution", {}).get("baselines", [])
        states: List[Dict[str, Any]] = []

        for cfg in baselines_cfg:
            if not cfg.get("enabled", True):
                continue
            model = cfg["model"]

            if model == "uxm":
                from baselines.deconvolution.uxm.uxm import load_atlas

                atlas_df, ref_cells_all = load_atlas(cfg["atlas_path"])
                ignore_cells = cfg.get("ignore_cells", [])
                ref_cells = [c for c in ref_cells_all if c not in ignore_cells]
                self.logger.info("UXM baseline: %d reference cell types", len(ref_cells))
                states.append({"name": "uxm", "atlas_df": atlas_df, "ref_cells": ref_cells})

            elif model in ("celfieish", "celfie"):
                from syto.data.atlases.celfieish_atlases import CpGBetaCountsMethylationAtlas

                atlas_path = cfg["atlas_path"]
                atlas = CpGBetaCountsMethylationAtlas(
                    atlas_name=cfg.get("atlas_name", Path(atlas_path).stem),
                    reference_genome=cfg.get("reference_genome", "hg38"),
                    atlas_path=atlas_path,
                )
                state: Dict[str, Any] = {
                    "name": model,
                    "atlas": atlas,
                    "ref_cells": atlas.ref_cells,
                    "em_checkpoints": cfg.get("em_checkpoints"),
                    "num_iterations": cfg.get("num_iterations", 50),
                    "convergence_criteria": cfg.get("convergence_criteria", 0.001),
                }
                if model == "celfie":
                    state["random_restarts"] = cfg.get("random_restarts", 1)
                self.logger.info(
                    "%s baseline: %d reference cell types", model.upper(), len(atlas.ref_cells)
                )
                states.append(state)

            else:
                self.logger.warning("Unknown baseline model %r in config; skipping.", model)

        return states

    def _run_uxm_deconvolution(self, baseline_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Run UXM baseline deconvolution using the pre-loaded atlas state."""
        from baselines.deconvolution.uxm.uxm import run_uxm_deconvolution

        aligned = run_uxm_deconvolution(
            self.prepared_reads,
            baseline_state["atlas_df"],
            baseline_state["ref_cells"],
            self.labels_dict_reversed,
            n_labels=self.num_labels,
        )
        if aligned is None:
            self.logger.warning("UXM: no overlapping reads, skipping.")
            return None
        proportions = np.round(np.array(aligned), 4)
        self.logger.debug("UXM proportions: %s", proportions)
        return proportions

    def _run_celfieish_baseline(
        self, baseline_state: Dict[str, Any]
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Run CelFiE-ISH baseline deconvolution; returns result tuples ready for results list."""
        from baselines.deconvolution.celfieish.celfieish import run_celfieish_deconvolution

        if self.processed_reads is None:
            self.logger.warning(
                "CelFiE-ISH: requires processed_reads (not available for predicted_reads input); skipping."
            )
            return []

        em_checkpoints = baseline_state.get("em_checkpoints")
        result = run_celfieish_deconvolution(
            self.processed_reads,
            baseline_state["atlas"],
            self.labels_dict_reversed,
            n_labels=self.num_labels,
            prepare_reads=True,
            num_iterations=baseline_state.get("num_iterations", 50),
            convergence_criteria=baseline_state.get("convergence_criteria", 0.001),
            checkpoints=em_checkpoints,
        )
        if result is None:
            self.logger.warning("CelFiE-ISH: no overlapping reads, skipping.")
            return []

        if em_checkpoints is not None:
            out = []
            for n_steps, aligned in result:
                name = f"celfieish_{n_steps}_steps"
                out.append((name, "None", np.round(np.array(aligned), 4)))
            return out

        self.logger.debug("CelFiE-ISH proportions: %s", result)
        return [("celfieish", "None", np.round(np.array(result), 4))]

    def _run_celfie_baseline(
        self, baseline_state: Dict[str, Any]
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Run CelFiE baseline deconvolution; returns result tuples ready for results list."""
        from baselines.deconvolution.celfie.celfie import run_celfie_deconvolution

        if self.processed_reads is None:
            self.logger.warning(
                "CelFiE: requires processed_reads (not available for predicted_reads input); skipping."
            )
            return []

        em_checkpoints = baseline_state.get("em_checkpoints")
        result = run_celfie_deconvolution(
            self.processed_reads,
            baseline_state["atlas"],
            self.labels_dict_reversed,
            n_labels=self.num_labels,
            prepare_reads=True,
            num_iterations=baseline_state.get("num_iterations", 50),
            convergence_criteria=baseline_state.get("convergence_criteria", 0.001),
            random_restarts=baseline_state.get("random_restarts", 1),
            checkpoints=em_checkpoints,
        )
        if result is None:
            self.logger.warning("CelFiE: no overlapping reads, skipping.")
            return []

        if em_checkpoints is not None:
            out = []
            for n_steps, aligned in result:
                name = f"celfie_{n_steps}_steps"
                out.append((name, "None", np.round(np.array(aligned), 4)))
            return out

        self.logger.debug("CelFiE proportions: %s", result)
        return [("celfie", "None", np.round(np.array(result), 4))]

    # ═══════════════════════════════════════════════════════════════════
    #  Output
    # ═══════════════════════════════════════════════════════════════════

    def _save_results(self) -> None:
        """Save all pipeline outputs to the configured output directory."""
        output_cfg = self.config.get("output", {})
        output_dir = self.config.get("output_dir", "./inference_output")
        os.makedirs(output_dir, exist_ok=True)

        # ── Processed reads (pickle) ───────────────────────────────────
        if (
            output_cfg.get("save_processed_reads", True)
            and self.processed_reads is not None
        ):
            path = os.path.join(output_dir, "processed_reads.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.processed_reads, f)
            self.logger.info(f"Saved processed reads to {path}")

        # ── Predictions with MethylBERT scores (pickle) ────────────────
        if output_cfg.get("save_predictions", True) and self.predictions_df is not None:
            path = os.path.join(output_dir, "predictions.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.predictions_df, f)
            self.logger.info(f"Saved predictions to {path}")

        # ── Deconvolution proportions (single long-format CSV) ──────────
        if output_cfg.get("save_deconvolution", True) and self.deconvolution_results:
            classifier_name = getattr(self, "classifier_type", "unknown")
            file_name = getattr(self, "file_name", "unknown")
            cell_types = list(self.labels_dict.values())

            rows = []
            for deconvolver, calibrator, proportions in self.deconvolution_results:
                props_flat = proportions.flatten()
                for ct, prop in zip(cell_types, props_flat):
                    rows.append(
                        {
                            "FileName": file_name,
                            "CellType": ct,
                            "Classifier": classifier_name,
                            "Deconvolver": deconvolver,
                            "Calibrator": calibrator,
                            "PredictedProportion": prop,
                        }
                    )

            results_df = pd.DataFrame(rows)
            path = os.path.join(output_dir, "deconvolution_results.csv")
            results_df.to_csv(path, index=False)
            self.logger.info(
                f"Saved deconvolution results ({len(self.deconvolution_results)} "
                f"method/calibrator combinations) to {path}"
            )

        # ── GR-aggregated predictions (pickle) ────────────────────────
        if self.dmr_aggregated is not None:
            path = os.path.join(output_dir, "dmr_aggregated.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.dmr_aggregated, f)
            self.logger.info(f"Saved DMR aggregated predictions to {path}")
