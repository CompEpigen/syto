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
import torch.nn as nn
import torch

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from copy import deepcopy

from methyldl.data import LOYFER_CELL_TYPE_MATCH_DICT
from methyldl.data.sequencing.bam_processing import process_bam_with_chunking
from methyldl.modelling.prediction_aggregation import (
    aggregate_predictions_by_dmr,
)
from methyldl.modelling.abstract_read_classifier import ClassifierAdapter

from methyldl.deconvolution.uxm import (
    prepare_reads_for_uxm,
    uxm_deconvolution,
    rearange_uxm_deconvolution_results,
    load_atlas,
)
from methyldl.deconvolution.least_squares_deconvolvers import (
    PSLSDeconvolver,
    NNLSDeconvolver,
)

from methyldl.deconvolution.xgbdeconvolver import (
    XGBoostDeconvolver,
)

from methyldl.calibration.linear_calibrator import LinearCalibrator
from methyldl.cross_validation_engine import CrossValidationEngine

LINEAR_NORM_METHODS = ["clip0-normalize", "simplex-projection"]


class InferencePipeline:
    """
    Orchestrates the full inference pipeline.

    Stages:
        1. BAM → processed reads (or load pre-processed)
        2. Reads × atlas → region-overlapped, annotated reads
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
        self.atlas = pd.read_csv(atlas_path, sep="\t")
        self.logger.info(
            f"Loaded atlas with {len(self.atlas)} regions from {atlas_path}"
        )

        # ── Placeholder attributes populated during run() ───────────────
        self.processed_reads: Optional[pd.DataFrame] = None
        self.prepared_reads: Optional[pd.DataFrame] = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.dmr_aggregated: Optional[pd.DataFrame] = None
        self.deconvolution_results: Dict[str, Any] = {}
        self.features_mask = np.load(config["features_mask_path"])["features_mask"]

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

        self.input_length = int(np.sum(self.features_mask))

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def run(self) -> List[Tuple[str, str, np.ndarray]]:
        """Execute the full inference pipeline end-to-end."""
        # pylint: disable=attribute-defined-outside-init

        self.skip_classification = False
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
        else:
            raise ValueError(
                f"Unknown input type: {input_cfg['type']}. "
                "Must be 'bam' or 'parsed_reads' or 'predicted_reads'."
            )
        if not self.processed_reads is None:
            if not self.skip_classification:
                self.logger.info(
                    f"Stage 1 complete: {len(self.processed_reads)} processed reads"
                )

                # ── Stage 2: overlap reads with atlas regions ───────────────────
                self.prepared_reads = self._prepare_reads()
                self.logger.info(
                    f"Stage 2 complete: {len(self.prepared_reads)} atlas-overlapped reads"
                )

                # ── Stage 3: classifier predictions ─────────────────────────────
                self.predictions_df = self._predict_classifier()
                self.logger.info(
                    f"Stage 3 complete: predictions for {len(self.predictions_df)} reads"
                )

        # ── Stage 4: aggregate to DMR level ────────────────────────────
        self.dmr_aggregated = self._aggregate_to_dmr()
        self.logger.info(
            f"Stage 4 complete: {len(self.dmr_aggregated)} DMR-level aggregations"
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
        """
        Overlap processed reads with atlas regions and resolve DMR labels.

        Uses ``prepare_reads_for_uxm`` which:
        - iterates atlas regions and scans sorted reads for overlaps
        - trims reads to region boundaries
        - computes M / U / X counts
        - resolves ``dmr_ctype_label`` via labels_dict_reversed and
          cell_type_match_dict

        Returns
        -------
        pd.DataFrame
            Prepared reads with atlas-region annotations.  Column names
            are kept as-is (``seq``, ``pattern``, …) so that the
            :class:`ClassifierAdapter` can handle any downstream
            transformations internally.
        """
        df = self.processed_reads.copy()

        # Sort by chromosome and start position (required by prepare_reads_for_uxm)
        df = df.sort_values(["chromosome", "read_start"]).reset_index(drop=True)

        # For inference from BAM, reads don't have ground-truth labels.
        # Add dummy columns so prepare_reads_for_uxm doesn't break.
        if "original_label" not in df.columns:
            df["original_label"] = -1
        if "label" not in df.columns:
            df["label"] = -1

        self.logger.info("Overlapping reads with atlas regions ...")
        prepared_uxm = prepare_reads_for_uxm(
            reads_data=df,
            atlas=self.atlas,
            labels_dict=self.labels_dict,
            cell_type_match_dict=self.cell_type_match_dict,
        )

        if len(prepared_uxm) == 0:
            raise RuntimeError(
                "No reads overlapped with atlas regions. "
                "Check that chromosome naming is consistent between BAM and atlas."
            )

        return prepared_uxm

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 3: classifier predictions
    # ═══════════════════════════════════════════════════════════════════

    def _build_classifier_adapter(self) -> ClassifierAdapter:
        """Build a :class:`ClassifierAdapter` from the pipeline config.

        Reads the ``classifier`` (or legacy ``model``) section of the
        configuration to determine which classifier backend to use and
        how to initialise it.

        Returns
        -------
        ClassifierAdapter
        """
        classifier_cfg = self.config.get("classifier", self.config.get("model", {}))
        classifier_type = classifier_cfg.get("classifier_type", "methylbert")
        self.classifier_type = classifier_type
        checkpoint_path = self.config["checkpoint_path"]
        seq_len = self.config.get("max_sequence_length", 150)
        batch_size = self.config.get("prediction_batch_size", 2200)

        adapter = ClassifierAdapter(
            classifier_type=classifier_type,
            checkpoint_path=checkpoint_path,
            labels_dict=self.labels_dict,
            num_labels=self.num_labels,
            seq_length=seq_len,
            foundation_model_path=classifier_cfg.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            classifier_head_implementation=classifier_cfg.get(
                "classifier_head_implementation", "dmr_attention_based"
            ),
            dmr_label_column=classifier_cfg.get("dmr_label_column", "dmr_ctype_label"),
            dismir_flavor=classifier_cfg.get("dismir_flavor", "lstm"),
            cancer_detector_prior_type=classifier_cfg.get(
                "cancer_detector_prior_type", "uniform"
            ),
            soft_labels=classifier_cfg.get("soft_labels", False),
            batch_size=batch_size,
        )

        self.logger.info(
            f"Built ClassifierAdapter: type={classifier_type}, "
            f"checkpoint={checkpoint_path}"
        )
        return adapter

    def _predict_classifier(self) -> pd.DataFrame:
        """
        Run the configured classifier on the prepared reads.

        Delegates to :class:`ClassifierAdapter` which supports
        MethylBERT, Dismir, CancerDetector, and LookupClassifier
        through a unified ``predict_split`` interface.

        Returns
        -------
        pd.DataFrame
            The prepared_reads DataFrame augmented with ``prediction_*``
            columns (one per cell type).
        """
        adapter = self._build_classifier_adapter()

        self.logger.info("Running classifier predictions ...")
        result_df = adapter.predict_split(self.prepared_reads)

        result_df = result_df.dropna(
            subset=result_df.columns.difference(["soft_label"])
        )
        if "M_rate" in result_df.columns:
            result_df.rename(columns={"M_rate": "methylation_level"}, inplace=True)

        return result_df

    def _load_or_compute_uniform_prior(self) -> pd.DataFrame:
        """Load or compute the uniform prior matrix for missing-label substitution.

        Resolution order:
        1. ``uniform_prior_path`` — load from pre-computed ``.npz``.
        2. ``pure_profiles_path`` — load pure profiles ``.pkl``, compute the
           prior on-the-fly, and optionally cache it next to the profiles.

        Raises
        ------
        ValueError
            If neither path is configured.
        """
        from methyldl.data.pure_profile_generation import (
            compute_uniform_prior_matrix,
            load_uniform_prior,
            save_uniform_prior,
        )

        uniform_prior_path = self.config.get("uniform_prior_path", None)
        pure_profiles_path = self.config.get("pure_profiles_path", None)

        # Option 1: direct .npz
        if uniform_prior_path and os.path.exists(uniform_prior_path):
            self.logger.info(f"Loading uniform prior from {uniform_prior_path}")
            return load_uniform_prior(uniform_prior_path)

        # Option 2: compute from pure profiles pickle
        if pure_profiles_path and os.path.exists(pure_profiles_path):
            self.logger.info(
                f"Computing uniform prior from pure profiles: " f"{pure_profiles_path}"
            )
            with open(pure_profiles_path, "rb") as f:
                pure_profiles = pickle.load(f)

            split_key = self.config.get("pure_profiles_split_key", "train")
            prior = compute_uniform_prior_matrix(
                pure_profiles,
                split_key=split_key,
                num_input_labels=len(self.labels_dict),
            )

            # Cache for future runs
            cache_path = str(Path(pure_profiles_path).parent / "uniform_prior.npz")
            save_uniform_prior(prior, cache_path)
            self.logger.info(f"Cached uniform prior to {cache_path}")
            return prior

        raise ValueError(
            f"missing_label_strategy='{self.missing_label_strategy}' requires "
            f"either 'uniform_prior_path' or 'pure_profiles_path' in config."
        )

    # ═════════════════════════════════════════════════════════════════
    #  Stage 4: aggregate predictions by DMR
    # ═════════════════════════════════════════════════════════════════

    def _aggregate_to_dmr(self) -> pd.DataFrame:
        """
        Aggregate read-level predictions to DMR level using
        ``aggregate_predictions_by_dmr`` from edautils.
        """
        self.logger.info("Aggregating predictions by DMR ...")

        aggregated = aggregate_predictions_by_dmr(
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
        methods = deconv_cfg.get("methods", [])

        results: List[Tuple[str, str, np.ndarray]] = []

        for method_cfg in methods:
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
                elif name == "uxm":
                    proportions = self._run_uxm_deconvolution(method_cfg)
                elif name in ["3Layer_MLP", "Shallow_Wide_Network"]:
                    proportions = self._run_nn_deconvolution(method_cfg)
                elif name == "ls":
                    proportions = self._run_ls_deconvolution(method_cfg)
                else:
                    self.logger.warning(
                        f"Unknown deconvolution method: {name}, skipping"
                    )
                    continue

                # Uncalibrated result
                results.append((base_name, "None", proportions))

                # ── Apply all discovered calibrators ────────────────────
                calibrators_dir = method_cfg.get("calibrators_dir", None)
                # Legacy single-calibrator fallback
                if calibrators_dir is None and method_cfg.get(
                    "use_callibration", False
                ):
                    calibrators_dir = str(Path(method_cfg["callibrator_path"]).parent)

                if calibrators_dir is not None:
                    calibrated = self._apply_all_calibrators(
                        proportions, calibrators_dir, base_name
                    )
                    results.extend(calibrated)

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error(
                    f"Deconvolution method '{name}' failed: {e}", exc_info=True
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
                vs_cal = CrossValidationEngine.load(vs_path)
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
        X = self._extract_features_by_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )
        X = X.flatten()
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
        runs prediction on the DMR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        self.logger.info(f"Loading XGBoostDeconvolver from {checkpoint_path}")

        deconvolver = XGBoostDeconvolver.load(checkpoint_path)

        # Build the prediction matrix from DMR-aggregated data
        # prediction_matrix = self._build_prediction_matrix()
        X = self._extract_features_by_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )
        deconv_preds = deconvolver.predict(X)
        proportions = np.round(deconv_preds, 4)
        self.logger.debug(f"XGBoost proportions: {proportions}")

        return proportions

    def _run_nn_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run torch-nn-based deconvolution.

        Loads a pre-trained model of specified architectur from checkpoint and
        runs prediction on the DMR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        architecture = method_cfg["name"]
        self.logger.info(f"Loading {architecture} from {checkpoint_path}")

        if architecture == "Shallow_Wide_Network":
            #     deconvolver  = nn.Sequential(
            #     nn.Linear(78, 1024),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(1024, 39),
            #     nn.Softmax(dim=-1)
            # )
            deconvolver = nn.Sequential(
                nn.Linear(self.input_length, 1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 39),
                nn.Softmax(dim=-1),
            )
        elif architecture == "3Layer_MLP":
            # deconvolver = nn.Sequential(
            #     nn.Linear(78, 128),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(128, 128),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(128, 64),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(64, 39),
            #     nn.Softmax(dim=-1)
            # )
            deconvolver = nn.Sequential(
                nn.Linear(self.input_length, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, self.input_length),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.input_length, 39),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(
                "Architecture for NN method should be either Shallow_Wide_Network or 3Layer_MLP"
            )
        device = "cuda"
        deconvolver.to(device)
        deconvolver.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        deconvolver.eval()
        X = torch.FloatTensor(
            self._extract_features_by_mask(
                np.array(
                    self.dmr_aggregated[
                        [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                    ]
                ),
                self.features_mask,
            )
        ).to("cuda")
        # X = torch.unsqueeze(X,0)
        # X = torch.concat([torch.diagonal(X[:, :, :39], dim1=1, dim2=2),X[:, :, -1]], dim=1)
        deconv_preds = deconvolver(X)
        proportions = np.round(deconv_preds.to("cpu").detach().numpy(), 4)
        self.logger.debug(f"{architecture} proportions: {proportions}")

        return proportions

    def _run_uxm_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run UXM deconvolution.
        """
        # Load atlas for UXM (uses its own loader)
        atlas_path = self.config["atlas_path"]
        uxm_atlas, ref_cells = load_atlas(atlas_path)

        # Build UXM-compatible input from the prepared reads
        uxm_input = self._build_uxm_input(uxm_atlas, ref_cells)

        # Run UXM deconvolution
        uxm_proportions = uxm_deconvolution(
            atlas=uxm_atlas,
            ref_cells=ref_cells,
            sf=uxm_input["scaling_factors"],
            counts=uxm_input["counts"],
            sample_names=["sample"],
        )[0]

        # Align proportions to match labels_dict order
        ref_cells = [x for x in ref_cells if x != "Megakaryocytes"]

        proportions_aligned = rearange_uxm_deconvolution_results(
            self.labels_dict_reversed, uxm_proportions, ref_cells
        )
        proportions_aligned = np.round(np.array(proportions_aligned), 4)

        self.logger.debug(f"UXM proportions: {proportions_aligned}")
        return proportions_aligned

    @staticmethod
    def _extract_diag_and_bckg(matrix):
        return np.reshape(np.concat([np.diag(matrix), matrix[:, -1]], axis=0), (1, 78))

    @staticmethod
    def _extract_features_by_mask(matrix, target_mask):
        return np.expand_dims(
            np.ma.masked_array(
                matrix, ~np.array(target_mask, dtype=np.bool)
            ).compressed(),
            0,
        )

    def _build_prediction_matrix(self) -> np.ndarray:
        """
        Build the deconvolution input matrix from DMR-aggregated predictions.

        Returns
        -------
        np.ndarray
            Shape: (1, n_dmrs, n_cell_types + 1) — a single sample with
            per-DMR predictions and a rejection column.
        """
        n_labels = len(self.labels_dict)
        pred_cols = [f"prediction_{i}" for i in range(n_labels)]

        # Pivot aggregated data: rows = DMR groups, cols = cell-type predictions
        agg = self.dmr_aggregated.copy()

        # Sort by dmr_ctype_label so matrix rows are in consistent order
        agg = agg.sort_values("dmr_ctype_label").reset_index(drop=True)

        # Extract prediction matrix
        if all(col in agg.columns for col in pred_cols):
            matrix = agg[pred_cols].values  # (n_dmrs, n_labels)
        else:
            # Auto-detect prediction columns
            available = [c for c in agg.columns if c.startswith("prediction_")]
            matrix = agg[available].values

        return self._extract_diag_and_bckg(matrix)

    def _build_uxm_input(
        self, uxm_atlas: pd.DataFrame, ref_cells: list
    ) -> Dict[str, Any]:
        """
        Build UXM-compatible input from prepared reads.

        Computes per-region scaling factors and methylation counts
        in the format expected by ``decon_single_samp``.
        """
        prepared = self.prepared_reads.copy()

        results_agg = (
            prepared[prepared["NCPGS"] > 3]
            .groupby("name")
            .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
            .reset_index()
        )
        results_agg["count"] = (
            results_agg["record_M"] + results_agg["record_U"] + results_agg["record_X"]
        )
        results_agg["sf"] = results_agg["record_U"] / results_agg["count"]
        # TODO: Derrive direction from the source
        results_agg["direction"] = "U"
        sample_name = "sample"

        sf = deepcopy(results_agg[["name", "direction"]])
        sf[sample_name] = results_agg["sf"]
        counts = results_agg[["name", "direction", "count"]]
        counts.columns = ["name", "direction", sample_name]

        return {
            "scaling_factors": sf,
            "counts": counts,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Output
    # ═══════════════════════════════════════════════════════════════════

    def _save_results(self) -> None:
        """Save all pipeline outputs to the configured output directory."""
        output_cfg = self.config.get("output", {})
        output_dir = output_cfg.get("output_dir", "./inference_output")
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

        # ── DMR-aggregated predictions (pickle) ────────────────────────
        if self.dmr_aggregated is not None:
            path = os.path.join(output_dir, "dmr_aggregated.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.dmr_aggregated, f)
            self.logger.info(f"Saved DMR aggregated predictions to {path}")
