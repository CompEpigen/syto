"""
Calibration pipeline:

This pipeline takes as input:
- path to the consolidated pseudobulk HDF5 file (output of the pseudobulk v2
  pipeline, see ``syto/app/pseudobulk_pipeline.py``)
- path to the feature-selection mask (output of the deconvolution pipeline)
- path to folder with saved deconvolvers (named expected to match the output of
the deconvolution pipeline)

A deconvolver may instead declare ``predictions_dir``, pointing at a directory
of already-computed long-format parquets written by the pseudobulk
deconvolution pipeline (see ``syto/app/pseudobulk_deconvolution_pipeline.py``).  This
is how the baselines are calibrated: their predictions and targets are read
straight off disk, so no HDF5 file, feature mask, or saved model is needed.
Those three inputs become optional whenever every configured deconvolver is
prediction-backed.

and outputs for each deconvolver:
- the weights of the linear calibrators and the weights of the VectorScalingCalibrator with CV
- the predictions of the deconvolver (on test and val set) without calibration,
with linear calibration (clip0 normalisation, simplex projection),
and with VectorScalingCalibrator with CV calibration


The pipeline proceeds in the following steps:
1. Read the pseudobulk feature matrices from the HDF5 file and apply the feature
   mask (skipped when every deconvolver is prediction-backed)
2. For each deconvolver:
    a. obtain validation and test predictions - either by loading the model and
    evaluating it on the feature matrices, or by reading a stored parquet -
    and save them
    b. fit a linear calibrator on the validation set
        i. save the predictions of the linear calibrator on the validation set and test set
        for both normalization methods (clip0-norm, simplex projection)
    c. fit a VectorScalingCalibrator with CV on the validation set
        i. save the predictions of the VectorScalingCalibrator with CV
        on the validation set and test set

"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from syto.data.pseudobulk_store import open_pseudobulk_store
from syto.deconvolution.evaluation import compute_deconvolution_metrics
from syto.deconvolution.feature_selection import apply_feature_mask
from syto.calibration.linear_calibrator import LinearCalibrator
from syto.calibration.vector_scaling_calibrator import VectorScalingCalibrator
from syto.cross_validation_engine import CrossValidationEngine
from syto.deconvolution.xgbdeconvolver import XGBoostDeconvolver
from syto.deconvolution.least_squares_deconvolvers import (
    NNLSDeconvolver,
    PSLSDeconvolver,
)
from syto.deconvolution.deep_deconvolvers.swn import SWNDeconvolver
from syto.deconvolution.deep_deconvolvers.mlp import MLPDeconvolver

LINEAR_NORM_METHODS = ["clip-normalize", "simplex-projection"]

# Columns every stored deconvolution parquet carries; the remaining column
# holds the model's predictions and is named after the model/checkpoint.
BASELINE_KEY_COLUMNS = ("pb_index", "cell_type", "target_proportion")


def load_predictions_from_parquet(
    predictions_dir: str,
    labels_dict: Dict[int, str],
    splits: Tuple[str, ...] = ("valid", "test"),
    prediction_column: str | None = None,
    num_output_labels: int | None = None,
    logger: logging.Logger | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read stored deconvolution results into prediction/target matrices.

    Reads ``{predictions_dir}/{split}_deconvolution_results.parquet`` - the
    long-format output of the pseudobulk deconvolution pipeline, one row per
    (pseudobulk, cell type) - and pivots it into the dense ``(n_samples,
    n_classes)`` layout the calibrators expect.

    Rows are ordered by ``pb_index`` and columns by ``labels_dict`` key, which
    reproduces the ordering of the corresponding HDF5 split (``pb_index`` is the
    position within the split), so the resulting matrices are row-aligned with
    those of the model-backed deconvolvers.

    Parameters
    ----------
    predictions_dir : str
        Directory holding the per-split parquets.
    labels_dict : dict
        ``{index: cell_type_name}`` mapping defining the column order.
    splits : tuple of str
        Splits to read.  All of them must be present.
    prediction_column : str, optional
        Name of the prediction column.  Inferred when the parquet has exactly
        one non-key column.
    num_output_labels : int, optional
        Keep only the first ``num_output_labels`` labels, matching the
        truncation applied to the HDF5 proportions.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        ``{split: (predictions, targets)}``, both ``(n_samples, n_classes)``.
    """
    class_names = [labels_dict[i] for i in sorted(labels_dict)]
    if num_output_labels is not None:
        class_names = class_names[:num_output_labels]

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in splits:
        path = os.path.join(predictions_dir, f"{split}_deconvolution_results.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No stored deconvolution results for split '{split}': {path}"
            )

        df = pd.read_parquet(path)

        missing_cols = [c for c in BASELINE_KEY_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{path} is missing required column(s) {missing_cols}; "
                f"found {list(df.columns)}"
            )

        column = prediction_column
        if column is None:
            candidates = [c for c in df.columns if c not in BASELINE_KEY_COLUMNS]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot infer the prediction column of {path}: expected exactly "
                    f"one non-key column but found {candidates}. Set "
                    "'prediction_column' on the deconvolver config to disambiguate."
                )
            column = candidates[0]
        elif column not in df.columns:
            raise ValueError(
                f"prediction_column '{column}' not in {path}; "
                f"found {list(df.columns)}"
            )

        if df.duplicated(subset=["pb_index", "cell_type"]).any():
            n_dup = int(df.duplicated(subset=["pb_index", "cell_type"]).sum())
            raise ValueError(
                f"{path} has {n_dup} duplicate (pb_index, cell_type) row(s); "
                "the file is not a well-formed deconvolution result."
            )

        wide = df.pivot(
            index="pb_index",
            columns="cell_type",
            values=[column, "target_proportion"],
        ).sort_index()

        absent = [c for c in class_names if c not in wide[column].columns]
        if absent:
            raise ValueError(
                f"{path} has no rows for cell type(s) {absent}; expected all of "
                f"{class_names}"
            )

        predictions = wide[column].reindex(columns=class_names).to_numpy(np.float64)
        targets = (
            wide["target_proportion"].reindex(columns=class_names).to_numpy(np.float64)
        )

        if np.isnan(predictions).any() or np.isnan(targets).any():
            raise ValueError(
                f"{path} does not cover every (pb_index, cell_type) pair; "
                "the pivoted matrices contain missing entries."
            )

        out[split] = (predictions, targets)
        if logger is not None:
            logger.info(
                f"    {split}: predictions={predictions.shape} "
                f"from column '{column}' of {path}"
            )

    return out


class CalibratorFittingPipeline:
    """Orchestrate calibrators fitting end-to-end.

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
        self.num_output_labels = config.get("num_output_labels", len(self.labels_dict))
        self.num_input_labels = config.get("num_input_labels", self.num_output_labels)

        self.deconvolvers_cfg: List[dict] = config.get("deconvolvers", [])
        self._check_unique_names()

        # Deconvolvers reading their predictions off disk need none of the
        # model-evaluation inputs below, so those are only required when at
        # least one deconvolver has to be loaded and run.
        self.model_backed_cfgs = [
            cfg for cfg in self.deconvolvers_cfg if not cfg.get("predictions_dir")
        ]
        required_for_models = bool(self.model_backed_cfgs)

        # Input: a pseudobulk store (pseudobulks + proportions), in either the
        # consolidated HDF5 or the columnar layout.
        self.pseudobulk_path = self._get_model_input(
            "pseudobulk_path", required_for_models
        )
        self.reader = (
            open_pseudobulk_store(self.pseudobulk_path, logger=self.logger)
            if self.pseudobulk_path
            else None
        )

        # Feature-selection mask (produced by the deconvolution pipeline)
        self.features_mask_path = self._get_model_input(
            "features_mask_path", required_for_models
        )

        # Splits to load from the HDF5 file
        self.splits = config.get("splits", ["train", "valid", "test"])

        # Output
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Deconvolver directory (where saved models live)
        self.deconvolvers_dir = self._get_model_input(
            "deconvolvers_dir", required_for_models
        )

    def _get_model_input(self, key: str, required: bool) -> Any:
        """Fetch a config key needed only for model-backed deconvolvers."""
        if key in self.config:
            return self.config[key]
        if required:
            names = [cfg.get("name") for cfg in self.model_backed_cfgs]
            raise ValueError(
                f"Missing '{key}', required because deconvolver(s) {names} have no "
                "'predictions_dir' and must be loaded and evaluated."
            )
        return None

    def _check_unique_names(self) -> None:
        """Reject duplicate deconvolver names.

        Each name becomes an output sub-directory, so duplicates would silently
        overwrite one another.  This bites in practice because the same model
        appears under several atlases (e.g. ``uxm`` and
        ``uxm_trainonly_atlas/uxm``) and must be given distinct names.
        """
        seen, duplicates = set(), []
        for cfg in self.deconvolvers_cfg:
            name = cfg["name"]
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate deconvolver name(s) {sorted(set(duplicates))} in config; "
                "names must be unique because each one names an output directory."
            )

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """Execute the full calibration fitting pipeline.

        Returns
        -------
        dict
            Summary of calibration results per deconvolver.
        """
        # ── Stage 1: Load data ────────────────────────────────────
        if self.model_backed_cfgs:
            features_dict, proportions_dict = self._stage1_load_features()
        else:
            self.logger.info(
                "Stage 1: every deconvolver is prediction-backed, skipping the "
                "pseudobulk HDF5 and feature-mask load"
            )
            features_dict, proportions_dict = {}, {}

        results: Dict[str, Any] = {}

        # ── Stage 2: For each deconvolver, calibrate ─────────────
        for deconv_cfg in self.deconvolvers_cfg:
            deconv_name = deconv_cfg["name"]
            self.logger.info(f"Processing deconvolver: {deconv_name}")
            try:
                deconv_results = self._process_deconvolver(
                    deconv_name,
                    deconv_cfg,
                    features_dict,
                    proportions_dict,
                )
                results[deconv_name] = deconv_results
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error(
                    f"  Failed to process '{deconv_name}': {e}", exc_info=True
                )

        # Log summary
        self._log_summary(results)

        # ── Stage 3: Save CSV summary ────────────────────────────
        self._save_summary_csv(results)

        return results

    # ═══════════════════════════════════════════════════════════════
    #  Stage 1: Load data and deconvolvers
    # ═══════════════════════════════════════════════════════════════

    def _stage1_load_features(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Read pseudobulk feature matrices from the HDF5 file and apply the mask.

        The feature mask (shape ``(n_gr_groups, n_pred_classes)``) is produced by
        the deconvolution pipeline.  It is applied to the pseudobulk matrices read
        from the consolidated HDF5 file so that the calibration features match the
        features the deconvolvers were trained on.

        Returns
        -------
        features_dict : dict
            ``{split_name: np.ndarray}`` of shape ``(n_samples, n_selected_features)``.
        proportions_dict : dict
            ``{split_name: np.ndarray}`` of shape ``(n_samples, n_output_labels)``.
        """
        self.logger.info(
            f"Stage 1a: Loading feature mask from {self.features_mask_path}"
        )
        mask = np.load(self.features_mask_path)["features_mask"]
        self.logger.info(f"  Mask shape={mask.shape}, selected={int(mask.sum())}")

        self.logger.info(
            f"Stage 1a: Reading pseudobulk matrices from {self.pseudobulk_path}"
        )
        features_dict: Dict[str, np.ndarray] = {}
        proportions_dict: Dict[str, np.ndarray] = {}
        for split_name in self.splits:
            features, proportions = self.reader.read_pseudobulk_matrices(
                split_name, num_pred_classes=self.num_input_labels
            )
            # features: (n_samples, n_gr_groups, n_pred_classes)
            features_dict[split_name] = apply_feature_mask(features, mask)
            proportions_dict[split_name] = proportions[:, : self.num_output_labels]
            self.logger.info(
                f"  {split_name}: features={features_dict[split_name].shape}, "
                f"proportions={proportions_dict[split_name].shape}"
            )

        if "valid" not in features_dict:
            raise ValueError(
                "'valid' split is required for calibration but was not found "
                f"in the loaded data. Available: {list(features_dict.keys())}"
            )

        return features_dict, proportions_dict

    def _load_deconvolver(self, name: str) -> Any:
        """Load a single saved deconvolver from ``deconvolvers_dir``."""
        self.logger.info(f"  Loading deconvolver '{name}' from {self.deconvolvers_dir}")

        if name == "xgb":
            return XGBoostDeconvolver.load(
                os.path.join(self.deconvolvers_dir, "xgb_deconvolver.joblib")
            )
        if name == "swn":
            return SWNDeconvolver.load(
                path=Path(self.deconvolvers_dir) / "swn_best_deconvolver.pt",
                metadata_path=Path(self.deconvolvers_dir)
                / "swn_architecture_meta.json",
            )
        if name == "mlp":
            return MLPDeconvolver.load(
                path=Path(self.deconvolvers_dir) / "mlp_best_deconvolver.pt",
                metadata_path=Path(self.deconvolvers_dir)
                / "mlp_architecture_meta.json",
            )
        if name == "nnls":
            return NNLSDeconvolver.load(
                os.path.join(self.deconvolvers_dir, "nnls_deconvolver.joblib")
            )
        if name == "psls":
            return PSLSDeconvolver.load(
                os.path.join(self.deconvolvers_dir, "psls_deconvolver.joblib")
            )
        raise ValueError(
            f"Unknown deconvolver '{name}' with no 'predictions_dir'; expected one of "
            "xgb, swn, mlp, nnls, psls, or a config pointing at stored predictions."
        )

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2: Per-deconvolver calibration
    # ═══════════════════════════════════════════════════════════════

    def _load_uncalibrated(
        self,
        name: str,
        cfg: dict,
        deconv_out: str,
        features_dict: Dict[str, np.ndarray],
        proportions_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Obtain uncalibrated val/test predictions and their targets.

        Resolution order:
          1. a cached ``uncalibrated_predictions.npz`` from a previous run
          2. stored parquet results, when the config gives ``predictions_dir``
          3. loading the saved model and evaluating it on the feature matrices

        Returns ``(val_pred, y_valid, test_pred, y_test)``.
        """
        uncalibrated_predictions_path = os.path.join(
            deconv_out, "uncalibrated_predictions.npz"
        )
        if os.path.exists(uncalibrated_predictions_path):
            self.logger.info(
                f"    Found existing uncalibrated predictions at {uncalibrated_predictions_path},"
                " loading instead of re-evaluating"
            )
            data = np.load(uncalibrated_predictions_path)
            return (
                data["val_pred"],
                data["val_target"],
                data["test_pred"],
                data["test_target"],
            )

        predictions_dir = cfg.get("predictions_dir")
        if predictions_dir:
            self.logger.info(f"    Reading stored predictions from {predictions_dir}")
            per_split = load_predictions_from_parquet(
                predictions_dir,
                self.labels_dict,
                splits=self.splits,
                prediction_column=cfg.get("prediction_column"),
                num_output_labels=self.num_output_labels,
                logger=self.logger,
            )
            val_pred, y_valid = per_split["valid"]
            test_pred, y_test = per_split.get("test", per_split["valid"])
            self._check_targets_match_hdf5(name, y_valid, y_test, proportions_dict)
        else:
            model = self._load_deconvolver(name)
            features_valid = features_dict["valid"]
            y_valid = proportions_dict["valid"]
            features_test = features_dict.get("test", features_valid)
            y_test = proportions_dict.get("test", y_valid)
            val_pred = model.predict(features_valid, **cfg.get("params", {}))
            test_pred = model.predict(features_test, **cfg.get("params", {}))

        np.savez_compressed(
            uncalibrated_predictions_path,
            val_pred=val_pred,
            test_pred=test_pred,
            val_target=y_valid,
            test_target=y_test,
        )
        return val_pred, y_valid, test_pred, y_test

    def _check_targets_match_hdf5(
        self,
        name: str,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        proportions_dict: Dict[str, np.ndarray],
    ) -> None:
        """Verify parquet targets agree with the HDF5 ones, when both are loaded.

        Only applies to mixed configs.  A mismatch means the stored predictions
        were produced from a different pseudobulk file than the one being
        calibrated against, which would make the comparison meaningless.
        """
        for split, y_parquet in (("valid", y_valid), ("test", y_test)):
            y_hdf5 = proportions_dict.get(split)
            if y_hdf5 is None:
                continue
            if y_hdf5.shape != y_parquet.shape:
                raise ValueError(
                    f"[{name}] stored {split} targets have shape {y_parquet.shape} but "
                    f"the pseudobulk HDF5 gives {y_hdf5.shape}; the stored predictions "
                    "were computed from a different pseudobulk file."
                )
            if not np.allclose(y_hdf5, y_parquet):
                max_diff = float(np.abs(y_hdf5 - y_parquet).max())
                raise ValueError(
                    f"[{name}] stored {split} target proportions disagree with the "
                    f"pseudobulk HDF5 (max abs diff {max_diff:.6g}); the stored "
                    "predictions were computed from a different pseudobulk file."
                )

    def _process_deconvolver(
        self,
        name: str,
        cfg: dict,
        features_dict: Dict[str, np.ndarray],
        proportions_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run the full calibration workflow for a single deconvolver.

        Steps:
          a. Obtain uncalibrated predictions on val and test
          b. Fit LinearCalibrator on val, evaluate with 2 normalisation methods
          c. Fit VectorScalingCalibrator with CV on val, evaluate on val and test
        """
        deconv_out = os.path.join(
            self.output_dir, name + "_calibrators_and_predictions"
        )
        os.makedirs(deconv_out, exist_ok=True)

        # ── 2a: Uncalibrated evaluation ───────────────────────────
        self.logger.info(f"  [{name}] Evaluating uncalibrated predictions ...")
        val_pred, y_valid, test_pred, y_test = self._load_uncalibrated(
            name, cfg, deconv_out, features_dict, proportions_dict
        )

        val_metrics = compute_deconvolution_metrics(val_pred, y_valid)
        test_metrics = compute_deconvolution_metrics(test_pred, y_test)

        self.logger.info(
            f"    Uncalibrated  val MSE={val_metrics['mse']:.6f}  "
            f"test MSE={test_metrics['mse']:.6f}"
        )

        result: Dict[str, Any] = {
            "uncalibrated": {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
        }

        # ── 2b: Linear calibration ───────────────────────────────
        self.logger.info(f"  [{name}] Fitting linear calibrator ...")
        linear_calibrator_path = os.path.join(deconv_out, "linear_calibrator.joblib")
        if os.path.exists(linear_calibrator_path):
            self.logger.info(
                f"    Found existing linear calibrator at {linear_calibrator_path},"
                " loading instead of re-fitting"
            )
            linear_calibrator = LinearCalibrator.load(linear_calibrator_path)
        else:
            linear_calibrator = LinearCalibrator()
            linear_calibrator.fit(val_pred, y_valid)
            linear_calibrator.save(linear_calibrator_path)
            self.logger.info(f"    Saved linear calibrator to {linear_calibrator_path}")

        for norm_method in LINEAR_NORM_METHODS:
            short_name = norm_method.replace("-", "_")
            predictions_path = os.path.join(
                deconv_out, f"linear_{short_name}_predictions.npz"
            )
            if os.path.exists(predictions_path):
                self.logger.info(
                    "    Found existing linear calibrated predictions for norm method "
                    f"'{norm_method}' at {predictions_path},"
                    " loading instead of re-predicting"
                )
                data = np.load(predictions_path)
                val_calib = data["val_pred"]
                test_calib = data["test_pred"]
            else:
                val_calib = linear_calibrator.predict(val_pred, norm_method=norm_method)
                test_calib = linear_calibrator.predict(
                    test_pred, norm_method=norm_method
                )
                np.savez_compressed(
                    predictions_path,
                    val_pred=val_calib,
                    test_pred=test_calib,
                )

            val_m = compute_deconvolution_metrics(val_calib, y_valid)
            test_m = compute_deconvolution_metrics(test_calib, y_test)

            self.logger.info(
                f"    Linear ({norm_method})  val MSE={val_m['mse']:.6f}  "
                f"test MSE={test_m['mse']:.6f}"
            )

            result[f"linear_{short_name}"] = {
                "val_metrics": val_m,
                "test_metrics": test_m,
            }

        # ── 2c: VectorScaling calibration ─────────────────────────
        self.logger.info(f"  [{name}] Fitting VectorScalingCalibrator with CV ...")

        vs_path = os.path.join(deconv_out, "vector_scaling_calibrator_with_cv.joblib")
        if os.path.exists(vs_path):
            self.logger.info(
                f"    Found existing VectorScalingCalibrator with CV at {vs_path},"
                " loading instead of re-fitting"
            )
            vs_calibrator = VectorScalingCalibrator.load(vs_path)
            best_params = {
                k: v for k, v in vs_calibrator.get_params().items() if k != "logger"
            }
            best_cv_val_loss = vs_calibrator.get_cv_metric(val_pred, y_valid)
        else:
            # get config for vector scaling calibrator,
            # log any missing parameters and their defaults
            vs_cfg = cfg.get("vector_scaling", {})
            if not vs_cfg:
                self.logger.info(
                    f"    Deconvolver config for '{name}' missing"
                    " 'vector_scaling' section, using defaults for all parameters:  "
                    "reg_lambda_list=[0.0, 1e-4, 1e-3], "
                    "lr_list=[1e-3, 1e-2], "
                    "max_iter_list=[1000], "
                    "optimizer='adam', "
                    "scheduler='plateau', "
                    "patience=50, "
                    "n_folds=5, "
                    "batch_size=None, "
                    "verbose=True, "
                    "plateau_factor=0.5, "
                    "plateau_patience=10"
                )
            else:
                if "reg_lambda_list" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'reg_lambda_list',"
                        " using default [0.0, 1e-4, 1e-3]"
                    )
                if "lr_list" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'lr_list',"
                        " using default [1e-3, 1e-2]"
                    )
                if "max_iter_list" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'max_iter_list',"
                        " using default [1000]"
                    )
                if "optimizer" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'optimizer',"
                        " using default 'adam'"
                    )
                if "scheduler" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'scheduler',"
                        " using default 'plateau'"
                    )
                if "patience" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'patience',"
                        " using default 50"
                    )
                if "n_folds" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'n_folds',"
                        " using default 5"
                    )
                if "batch_size" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'batch_size',"
                        " using default None (full batch)"
                    )
                if "verbose" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'verbose',"
                        " using default True"
                    )

            vs_calibrator = CrossValidationEngine()
            vs_calibrator.fit(
                X=val_pred,
                y=y_valid,
                model_class=VectorScalingCalibrator,
                n_folds=3,
                model_param_grid={
                    "reg_lambda": vs_cfg.get("reg_lambda_list", [0.0, 1e-4, 1e-3]),
                    "lr": vs_cfg.get("lr_list", [1e-3, 1e-2]),
                    "max_iter": vs_cfg.get("max_iter_list", [1000]),
                    "optimizer": vs_cfg.get("optimizer", "adam"),
                    "scheduler": vs_cfg.get("scheduler", "plateau"),
                    "patience": vs_cfg.get("patience", 50),
                    "batch_size": vs_cfg.get("batch_size", None),
                    "plateau_factor": vs_cfg.get("plateau_factor", 0.5),
                    "plateau_patience": vs_cfg.get("plateau_patience", 10),
                },
                random_state=42,
                disable_pbar=False,
            )
            best_params = vs_calibrator.best_params_
            best_cv_val_loss = vs_calibrator.best_metric_

            # Keep only the cross-validated model going forward and persist it
            # (default save mode saves only the model, not the whole CV engine).
            vs_calibrator.save(vs_path)
            vs_calibrator = vs_calibrator.best_model_

            self.logger.info(f"    Saved VectorScaling calibrator to {vs_path}")

        self.logger.info(
            f"    Best CV loss: {best_cv_val_loss:.6f}, " f"params: {best_params}"
        )

        vs_pred_path = os.path.join(deconv_out, "vector_scaling_predictions.npz")
        if os.path.exists(vs_pred_path):
            self.logger.info(
                f"    Found existing VectorScalingCalibrator CV predictions at {vs_pred_path},"
                " loading instead of re-predicting"
            )
            data = np.load(vs_pred_path)
            val_vs = data["val_pred"]
            test_vs = data["test_pred"]
        else:
            val_vs = vs_calibrator.predict(val_pred)
            test_vs = vs_calibrator.predict(test_pred)
            np.savez_compressed(
                vs_pred_path,
                val_pred=val_vs,
                test_pred=test_vs,
            )

        val_vs_m = compute_deconvolution_metrics(val_vs, y_valid)
        test_vs_m = compute_deconvolution_metrics(test_vs, y_test)

        self.logger.info(
            f"    VectorScaling  val MSE={val_vs_m['mse']:.6f}  "
            f"test MSE={test_vs_m['mse']:.6f}"
        )

        result["vector_scaling"] = {
            "val_metrics": val_vs_m,
            "test_metrics": test_vs_m,
            "best_params": best_params,
            "best_cv_val_loss": best_cv_val_loss,
        }

        return result

    # ═══════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of all calibration results."""
        self.logger.info("=" * 70)
        self.logger.info("CALIBRATION FITTING SUMMARY")
        self.logger.info("=" * 70)

        for deconv_name, deconv_results in results.items():
            self.logger.info(f"  Deconvolver: {deconv_name}")
            for calib_name, data in deconv_results.items():
                if "test_metrics" in data:
                    m = data["test_metrics"]
                    self.logger.info(
                        f"    {calib_name:30s}  "
                        f"R2={m['r2']:.6f}  "
                        f"LoA=[{m['loa_lower']:.6f}, {m['loa_upper']:.6f}]  "
                        f"LoA(worst)=[{m['worst_class_loa_lower']:.6f},"
                        f" {m['worst_class_loa_upper']:.6f}]  "
                        f"MAE={m['mae']:.6f}  "
                        f"MSE={m['mse']:.6f}  "
                        f"KLDiv={m['kl']:.6f}"
                    )

        self.logger.info("=" * 70)
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info("=" * 70)

    def _save_summary_csv(self, results: Dict[str, Any]) -> None:
        """Save a CSV with one row per (deconvolver, calibration_method) pair."""
        rows: list = []
        for deconv_name, deconv_results in results.items():
            for calib_name, data in deconv_results.items():
                if "test_metrics" not in data:
                    continue
                m = data["test_metrics"]
                rows.append(
                    {
                        "deconvolver": deconv_name,
                        "calibration_method": calib_name,
                        "r2": m["r2"],
                        "loa_lower": m["loa_lower"],
                        "loa_upper": m["loa_upper"],
                        "worst_class_loa_lower": m["worst_class_loa_lower"],
                        "worst_class_loa_upper": m["worst_class_loa_upper"],
                        "mae": m["mae"],
                        "mse": m["mse"],
                        "kl": m["kl"],
                    }
                )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "calibration_summary.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved calibration summary CSV to {csv_path}")
