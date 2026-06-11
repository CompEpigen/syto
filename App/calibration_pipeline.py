"""
Calibration pipeline:

This pipeline takes as input:
- path to the consolidated pseudobulk HDF5 file (output of the pseudobulk v2
  pipeline, see ``App/pseudobulk_pipeline.py``)
- path to the feature-selection mask (output of the deconvolution pipeline)
- path to folder with saved deconvolvers (named expected to match the output of
the deconvolution pipeline)

and outputs for each deconvolver:
- the weights of the linear calibrators and the weights of the VectorScalingCalibrator with CV
- the predictions of the deconvolver (on test and val set) without calibration,
with linear calibration (clip0 normalisation, simplex projection),
and with VectorScalingCalibrator with CV calibration


The pipeline proceeds in the following steps:
1. Read the pseudobulk feature matrices from the HDF5 file, apply the feature
   mask, and load the deconvolvers
2. For each deconvolver:
    a. evaluate the deconvolver on the validation set and test set, saving the predictions
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

from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader
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

        # Input: consolidated pseudobulk HDF5 file (pseudobulks + proportions)
        self.pseudobulk_h5_path = config["pseudobulk_h5_path"]
        self.reader = PseudobulkHDF5Reader(self.pseudobulk_h5_path, logger=self.logger)

        # Feature-selection mask (produced by the deconvolution pipeline)
        self.features_mask_path = config["features_mask_path"]

        # Splits to load from the HDF5 file
        self.splits = config.get("splits", ["train", "valid", "test"])

        # Output
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Deconvolver directory (where saved models live)
        self.deconvolvers_dir = config["deconvolvers_dir"]

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
        # ── Stage 1: Load data and deconvolvers ───────────────────
        features_dict, proportions_dict = self._stage1_load_features()
        deconvolvers = self._stage1_load_deconvolvers()

        features_valid = features_dict["valid"]
        y_valid = proportions_dict["valid"]

        features_test = features_dict.get("test", features_valid)
        y_test = proportions_dict.get("test", y_valid)

        results: Dict[str, Any] = {}

        # ── Stage 2: For each deconvolver, calibrate ─────────────
        for deconv_name, deconv_cfg, model in deconvolvers:
            self.logger.info(f"Processing deconvolver: {deconv_name}")
            try:
                deconv_results = self._process_deconvolver(
                    deconv_name,
                    deconv_cfg,
                    model,
                    features_valid,
                    y_valid,
                    features_test,
                    y_test,
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
            f"Stage 1a: Reading pseudobulk matrices from {self.pseudobulk_h5_path}"
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

    def _stage1_load_deconvolvers(
        self,
    ) -> List[Tuple[str, dict, Any]]:
        """Load all configured deconvolvers from disk.

        Returns list of (name, config_dict, loaded_model) tuples.
        """
        self.logger.info(f"Stage 1b: Loading deconvolvers from {self.deconvolvers_dir}")

        deconvolvers_cfg = self.config.get("deconvolvers", [])
        loaded: List[Tuple[str, dict, Any]] = []

        for cfg in deconvolvers_cfg:
            name = cfg["name"]
            self.logger.info(f"  Loading deconvolver: {name}")

            if name == "xgb":
                model = XGBoostDeconvolver.load(
                    os.path.join(self.deconvolvers_dir, "xgb_deconvolver.joblib")
                )
            elif name == "swn":
                weights_path = Path(self.deconvolvers_dir) / "swn_best_deconvolver.pt"
                metadata_path = (
                    Path(self.deconvolvers_dir) / "swn_architecture_meta.json"
                )
                model = SWNDeconvolver.load(
                    path=weights_path,
                    metadata_path=metadata_path,
                )
            elif name == "mlp":
                weights_path = Path(self.deconvolvers_dir) / "mlp_best_deconvolver.pt"
                metadata_path = (
                    Path(self.deconvolvers_dir) / "mlp_architecture_meta.json"
                )
                model = MLPDeconvolver.load(
                    path=weights_path,
                    metadata_path=metadata_path,
                )
            elif name == "nnls":
                model = NNLSDeconvolver.load(
                    os.path.join(self.deconvolvers_dir, "nnls_deconvolver.joblib")
                )
            elif name == "psls":
                model = PSLSDeconvolver.load(
                    os.path.join(self.deconvolvers_dir, "psls_deconvolver.joblib")
                )
            else:
                self.logger.warning(f"  Unknown deconvolver '{name}', skipping")
                continue

            loaded.append((name, cfg, model))

        self.logger.info(f"  Loaded {len(loaded)} deconvolvers")
        return loaded

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2: Per-deconvolver calibration
    # ═══════════════════════════════════════════════════════════════

    def _process_deconvolver(
        self,
        name: str,
        cfg: dict,
        model: Any,
        features_valid: np.ndarray,
        y_valid: np.ndarray,
        features_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Run the full calibration workflow for a single deconvolver.

        Steps:
          a. Evaluate uncalibrated predictions on val and test
          b. Fit LinearCalibrator on val, evaluate with 2 normalisation methods
          c. Fit VectorScalingCalibrator with CV on val, evaluate on val and test
        """
        deconv_out = os.path.join(
            self.output_dir, name + "_calibrators_and_predictions"
        )
        os.makedirs(deconv_out, exist_ok=True)

        # ── 2a: Uncalibrated evaluation ───────────────────────────
        self.logger.info(f"  [{name}] Evaluating uncalibrated predictions ...")
        uncalibrated_predictions_path = os.path.join(
            deconv_out, "uncalibrated_predictions.npz"
        )
        if os.path.exists(uncalibrated_predictions_path):
            self.logger.info(
                f"    Found existing uncalibrated predictions at {uncalibrated_predictions_path},"
                " loading instead of re-evaluating"
            )
            data = np.load(uncalibrated_predictions_path)
            val_pred = data["val_pred"]
            test_pred = data["test_pred"]
        else:
            val_pred = model.predict(features_valid, **cfg.get("params", {}))
            test_pred = model.predict(features_test, **cfg.get("params", {}))

            np.savez_compressed(
                os.path.join(deconv_out, "uncalibrated_predictions.npz"),
                val_pred=val_pred,
                test_pred=test_pred,
                val_target=y_valid,
                test_target=y_test,
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
