"""
Calibration pipeline:

This pipeline takes as input:
- path to ios (already feature selected)
- path to folder with saved deconvolvers (named expected to match the output of
the deconvolution pipeline)

and outputs for each deconvolver:
- the weights of the linear calibrators and the weights of the VectorScalingCalibratorCV
- the predictions of the deconvolver (on test and val set) without calibration,
with linear calibration (clip01 normalisation, clip0 normalisation, simplex projection),
and with VectorScalingCalibratorCV calibration


The pipeline proceeds in the following steps:
1. Load the ios (already feature selected) and deconvolvers
2. For each deconvolver:
    a. evaluate the deconvolver on the validation set and test set, saving the predictions
    b. fit a linear calibrator on the validation set
        i. save the predictions of the linear calibrator on the validation set and test set
        for all 3 normalization methods (clip01-norm, clip0-norm, simplex projection)
    c. fit a VectorScalingCalibratorCV on the validation set
        i. save the predictions of the VectorScalingCalibratorCV
        on the validation set and test set

"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from methyldl.deconvolution.evaluation import compute_deconvolution_metrics
from methyldl.deconvolution.linear_calibrator import LinearCalibrator
from methyldl.deconvolution.vector_scaling_calibrator import (
    VectorScalingCalibratorCV,
)
from methyldl.deconvolution.xgbdeconvolver import XGBoostDeconvolver
from methyldl.deconvolution.least_squares_deconvolvers import (
    NNLSDeconvolver,
    PSLSDeconvolver,
)

LINEAR_NORM_METHODS = ["clip01-normalize", "clip0-normalize", "simplex-projection"]


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
        ios_data = self._stage1_load_data()
        deconvolvers = self._stage1_load_deconvolvers()

        # Extract features per split, resolving variant-aware keys
        features_dict, proportions_dict = self._resolve_features_and_proportions(ios_data)

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

    def _stage1_load_data(self) -> Dict[str, np.ndarray]:
        """Load the IOs (already feature selected).

        Supports both legacy key layout (``proportions``,
        ``features_{split}``) and the new variant-aware layout
        (``proportions_{split}``, ``features_{split}_{variant}``).
        """
        ios_path = self.config["ios_feature_selected_path"]
        self.logger.info(f"Stage 1a: Loading IOs from {ios_path}")

        data = np.load(ios_path)
        ios_data = {k: data[k] for k in data.files}
        self.logger.info(f"  Available keys: {list(ios_data.keys())}")
        return ios_data

    def _resolve_features_and_proportions(
        self, ios_data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resolve variant-aware feature and per-split proportion keys.

        Config options
        --------------
        preferred_dmr_variant : str or dict, optional
            A global variant name (e.g. ``"uniform"``) or a per-split
            mapping (e.g. ``{"valid": "uniform", "test": "uniform_multimodal"}``).
            When omitted, the first variant found alphabetically is used.

        Returns
        -------
        features_dict : dict
            ``{split_name: np.ndarray}``
        proportions_dict : dict
            ``{split_name: np.ndarray}``
        """
        preferred_variant = self.config.get("preferred_dmr_variant")

        # ── Features ──────────────────────────────────────────────
        features_dict: Dict[str, np.ndarray] = {}
        for key, val in ios_data.items():
            if not key.startswith("features_"):
                continue
            remainder = key[len("features_"):]
            parts = remainder.split("_")

            if len(parts) > 1:
                # Variant-aware key: features_{split}_{variant}
                split_name, variant = parts[0], "_".join(parts[1:])

                target_variant = preferred_variant
                if isinstance(preferred_variant, dict):
                    target_variant = preferred_variant.get(split_name)

                if target_variant and variant != target_variant:
                    continue
                if split_name not in features_dict:
                    features_dict[split_name] = val
            else:
                # Legacy key: features_{split}
                features_dict[remainder] = val

        self.logger.info(f"  Resolved feature splits: {list(features_dict.keys())}")

        if "valid" not in features_dict:
            raise ValueError(
                "'valid' split is required for calibration but was not found "
                f"in the loaded data. Available: {list(features_dict.keys())}"
            )

        # ── Proportions ───────────────────────────────────────────
        splits = self.config.get("splits", ["train", "valid", "test"])
        proportions_dict: Dict[str, np.ndarray] = {}
        for split_name in splits:
            pkey = f"proportions_{split_name}"
            if pkey in ios_data:
                proportions_dict[split_name] = ios_data[pkey]
            elif "proportions" in ios_data:
                proportions_dict[split_name] = ios_data["proportions"]

        if "valid" not in proportions_dict:
            raise ValueError(
                "Proportions for 'valid' split are required for calibration."
            )

        self.logger.info(
            f"  Resolved proportion splits: {list(proportions_dict.keys())}"
        )

        for split_name in features_dict:
            f_shape = features_dict[split_name].shape
            if split_name in proportions_dict:
                p_shape = proportions_dict[split_name].shape
                self.logger.info(
                    f"  {split_name}: features={f_shape}, proportions={p_shape}"
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
            elif name in ("swn", "mlp"):
                model = self._load_nn_model(name, cfg)
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

    def _load_nn_model(self, name: str, cfg: dict) -> nn.Module:
        """Load a PyTorch neural network deconvolver from saved state dict + metadata."""
        meta_path = os.path.join(
            self.deconvolvers_dir, f"{name}_architecture_meta.json"
        )
        weights_path = os.path.join(
            self.deconvolvers_dir, f"{name}_best_deconvolver.pt"
        )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        n_input_features = meta["n_input_features"]
        n_cell_types = meta["n_cell_types"]
        params = meta.get("params", {})

        model = self._build_nn_model(
            name, n_input_features, n_cell_types, params, self.logger
        )

        if not "params" in cfg:
            self.logger.warning(
                f"Deconvolver config for '{name}' missing 'params' section, using empty dict"
            )
        elif not "device" in cfg["params"]:
            self.logger.warning(
                f"Deconvolver config for '{name}' missing 'device' param, using cuda if available else cpu"
            )
        device = cfg.get("params", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        model.eval()
        return model

    @staticmethod
    def _build_nn_model(
        name: str,
        n_input_features: int,
        n_cell_types: int,
        params: dict,
        logger: logging.Logger,
    ) -> nn.Module:
        """Reconstruct a neural network architecture from metadata."""
        if name == "swn":
            if not "hidden_dim" in params:
                logger.warning(
                    "SWN architecture metadata missing 'hidden_dim', using default 1024"
                )
            if not "dropout" in params:
                logger.warning(
                    "SWN architecture metadata missing 'dropout', using default 0.2"
                )
            hidden_dim = params.get("hidden_dim", 1024)
            model = nn.Sequential(
                nn.Linear(n_input_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(params.get("dropout", 0.2)),
                nn.Linear(hidden_dim, n_cell_types),
                nn.Softmax(dim=-1),
            )
        elif name == "mlp":
            if not "hidden_dims" in params:
                logger.warning(
                    "MLP architecture metadata missing 'hidden_dims', using default [512, 256]"
                )
            if not "dropout" in params:
                logger.warning(
                    "MLP architecture metadata missing 'dropout', using default 0.2"
                )
            if not "final_dropout" in params:
                logger.warning(
                    "MLP architecture metadata missing 'final_dropout', using default 0.1"
                )
            hidden_dims = params.get("hidden_dims", [512, 256])
            layers: list = []
            in_dim = n_input_features
            for h_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(in_dim, h_dim),
                        nn.GELU(),
                        nn.Dropout(params.get("dropout", 0.2)),
                    ]
                )
                in_dim = h_dim
            layers.extend(
                [
                    nn.Linear(in_dim, n_input_features),
                    nn.GELU(),
                    nn.Dropout(params.get("final_dropout", 0.1)),
                ]
            )
            layers.append(nn.Linear(n_input_features, n_cell_types))
            layers.append(nn.Softmax(dim=-1))
            model = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown NN architecture: {name}")
        return model

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
          b. Fit LinearCalibrator on val, evaluate with 3 normalisation methods
          c. Fit VectorScalingCalibratorCV on val, evaluate on val and test
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
                f"    Found existing uncalibrated predictions at {uncalibrated_predictions_path}, loading instead of re-evaluating"
            )
            data = np.load(uncalibrated_predictions_path)
            val_pred = data["val_pred"]
            test_pred = data["test_pred"]
        else:
            val_pred = self._predict_with_model(name, model, features_valid, cfg)
            test_pred = self._predict_with_model(name, model, features_test, cfg)

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
        linear_calibrator_path = os.path.join(deconv_out, "linear_calibrator.npz")
        if os.path.exists(linear_calibrator_path):
            self.logger.info(
                f"    Found existing linear calibrator at {linear_calibrator_path}, loading instead of re-fitting"
            )
            linear_calibrator = LinearCalibrator()
            linear_calibrator.load_calibration_parameters(linear_calibrator_path)
        else:
            linear_calibrator = LinearCalibrator()
            linear_calibrator.fit(val_pred, y_valid)
            linear_calibrator.save_calibration_parameters(linear_calibrator_path)
            self.logger.info(f"    Saved linear calibrator to {linear_calibrator_path}")

        for norm_method in LINEAR_NORM_METHODS:
            short_name = norm_method.replace("-", "_")
            predictions_path = os.path.join(
                deconv_out, f"linear_{short_name}_predictions.npz"
            )
            if os.path.exists(predictions_path):
                self.logger.info(
                    f"    Found existing linear calibrated predictions for norm method '{norm_method}' at {predictions_path}, loading instead of re-predicting"
                )
                data = np.load(predictions_path)
                val_calib = data["val_pred"]
                test_calib = data["test_pred"]
            else:
                val_calib, _ = linear_calibrator.predict(
                    val_pred, norm_method=norm_method
                )
                test_calib, _ = linear_calibrator.predict(
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
        self.logger.info(f"  [{name}] Fitting VectorScalingCalibratorCV ...")

        vs_path = os.path.join(deconv_out, "vector_scaling_calibrator.npz")
        if os.path.exists(vs_path):
            self.logger.info(
                f"    Found existing VectorScalingCalibratorCV at {vs_path}, loading instead of re-fitting"
            )
            vs_calibrator = VectorScalingCalibratorCV()
            vs_calibrator.load(vs_path)
        else:
            # get config for vector scaling calibrator, log any missing parameters and their defaults
            vs_cfg = cfg.get("vector_scaling", {})
            if not vs_cfg:
                self.logger.info(
                    f"    Deconvolver config for '{name}' missing 'vector_scaling' section, using defaults for all parameters:  "
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
                        f"    Deconvolver config for '{name}' missing 'reg_lambda_list', using default [0.0, 1e-4, 1e-3]"
                    )
                if "lr_list" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'lr_list', using default [1e-3, 1e-2]"
                    )
                if "max_iter_list" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'max_iter_list', using default [1000]"
                    )
                if "optimizer" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'optimizer', using default 'adam'"
                    )
                if "scheduler" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'scheduler', using default 'plateau'"
                    )
                if "patience" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'patience', using default 50"
                    )
                if "n_folds" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'n_folds', using default 5"
                    )
                if "batch_size" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'batch_size', using default None (full batch)"
                    )
                if "verbose" not in vs_cfg:
                    self.logger.info(
                        f"    Deconvolver config for '{name}' missing 'verbose', using default True"
                    )

            vs_calibrator = VectorScalingCalibratorCV(
                reg_lambda_list=vs_cfg.get("reg_lambda_list", [0.0, 1e-4, 1e-3]),
                lr_list=vs_cfg.get("lr_list", [1e-3, 1e-2]),
                max_iter_list=vs_cfg.get("max_iter_list", [1000]),
                optimizer=vs_cfg.get("optimizer", "adam"),
                scheduler=vs_cfg.get("scheduler", "plateau"),
                patience=vs_cfg.get("patience", 50),
                n_folds=vs_cfg.get("n_folds", 5),
                batch_size=vs_cfg.get("batch_size", None),
                verbose=vs_cfg.get("verbose", True),
                plateau_factor=vs_cfg.get("plateau_factor", 0.5),
                plateau_patience=vs_cfg.get("plateau_patience", 10),
            )

            vs_calibrator.fit(val_pred, y_valid)
            vs_calibrator.save(vs_path)

            self.logger.info(f"    Saved VectorScaling calibrator to {vs_path}")

        self.logger.info(
            f"    Best CV loss: {vs_calibrator.best_cv_val_loss_:.6f}, "
            f"params: {vs_calibrator.best_params_}"
        )

        vs_pred_path = os.path.join(deconv_out, "vector_scaling_predictions.npz")
        if os.path.exists(vs_pred_path):
            self.logger.info(
                f"    Found existing VectorScalingCalibratorCV predictions at {vs_pred_path}, loading instead of re-predicting"
            )
            data = np.load(vs_pred_path)
            val_vs = data["val_pred"]
            test_vs = data["test_pred"]
        else:
            val_vs = vs_calibrator.predict_proba(val_pred)
            test_vs = vs_calibrator.predict_proba(test_pred)
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
            "best_params": vs_calibrator.best_params_,
            "best_cv_val_loss": vs_calibrator.best_cv_val_loss_,
        }

        return result

    # ═══════════════════════════════════════════════════════════════
    #  Prediction helper
    # ═══════════════════════════════════════════════════════════════

    def _predict_with_model(
        self,
        name: str,
        model: Any,
        X: np.ndarray,
        cfg: dict,
    ) -> np.ndarray:
        """Run inference with a fitted deconvolver."""
        params = cfg.get("params", {})

        if name == "xgb":
            # pylint: disable=protected-access
            raw = model._predict_raw(X)
            return model._transform_output(raw)

        elif name in ("swn", "mlp"):
            device = params.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
            model.eval()
            with torch.inference_mode():
                X_t = torch.FloatTensor(X).to(device)
                pred = model(X_t).cpu().numpy()
            return pred

        elif name == "nnls":
            pred, _, _ = model.predict(X, n_workers=1)
            return pred

        elif name == "psls":
            if not "n_workers" in params:
                self.logger.warning(
                    f"Deconvolver config for '{name}' missing 'n_workers' param, using default 2"
                )
            n_workers = params.get("n_workers", 2)
            return model.predict(X, n_workers=n_workers)

        else:
            raise ValueError(f"Cannot predict with model type '{name}'")

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
                        f"R2={m['overall_r2']:.6f}  "
                        f"LoA=[{m['loa_lower']:.6f}, {m['loa_upper']:.6f}]  "
                        f"LoA(worst)=[{m['worst_class_loa_lower']:.6f}, {m['worst_class_loa_upper']:.6f}]  "
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
                        "overall_r2": m["overall_r2"],
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
