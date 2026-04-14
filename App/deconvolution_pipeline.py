"""
Deconvolution Model Fitting Pipeline
=====================================

End-to-end pipeline orchestrating:
    1. Pure cell-type profile generation from predicted splits
    2. Feature selection via cutoff-based mask
    3. Fitting / training deconvolution models (XGB, SWN, MLP, NNLS, PSLS)
       with optional linear calibration
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from methyldl.data.pure_profile_generation import generate_pure_profiles
from methyldl.deconvolution.evaluation import compute_deconvolution_metrics
from methyldl.deconvolution.feature_selection import (
    apply_feature_mask,
    apply_mask_to_ios,
    compute_feature_mask,
    compute_feature_ratios,
    extract_pure_feature_matrix,
    generate_feature_selection_plot,
)
from methyldl.deconvolution.least_squares_deconvolvers import (
    NNLSDeconvolver,
    PSLSDeconvolver,
)
from methyldl.deconvolution.linear_calibrator import LinearCalibrator
from methyldl.deconvolution.xgbdeconvolver import (
    XGBoostDeconvolver,
    XGBDeconvolverConfig,
)
from methyldl.deconvolution.deep_deconvolvers.training import (
    train_matrix_deconvolver,
)


class DeconvolutionFittingPipeline:
    """Orchestrate deconvolution model fitting end-to-end.

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
        with open(labels_dict_path, "r") as f:
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.num_output_labels = config.get("num_output_labels", len(self.labels_dict))
        self.num_input_labels = config.get("num_input_labels", self.num_output_labels)

        # Output
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Feature selection
        self.cutoff = config.get("feature_cutoff", 1.1)
        self.guarantee_diagonal = config.get("guarantee_diagonal_selection", False)
        self.guarantee_columns = config.get("guarantee_columns_selection", [])

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """Execute the full deconvolution fitting pipeline.

        Returns
        -------
        dict
            Summary of fitted models and evaluation metrics.
        """
        # ── Stage 1: Pure profiles ────────────────────────────────
        pure_profiles = self._stage1_pure_profiles()

        # ── Stage 2: Feature selection ────────────────────────────
        feature_data = self._stage2_feature_selection(pure_profiles)

        # ── Stage 3: Fit deconvolvers ─────────────────────────────
        results = self._stage3_fit_deconvolvers(pure_profiles, feature_data)

        return results

    # ═══════════════════════════════════════════════════════════════
    #  Stage 1: Pure profile generation
    # ═══════════════════════════════════════════════════════════════

    def _stage1_pure_profiles(self) -> list:
        """Generate or load purified cell-type profiles."""
        pure_profiles_path = self.config.get("pure_profiles_path")

        if pure_profiles_path and os.path.exists(pure_profiles_path):
            self.logger.info(
                f"Stage 1: Loading pre-computed pure profiles from "
                f"{pure_profiles_path}"
            )
            with open(pure_profiles_path, "rb") as f:
                pure_profiles = pickle.load(f)
            self.logger.info(
                f"  Loaded {len(pure_profiles)} pure profiles"
            )
            return pure_profiles

        self.logger.info("Stage 1: Generating purified cell-type profiles ...")

        # Load predicted splits
        train, valid, test = self._load_predicted_splits()

        n_read_per_split = self.config.get("n_read_per_split", 475_000)

        pure_profiles = generate_pure_profiles(
            train_data=train,
            valid_data=valid,
            test_data=test,
            num_output_labels=self.num_output_labels,
            num_input_labels=self.num_input_labels,
            n_read_per_split=n_read_per_split,
        )

        # Save
        save_path = os.path.join(self.output_dir, "pure_profiles.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pure_profiles, f)
        self.logger.info(
            f"  Generated and saved {len(pure_profiles)} pure profiles "
            f"to {save_path}"
        )

        return pure_profiles

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2: Feature selection
    # ═══════════════════════════════════════════════════════════════

    def _stage2_feature_selection(
        self, pure_profiles: list
    ) -> Dict[str, np.ndarray]:
        """Compute feature mask and apply to full IO matrices."""
        self.logger.info(
            f"Stage 2: Feature selection with cutoff={self.cutoff} ..."
        )

        # Compute mask from validation split
        pure_valid = extract_pure_feature_matrix(
            pure_profiles, self.num_input_labels, self.num_output_labels, split_idx=1
        )
        valid_ratios = compute_feature_ratios(pure_valid)
        mask = compute_feature_mask(
            valid_ratios,
            self.cutoff,
            guarantee_diagonal_selection=self.guarantee_diagonal,
            guarantee_columns_selection=self.guarantee_columns,
        )

        n_selected = int(mask.sum())
        total = self.num_output_labels * self.num_input_labels
        self.logger.info(
            f"  Selected {n_selected}/{total} features "
            f"(cutoff={self.cutoff})"
        )

        # Save mask
        mask_path = os.path.join(
            self.output_dir, f"features_mask_{self.cutoff}_cutoff.npz"
        )
        np.savez_compressed(mask_path, features_mask=mask)
        self.logger.info(f"  Saved feature mask to {mask_path}")

        # Apply mask to full IO matrices
        ios_path = self.config["ios_full_matrices_path"]
        filtered_path = os.path.join(
            self.output_dir, f"features_cutoff_{self.cutoff}.npz"
        )
        feature_data = apply_mask_to_ios(
            ios_path=ios_path,
            mask=mask,
            output_path=filtered_path,
            cutoff=self.cutoff,
        )
        feature_data["mask"] = mask

        # Optional plot
        if self.config.get("generate_feature_selection_plot", True):
            master_names = [
                self.labels_dict.get(i, str(i))
                for i in range(self.num_output_labels)
            ]
            plot_path = os.path.join(
                self.output_dir,
                f"feature_selection_cutoff_{self.cutoff}.png",
            )
            generate_feature_selection_plot(
                pure_profiles=pure_profiles,
                num_input_labels=self.num_input_labels,
                num_output_labels=self.num_output_labels,
                cutoff=self.cutoff,
                output_path=plot_path,
                master_names=master_names,
                guarantee_diagonal_selection=self.guarantee_diagonal,
                guarantee_columns_selection=self.guarantee_columns,
            )

        return feature_data

    # ═══════════════════════════════════════════════════════════════
    #  Stage 3: Fit deconvolvers
    # ═══════════════════════════════════════════════════════════════

    def _stage3_fit_deconvolvers(
        self,
        pure_profiles: list,
        feature_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Fit all configured deconvolvers and optional calibrators."""
        self.logger.info("Stage 3: Fitting deconvolvers ...")

        deconvolvers_cfg = self.config.get("deconvolvers", [])
        features_train = feature_data["features_train"]
        features_valid = feature_data["features_valid"]
        features_test = feature_data["features_test"]
        proportions = feature_data["proportions"]
        mask = feature_data["mask"]

        results: Dict[str, Any] = {}

        for deconv_cfg in deconvolvers_cfg:
            name = deconv_cfg["name"]
            self.logger.info(f"  Fitting deconvolver: {name}")

            try:
                if name == "xgb":
                    model, metrics = self._fit_xgb(
                        deconv_cfg, features_train, proportions,
                        features_valid, proportions, features_test, proportions,
                    )
                elif name in ("swn", "mlp"):
                    model, metrics = self._fit_nn(
                        name, deconv_cfg, features_train, proportions,
                        features_valid, proportions, features_test, proportions,
                    )
                elif name in ("nnls", "psls"):
                    model, metrics = self._fit_ls(
                        name, deconv_cfg, pure_profiles, mask,
                        features_train, proportions,
                        features_valid, proportions,
                        features_test, proportions,
                    )
                else:
                    self.logger.warning(
                        f"  Unknown deconvolver '{name}', skipping"
                    )
                    continue

                results[name] = {"metrics": metrics}

                # Optional calibration
                if deconv_cfg.get("calibrate", False):
                    self.logger.info(f"  Fitting linear calibrator for {name}")
                    calib_metrics = self._fit_calibrator(
                        name, model, deconv_cfg,
                        features_valid, proportions,
                        features_test, proportions,
                    )
                    results[f"{name}_calibrated"] = {
                        "metrics": calib_metrics
                    }

            except Exception as e:
                self.logger.error(
                    f"  Failed to fit '{name}': {e}", exc_info=True
                )

        # Log summary
        self._log_summary(results)

        return results

    # ───────────────────────────────────────────────────────────────
    #  XGBoost
    # ───────────────────────────────────────────────────────────────

    def _fit_xgb(
        self,
        cfg: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[XGBoostDeconvolver, dict]:
        """Fit XGBoost deconvolver."""
        params = cfg.get("params", {})
        xgb_config = XGBDeconvolverConfig(**params)

        n_features = X_train.shape[1]
        
        model = XGBoostDeconvolver(
            config=xgb_config,
            output_transform="clip_normalize",
            n_dmr_groups=self.num_output_labels,
            n_pred_classes=self.num_output_labels,
            n_cell_types=self.num_output_labels,
            with_reject_features=False,
            process_inputs=False,
        )

        model.fit(
            X_train, y_train,
            X_val, y_val,
            verbose=1,
        )

        # Evaluate on test
        test_pred_raw = model._predict_raw(X_test)
        test_pred = model._transform_output(test_pred_raw)
        metrics = compute_deconvolution_metrics(test_pred, y_test)
        self.logger.info(f"    XGB test MAE: {metrics['mae']:.6f}")

        # Save
        save_path = os.path.join(self.output_dir, "xgb_deconvolver.joblib")
        model.save(save_path)
        self.logger.info(f"    Saved XGB model to {save_path}")

        return model, metrics

    # ───────────────────────────────────────────────────────────────
    #  Neural Network (SWN / MLP)
    # ───────────────────────────────────────────────────────────────

    def _build_nn_model(
        self,
        name: str,
        n_input_features: int,
        params: dict,
    ) -> nn.Module:
        """Build a neural network deconvolver from config."""
        n_cell_types = self.num_output_labels

        if name == "swn":
            hidden_dim = params.get("hidden_dim", 1024)
            model = nn.Sequential(
                nn.Linear(n_input_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(params.get("dropout", 0.2)),
                nn.Linear(hidden_dim, n_cell_types),
                nn.Softmax(dim=-1),
            )
        elif name == "mlp":
            hidden_dims = params.get("hidden_dims", [512, 256])
            layers = []
            in_dim = n_input_features
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                ])
                in_dim = h_dim
            # Final extra hidden layer matching the input
            layers.extend([
                nn.Linear(in_dim, n_input_features),
                nn.GELU(),
                nn.Dropout(params.get("final_dropout", 0.1)),
            ])
            layers.append(nn.Linear(n_input_features, n_cell_types))
            layers.append(nn.Softmax(dim=-1))
            model = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown NN architecture: {name}")

        return model

    def _fit_nn(
        self,
        name: str,
        cfg: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[nn.Module, dict]:
        """Fit a neural network deconvolver (SWN or MLP)."""
        params = cfg.get("params", {})
        n_input_features = X_train.shape[1]

        model = self._build_nn_model(name, n_input_features, params)

        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        trained_model, history = train_matrix_deconvolver(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_epochs=params.get("n_epochs", 100),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
            device=device,
            early_stopping_metric=params.get(
                "early_stopping_metric", "val_mae"
            ),
            early_stopping_patience=params.get(
                "early_stopping_patience", 15
            ),
            scheduler_type=params.get("scheduler_type", "plateau"),
            verbose=params.get("verbose", 1),
        )

        # Evaluate on test
        trained_model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            test_pred = trained_model(X_test_t).cpu().numpy()
        metrics = compute_deconvolution_metrics(test_pred, y_test)
        self.logger.info(f"    {name.upper()} test MAE: {metrics['mae']:.6f}")

        # Save
        save_path = os.path.join(
            self.output_dir, f"{name}_best_deconvolver.pt"
        )
        torch.save(trained_model.state_dict(), save_path)
        self.logger.info(f"    Saved {name.upper()} model to {save_path}")

        # Also save architecture metadata for later loading
        meta_path = os.path.join(
            self.output_dir, f"{name}_architecture_meta.json"
        )
        meta = {
            "name": name,
            "n_input_features": n_input_features,
            "n_cell_types": self.num_output_labels,
            "params": params,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return trained_model, metrics

    # ───────────────────────────────────────────────────────────────
    #  Least Squares (NNLS / PSLS)
    # ───────────────────────────────────────────────────────────────

    def _fit_ls(
        self,
        name: str,
        cfg: dict,
        pure_profiles: list,
        mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Any, dict]:
        """Fit NNLS or PSLS deconvolver.

        The LS deconvolvers use a reference matrix from the **train**
        split of the purified profiles (after applying the feature mask
        derived from validation data).
        """
        params = cfg.get("params", {})

        # Build reference matrix from train-split pure profiles
        pure_train = extract_pure_feature_matrix(
            pure_profiles, self.num_input_labels, self.num_output_labels, split_idx=0
        )
        # pure_train shape: (n_cell_types, n_dmr_groups, n_pred_classes)
        # Apply mask to each cell type's profile
        reference_features = apply_feature_mask(pure_train, mask)
        # reference_features shape: (n_cell_types, n_selected_features)

        labels = np.arange(self.num_output_labels)

        if name == "nnls":
            model = NNLSDeconvolver()
        elif name == "psls":
            solver_type = params.get("solver_type", "cvxpy")
            model = PSLSDeconvolver(solver_type=solver_type)
        else:
            raise ValueError(f"Unknown LS variant: {name}")

        model.fit(reference_features, labels)

        # Evaluate on test
        if name == "nnls":
            test_pred, _, _ = model.predict(X_test, n_workers=1)
        else:
            n_workers = params.get("n_workers", 2)
            test_pred = model.predict(X_test, n_workers=n_workers)

        metrics = compute_deconvolution_metrics(test_pred, y_test)
        self.logger.info(
            f"    {name.upper()} test MAE: {metrics['mae']:.6f}"
        )

        # Save
        save_path = os.path.join(
            self.output_dir, f"{name}_deconvolver.joblib"
        )
        model.save(save_path)
        self.logger.info(f"    Saved {name.upper()} model to {save_path}")

        return model, metrics

    # ───────────────────────────────────────────────────────────────
    #  Linear Calibration
    # ───────────────────────────────────────────────────────────────

    def _fit_calibrator(
        self,
        deconv_name: str,
        model: Any,
        cfg: dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Fit a linear calibrator on validation predictions.

        1. Run the deconvolver on the validation set.
        2. Fit ``LinearCalibrator(predicted, ground_truth)``.
        3. Apply calibration to the test predictions.
        4. Evaluate and save.
        """
        # Get validation predictions
        val_pred = self._predict_with_model(
            deconv_name, model, X_val, cfg
        )

        # Fit calibrator
        calibrator = LinearCalibrator()
        calibrator.fit(val_pred, y_val)

        # Get test predictions and calibrate
        test_pred = self._predict_with_model(
            deconv_name, model, X_test, cfg
        )
        calibrated_pred, _ = calibrator.predict(test_pred)

        metrics = compute_deconvolution_metrics(calibrated_pred, y_test)
        self.logger.info(
            f"    {deconv_name}_calibrated test MAE: "
            f"{metrics['mae']:.6f}"
        )

        # Save calibrator
        calib_path = os.path.join(
            self.output_dir, f"{deconv_name}_linear_calibrator.npz"
        )
        calibrator.save_calibration_parameters(calib_path)
        self.logger.info(f"    Saved calibrator to {calib_path}")

        return metrics

    def _predict_with_model(
        self,
        name: str,
        model: Any,
        X: np.ndarray,
        cfg: dict,
    ) -> np.ndarray:
        """Run inference with a fitted deconvolver.

        Handles the different prediction interfaces of XGB, NN, and LS
        models uniformly.
        """
        params = cfg.get("params", {})

        if name == "xgb":
            raw = model._predict_raw(X)
            return model._transform_output(raw)

        elif name in ("swn", "mlp"):
            device = params.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                pred = model(X_t).cpu().numpy()
            return pred

        elif name == "nnls":
            pred, _, _ = model.predict(X, n_workers=1)
            return pred

        elif name == "psls":
            n_workers = params.get("n_workers", 2)
            return model.predict(X, n_workers=n_workers)

        else:
            raise ValueError(f"Cannot predict with model type '{name}'")

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    def _load_predicted_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load predicted train/valid/test splits from pickle."""
        paths = self.config["predicted_splits"]

        self.logger.info("  Loading predicted splits ...")
        with open(paths["train"], "rb") as f:
            train = pickle.load(f)
        with open(paths["valid"], "rb") as f:
            valid = pickle.load(f)
        with open(paths["test"], "rb") as f:
            test = pickle.load(f)

        self.logger.info(
            f"    train={len(train)}, valid={len(valid)}, test={len(test)}"
        )
        return train, valid, test

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of all fitted models and their test metrics."""
        self.logger.info("=" * 60)
        self.logger.info("DECONVOLUTION FITTING SUMMARY")
        self.logger.info("=" * 60)

        for model_name, data in results.items():
            metrics = data["metrics"]
            self.logger.info(
                f"  {model_name:25s}  "
                f"MAE={metrics['mae']:.6f}  "
                f"MSE={metrics['mse']:.6f}  "
                f"CosSim={metrics['cosine_sim']:.6f}"
            )

        self.logger.info("=" * 60)
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info("=" * 60)
