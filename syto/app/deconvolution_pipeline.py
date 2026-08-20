"""
Deconvolution Model Fitting Pipeline
=====================================

End-to-end pipeline orchestrating:
    1. Pure cell-type profile loading from a consolidated pseudobulk HDF5 file
    2. Feature selection via cutoff-based mask
    3. Fitting / training deconvolution models (XGB, SWN, MLP, NNLS, PSLS)

The pipeline consumes the single ``pseudobulk.h5`` file produced by the new
:class:`~syto.data.pseudobulk_generator.PseudobulkGenerator` (see
``syto/app/pseudobulk_pipeline.py``).  Both the pseudobulk samples (model inputs /
outputs) and the pure cell-type profiles (used for feature selection and the
least-squares reference matrix) are read from this single file.
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from syto.data.pseudobulk_store import open_pseudobulk_store
from syto.deconvolution.evaluation import compute_deconvolution_metrics
from syto.deconvolution.feature_selection import (
    apply_feature_mask,
    compute_feature_mask,
    compute_feature_ratios,
    generate_feature_selection_plot,
)
from syto.deconvolution.least_squares_deconvolvers import (
    NNLSDeconvolver,
    PSLSDeconvolver,
)
from syto.deconvolution.xgbdeconvolver import (
    XGBoostDeconvolver,
    XGBDeconvolverConfig,
)
from syto.deconvolution.deep_deconvolvers.mlp import MLPDeconvolver
from syto.deconvolution.deep_deconvolvers.swn import SWNDeconvolver


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
        with open(labels_dict_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.num_output_labels = config.get("num_output_labels", len(self.labels_dict))
        self.num_input_labels = config.get("num_input_labels", self.num_output_labels)

        # Input: a pseudobulk store (pure profiles + pseudobulks), in either the
        # consolidated HDF5 or the columnar layout.
        self.pseudobulk_path = config["pseudobulk_path"]
        self.reader = open_pseudobulk_store(self.pseudobulk_path, logger=self.logger)

        # Output
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Feature selection
        self.top_features = config.get("top_features", None)
        self.cutoff = config.get("feature_cutoff", 1.1)
        self.splits = config.get("splits", ["train", "valid", "test"])
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
    #  Stage 1: Pure profile loading
    # ═══════════════════════════════════════════════════════════════

    def _stage1_pure_profiles(self) -> Dict[str, np.ndarray]:
        """Load purified cell-type profiles from the pseudobulk HDF5 file.

        Returns
        -------
        dict
            Mapping ``split_name -> pure feature matrix`` of shape
            ``(n_cell_types, n_gr_groups, n_pred_classes)`` containing only the
            ``prediction_{i}_wavg`` columns.
        """
        self.logger.info(
            f"Stage 1: Loading pure cell-type profiles from {self.pseudobulk_path}"
        )

        pure_profiles: Dict[str, np.ndarray] = {}
        for split_name in self.splits:
            matrix = self.reader.read_pure_feature_matrix(
                split_name, num_pred_classes=self.num_input_labels
            )
            pure_profiles[split_name] = matrix
            self.logger.info(
                f"  {split_name}: pure profiles {matrix.shape} "
                f"(cell_types, gr_groups, pred_classes)"
            )

        return pure_profiles

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2: Feature selection
    # ═══════════════════════════════════════════════════════════════

    def _stage2_feature_selection(
        self, pure_profiles: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute feature mask and apply it to the pseudobulk feature matrices."""
        ios_feature_selected_path = self.config.get("ios_feature_selected_path")

        if ios_feature_selected_path and os.path.exists(ios_feature_selected_path):
            self.logger.info(
                f"Stage 2: Loading already computed features from {ios_feature_selected_path}"
            )
            data = np.load(ios_feature_selected_path)
            feature_data = {k: data[k] for k in data.files}

            # Load the mask if provided explicitly or inside the npz
            mask_path = self.config.get("features_mask_path")
            if mask_path and os.path.exists(mask_path):
                self.logger.info(f"  Loading feature mask from {mask_path}")
                feature_data["mask"] = np.load(mask_path)["features_mask"]
            elif "mask" not in feature_data and "features_mask" in data.files:
                feature_data["mask"] = data["features_mask"]
            elif "mask" not in feature_data:
                self.logger.warning(
                    "  No feature mask loaded. LS deconvolvers may fail if used."
                )

            return feature_data

        if getattr(self, "top_features", None) is not None:
            self.logger.info(
                f"Stage 2: Feature selection for top {self.top_features} features ..."
            )
        else:
            self.logger.info(
                f"Stage 2: Feature selection with cutoff={self.cutoff} ..."
            )

        # Compute mask from the validation split's pure profiles
        if "valid" not in pure_profiles:
            raise ValueError(
                "A 'valid' split is required to compute the feature mask, "
                f"but only splits {list(pure_profiles)} were loaded."
            )
        pure_valid = pure_profiles["valid"]
        valid_ratios = compute_feature_ratios(pure_valid)

        if getattr(self, "top_features", None) is not None:
            mask, calc_cutoff = compute_feature_mask(
                valid_ratios,
                cutoff=None,
                guarantee_diagonal_selection=self.guarantee_diagonal,
                guarantee_columns_selection=self.guarantee_columns,
                top_features=self.top_features,
                return_cutoff=True,
            )
            self.cutoff = calc_cutoff
            self.logger.info(f"  Automatic cutoff evaluated as {self.cutoff:.4f}")
        else:
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
            f"(dynamic_cutoff={self.cutoff:.4f})"
        )

        # Save mask
        mask_path = os.path.join(
            self.output_dir, f"features_mask_{self.cutoff}_cutoff.npz"
        )
        np.savez_compressed(mask_path, features_mask=mask)
        self.logger.info(f"  Saved feature mask to {mask_path}")

        # Read pseudobulk matrices per split and apply the mask
        feature_data: Dict[str, np.ndarray] = {"mask": mask}
        for split_name in self.splits:
            features, proportions = self.reader.read_pseudobulk_matrices(
                split_name, num_pred_classes=self.num_input_labels
            )
            # features: (n_samples, n_gr_groups, n_pred_classes)
            feature_data[f"features_{split_name}"] = apply_feature_mask(features, mask)
            feature_data[f"proportions_{split_name}"] = proportions
            self.logger.info(
                f"  {split_name}: {features.shape[0]} pseudobulks -> "
                f"{feature_data[f'features_{split_name}'].shape[1]} selected features"
            )

        # Persist the feature-selected matrices for reuse / inspection
        filtered_path = os.path.join(
            self.output_dir, f"features_cutoff_{self.cutoff}.npz"
        )
        np.savez_compressed(filtered_path, **feature_data)
        self.logger.info(f"  Saved feature-selected matrices to {filtered_path}")

        # Optional plot
        if self.config.get("generate_feature_selection_plot", True):
            master_names = [
                self.labels_dict.get(i, str(i)) for i in range(self.num_output_labels)
            ]
            plot_path = os.path.join(
                self.output_dir,
                f"feature_selection_cutoff_{self.cutoff}.png",
            )
            generate_feature_selection_plot(
                pure_matrices_per_split=pure_profiles,
                num_input_labels=self.num_input_labels,
                num_output_labels=self.num_output_labels,
                cutoff=self.cutoff,
                output_path=plot_path,
                master_names=master_names,
                guarantee_diagonal_selection=self.guarantee_diagonal,
                guarantee_columns_selection=self.guarantee_columns,
                splits=self.splits,
                top_features=getattr(self, "top_features", None),
            )

        return feature_data

    # ═══════════════════════════════════════════════════════════════
    #  Stage 3: Fit deconvolvers
    # ═══════════════════════════════════════════════════════════════

    def _stage3_fit_deconvolvers(
        self,
        pure_profiles: Dict[str, np.ndarray],
        feature_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Fit all configured deconvolvers."""
        self.logger.info("Stage 3: Fitting deconvolvers ...")

        deconvolvers_cfg = self.config.get("deconvolvers", [])

        # Extract feature dict mapping split names to arrays.
        features_dict = {
            key[len("features_") :]: val
            for key, val in feature_data.items()
            if key.startswith("features_")
        }

        if "train" not in features_dict or "valid" not in features_dict:
            raise ValueError(
                "Both 'train' and 'valid' splits are required for fitting deconvolvers."
            )

        proportions_dict = {}
        for split_name in self.splits:
            pkey = f"proportions_{split_name}"
            if pkey in feature_data:
                proportions_dict[split_name] = feature_data[pkey]
            elif "proportions" in feature_data:
                proportions_dict[split_name] = feature_data["proportions"]

        if "train" not in proportions_dict or "valid" not in proportions_dict:
            raise ValueError(
                "Proportions for both 'train' and 'valid' splits are required."
            )

        mask = feature_data["mask"]

        results: Dict[str, Any] = {}

        for deconv_cfg in deconvolvers_cfg:
            name = deconv_cfg["name"]
            self.logger.info(f"  Fitting deconvolver: {name}")

            try:
                if name == "xgb":
                    metrics = self._fit_xgb(deconv_cfg, features_dict, proportions_dict)
                elif name in ("swn", "mlp"):
                    metrics = self._fit_nn(
                        name, deconv_cfg, features_dict, proportions_dict
                    )
                elif name in ("nnls", "psls"):
                    metrics = self._fit_ls(
                        name,
                        deconv_cfg,
                        pure_profiles,
                        mask,
                        features_dict,
                        proportions_dict,
                        self.splits,
                    )
                else:
                    self.logger.warning(f"  Unknown deconvolver '{name}', skipping")
                    continue

                results[name] = {"metrics": metrics}

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error(f"  Failed to fit '{name}': {e}", exc_info=True)

        # Log summary
        self._log_summary(results)
        self._save_summary_csv(results)

        return results

    # ───────────────────────────────────────────────────────────────
    #  XGBoost
    # ───────────────────────────────────────────────────────────────

    def _fit_xgb(
        self,
        cfg: dict,
        features: Dict[str, np.ndarray],
        proportions: Dict[str, np.ndarray],
    ) -> dict:
        """Fit XGBoost deconvolver."""
        # pylint: disable=invalid-name
        save_path = os.path.join(self.output_dir, "xgb_deconvolver.joblib")

        X_val = features["valid"]
        X_test = features.get("test")

        y_val = proportions["valid"]
        y_test = proportions.get("test")

        if os.path.exists(save_path):
            self.logger.info(
                f"  XGB model already exists at {save_path}, loading instead of fitting."
            )
            model = XGBoostDeconvolver.load(save_path)
        else:
            params = cfg.get("params", {})
            xgb_config = XGBDeconvolverConfig(**params)

            model = XGBoostDeconvolver(
                config=xgb_config,
                output_transform="clip_normalize",
                n_gr_groups=self.num_output_labels,
                n_pred_classes=self.num_output_labels,
                n_cell_types=self.num_output_labels,
            )

            X_train = features["train"]
            y_train = proportions["train"]

            model.fit(
                X=X_train,
                y=y_train,
                X_val=X_val,
                y_val=y_val,
                verbose=1,
            )

            # Save
            model.save(save_path)
            self.logger.info(f"    Saved XGB model to {save_path}")

        # Evaluate on test or fallback to valid
        eval_X = X_test if X_test is not None else X_val
        eval_y = y_test if y_test is not None else y_val
        eval_name = "test" if X_test is not None else "valid"

        test_pred = model.predict(eval_X)
        metrics = compute_deconvolution_metrics(test_pred, eval_y)
        self.logger.info(f"    XGB {eval_name} MAE: {metrics['mae']:.6f}")

        return metrics

    # ───────────────────────────────────────────────────────────────
    #  Neural Network (SWN / MLP)
    # ───────────────────────────────────────────────────────────────

    def _fit_nn(
        self,
        name: str,
        cfg: dict,
        features: Dict[str, np.ndarray],
        proportions: Dict[str, np.ndarray],
    ) -> dict:
        """Fit a neural network deconvolver (SWN or MLP)."""
        assert name in ("swn", "mlp"), f"Unknown NN architecture: {name}"
        # pylint: disable=invalid-name
        save_path = os.path.join(self.output_dir, f"{name}_best_deconvolver.pt")
        meta_path = os.path.join(self.output_dir, f"{name}_architecture_meta.json")

        params = cfg.get("params", {})
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        X_val = features["valid"]
        X_test = features.get("test")

        y_val = proportions["valid"]
        y_test = proportions.get("test")

        if os.path.exists(save_path) and os.path.exists(meta_path):
            self.logger.info(
                f"  {name.upper()} model already exists at {save_path} with metadata at {meta_path}, "
                "loading instead of fitting."
            )
            if name == "swn":
                model = SWNDeconvolver.load(path=save_path, metadata_path=meta_path)
            else:
                model = MLPDeconvolver.load(path=save_path, metadata_path=meta_path)
        else:

            X_train = features["train"]
            y_train = proportions["train"]

            n_input_features = X_train.shape[1]

            if name == "swn":
                model = SWNDeconvolver(
                    n_input_features, self.num_output_labels, logger=self.logger
                )
            else:
                model = MLPDeconvolver(
                    n_input_features, self.num_output_labels, logger=self.logger
                )

            model.fit(
                X=X_train,
                y=y_train,
                X_val=X_val,
                y_val=y_val,
                n_epochs=params.get("n_epochs", 100),
                batch_size=params.get("batch_size", 64),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 1e-4),
                early_stopping_metric=params.get("early_stopping_metric", "val_mae"),
                early_stopping_patience=params.get("early_stopping_patience", 15),
                scheduler_type=params.get("scheduler_type", "plateau"),
                verbose=params.get("verbose", 1),
                device=device,
            )

            # Save
            model.save(save_path, metadata_path=meta_path)
            self.logger.info(
                f"    Saved {name.upper()} model to {save_path} and metadata to {meta_path}"
            )

        # Evaluate on test or fallback to valid
        eval_X = X_test if X_test is not None else X_val
        eval_y = y_test if y_test is not None else y_val
        eval_name = "test" if X_test is not None else "valid"

        test_pred = model.predict(eval_X, device=device)
        metrics = compute_deconvolution_metrics(test_pred, eval_y)
        self.logger.info(f"    {name.upper()} {eval_name} MAE: {metrics['mae']:.6f}")

        return metrics

    # ───────────────────────────────────────────────────────────────
    #  Least Squares (NNLS / PSLS)
    # ───────────────────────────────────────────────────────────────

    def _fit_ls(
        self,
        name: str,
        cfg: dict,
        pure_profiles: Dict[str, np.ndarray],
        mask: np.ndarray,
        features: Dict[str, np.ndarray],
        proportions: Dict[str, np.ndarray],
        splits: List,
    ) -> dict:
        """Fit NNLS or PSLS deconvolver.

        The LS deconvolvers use a reference matrix from the **train**
        split of the purified profiles (after applying the feature mask
        derived from validation data).
        """
        # pylint: disable=invalid-name
        save_path = os.path.join(self.output_dir, f"{name}_deconvolver.joblib")
        params = cfg.get("params", {})

        if os.path.exists(save_path):
            self.logger.info(
                f"  {name.upper()} model already exists at {save_path}, loading instead of fitting."
            )
            if name == "nnls":
                model = NNLSDeconvolver.load(save_path)
            else:
                model = PSLSDeconvolver.load(save_path)
        else:

            # Build reference matrix from train-split pure profiles
            train_split = splits[0]
            pure_train = pure_profiles[train_split]
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

            # Save
            model.save(save_path)
            self.logger.info(f"    Saved {name.upper()} model to {save_path}")

        # Evaluate on test or fallback to valid
        X_test = features.get("test")
        X_val = features["valid"]
        eval_X = X_test if X_test is not None else X_val

        y_test = proportions.get("test")
        y_val = proportions["valid"]
        eval_y = y_test if y_test is not None else y_val

        eval_name = "test" if X_test is not None else "valid"

        if name == "nnls":
            test_pred = model.predict(eval_X, n_workers=1)
        else:
            n_workers = params.get("n_workers", 2)
            test_pred = model.predict(eval_X, n_workers=n_workers)

        metrics = compute_deconvolution_metrics(test_pred, eval_y)
        self.logger.info(f"    {name.upper()} {eval_name} MAE: {metrics['mae']:.6f}")

        return metrics

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of all fitted models and their test metrics."""
        self.logger.info("=" * 70)
        self.logger.info("DECONVOLUTION FITTING SUMMARY")
        self.logger.info("=" * 70)

        for model_name, data in results.items():
            if "metrics" in data:
                m = data["metrics"]
                self.logger.info(
                    f"  {model_name:30s}  "
                    f"R2={m.get('r2', 0.0):.6f}  "
                    f"LoA=[{m.get('loa_lower', 0.0):.6f}, {m.get('loa_upper', 0.0):.6f}]  "
                    f"LoA(worst)=[{m.get('worst_class_loa_lower', 0.0):.6f}, "
                    f"{m.get('worst_class_loa_upper', 0.0):.6f}]  "
                    f"MAE={m['mae']:.6f}  "
                    f"MSE={m['mse']:.6f}  "
                    f"KLDiv={m.get('kl', 0.0):.6f}"
                )

        self.logger.info("=" * 70)
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info("=" * 70)

    def _save_summary_csv(self, results: Dict[str, Any]) -> None:
        """Save a CSV with one row per model."""

        rows: list = []
        for model_name, data in results.items():
            if "metrics" not in data:
                continue
            m = data["metrics"]
            rows.append(
                {
                    "model": model_name,
                    "r2": m.get("r2"),
                    "loa_lower": m.get("loa_lower"),
                    "loa_upper": m.get("loa_upper"),
                    "worst_class_loa_lower": m.get("worst_class_loa_lower"),
                    "worst_class_loa_upper": m.get("worst_class_loa_upper"),
                    "mae": m["mae"],
                    "mse": m["mse"],
                    "kl": m.get("kl"),
                }
            )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "deconvolution_summary.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved deconvolution summary CSV to {csv_path}")
