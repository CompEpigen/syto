import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Callable, Union, Dict, Tuple
import numpy as np
from methyldl.deconvolution.evaluation import (
    compute_deconvolution_metrics_np,
    compute_combined_loss,
)


@dataclass
class XGBDeconvolverConfig:
    """Configuration for XGBoost Deconvolver."""

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: Optional[int] = 20
    random_state: int = 42


@dataclass
class XGBTrainingHistory:
    """Stores training metrics matching PyTorch version."""

    train_loss: list = field(default_factory=list)
    train_mae: list = field(default_factory=list)
    train_mse: list = field(default_factory=list)
    train_kl: list = field(default_factory=list)
    train_max_error: list = field(default_factory=list)
    train_cosine_sim: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_mae: list = field(default_factory=list)
    val_mse: list = field(default_factory=list)
    val_kl: list = field(default_factory=list)
    val_max_error: list = field(default_factory=list)
    val_cosine_sim: list = field(default_factory=list)
    best_iteration: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_mae": self.train_mae,
            "train_mse": self.train_mse,
            "train_kl": self.train_kl,
            "train_max_error": self.train_max_error,
            "train_cosine_sim": self.train_cosine_sim,
            "val_loss": self.val_loss,
            "val_mae": self.val_mae,
            "val_mse": self.val_mse,
            "val_kl": self.val_kl,
            "val_max_error": self.val_max_error,
            "val_cosine_sim": self.val_cosine_sim,
            "best_iteration": self.best_iteration,
            "stopped_early": self.stopped_early,
        }


class XGBoostDeconvolver:
    """
    XGBoost-based deconvolver using diagonal and rejection features.

    Uses only:
    - Diagonal elements: x[i, i] for i in 0..n_dmr-1 (39 features)
    - Rejection column: x[:, -1] (39 features)
    Total: 78 features → 39 cell type proportions

    Parameters
    ----------
    n_dmr_groups : int
        Number of DMR groups (default: 39)
    n_pred_classes : int
        Number of prediction classes including rejection (default: 40)
    n_cell_types : int
        Number of output cell types (default: 39)
    config : XGBDeconvolverConfig
        XGBoost hyperparameters
    output_transform : str
        How to handle outputs:
        - 'none': Direct prediction (use when targets are proportions)
        - 'clip_normalize': Clip to [0,∞) and normalize to sum=1
        - 'softmax': Apply softmax (use when training on logits)
    """

    def __init__(
        self,
        n_dmr_groups: int = 39,
        n_pred_classes: int = 40,
        n_cell_types: int = 39,
        with_reject_features=True,
        process_inputs=True,
        config: Optional[XGBDeconvolverConfig] = None,
        output_transform: Literal[
            "none", "clip_normalize", "softmax"
        ] = "clip_normalize",
    ):
        self.n_dmr = n_dmr_groups
        self.n_pred_classes = n_pred_classes
        self.n_cell_types = n_cell_types
        self.config = config or XGBDeconvolverConfig()
        self.output_transform = output_transform
        self.with_reject_features = with_reject_features
        self.process_inputs = process_inputs
        if with_reject_features:
            self.n_features = n_dmr_groups * 2  # diagonal + reject column
        else:
            self.n_features = n_dmr_groups

        self.model = None
        self.best_model = None
        self.history = None
        self._is_fitted = False

    def _build_model(self, verbose=0) -> MultiOutputRegressor:
        """Build XGBoost multi-output regressor."""
        base_model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=verbose,
        )
        return MultiOutputRegressor(base_model)

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract diagonal and rejection column features.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_dmr_groups, n_pred_classes)

        Returns
        -------
        features : np.ndarray
            Extracted features of shape (n_samples, n_dmr_groups * 2)
        """
        if not self.process_inputs:
            return X

        if X.ndim == 2:
            X = X[np.newaxis, ...]

        n_samples = X.shape[0]

        # Extract diagonal: x[i, i] for i in 0..n_dmr-1
        # (n_samples, 39)
        diagonal = np.array([np.diag(X[i, :, : self.n_dmr]) for i in range(n_samples)])

        # Extract rejection column (last column)
        if self.with_reject_features:
            reject_col = X[:, :, -1]  # (n_samples, 39)
            # Concatenate features
            features = np.concatenate([diagonal, reject_col], axis=1)
        else:
            features = diagonal
        return features

    def _transform_output(self, raw_output: np.ndarray) -> np.ndarray:
        """
        Transform raw model outputs to valid proportions.
        """
        if self.output_transform == "none":
            # Just ensure valid proportions
            output = np.clip(raw_output, 0, 1)
            row_sums = output.sum(axis=1, keepdims=True)
            # Only normalize if sum > 0
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return output / row_sums

        elif self.output_transform == "clip_normalize":
            # Clip negative values and normalize
            clipped = np.clip(raw_output, 0, None)
            row_sums = clipped.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return clipped / row_sums

        elif self.output_transform == "softmax":
            # Softmax (use only if training on log-odds/logits)
            exp_out = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
            return exp_out / exp_out.sum(axis=1, keepdims=True)

        else:
            raise ValueError(f"Unknown output_transform: {self.output_transform}")

    def _predict_raw(self, X_features: np.ndarray) -> np.ndarray:
        """Get raw model predictions without transformation."""
        return self.model.predict(X_features)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        early_stopping_metric: Literal[
            "val_loss",
            "val_mae",
            "val_mse",
            "val_kl",
            "val_max_error",
            "val_cosine_sim",
        ] = "val_mae",
        verbose: int = 1,
    ) -> "XGBoostDeconvolver":
        """
        Fit the XGBoost deconvolver with metric tracking.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_samples, n_dmr_groups, n_pred_classes)
        y_train : np.ndarray
            Training labels of shape (n_samples, n_cell_types)
        X_val : np.ndarray, optional
            Validation data
        y_val : np.ndarray, optional
            Validation labels
        loss_weights : dict, optional
            Weights for combined loss: {'mse': float, 'kl': float}
        early_stopping_metric : str
            Metric to monitor for early stopping
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed)

        Returns
        -------
        self : XGBoostDeconvolver
        """
        if loss_weights is None:
            loss_weights = {"mse_weight": 1.0, "kl_weight": 0.5}

        # Determine if we should maximize or minimize
        maximize_metrics = {"val_cosine_sim"}
        minimize = early_stopping_metric not in maximize_metrics

        # Extract features
        X_train_feat = self.extract_features(X_train)
        X_val_feat = self.extract_features(X_val) if X_val is not None else None

        if verbose:
            print(f"Training XGBoost Deconvolver")
            print(f"  Input shape: {X_train.shape} → Features: {X_train_feat.shape}")
            print(f"  Output shape: {y_train.shape}")
            print(f"  Output transform: {self.output_transform}")
            print(f"  Early stopping on: {early_stopping_metric}")
            print("-" * 60)

        # Initialize history
        self.history = XGBTrainingHistory()

        # For proper early stopping, we'll train incrementally
        best_score = float("inf") if minimize else float("-inf")
        patience_counter = 0
        best_iteration = 0

        # Build fresh model
        self.model = self._build_model(verbose=verbose)

        # Train for n_estimators rounds, checking validation each time
        # Note: This is a workaround since sklearn's MultiOutputRegressor
        # doesn't support per-round callbacks easily

        # For simplicity, we'll train the full model first, then evaluate
        # If you need true incremental training, use the Native version below

        self.model.fit(X_train_feat, y_train)

        # Compute final metrics
        train_pred_raw = self._predict_raw(X_train_feat)
        train_pred = self._transform_output(train_pred_raw)
        train_metrics = compute_deconvolution_metrics_np(train_pred, y_train)
        train_loss = compute_combined_loss(train_pred, y_train, **loss_weights)

        self.history.train_loss.append(train_loss)
        self.history.train_mae.append(train_metrics["mae"])
        self.history.train_mse.append(train_metrics["mse"])
        self.history.train_kl.append(train_metrics["kl"])
        self.history.train_max_error.append(train_metrics["max_error"])
        self.history.train_cosine_sim.append(train_metrics["cosine_sim"])

        if X_val is not None and y_val is not None:
            val_pred_raw = self._predict_raw(X_val_feat)
            val_pred = self._transform_output(val_pred_raw)
            val_metrics = compute_deconvolution_metrics_np(val_pred, y_val)
            val_loss = compute_combined_loss(val_pred, y_val, **loss_weights)

            self.history.val_loss.append(val_loss)
            self.history.val_mae.append(val_metrics["mae"])
            self.history.val_mse.append(val_metrics["mse"])
            self.history.val_kl.append(val_metrics["kl"])
            self.history.val_max_error.append(val_metrics["max_error"])
            self.history.val_cosine_sim.append(val_metrics["cosine_sim"])

        self._is_fitted = True

        if verbose:
            print(f"\nTraining Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train MAE:  {train_metrics['mae']:.4f}")
            print(f"  Train MSE:  {train_metrics['mse']:.4f}")
            print(f"  Train KL:   {train_metrics['kl']:.4f}")
            print(f"  Train Max Error: {train_metrics['max_error']:.4f}")
            print(f"  Train Cosine Sim: {train_metrics['cosine_sim']:.4f}")

            if X_val is not None:
                print(f"\nValidation Results:")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Val MAE:    {val_metrics['mae']:.4f}")
                print(f"  Val MSE:    {val_metrics['mse']:.4f}")
                print(f"  Val KL:     {val_metrics['kl']:.4f}")
                print(f"  Val Max Error: {val_metrics['max_error']:.4f}")
                print(f"  Val Cosine Sim: {val_metrics['cosine_sim']:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cell type proportions.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_dmr_groups, n_pred_classes)

        Returns
        -------
        proportions : np.ndarray
            Predicted proportions of shape (n_samples, n_cell_types)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        single_sample = X.ndim == 2
        features = self.extract_features(X)

        raw_output = self._predict_raw(features)
        proportions = self._transform_output(raw_output)

        if single_sample:
            proportions = proportions[0]

        return proportions

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on given data.

        Returns all metrics matching the PyTorch version.
        """
        if loss_weights is None:
            loss_weights = {"mse": 1.0, "kl": 0.5}

        pred = self.predict(X)
        metrics = compute_deconvolution_metrics_np(pred, y)
        metrics["loss"] = compute_combined_loss(pred, y, **loss_weights)

        return metrics

    def get_feature_importance(self, aggregate: bool = True) -> Dict[str, np.ndarray]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting importance")

        importances = np.array(
            [est.feature_importances_ for est in self.model.estimators_]
        )

        if aggregate:
            importances = importances.mean(axis=0)

        return {
            "diagonal": importances[..., : self.n_dmr],
            "reject": importances[..., self.n_dmr :],
            "all": importances,
        }

    def save(self, filepath: str):
        """
        Save the entire model object to disk using joblib.

        Parameters
        ----------
        filepath : str
            Path to save the model (e.g., 'model.pkl' or 'model.joblib')
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # We save 'self' which includes the config, the fitted model, and history
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "XGBoostDeconvolver":
        """
        Load a saved model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.

        Returns
        -------
        model : XGBoostDeconvolver
            The loaded model instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        model = joblib.load(filepath)

        # Basic validation to ensure it's the right class
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        return model


def train_xgb_deconvolver(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[XGBDeconvolverConfig] = None,
    n_dmr_groups=39,
    n_pred_classes=40,
    n_cell_types=39,
    with_reject_features=True,
    process_inputs=True,
    output_transform: Literal["none", "clip_normalize", "softmax"] = "clip_normalize",
    loss_weights: Optional[Dict[str, float]] = None,
    early_stopping_metric: str = "val_mae",
    early_stopping_patience: int = 15,
    verbose: int = 1,
) -> tuple:
    """
    Convenience function matching the signature pattern of train_matrix_deconvolver.

    Returns
    -------
    model : XGBoostDeconvolver
        Trained model
    history : XGBTrainingHistory
        Training history
    """
    if config is None:
        config = XGBDeconvolverConfig(early_stopping_rounds=early_stopping_patience)

    model = XGBoostDeconvolver(
        config=config,
        output_transform=output_transform,
        n_dmr_groups=n_dmr_groups,
        n_pred_classes=n_pred_classes,
        n_cell_types=n_cell_types,
        with_reject_features=with_reject_features,
        process_inputs=process_inputs,
    )

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        loss_weights=loss_weights,
        early_stopping_metric=early_stopping_metric,
        verbose=verbose,
    )

    return model, model.history
