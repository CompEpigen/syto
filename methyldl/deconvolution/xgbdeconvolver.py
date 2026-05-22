import os
from typing import Literal, Optional, Dict
import logging

import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import joblib
from dataclasses import dataclass

from methyldl.deconvolution.evaluation import compute_deconvolution_metrics

from methyldl.deconvolution.history import DeconvolutionHistory
from methyldl.deconvolution.loss import build_loss_from_config
from methyldl.deconvolution.abstract_deconvolver import AbstractDeconvolver

_module_logger = logging.getLogger(__name__)


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
    # useless but kept for backwards compatibility
    early_stopping_rounds: Optional[int] = 20
    random_state: int = 42


# XGBTrainingHistory has been replaced by DeconvolutionHistory
# (imported from methyldl.deconvolution.history)


class XGBoostDeconvolver(AbstractDeconvolver):
    """
    XGBoost-based deconvolver using diagonal and rejection features.

    Uses only:
    - Diagonal elements: x[i, i] for i in 0..n_dmr-1 (39 features)
    - Rejection column: x[:, -1] (39 features)
    Total: 78 features → 39 cell type proportions

    Parameters
    ----------
    n_gr_groups : int
        Number of GR groups (default: 39)
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
        n_gr_groups: int = 39,
        n_pred_classes: int = 40,
        n_cell_types: int = 39,
        config: Optional[XGBDeconvolverConfig] = None,
        output_transform: Literal[
            "none", "clip_normalize", "softmax"
        ] = "clip_normalize",
        logger=None,
    ):
        super().__init__()
        self.n_gr = n_gr_groups
        self.n_pred_classes = n_pred_classes
        self.n_cell_types = n_cell_types
        self.config = config or XGBDeconvolverConfig()
        self.output_transform = output_transform
        self.logger = logger if logger is not None else _module_logger

        self.model = None
        self.best_model = None
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

    def _predict_raw(
        self, X_features: np.ndarray  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """Get raw model predictions without transformation."""
        return self.model.predict(X_features)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostDeconvolver":
        """
        Fit the XGBoost deconvolver with metric tracking.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_samples, n_gr_groups, n_pred_classes)
        y_train : np.ndarray
            Training labels of shape (n_samples, n_cell_types)
        X_val : np.ndarray, optional
            Validation data
        y_val : np.ndarray, optional
            Validation labels
        loss: str, optional
            Selected loss for evaluation. Dedaults to the one set in
            build_loss_from_config
        loss_weights : dict, optional
            Weights for combined loss: {'mse': float, 'kl': float}
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed)

        Returns
        -------
        self : XGBoostDeconvolver
        """
        # pylint: disable=invalid-name
        criterion = build_loss_from_config(**kwargs)

        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)
        verbose = kwargs.get("verbose", 1)

        X_train_feat = X
        X_val_feat = X_val if X_val is not None else None

        if verbose:
            self.logger.info("Training XGBoost Deconvolver")
            self.logger.info(
                "  Input shape: %s → Features: %s", X.shape, X_train_feat.shape
            )
            self.logger.info("  Output shape: %s", y.shape)
            self.logger.info("  Output transform: %s", self.output_transform)
            self.logger.info("-" * 60)

        # Initialize history
        self.history = DeconvolutionHistory()

        # Build fresh model
        self.model = self._build_model(verbose=verbose)

        # Train for n_estimators rounds, checking validation each time
        # Note: This is a workaround since sklearn's MultiOutputRegressor
        # doesn't support per-round callbacks easily

        self.model.fit(X_train_feat, y)

        # Compute final metrics
        train_pred_raw = self._predict_raw(X_train_feat)
        train_pred = self._transform_output(train_pred_raw)
        train_metrics = compute_deconvolution_metrics(train_pred, y)
        train_loss = criterion(train_pred, y)

        self.cv_metric = train_loss  # For cross-validation model selection

        self.history.train_loss.append(train_loss)
        self.history.record_metrics(train_metrics, "train")

        if X_val is not None and y_val is not None:
            val_pred_raw = self._predict_raw(X_val_feat)
            val_pred = self._transform_output(val_pred_raw)
            val_metrics = compute_deconvolution_metrics(val_pred, y_val)
            val_loss = criterion(val_pred, y_val)
            self.cv_metric = val_loss  # For cross-validation model selection

            self.history.val_loss.append(val_loss)
            self.history.record_metrics(val_metrics, "val")

        self._is_fitted = True

        if verbose:
            self.logger.info("Training Results:")
            self.logger.info("  Train Loss: %f", train_loss)
            self.logger.info("  Train MAE:  %f", train_metrics.get("mae", float("nan")))
            self.logger.info("  Train MSE:  %f", train_metrics.get("mse", float("nan")))
            self.logger.info("  Train KL:   %f", train_metrics.get("kl", float("nan")))
            self.logger.info(
                "  Train Max Error: %f", train_metrics.get("max_error", float("nan"))
            )
            self.logger.info(
                "  Train Cosine Sim: %f", train_metrics.get("cosine_sim", float("nan"))
            )

            if X_val is not None:
                self.logger.info("Validation Results:")
                self.logger.info("  Val Loss:   %f", val_loss)
                self.logger.info(
                    "  Val MAE:    %f", val_metrics.get("mae", float("nan"))
                )
                self.logger.info(
                    "  Val MSE:    %f", val_metrics.get("mse", float("nan"))
                )
                self.logger.info(
                    "  Val KL:     %f", val_metrics.get("kl", float("nan"))
                )
                self.logger.info(
                    "  Val Max Error: %f", val_metrics.get("max_error", float("nan"))
                )
                self.logger.info(
                    "  Val Cosine Sim: %f", val_metrics.get("cosine_sim", float("nan"))
                )

        return self

    def get_cv_metric(self, X, y, **kwargs):
        return self.cv_metric

    @property
    def cv_metric_name(self) -> str:
        return "Validation Loss" if self.history.val_loss else "Training Loss"

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict cell type proportions.

        Parameters
        ----------
        X : np.ndarray
            Input data of target shape (features already extracted)

        Returns
        -------
        proportions : np.ndarray
            Predicted proportions of shape (n_samples, n_cell_types)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        single_sample = X.shape[0] == 1

        raw_output = self._predict_raw(X)
        proportions = self._transform_output(raw_output)

        if single_sample:
            proportions = proportions[0]

        return proportions


    def save(self, path: str, **kwargs) -> None:
        """
        Save the entire model object to disk using joblib.

        Parameters
        ----------
        path : str
            Path to save the model (e.g., 'model.pkl' or 'model.joblib')
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # We save 'self' which includes the config, the fitted model, and history
        joblib.dump(self, path)
        self.logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "XGBoostDeconvolver":
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Returns
        -------
        model : XGBoostDeconvolver
            The loaded model instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        model = joblib.load(path)

        # Basic validation to ensure it's the right class
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        if not hasattr(model, "logger"):
            model.logger = _module_logger

        model.logger.info("Model loaded from %s", path)
        return model


def train_xgb_deconvolver(
    X_train: np.ndarray,  # pylint: disable=invalid-name
    y_train: np.ndarray,
    X_val: np.ndarray,  # pylint: disable=invalid-name
    y_val: np.ndarray,
    config: Optional[XGBDeconvolverConfig] = None,
    n_gr_groups=39,
    n_pred_classes=40,
    n_cell_types=39,
    output_transform: Literal["none", "clip_normalize", "softmax"] = "clip_normalize",
    loss_weights: Optional[Dict[str, float]] = None,
    verbose: int = 1,
) -> tuple:
    """
    Convenience function matching the signature pattern of train_matrix_deconvolver.

    Returns
    -------
    model : XGBoostDeconvolver
        Trained model
    history : DeconvolutionHistory
        Training history
    """
    if config is None:
        config = XGBDeconvolverConfig()

    model = XGBoostDeconvolver(
        config=config,
        output_transform=output_transform,
        n_gr_groups=n_gr_groups,
        n_pred_classes=n_pred_classes,
        n_cell_types=n_cell_types,
    )

    model.fit(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        loss_weights=loss_weights,
        verbose=verbose,
    )

    return model, model.history
