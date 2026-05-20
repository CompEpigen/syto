"""
For now, placeholder for cross-validation engine.

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Type

from sklearn.model_selection import KFold, ParameterGrid
import numpy as np
import joblib
from tqdm import tqdm


class CrossValidationCompatibleModel(ABC):
    """Interface for models that can be used in the cross-validation engine."""

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, **kwargs  # pylint: disable=invalid-name
    ) -> "CrossValidationCompatibleModel":
        """Fit the model to the training data."""

    @abstractmethod
    def predict(
        self, X: np.ndarray, **kwargs  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """Predict on the test data."""

    @abstractmethod
    def get_cv_metric(
        self, X: np.ndarray, y: np.ndarray, **kwargs  # pylint: disable=invalid-name
    ) -> float:
        """Return a metric that can be used for model selection during cross-validation.
        The metric should be best when it is lower.
        The input values are the same as the validation data passed to the last call
        to fit(), so the model can use any relevant attributes set during fitting to
        return the metric.
        """

    @property
    @abstractmethod
    def cv_metric_name(self) -> str:
        """Return the name of the CV metric used for model selection, e.g. "val_loss"."""

    @abstractmethod
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "CrossValidationCompatibleModel":
        """Load the model from disk and return an instance of the model."""


class CrossValidationEngine(CrossValidationCompatibleModel):
    """
    Cross-validation engine that wraps the hyperparameter tuning and training
    of deconvolvers and calibrators.

    Grid-search with k-fold cross-validation for :class:`VectorScalingCalibrator`.

    Evaluates every combination of hyperparameters via k-fold CV, and
    retrains a final calibrator on the full dataset using the
    best-performing hyperparameters.
    """

    def __init__(self):
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, y: np.ndarray, **kwargs  # pylint: disable=invalid-name
    ) -> CrossValidationCompatibleModel:
        """Evaluates every hyperparameter combination via k-fold CV,
        then retrains a final :class:`VectorScalingCalibrator` on the full
        dataset with the best hyperparameters.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            **kwargs: Additional keyword arguments for configuring the CV engine:
                - model_class: The class of the model to be tuned (must implement
                  CrossValidationCompatibleModel).
                - model_param_grid: A dict specifying the hyperparameter grid to search.
                  Example: {"reg_lambda": [0.0, 1e-4], "lr": [1e-3, 1e-2]}.
                - n_folds: Number of folds for k-fold CV (default: 3).
                - random_state: Random seed for reproducibility (default: 42).
                - disable_pbar: If True, disables the tqdm progress bar (default: False).
        """
        # pylint: disable=attribute-defined-outside-init, invalid-name

        ## Load arguments from kwargs with assertions
        assert "model_class" in kwargs, "model_class must be provided in kwargs"
        assert (
            "model_param_grid" in kwargs
        ), "model_param_grid must be provided in kwargs"
        assert "n_folds" in kwargs, "n_folds must be provided in kwargs"
        self.model_class_: Type[CrossValidationCompatibleModel] = kwargs["model_class"]
        self.model_param_grid_: dict = kwargs["model_param_grid"]
        self.model_param_grid_ = {
            k: v if isinstance(v, list) else [v]
            for k, v in self.model_param_grid_.items()
        }
        self.n_folds_: int = kwargs.get("n_folds", 3)
        self.random_state_: int = kwargs.get("random_state", 42)
        disable_pbar = kwargs.get("disable_pbar", False)

        self.best_metric_ = float("inf")
        self.best_params_ = None
        self.metrics_per_param_ = {}

        kf = KFold(
            n_splits=self.n_folds_, shuffle=True, random_state=self.random_state_
        )
        param_grid = ParameterGrid(self.model_param_grid_)

        # grid search with k-fold CV
        with tqdm(
            total=len(param_grid) * self.n_folds_,
            desc="CV grid search",
            disable=disable_pbar,
        ) as pbar:
            for params in param_grid:
                fold_metrics = []
                for train_idx, val_idx in kf.split(X):
                    X_tr_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

                    model = self.model_class_(**params)
                    model.fit(X_tr_fold, y_tr_fold, X_val=X_val_fold, y_val=y_val_fold)
                    metric = model.get_cv_metric(X_val_fold, y_val_fold)
                    fold_metrics.append(metric)
                    pbar.update(1)
                mean_metric = np.mean(fold_metrics)
                self.metrics_per_param_[tuple(params.items())] = mean_metric
                if mean_metric < self.best_metric_:
                    self.best_metric_ = mean_metric
                    self.best_params_ = params
                    pbar.set_postfix(
                        {
                            "best_metric": f"{self.best_metric_:.7e}",
                            "best_params": self.best_params_,
                        }
                    )

        # train on the full dataset with the best hyperparameters
        self.best_model_ = self.model_class_(**self.best_params_)
        self.best_model_.fit(X, y)
        self.is_fitted = True
        return self

    @property
    def cv_metric_name(self) -> str:
        """Only present for compatibility with CrossValidationCompatibleModel interface."""
        if not self.is_fitted:
            return "Not fitted"
        return self.best_model_.cv_metric_name

    def get_cv_metric(self, X, y, **kwargs):
        """Only present for compatibility with CrossValidationCompatibleModel interface."""
        return self.best_metric_

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict using the best model found during cross-validation."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.best_model_.predict(X, **kwargs)

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the whole CV engine, including the best model and all relevant attributes."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        path = Path(path)
        assert (
            path.suffix == ".joblib"
        ), "Only .joblib format is supported for saving the CV engine"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=self, filename=path)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "CrossValidationEngine":
        path = Path(path)
        assert (
            path.suffix == ".joblib"
        ), "Only .joblib format is supported for loading the CV engine"
        return joblib.load(filename=path)
