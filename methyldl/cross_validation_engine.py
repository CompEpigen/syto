"""
For now, placeholder for cross-validation engine.

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

# from sklearn.model_selection import KFold, ParameterGrid
import numpy as np


class CrossValidationCompatibleModel(ABC):
    """Interface for models that can be used in the cross-validation engine."""

    @abstractmethod  # pylint: disable-next=invalid-name
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CrossValidationCompatibleModel":
        """Fit the model to the training data."""

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:  # pylint: disable=invalid-name
        """Predict on the test data."""

    @abstractmethod
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "CrossValidationCompatibleModel":
        """Load the model from disk and return an instance of the model."""


class CrossValidationEngine:
    """
    Cross-validation engine that wraps the hyperparameter tuning and training
    of deconvolvers and calibrators.
    """

    def __init__(self):
        pass


# class VectorScalingCalibratorCV():
#     """Grid-search with k-fold cross-validation for :class:`VectorScalingCalibrator`.

#     Evaluates every combination of hyperparameters via k-fold CV, and
#     retrains a final calibrator on the full dataset using the
#     best-performing hyperparameters.

#     Args:
#         reg_lambda_list: List of L2 regularisation strengths to search over.
#         lr_list: List of learning rates to search over.
#         max_iter_list: List of maximum epoch counts to search over.
#         optimizer: Optimizer passed to each inner calibrator ("adam" or "sgd").
#         scheduler: LR scheduler passed to each inner calibrator
#             ("plateau", "constant", "linear", or "cosine").
#         patience: Early-stopping patience passed to each inner calibrator.
#         tol: Minimum loss improvement for early stopping.
#         n_folds: Number of cross-validation folds (default 5).

#     Attributes:
#         best_calibrator_: The :class:`VectorScalingCalibrator` retrained on the
#             full dataset with the best hyperparameters. ``None`` before fitting.
#         best_params_: Dictionary of the best hyperparameter combination found
#             by CV. ``None`` before fitting.
#         best_cv_val_loss_: Mean validation loss across folds for the best
#             hyperparameter combination. ``None`` before fitting.
#     """

#     def __init__(
#         self,
#         reg_lambda_list: list[float] = None,
#         lr_list: list[float] = None,
#         max_iter_list: list[int] = None,
#         optimizer: str = "adam",
#         scheduler: str = "plateau",
#         patience: int = 50,
#         tol: float = 1e-7,
#         n_folds: int = 5,
#         batch_size: int = None,
#         verbose: bool = False,
#         plateau_factor: float = 0.5,
#         plateau_patience: int = 10,
#     ):
#         if reg_lambda_list is None:
#             reg_lambda_list = [0.0, 1e-4, 1e-3, 1e-2]
#         if lr_list is None:
#             lr_list = [1e-4, 1e-3, 1e-2]
#         if max_iter_list is None:
#             max_iter_list = [1000]
#         self.reg_lambda_list = reg_lambda_list
#         self.lr_list = lr_list
#         self.max_iter_list = max_iter_list
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.patience = patience
#         self.tol = tol
#         self.n_folds = n_folds
#         self.batch_size = batch_size
#         self.verbose = verbose
#         self.plateau_factor = plateau_factor
#         self.plateau_patience = plateau_patience

#         self.best_calibrator_: Optional[VectorScalingCalibrator] = None
#         self.best_params_: Optional[dict] = None
#         self.best_cv_val_loss_: Optional[float] = None
#         self.best_metrics_per_param_: Optional[dict] = None

#     def fit(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#     ) -> "VectorScalingCalibratorCV":
#         """Run k-fold CV grid search, and retrain on all data.

#         Evaluates every hyperparameter combination via k-fold CV,
#         then retrains a final :class:`VectorScalingCalibrator` on the full
#         dataset with the best hyperparameters.

#         Args:
#             X: Probability predictions, shape (n_samples, n_classes).
#             y: Labels (integer, one-hot, or soft), shape (n_samples, n_classes).

#         Returns:
#             self, with ``best_calibrator_``, ``best_params_``, and
#             ``best_cv_val_loss_`` set.
#         """
#         param_grid = {
#             "reg_lambda": self.reg_lambda_list,
#             "lr": self.lr_list,
#             "max_iter": self.max_iter_list,
#             "optimizer": [self.optimizer],
#             "scheduler": [self.scheduler],
#             "patience": [self.patience],
#             "tol": [self.tol],
#             "batch_size": [self.batch_size],
#             "plateau_factor": [self.plateau_factor],
#             "plateau_patience": [self.plateau_patience],
#         }

#         # n-fold CV to find best hyperparameters
#         best_mean_val_loss = float("inf")
#         best_params = None
#         kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

#         # Store best metrics for each hyperparameter combination
#         best_metrics_per_param = {}
#         with tqdm(
#             total=len(ParameterGrid(param_grid)) * self.n_folds,
#             desc="CV grid search",
#             disable=not self.verbose,
#         ) as pbar:
#             for params in ParameterGrid(param_grid):
#                 fold_val_losses = []
#                 for train_idx, val_idx in kf.split(X):
#                     X_tr_fold, X_val_fold = X[train_idx], X[val_idx]
#                     y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

#                     calibrator = VectorScalingCalibrator(**params, verbose=False)
#                     calibrator.fit(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)
#                     best_metrics_per_param[tuple(params.items())] = (
#                         calibrator.best_metrics_
#                     )
#                     val_loss = calibrator.best_metrics_.get("val_loss", float("inf"))
#                     fold_val_losses.append(val_loss)
#                     pbar.update(1)

#                 mean_val_loss = np.mean(fold_val_losses)
#                 if mean_val_loss < best_mean_val_loss:
#                     best_mean_val_loss = mean_val_loss
#                     best_params = params
#                     pbar.set_postfix(
#                         {
#                             "best_val_loss": f"{best_mean_val_loss:.7e}",
#                             "best_params": best_params,
#                         }
#                     )

#         self.best_metrics_per_param_ = best_metrics_per_param

#         # Retrain on the full dataset with the best hyperparameters
#         final_calibrator = VectorScalingCalibrator(**best_params, verbose=False)
#         final_calibrator.fit(X, y)

#         self.best_calibrator_ = final_calibrator
#         self.best_params_ = best_params
#         self.best_cv_val_loss_ = best_mean_val_loss
#         return self

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Apply the best calibration map

#         Args:
#             X: Uncalibrated probability predictions, shape (n_samples, n_classes).

#         Returns:
#             Calibrated probabilities, shape (n_samples, n_classes).

#         Raises:
#             RuntimeError: If :meth:`fit` has not been called yet.
#         """
#         if self.best_calibrator_ is None:
#             raise RuntimeError("Calibrator not fitted yet. Call fit() first.")
#         return self.best_calibrator_.predict(X)

#     def save(self, path: Union[str, Path]) -> None:
#         """Save the fitted CV calibrator to a .npz file.

#         Persists all constructor parameters, CV results, and the best
#         inner calibrator (including its model weights).

#         Args:
#             path: File path to save to (typically .npz).

#         Raises:
#             RuntimeError: If the calibrator has not been fitted yet.
#         """
#         if self.best_calibrator_ is None:
#             raise RuntimeError("Cannot save an unfitted calibrator. Call fit() first.")

#         # Serialise best_metrics_per_param_ with string keys
#         serialisable_metrics = None
#         if self.best_metrics_per_param_ is not None:
#             serialisable_metrics = {
#                 json.dumps(list(k)): v for k, v in self.best_metrics_per_param_.items()
#             }

#         # Inner calibrator metadata (same structure as VectorScalingCalibrator.save)
#         inner = self.best_calibrator_
#         inner_metadata = {
#             "params": inner.get_params(),
#             "n_classes_": inner.n_classes_,
#             "final_loss_": inner.final_loss_,
#             "best_epoch_": inner.best_epoch_,
#             "best_metrics_": inner.best_metrics_,
#             "history_": inner.history_,
#             "model_method": inner.model_.method.value,
#         }

#         metadata = {
#             "cv_params": self.get_params(),
#             "best_params_": self.best_params_,
#             "best_cv_val_loss_": self.best_cv_val_loss_,
#             "best_metrics_per_param_": serialisable_metrics,
#             "inner_calibrator": inner_metadata,
#         }

#         arrays = {
#             name: param.detach().cpu().numpy()
#             for name, param in inner.model_.state_dict().items()
#         }
#         arrays["_metadata_json"] = np.array(json.dumps(metadata))
#         np.savez(path, **arrays)

#     def load(self, path: Union[str, Path]) -> "VectorScalingCalibratorCV":
#         """Load a fitted CV calibrator from a .npz file.

#         Args:
#             path: File path to load from.

#         Returns:
#             self, with all fitted attributes restored.
#         """
#         data = np.load(path, allow_pickle=False)
#         metadata = json.loads(str(data["_metadata_json"]))

#         # Restore constructor params
#         cv_params = metadata["cv_params"]
#         for key, value in cv_params.items():
#             setattr(self, key, value)

#         self.best_params_ = metadata["best_params_"]
#         self.best_cv_val_loss_ = metadata["best_cv_val_loss_"]

#         # Restore best_metrics_per_param_ with tuple keys
#         raw = metadata.get("best_metrics_per_param_")
#         if raw is not None:
#             self.best_metrics_per_param_ = {
#                 tuple(tuple(pair) for pair in json.loads(k)): v for k, v in raw.items()
#             }

#         # Reconstruct inner calibrator
#         inner_meta = metadata["inner_calibrator"]
#         inner = VectorScalingCalibrator(**inner_meta["params"])
#         inner.n_classes_ = inner_meta["n_classes_"]
#         inner.final_loss_ = inner_meta["final_loss_"]
#         inner.best_epoch_ = inner_meta["best_epoch_"]
#         inner.best_metrics_ = inner_meta["best_metrics_"]
#         inner.history_ = inner_meta["history_"]

#         method_enum = CalibrationMethod(inner_meta["model_method"])
#         model = _TrainedLinearCalibrationModel(inner.n_classes_, method_enum)
#         state_dict = {name: torch.as_tensor(data[name]) for name in model.state_dict()}
#         model.load_state_dict(state_dict)
#         model.to(torch.device(inner.device))
#         inner.model_ = model

#         self.best_calibrator_ = inner
#         return self
