"""Linear post-processing utilities for deconvolution predictions."""

from typing import Union
from pathlib import Path
import logging

import joblib
import numpy as np
from scipy.stats import linregress
from sklearn.utils.validation import check_is_fitted

from methyldl.calibration.abstract_calibrator import AbstractCalibrator


def _project_onto_simplex(unnorm_pred: np.ndarray) -> np.ndarray:
    """Project each row of v onto the probability simplex (Duchi et al., 2008).

    Args:
        unnorm_pred: A 2D array of shape (n_samples, n_classes) containing the unnormalized
        calibrated predictions.
    """
    n, d = unnorm_pred.shape
    u = np.sort(unnorm_pred, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    j = np.arange(1, d + 1)
    cond = u * j > (cssv - 1)
    rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = (cssv[np.arange(n), rho] - 1) / (rho + 1.0)
    return np.maximum(unnorm_pred - theta[:, np.newaxis], 0)


def _clip0_normalize(unnorm_pred: np.ndarray) -> np.ndarray:
    """Clip negative values to zero and normalize each row to sum to 1.

    Args:
        unnorm_pred: A 2D array of shape (n_samples, n_classes) containing the unnormalized
        calibrated predictions.
    """
    clipped = np.maximum(unnorm_pred, 0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    return clipped / np.maximum(row_sums, 1e-12)


class LinearCalibrator(AbstractCalibrator):
    """
    For each cell type, we fit a linear regression between the predicted and true proportions
    on the validation set, and then we use the fitted slopes and intercepts to adjust
    the predictions on the test set.
    """

    _VALID_NORM_METHODS = (
        "clip0-normalize",
        "simplex-projection",
    )

    def __init__(self, logger: Union[logging.Logger, None] = None):
        """Initialize calibration statistics for each cell type."""
        self.slopes = None
        self.intercepts = None
        self.r_values = None
        self.p_values = None
        self.std_errs = None
        self.n_cell_types = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit one linear calibration model per cell type.

        Args:
            X: Predicted proportions with shape ``(n_samples, n_cell_types)``.
            y: Ground-truth proportions with the same shape as ``X``.
        """
        assert X.shape == y.shape

        self.n_cell_types = X.shape[1]

        self.slopes = []
        self.intercepts = []
        self.r_values = []
        self.p_values = []
        self.std_errs = []
        for cell_type_idx in range(self.n_cell_types):
            # Each cell type is calibrated independently so a biased output head
            # does not distort the correction applied to other cell types.
            cell_type_pred = X[:, cell_type_idx]
            cell_type_true = y[:, cell_type_idx]
            slope, intercept, r_value, p_value, std_err = linregress(
                cell_type_pred, cell_type_true
            )
            self.slopes.append(slope)
            self.intercepts.append(intercept)
            self.r_values.append(r_value)
            self.p_values.append(p_value)
            self.std_errs.append(std_err)

    def __sklearn_is_fitted__(self):
        """Report whether scikit-learn can treat this estimator as fitted."""
        return self.slopes is not None and self.intercepts is not None

    def predict(
        self, X: np.ndarray, norm_method: str = "simplex-projection", **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Apply the learned calibration and return corrected predictions.

        Args:
            X: Raw predicted proportions with shape ``(n_samples, n_cell_types)``.
            norm_method: One of ``"clip0-normalize"``, ``"simplex-projection"``.

        Returns:
            A tuple containing the normalized predictions, followed by
            the raw affine-adjusted predictions before normalization.
        """
        check_is_fitted(self)
        assert X.shape[1] == len(
            self.slopes
        ), "Number of cell types in predictions must match number of slopes/intercepts"
        assert (
            norm_method in self._VALID_NORM_METHODS
        ), f"norm_method must be one of {self._VALID_NORM_METHODS}"

        adjusted_predictions = np.zeros_like(X)
        for cell_type_idx in range(self.n_cell_types):
            adjusted_predictions[:, cell_type_idx] = (
                self.slopes[cell_type_idx] * X[:, cell_type_idx]
                + self.intercepts[cell_type_idx]
            )

        # Linear correction can push some values slightly outside the simplex.
        # Project back onto the simplex using the chosen method.
        final_predictions = None
        if norm_method == "clip0-normalize":
            final_predictions = _clip0_normalize(adjusted_predictions)
        elif norm_method == "simplex-projection":
            final_predictions = _project_onto_simplex(adjusted_predictions)

        return final_predictions, adjusted_predictions

    def save(self, path: Union[str, Path], **kwargs):
        """Save the fitted calibration parameters to a file."""
        check_is_fitted(self)
        path = Path(path)
        file_extension = path.suffix
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_extension == ".joblib":
            joblib.dump(value=self, filename=path)
        elif file_extension == ".npz":
            self.logger.warning(
                "Saving as .npz is deprecated and is left only for backward compatibility. "
                "Please switch to .joblib."
            )
            np.savez(
                path,
                slopes=self.slopes,
                intercepts=self.intercepts,
                r_values=self.r_values,
                p_values=self.p_values,
                std_errs=self.std_errs,
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs):
        """Load calibration parameters from a file and set the fitted attributes."""
        logger = kwargs.get("logger", logging.getLogger(__name__))
        path = Path(path)
        file_extension = path.suffix

        if file_extension == ".joblib":
            model = joblib.load(path)
            if not isinstance(model, cls):
                raise ValueError(f"Loaded object is not a {cls.__name__} instance")
            return model
        elif file_extension == ".npz":
            logger.warning(
                "Loading from .npz is deprecated and is left only for backward compatibility. "
                "Please switch to .joblib."
            )
            model_parameters = np.load(path)
            model = cls()
            model.slopes = model_parameters["slopes"].tolist()
            model.intercepts = model_parameters["intercepts"].tolist()
            model.r_values = model_parameters["r_values"].tolist()
            model.p_values = model_parameters["p_values"].tolist()
            model.std_errs = model_parameters["std_errs"].tolist()
            model.n_cell_types = len(model.slopes)
            model.logger = logger
            return model
