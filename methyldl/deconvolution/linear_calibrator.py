"""Linear post-processing utilities for deconvolution predictions."""

import numpy as np
from scipy.stats import linregress
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class LinearCalibrator(BaseEstimator, RegressorMixin):
    """
    For each cell type, we fit a linear regression between the predicted and true proportions
    on the validation set, and then we use the fitted slopes and intercepts to adjust
    the predictions on the test set.
    """

    def __init__(self):
        """Initialize calibration statistics for each cell type."""
        self.slopes = None
        self.intercepts = None
        self.r_values = None
        self.p_values = None
        self.std_errs = None
        self.n_cell_types = None

    def fit(self, X: np.ndarray, y: np.ndarray):
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned calibration and return corrected predictions.

        Args:
            X: Raw predicted proportions with shape ``(n_samples, n_cell_types)``.

        Returns:
            A tuple containing the clipped and row-normalized predictions, followed by
            the raw affine-adjusted predictions before clipping.
        """
        check_is_fitted(self)
        assert X.shape[1] == len(
            self.slopes
        ), "Number of cell types in predictions must match number of slopes/intercepts"

        adjusted_predictions = np.zeros_like(X)
        for cell_type_idx in range(self.n_cell_types):
            adjusted_predictions[:, cell_type_idx] = (
                self.slopes[cell_type_idx] * X[:, cell_type_idx]
                + self.intercepts[cell_type_idx]
            )

        # Linear correction can push some values slightly outside the simplex.
        # We first enforce valid proportion bounds, then renormalize each sample
        # so the calibrated proportions still sum to one.
        clipped_adjusted_predictions = np.clip(adjusted_predictions, 0, 1)
        clipped_norm_adjusted_predictions = clipped_adjusted_predictions / np.clip(
            clipped_adjusted_predictions.sum(axis=1, keepdims=True), 1e-8, None
        )
        return clipped_norm_adjusted_predictions, adjusted_predictions

    def save_calibration_parameters(self, filepath: str):
        """Save the fitted calibration parameters to a file."""
        check_is_fitted(self)
        np.savez(
            filepath,
            slopes=self.slopes,
            intercepts=self.intercepts,
            r_values=self.r_values,
            p_values=self.p_values,
            std_errs=self.std_errs,
        )

    def load_calibration_parameters(self, filepath: str):
        """Load calibration parameters from a file and set the fitted attributes."""
        loaded = np.load(filepath)
        self.slopes = loaded["slopes"].tolist()
        self.intercepts = loaded["intercepts"].tolist()
        self.r_values = loaded["r_values"].tolist()
        self.p_values = loaded["p_values"].tolist()
        self.std_errs = loaded["std_errs"].tolist()
        self.n_cell_types = len(self.slopes)
