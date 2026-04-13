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

    @staticmethod
    def _project_onto_simplex(v):
        """Project each row of v onto the probability simplex (Duchi et al., 2008)."""
        n, d = v.shape
        u = np.sort(v, axis=1)[:, ::-1]
        cssv = np.cumsum(u, axis=1)
        j = np.arange(1, d + 1)
        cond = u * j > (cssv - 1)
        rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
        theta = (cssv[np.arange(n), rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta[:, np.newaxis], 0)

    @staticmethod
    def _entmax(z, alpha):
        """Alpha-entmax mapping (Peters et al., 2019)."""
        n, d = z.shape
        result = np.zeros_like(z)
        for i in range(n):
            zi = z[i]
            lo, hi = zi.min() - 1.0 / (alpha - 1), zi.max()
            for _ in range(50):
                mid = (lo + hi) / 2
                p = np.maximum((alpha - 1) * (zi - mid), 0) ** (1 / (alpha - 1))
                if p.sum() > 1:
                    lo = mid
                else:
                    hi = mid
            tau = (lo + hi) / 2
            result[i] = np.maximum((alpha - 1) * (zi - tau), 0) ** (1 / (alpha - 1))
            s = result[i].sum()
            if s > 0:
                result[i] /= s
        return result

    _VALID_NORM_METHODS = (
        "clip01-normalize",
        "clip0-normalize",
        "softmax",
        "simplex-projection",
        "shift-normalize",
        "entmax",
    )

    def predict(
        self,
        X: np.ndarray,
        norm_method="clip01-normalize",
        entmax_alpha=1.5,
    ) -> np.ndarray:
        """Apply the learned calibration and return corrected predictions.

        Args:
            X: Raw predicted proportions with shape ``(n_samples, n_cell_types)``.
            norm_method: One of ``"clip01-normalize"``, ``"clip0-normalize"``, ``"softmax"``,
                ``"simplex-projection"``, ``"shift-normalize"``,
                or ``"entmax"``.
            entmax_alpha: Alpha parameter for entmax (only used when
                ``norm_method="entmax"``). 1 = softmax, 2 = sparsemax.

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
        if norm_method == "clip01-normalize":
            clipped = np.clip(adjusted_predictions, 0, 1)
            final_predictions = clipped / np.clip(
                clipped.sum(axis=1, keepdims=True), 1e-8, None
            )
        elif norm_method == "clip0-normalize":
            clipped = np.clip(adjusted_predictions, 0, None)
            final_predictions = clipped / np.clip(
                clipped.sum(axis=1, keepdims=True), 1e-8, None
            )
        elif norm_method == "softmax":
            exp_vals = np.exp(adjusted_predictions)
            final_predictions = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        elif norm_method == "simplex-projection":
            final_predictions = self._project_onto_simplex(adjusted_predictions)
        elif norm_method == "shift-normalize":
            shifted = adjusted_predictions - adjusted_predictions.min(
                axis=1, keepdims=True
            )
            final_predictions = shifted / np.clip(
                shifted.sum(axis=1, keepdims=True), 1e-8, None
            )
        elif norm_method == "entmax":
            assert entmax_alpha > 1, "entmax_alpha must be > 1"
            if entmax_alpha == 1:
                exp_vals = np.exp(adjusted_predictions)
                final_predictions = exp_vals / exp_vals.sum(axis=1, keepdims=True)
            elif entmax_alpha == 2:
                final_predictions = self._project_onto_simplex(adjusted_predictions)
            else:
                final_predictions = self._entmax(adjusted_predictions, entmax_alpha)

        return final_predictions, adjusted_predictions

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
