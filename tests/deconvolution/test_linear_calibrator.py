import unittest
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import linregress
from sklearn.exceptions import NotFittedError

from methyldl.deconvolution.linear_calibrator import LinearCalibrator


class TestLinearCalibrator(unittest.TestCase):
    """Tests covering every execution path in the linear calibrator."""

    @staticmethod
    def _make_fitted_calibrator(slopes, intercepts):
        """Create a calibrator with deterministic fitted coefficients."""
        calibrator = LinearCalibrator()
        calibrator.slopes = list(slopes)
        calibrator.intercepts = list(intercepts)
        calibrator.r_values = [0.0] * len(slopes)
        calibrator.p_values = [1.0] * len(slopes)
        calibrator.std_errs = [0.0] * len(slopes)
        calibrator.n_cell_types = len(slopes)
        return calibrator

    def test_initialization_starts_in_unfitted_state(self):
        """A new calibrator should expose empty fitted statistics."""
        calibrator = LinearCalibrator()

        self.assertIsNone(calibrator.slopes)
        self.assertIsNone(calibrator.intercepts)
        self.assertIsNone(calibrator.r_values)
        self.assertIsNone(calibrator.p_values)
        self.assertIsNone(calibrator.std_errs)
        self.assertIsNone(calibrator.n_cell_types)
        self.assertFalse(calibrator.__sklearn_is_fitted__())

    def test_fit_populates_regression_statistics_for_each_cell_type(self):
        """fit should store one linear regression summary per cell type."""
        X = np.array(
            [
                [0.10, 0.20],
                [0.30, 0.40],
                [0.50, 0.60],
                [0.70, 0.80],
            ],
            dtype=float,
        )
        y = np.array(
            [
                [0.25, 0.55],
                [0.55, 0.45],
                [0.85, 0.35],
                [1.15, 0.25],
            ],
            dtype=float,
        )
        calibrator = LinearCalibrator()

        fit_result = calibrator.fit(X, y)

        self.assertIsNone(fit_result)
        self.assertTrue(calibrator.__sklearn_is_fitted__())
        self.assertEqual(calibrator.n_cell_types, 2)
        self.assertEqual(len(calibrator.slopes), 2)
        self.assertEqual(len(calibrator.intercepts), 2)
        self.assertEqual(len(calibrator.r_values), 2)
        self.assertEqual(len(calibrator.p_values), 2)
        self.assertEqual(len(calibrator.std_errs), 2)

        for cell_type_idx in range(calibrator.n_cell_types):
            expected = linregress(X[:, cell_type_idx], y[:, cell_type_idx])
            self.assertAlmostEqual(calibrator.slopes[cell_type_idx], expected.slope)
            self.assertAlmostEqual(
                calibrator.intercepts[cell_type_idx], expected.intercept
            )
            self.assertAlmostEqual(calibrator.r_values[cell_type_idx], expected.rvalue)
            self.assertAlmostEqual(calibrator.p_values[cell_type_idx], expected.pvalue)
            self.assertAlmostEqual(calibrator.std_errs[cell_type_idx], expected.stderr)

    def test_fit_rejects_shape_mismatch(self):
        """fit should fail when predictions and targets do not share the same shape."""
        calibrator = LinearCalibrator()

        with self.assertRaises(AssertionError):
            calibrator.fit(
                np.array([[0.2, 0.8], [0.4, 0.6]], dtype=float),
                np.array([[0.2, 0.8]], dtype=float),
            )

    def test_predict_raises_when_not_fitted(self):
        """predict should delegate unfitted checks to scikit-learn validation."""
        calibrator = LinearCalibrator()

        with self.assertRaises(NotFittedError):
            calibrator.predict(np.array([[0.2, 0.8]], dtype=float))

    def test_save_calibration_parameters_raises_when_not_fitted(self):
        """Saving should reject calibrators that have not been fit yet."""
        calibrator = LinearCalibrator()

        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotFittedError):
                calibrator.save_calibration_parameters(
                    f"{tmp_dir}/unfitted_calibrator.npz"
                )

    def test_predict_rejects_mismatched_number_of_cell_types(self):
        """predict should validate the feature dimension against fitted slopes."""
        calibrator = self._make_fitted_calibrator(
            slopes=[1.0, 1.0],
            intercepts=[0.0, 0.0],
        )

        with self.assertRaisesRegex(
            AssertionError,
            "Number of cell types in predictions must match number of slopes/intercepts",
        ):
            calibrator.predict(np.array([[0.2, 0.3, 0.5]], dtype=float))

    def test_predict_applies_affine_adjustment_and_row_normalization(self):
        """predict should return raw affine outputs plus normalized calibrated rows."""
        calibrator = self._make_fitted_calibrator(
            slopes=[2.0, 0.5],
            intercepts=[0.1, 0.2],
        )
        X = np.array(
            [
                [0.20, 0.40],
                [0.50, 0.20],
            ],
            dtype=float,
        )

        normalized, adjusted = calibrator.predict(X)

        expected_adjusted = np.array(
            [
                [0.50, 0.40],
                [1.10, 0.30],
            ],
            dtype=float,
        )
        expected_normalized = np.array(
            [
                [5.0 / 9.0, 4.0 / 9.0],
                [10.0 / 13.0, 3.0 / 13.0],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(adjusted, expected_adjusted, atol=1e-10)
        np.testing.assert_allclose(normalized, expected_normalized, atol=1e-10)
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(X.shape[0]))

    def test_predict_clips_out_of_bounds_values_and_preserves_zero_rows(self):
        """predict should clip to [0, 1] and avoid dividing by zero on empty rows."""
        calibrator = self._make_fitted_calibrator(
            slopes=[2.0, -3.0],
            intercepts=[0.8, -0.1],
        )
        X = np.array(
            [
                [0.30, 0.20],
                [0.00, 1.00],
            ],
            dtype=float,
        )

        normalized, adjusted = calibrator.predict(X)

        expected_adjusted = np.array(
            [
                [1.40, -0.70],
                [0.80, -3.10],
            ],
            dtype=float,
        )
        expected_normalized = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(adjusted, expected_adjusted, atol=1e-10)
        np.testing.assert_allclose(normalized, expected_normalized, atol=1e-10)

        zero_row_calibrator = self._make_fitted_calibrator(
            slopes=[1.0, 1.0],
            intercepts=[-1.0, -1.0],
        )

        zero_row_normalized, zero_row_adjusted = zero_row_calibrator.predict(
            np.array([[0.1, 0.2]], dtype=float)
        )

        np.testing.assert_allclose(zero_row_adjusted, np.array([[-0.9, -0.8]]))
        np.testing.assert_allclose(zero_row_normalized, np.zeros((1, 2)))

    def test_save_calibration_parameters_persists_all_fitted_statistics(self):
        """Saving should serialize every fitted calibration array to an NPZ file."""
        calibrator = self._make_fitted_calibrator(
            slopes=[0.5, 1.5],
            intercepts=[0.1, -0.2],
        )
        calibrator.r_values = [0.25, -0.75]
        calibrator.p_values = [0.01, 0.99]
        calibrator.std_errs = [0.05, 0.15]

        with TemporaryDirectory() as tmp_dir:
            filepath = f"{tmp_dir}/calibration_params.npz"

            calibrator.save_calibration_parameters(filepath)

            saved = np.load(filepath)
            np.testing.assert_allclose(saved["slopes"], np.array([0.5, 1.5]))
            np.testing.assert_allclose(saved["intercepts"], np.array([0.1, -0.2]))
            np.testing.assert_allclose(saved["r_values"], np.array([0.25, -0.75]))
            np.testing.assert_allclose(saved["p_values"], np.array([0.01, 0.99]))
            np.testing.assert_allclose(saved["std_errs"], np.array([0.05, 0.15]))

    def test_load_calibration_parameters_restores_fitted_state(self):
        """Loading should repopulate all fitted attributes from a saved NPZ file."""
        source = self._make_fitted_calibrator(
            slopes=[0.4, 1.2, -0.6],
            intercepts=[0.3, -0.1, 0.8],
        )
        source.r_values = [0.9, 0.1, -0.4]
        source.p_values = [0.001, 0.25, 0.8]
        source.std_errs = [0.02, 0.03, 0.07]
        loaded = LinearCalibrator()

        with TemporaryDirectory() as tmp_dir:
            filepath = f"{tmp_dir}/calibration_params.npz"
            source.save_calibration_parameters(filepath)

            load_result = loaded.load_calibration_parameters(filepath)

        self.assertIsNone(load_result)
        self.assertTrue(loaded.__sklearn_is_fitted__())
        self.assertEqual(loaded.n_cell_types, 3)
        self.assertEqual(loaded.slopes, [0.4, 1.2, -0.6])
        self.assertEqual(loaded.intercepts, [0.3, -0.1, 0.8])
        self.assertEqual(loaded.r_values, [0.9, 0.1, -0.4])
        self.assertEqual(loaded.p_values, [0.001, 0.25, 0.8])
        self.assertEqual(loaded.std_errs, [0.02, 0.03, 0.07])
