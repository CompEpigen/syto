import os
import tempfile
import unittest
from unittest.mock import patch

import cvxpy as cp
import joblib
import numpy as np
from sklearn.exceptions import NotFittedError

from methyldl.deconvolution.least_squares_deconvolvers import (
    AbstractLSDeconvolver,
    NNLSDeconvolver,
    PSLSDeconvolver,
    _project_on_simplex,
    _project_rows_on_simplex,
)


def _passthrough_tqdm(iterable, **_kwargs):
    """Return the iterable unchanged so tests stay deterministic and quiet."""
    return iterable


# OSQP can return tiny non-zero coefficients for entries that are theoretically zero.
CVXPY_ATOL = 5e-5


def _make_reference_and_samples():
    """Build a small exact deconvolution problem with label order intentionally shuffled."""
    reference_predictions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([2, 0, 1], dtype=int)
    samples = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.3, 0.5],
            [0.7, 0.2, 0.1],
            [0.0, 0.4, 0.6],
            [0.5, 0.25, 0.25],
        ],
        dtype=float,
    )
    return reference_predictions, labels, samples


class LeastSquaresDeconvolverAssertions:
    """Shared assertions for fit-time validation across both deconvolver classes."""

    def make_deconvolver(self):
        raise NotImplementedError

    def assert_invalid_fit_inputs_raise(self):
        """Assert that malformed fit inputs fail on the expected validation checks."""
        with self.assertRaisesRegex(AssertionError, "X must be a 2D matrix"):
            self.make_deconvolver().fit(np.array([1.0, 0.0, 0.0]), np.array([0]))

        with self.assertRaises(AssertionError):
            self.make_deconvolver().fit(
                np.eye(2, dtype=float),
                np.array([0, 2], dtype=int),
            )

        with self.assertRaises(AssertionError):
            self.make_deconvolver().fit(
                np.eye(3, dtype=float),
                np.array([0, 1], dtype=int),
            )


class _DummyLSDeconvolver(AbstractLSDeconvolver):
    """Concrete helper used to exercise the abstract base-class helper paths."""

    def predict_single_sample(self, x: np.ndarray) -> np.ndarray:
        return x


class TestAbstractLSDeconvolver(unittest.TestCase):
    """Tests for the non-solver-specific base-class behavior."""

    def test_predict_single_sample_is_abstract(self):
        """The abstract base class should force subclasses to implement prediction."""
        with self.assertRaisesRegex(NotImplementedError, "Subclasses must implement"):
            AbstractLSDeconvolver().predict_single_sample(np.array([1.0], dtype=float))

    def test_predict_chunk_sequential_uses_single_sample_predictions(self):
        """The shared chunk helper should stack per-sample predictions row-wise."""
        deconvolver = _DummyLSDeconvolver()
        deconvolver.n_cell_types_ = 3

        chunk = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
            ],
            dtype=float,
        )

        predictions = deconvolver._predict_chunk_sequential(chunk)

        np.testing.assert_allclose(predictions, chunk, atol=1e-10)


class TestSimplexProjectionHelpers(unittest.TestCase):
    """Tests for the simplex projection helper functions."""

    def test_project_on_simplex_returns_probability_vector(self):
        """Projection of an arbitrary vector should be non-negative and sum to one."""
        projected = _project_on_simplex(np.array([0.2, -0.4, 2.1], dtype=float))

        self.assertEqual(projected.shape, (3,))
        self.assertTrue(np.all(projected >= 0.0))
        np.testing.assert_allclose(projected.sum(), 1.0, atol=1e-10)

    def test_project_on_simplex_is_identity_on_simplex(self):
        """Projection should leave a vector that is already on the simplex unchanged."""
        simplex_vector = np.array([0.2, 0.3, 0.5], dtype=float)
        projected = _project_on_simplex(simplex_vector)

        np.testing.assert_allclose(projected, simplex_vector, atol=1e-10)

    def test_project_rows_on_simplex_matches_rowwise_projection(self):
        """The batched projection should match applying the scalar helper row by row."""
        matrix = np.array(
            [[0.2, -0.4, 2.1], [0.5, 0.3, 0.2], [-1.0, 4.0, -2.0]],
            dtype=float,
        )

        projected = _project_rows_on_simplex(matrix)
        # This checks the vectorized implementation against the simpler scalar reference.
        expected = np.vstack([_project_on_simplex(row) for row in matrix])

        np.testing.assert_allclose(projected, expected, atol=1e-10)
        np.testing.assert_allclose(projected.sum(axis=1), np.ones(matrix.shape[0]))
        self.assertTrue(np.all(projected >= 0.0))

    def test_project_rows_on_simplex_requires_2d_input(self):
        """The row-wise projection helper should reject non-matrix inputs."""
        with self.assertRaisesRegex(ValueError, "v must be a 2D array"):
            _project_rows_on_simplex(np.array([0.2, 0.3, 0.5], dtype=float))


class TestNNLSDeconvolver(unittest.TestCase, LeastSquaresDeconvolverAssertions):
    """Tests covering every execution path in the NNLS deconvolver."""

    def make_deconvolver(self):
        return NNLSDeconvolver()

    def setUp(self):
        """Create a fitted NNLS deconvolver on an exact synthetic problem."""
        self.reference_predictions, self.labels, self.samples = (
            _make_reference_and_samples()
        )
        self.deconvolver = self.make_deconvolver().fit(
            self.reference_predictions,
            self.labels,
        )

    def test_fit_reorders_reference_matrix_and_sets_attributes(self):
        """Fit should reorder rows by label and populate fitted-shape attributes."""
        self.assertEqual(self.deconvolver.n_cell_types_, 3)
        self.assertEqual(self.deconvolver.n_features_, 3)
        np.testing.assert_allclose(
            self.deconvolver.reference_prediction_matrix_,
            np.eye(3, dtype=float),
        )

    def test_fit_rejects_invalid_inputs(self):
        """Fit should reject malformed shapes and invalid label sets."""
        self.assert_invalid_fit_inputs_raise()

    def test_predict_single_sample_returns_normalized_and_unnormalized_outputs(self):
        """Single-sample NNLS should return matching normalized and raw solutions here."""
        normalized, unnormalized, residual = self.deconvolver.predict_single_sample(
            self.samples[1]
        )

        np.testing.assert_allclose(normalized, self.samples[1], atol=1e-10)
        np.testing.assert_allclose(unnormalized, self.samples[1], atol=1e-10)
        self.assertAlmostEqual(float(residual), 0.0, places=10)

    def test_predict_single_sample_preserves_zero_solution(self):
        """A zero NNLS solution should stay zero instead of dividing by zero on normalization."""
        normalized, unnormalized, residual = self.deconvolver.predict_single_sample(
            np.zeros(3, dtype=float)
        )

        np.testing.assert_allclose(normalized, np.zeros(3, dtype=float))
        np.testing.assert_allclose(unnormalized, np.zeros(3, dtype=float))
        self.assertAlmostEqual(float(residual), 0.0, places=10)

    def test_predict_single_sample_requires_expected_shape(self):
        """Single-sample prediction should enforce the fitted feature dimension."""
        with self.assertRaises(AssertionError):
            self.deconvolver.predict_single_sample(np.array([1.0, 0.0], dtype=float))

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_chunk_sequential_matches_expected_outputs(self):
        """Chunk-local sequential prediction should preserve the exact synthetic solution."""
        normalized, unnormalized, residuals = (
            self.deconvolver._predict_chunk_sequential(self.samples[:3])
        )

        np.testing.assert_allclose(normalized, self.samples[:3], atol=1e-10)
        np.testing.assert_allclose(unnormalized, self.samples[:3], atol=1e-10)
        np.testing.assert_allclose(residuals, np.zeros(3), atol=1e-10)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_sequential_matches_expected_outputs(self):
        """Sequential batch prediction should recover the known proportions exactly."""
        normalized, unnormalized, residuals = self.deconvolver._predict_sequential(
            self.samples
        )

        np.testing.assert_allclose(normalized, self.samples, atol=1e-10)
        np.testing.assert_allclose(unnormalized, self.samples, atol=1e-10)
        np.testing.assert_allclose(residuals, np.zeros(len(self.samples)), atol=1e-10)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_parallel_matches_sequential_outputs(self):
        """Parallel NNLS should match the sequential implementation on the same inputs."""
        sequential = self.deconvolver.predict(self.samples, n_workers=1)
        parallel = self.deconvolver.predict(self.samples, n_workers=2, chunk_size=2)

        for sequential_output, parallel_output in zip(sequential, parallel):
            np.testing.assert_allclose(parallel_output, sequential_output, atol=1e-10)

    def test_predict_parallel_rejects_non_positive_chunk_size(self):
        """Parallel NNLS should reject invalid chunk sizes before dispatching work."""
        with self.assertRaisesRegex(
            ValueError, "chunk_size must be a positive integer"
        ):
            self.deconvolver._predict_parallel(self.samples, n_workers=2, chunk_size=0)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_dispatches_between_sequential_and_parallel_paths(self):
        """Public prediction should switch between sequential and parallel backends."""
        sequential = self.deconvolver.predict(self.samples, n_workers=1)
        parallel = self.deconvolver.predict(self.samples, n_workers=3, chunk_size=1)

        for sequential_output, parallel_output in zip(sequential, parallel):
            np.testing.assert_allclose(parallel_output, sequential_output, atol=1e-10)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_save_and_load_round_trip(self):
        """save/load should round-trip a fitted NNLS model and produce identical predictions."""
        expected = self.deconvolver.predict(self.samples, n_workers=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nnls.joblib")
            self.deconvolver.save(path)
            loaded = NNLSDeconvolver.load(path)

        self.assertIsInstance(loaded, NNLSDeconvolver)
        np.testing.assert_allclose(
            loaded.reference_prediction_matrix_,
            self.deconvolver.reference_prediction_matrix_,
        )
        actual = loaded.predict(self.samples, n_workers=1)
        for exp_out, act_out in zip(expected, actual):
            np.testing.assert_allclose(act_out, exp_out, atol=1e-10)

    def test_load_raises_file_not_found(self):
        """load should raise FileNotFoundError for a missing path."""
        with self.assertRaises(FileNotFoundError):
            NNLSDeconvolver.load("/definitely/not/present/model.joblib")

    def test_load_raises_type_error_for_wrong_object(self):
        """load should raise TypeError when the file does not contain this model class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.joblib")
            joblib.dump({"not": "a model"}, path)
            with self.assertRaises(TypeError):
                NNLSDeconvolver.load(path)

    def test_save_raises_not_fitted_error(self):
        """save should raise NotFittedError if the model has not been fit."""
        unfitted_model = self.make_deconvolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nnls_unfitted.joblib")
            with self.assertRaises(NotFittedError):
                unfitted_model.save(path)

    def test_load_raises_not_fitted_error(self):
        """load should raise NotFittedError if the loaded model lacks fitted attributes."""
        unfitted_model = self.make_deconvolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nnls_unfitted.joblib")
            # Force save the unfitted model using base joblib to test load validation
            joblib.dump(unfitted_model, path)
            with self.assertRaises(NotFittedError):
                NNLSDeconvolver.load(path)


class TestPSLSDeconvolverCVXPY(unittest.TestCase, LeastSquaresDeconvolverAssertions):
    """Tests covering the CVXPY-backed PSLS execution paths."""

    def make_deconvolver(self):
        return PSLSDeconvolver(solver_type="cvxpy")

    def setUp(self):
        """Create a fitted CVXPY PSLS deconvolver on the exact synthetic problem."""
        self.reference_predictions, self.labels, self.samples = (
            _make_reference_and_samples()
        )
        self.deconvolver = self.make_deconvolver().fit(
            self.reference_predictions,
            self.labels,
        )

    def test_initialization_sets_default_solver_and_generation_sentinel(self):
        """Initialization should default to CVXPY and start with an unset cache generation."""
        deconvolver = PSLSDeconvolver()

        self.assertEqual(deconvolver.solver_type, "cvxpy")
        self.assertEqual(deconvolver._cvxpy_fit_generation_, -1)

    def test_init_rejects_unsupported_solver_type(self):
        """Initialization should reject unknown solver identifiers."""
        with self.assertRaisesRegex(AssertionError, "Unsupported solver_type: bogus"):
            PSLSDeconvolver(solver_type="bogus")

    def test_fit_sets_attributes_and_cvxpy_cache_state(self):
        """Fit should build the reordered reference matrix and CVXPY worker cache state."""
        self.assertEqual(self.deconvolver.n_cell_types_, 3)
        self.assertEqual(self.deconvolver.n_features_, 3)
        self.assertEqual(self.deconvolver._cvxpy_fit_generation_, 0)
        self.assertIsNotNone(self.deconvolver._cvxpy_thread_cache_)
        self.assertTrue(
            self.deconvolver.reference_prediction_matrix_.flags.c_contiguous
        )
        np.testing.assert_allclose(
            self.deconvolver.reference_prediction_matrix_,
            np.eye(3, dtype=float),
        )

    def test_fit_rejects_invalid_inputs(self):
        """Fit should reject malformed shapes and invalid label sets."""
        self.assert_invalid_fit_inputs_raise()

    def test_fit_rejects_invalid_solver_type_after_mutation(self):
        """Fit should still guard against an invalid solver_type value at runtime."""
        deconvolver = self.make_deconvolver()
        deconvolver.solver_type = "bogus"

        with self.assertRaisesRegex(ValueError, "Unsupported solver_type: bogus"):
            deconvolver.fit(self.reference_predictions, self.labels)

    def test_build_cvxpy_problem_uses_expected_shapes(self):
        """CVXPY problem construction should use one parameter and one variable per feature set."""
        x_param, w, problem = self.deconvolver._build_cvxpy_problem()

        self.assertIsInstance(x_param, cp.Parameter)
        self.assertIsInstance(w, cp.Variable)
        self.assertIsInstance(problem, cp.Problem)
        self.assertEqual(x_param.shape, (3,))
        self.assertEqual(w.shape, (3,))

    def test_get_cvxpy_worker_problem_reuses_cache_until_refit(self):
        """The thread-local CVXPY cache should be reused until fit invalidates it."""
        first = self.deconvolver._get_cvxpy_worker_problem()
        second = self.deconvolver._get_cvxpy_worker_problem()

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertIs(first[2], second[2])

        self.deconvolver.fit(self.reference_predictions, self.labels)
        third = self.deconvolver._get_cvxpy_worker_problem()

        self.assertIsNot(first[0], third[0])
        self.assertIsNot(first[1], third[1])
        self.assertIsNot(first[2], third[2])
        self.assertEqual(self.deconvolver._cvxpy_fit_generation_, 1)

    def test_predict_single_sample_cvxpy_returns_expected_solution(self):
        """CVXPY single-sample prediction should recover the exact mixture proportions."""
        prediction = self.deconvolver.predict_single_sample_cvxpy(self.samples[2])

        np.testing.assert_allclose(prediction, self.samples[2], atol=CVXPY_ATOL)
        np.testing.assert_allclose(prediction.sum(), 1.0, atol=1e-8)

    def test_predict_single_sample_cvxpy_requires_expected_shape(self):
        """CVXPY single-sample prediction should enforce the fitted feature dimension."""
        with self.assertRaises(AssertionError):
            self.deconvolver.predict_single_sample_cvxpy(np.array([1.0, 0.0]))

    def test_predict_single_sample_dispatches_to_cvxpy_solver(self):
        """Single-sample prediction should dispatch through the configured solver_type."""
        prediction = self.deconvolver.predict_single_sample(self.samples[3])

        np.testing.assert_allclose(prediction, self.samples[3], atol=CVXPY_ATOL)

    def test_predict_single_sample_rejects_unsupported_solver_type(self):
        """Single-sample PSLS prediction should reject unknown solver names."""
        self.deconvolver.solver_type = "bogus"

        with self.assertRaisesRegex(ValueError, "Unsupported solver_type: bogus"):
            self.deconvolver.predict_single_sample(self.samples[0])

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_sequential_uses_cvxpy_solver(self):
        """Sequential PSLS prediction should honor the configured CVXPY backend."""
        predictions = self.deconvolver._predict_sequential(self.samples)

        np.testing.assert_allclose(predictions, self.samples, atol=CVXPY_ATOL)

    def test_predict_chunk_sequential_uses_single_sample_path_for_cvxpy(self):
        """Chunk prediction should use per-sample inference on the CVXPY path."""
        predictions = self.deconvolver._predict_chunk_sequential(self.samples[:2])

        np.testing.assert_allclose(predictions, self.samples[:2], atol=CVXPY_ATOL)

    def test_predict_parallel_rejects_non_positive_chunk_size(self):
        """Parallel PSLS should reject invalid chunk sizes before creating workers."""
        with self.assertRaisesRegex(
            ValueError, "chunk_size must be a positive integer"
        ):
            self.deconvolver._predict_parallel(
                self.samples,
                n_workers=2,
                chunk_size=0,
            )

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_parallel_matches_expected_outputs(self):
        """Parallel CVXPY PSLS should recover the known simplex solutions."""
        predictions = self.deconvolver._predict_parallel(
            self.samples,
            n_workers=2,
            chunk_size=2,
        )

        np.testing.assert_allclose(predictions, self.samples, atol=CVXPY_ATOL)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_dispatches_between_sequential_and_parallel_paths(self):
        """Public PSLS prediction should switch between sequential and parallel backends."""
        sequential = self.deconvolver.predict(self.samples, n_workers=1)
        parallel = self.deconvolver.predict(self.samples, n_workers=2, chunk_size=2)

        np.testing.assert_allclose(sequential, self.samples, atol=CVXPY_ATOL)
        np.testing.assert_allclose(parallel, self.samples, atol=CVXPY_ATOL)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_save_and_load_round_trip(self):
        """save/load should round-trip a fitted CVXPY PSLS model and produce matching predictions."""
        expected = self.deconvolver.predict(self.samples, n_workers=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "psls_cvxpy.joblib")
            self.deconvolver.save(path)
            loaded = PSLSDeconvolver.load(path)

        self.assertIsInstance(loaded, PSLSDeconvolver)
        self.assertEqual(loaded.solver_type, "cvxpy")
        np.testing.assert_allclose(
            loaded.reference_prediction_matrix_,
            self.deconvolver.reference_prediction_matrix_,
        )
        actual = loaded.predict(self.samples, n_workers=1)
        np.testing.assert_allclose(actual, expected, atol=CVXPY_ATOL)


class TestPSLSDeconvolverPGD(unittest.TestCase, LeastSquaresDeconvolverAssertions):
    """Tests covering the PGD-backed PSLS execution paths."""

    def make_deconvolver(self):
        return PSLSDeconvolver(solver_type="pgd")

    def setUp(self):
        """Create a fitted PGD PSLS deconvolver on the exact synthetic problem."""
        self.reference_predictions, self.labels, self.samples = (
            _make_reference_and_samples()
        )
        self.deconvolver = self.make_deconvolver().fit(
            self.reference_predictions,
            self.labels,
        )

    def test_fit_sets_attributes_and_pgd_precomputed_state(self):
        """Fit should build the reordered reference matrix and PGD precomputations."""
        self.assertEqual(self.deconvolver.n_cell_types_, 3)
        self.assertEqual(self.deconvolver.n_features_, 3)
        np.testing.assert_allclose(
            self.deconvolver.reference_prediction_matrix_,
            np.eye(3, dtype=float),
        )
        np.testing.assert_allclose(self.deconvolver._pgd_ptp_, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(
            self.deconvolver._pgd_xtp_multiplier_,
            np.eye(3),
            atol=1e-10,
        )
        self.assertAlmostEqual(float(self.deconvolver._pgd_step_size_), 1.0, places=10)

    def test_fit_rejects_invalid_inputs(self):
        """Fit should reject malformed shapes and invalid label sets."""
        self.assert_invalid_fit_inputs_raise()

    def test_predict_single_sample_pgd_converges_to_expected_solution(self):
        """PGD single-sample prediction should converge to the known simplex solution."""
        with patch("builtins.print") as mock_print:
            prediction = self.deconvolver.predict_single_sample_pgd(
                self.samples[1],
                verbose=True,
            )

        np.testing.assert_allclose(prediction, self.samples[1], atol=1e-4)
        mock_print.assert_called_once()
        self.assertIn("Converged in", mock_print.call_args[0][0])

    def test_predict_single_sample_pgd_warns_when_max_iterations_reached(self):
        """PGD should warn and return the current iterate when no iteration budget is available."""
        with patch("builtins.print") as mock_print:
            prediction = self.deconvolver.predict_single_sample_pgd(
                self.samples[1],
                max_iter=0,
            )

        np.testing.assert_allclose(
            prediction,
            np.full(3, 1.0 / 3.0),
            atol=1e-10,
        )
        mock_print.assert_called_once_with(
            "Warning: Reached max iterations without full convergence."
        )

    def test_predict_single_sample_pgd_requires_expected_shape(self):
        """PGD single-sample prediction should enforce the fitted feature dimension."""
        with self.assertRaises(AssertionError):
            self.deconvolver.predict_single_sample_pgd(np.array([1.0, 0.0]))

    def test_predict_batch_pgd_converges_to_expected_solution(self):
        """Batched PGD prediction should converge to the known mixture proportions."""
        with patch("builtins.print") as mock_print:
            predictions = self.deconvolver.predict_batch_pgd(
                self.samples,
                verbose=True,
            )

        np.testing.assert_allclose(predictions, self.samples, atol=1e-4)
        mock_print.assert_called_once()
        self.assertIn("Converged in", mock_print.call_args[0][0])

    def test_predict_batch_pgd_warns_when_max_iterations_reached(self):
        """Batched PGD should warn and keep the initial simplex iterate when max_iter is zero."""
        with patch("builtins.print") as mock_print:
            predictions = self.deconvolver.predict_batch_pgd(self.samples, max_iter=0)

        np.testing.assert_allclose(
            predictions,
            np.full_like(self.samples, 1.0 / 3.0),
            atol=1e-10,
        )
        mock_print.assert_called_once_with(
            "Warning: Reached max iterations 0 without full convergence."
        )

    def test_predict_batch_pgd_requires_2d_input(self):
        """Batched PGD should reject vector inputs because it expects a sample matrix."""
        with self.assertRaisesRegex(ValueError, "X must be a 2D array"):
            self.deconvolver.predict_batch_pgd(np.array([1.0, 0.0, 0.0]))

    def test_predict_batch_pgd_requires_matching_feature_count(self):
        """Batched PGD should reject matrices whose feature count differs from fit."""
        with self.assertRaisesRegex(ValueError, "X must have shape"):
            self.deconvolver.predict_batch_pgd(np.ones((2, 2), dtype=float))

    def test_predict_single_sample_dispatches_to_pgd_solver(self):
        """Single-sample prediction should dispatch through the configured PGD solver."""
        prediction = self.deconvolver.predict_single_sample(self.samples[3])

        np.testing.assert_allclose(prediction, self.samples[3], atol=1e-4)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_sequential_uses_pgd_solver(self):
        """Sequential PSLS prediction should honor the configured PGD backend."""
        predictions = self.deconvolver._predict_sequential(self.samples)

        np.testing.assert_allclose(predictions, self.samples, atol=1e-4)

    def test_predict_chunk_sequential_uses_batch_pgd_path(self):
        """Chunk prediction should delegate the PGD path to the dedicated batched solver."""
        with patch.object(
            self.deconvolver,
            "predict_batch_pgd",
            return_value=np.full((2, 3), 1.0 / 3.0),
        ) as mock_predict_batch_pgd:
            predictions = self.deconvolver._predict_chunk_sequential(self.samples[:2])

        np.testing.assert_allclose(predictions, np.full((2, 3), 1.0 / 3.0))
        mock_predict_batch_pgd.assert_called_once()
        np.testing.assert_allclose(
            mock_predict_batch_pgd.call_args.args[0],
            self.samples[:2],
            atol=1e-10,
        )

    def test_predict_parallel_rejects_non_positive_chunk_size(self):
        """Parallel PSLS should reject invalid chunk sizes before creating workers."""
        with self.assertRaisesRegex(
            ValueError, "chunk_size must be a positive integer"
        ):
            self.deconvolver._predict_parallel(
                self.samples,
                n_workers=2,
                chunk_size=0,
            )

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_parallel_matches_expected_outputs(self):
        """Parallel PGD PSLS should recover the known simplex solutions."""
        predictions = self.deconvolver._predict_parallel(
            self.samples,
            n_workers=2,
            chunk_size=2,
        )

        np.testing.assert_allclose(predictions, self.samples, atol=1e-4)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_predict_dispatches_between_sequential_and_parallel_paths(self):
        """Public PSLS prediction should switch between sequential and parallel backends."""
        sequential = self.deconvolver.predict(self.samples, n_workers=1)
        parallel = self.deconvolver.predict(self.samples, n_workers=2, chunk_size=2)

        np.testing.assert_allclose(sequential, self.samples, atol=1e-4)
        np.testing.assert_allclose(parallel, self.samples, atol=1e-4)

    @patch("methyldl.deconvolution.least_squares_deconvolvers.tqdm", _passthrough_tqdm)
    def test_save_and_load_round_trip(self):
        """save/load should round-trip a fitted PGD PSLS model and produce matching predictions."""
        expected = self.deconvolver.predict(self.samples, n_workers=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "psls_pgd.joblib")
            self.deconvolver.save(path)
            loaded = PSLSDeconvolver.load(path)

        self.assertIsInstance(loaded, PSLSDeconvolver)
        self.assertEqual(loaded.solver_type, "pgd")
        np.testing.assert_allclose(
            loaded.reference_prediction_matrix_,
            self.deconvolver.reference_prediction_matrix_,
        )
        actual = loaded.predict(self.samples, n_workers=1)
        np.testing.assert_allclose(actual, expected, atol=1e-4)
