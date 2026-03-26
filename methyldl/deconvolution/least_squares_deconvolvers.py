"""
Implementation of NNLS and PSLS (probability simplex least squares) deconvolution methods.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import threading

import cvxpy as cp
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
from tqdm import tqdm
from scipy.optimize import nnls


class AbstractLSDeconvolver(BaseEstimator, RegressorMixin):
    """
    Parent class of least-squares-based deconvolution methods.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store the reference prediction matrix built from pure reference predictions.

        Args:
            X: Array of shape (n_cell_types, n_features) containing one flattened
                reference prediction vector per cell type.
            y: Array of shape (n_cell_types,) containing the integer label associated
                with each row of X. Labels must be unique and span
                0, ..., n_cell_types - 1.

        Returns:
            The fitted AbstractLSDeconvolver instance.
        """
        assert X.ndim == 2, "X must be a 2D matrix of shape (n_cell_types, n_features)"
        self.n_cell_types_ = len(y)
        self.n_features_ = X.shape[1]
        assert set(y) == set(range(self.n_cell_types_))
        assert X.shape[0] == self.n_cell_types_
        self.reference_prediction_matrix_ = np.hstack(
            [
                X[y == cell_type_label].reshape(-1, 1)
                for cell_type_label in range(self.n_cell_types_)
            ]
        )
        return self

    def save(self, filepath: str):
        """Save the fitted model to disk using joblib.

        Args:
            filepath: Destination path (e.g. ``'model.joblib'``).

        Raises:
            sklearn.exceptions.NotFittedError: If the model has not been fit.
        """
        check_is_fitted(self)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AbstractLSDeconvolver":
        """Load a saved model from disk.

        Args:
            filepath: Path to the saved model file.

        Returns:
            The loaded model instance.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            TypeError: If the loaded object is not an instance of this class.
            sklearn.exceptions.NotFittedError: If the loaded model was not fit.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        model = joblib.load(filepath)
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is not of type {cls.__name__}"
            )
        check_is_fitted(model)
        return model

    def predict_single_sample(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement predict_single_sample")

    def _predict_chunk_sequential(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Predict NNLS outputs for a chunk of samples sequentially.

        Args:
            X_chunk: Array of shape (chunk_size, n_features) containing a subset of
                samples to process in the current worker.

        Returns:
            A tuple ``(normalized, unnormalized, residuals)`` where the first two
            arrays have shape (chunk_size, n_cell_types) and ``residuals`` has shape
            (chunk_size,).
        """
        n_samples = X_chunk.shape[0]
        chunk_predictions = np.zeros((n_samples, self.n_cell_types_))

        for i, sample in enumerate(X_chunk):
            chunk_predictions[i] = self.predict_single_sample(sample)

        return chunk_predictions


class NNLSDeconvolver(AbstractLSDeconvolver):
    """
    In this implementation, we use NNLS (Non Negative Least Squares)
    to predict the mixture proportions for each sample, given a reference prediction matrix
    containing the pure predictions for each cell type.
    """

    def predict_single_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Predict mixture proportions for a single sample.

        Args:
                x: Array of shape (n_features,) containing the sample prediction vector.

        Returns:
                A tuple ``(normalized, unnormalized, residual)`` where:
                - ``normalized`` has shape (n_cell_types,) and sums to 1 when possible.
                - ``unnormalized`` has shape (n_cell_types,) and contains the raw NNLS
                    coefficients.
                - ``residual`` is the scalar residual returned by ``scipy.optimize.nnls``.
        """
        assert x.shape == (self.n_features_,)
        unnorm_mixture_proportions, residual = nnls(
            self.reference_prediction_matrix_, x
        )
        mixture_proportions_sum = unnorm_mixture_proportions.sum()
        if mixture_proportions_sum > 0:
            norm_mixture_proportions = (
                unnorm_mixture_proportions / mixture_proportions_sum
            )
        else:
            norm_mixture_proportions = unnorm_mixture_proportions
        return norm_mixture_proportions, unnorm_mixture_proportions, residual

    def _predict_chunk_sequential(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Predict NNLS outputs for a chunk of samples sequentially.

        Args:
            X_chunk: Array of shape (chunk_size, n_features) containing a subset of
                samples to process in the current worker.

        Returns:
            A tuple ``(normalized, unnormalized, residuals)`` where the first two
            arrays have shape (chunk_size, n_cell_types) and ``residuals`` has shape
            (chunk_size,).
        """
        n_samples = X_chunk.shape[0]
        mixture_prop_pred = np.zeros((n_samples, self.n_cell_types_))
        mixture_prop_unnorm_pred = np.zeros((n_samples, self.n_cell_types_))
        residuals = np.zeros(n_samples)

        for i, sample in enumerate(X_chunk):
            mixture_prop_pred[i], mixture_prop_unnorm_pred[i], residuals[i] = (
                self.predict_single_sample(sample)
            )

        return mixture_prop_pred, mixture_prop_unnorm_pred, residuals

    def _predict_parallel(
        self, X: np.ndarray, n_workers: int = 1, chunk_size: int = 100
    ) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples in parallel.

        The input batch is split into chunks, and each worker processes one chunk
        sequentially to reduce thread-pool overhead.

        Args:
            X: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.
            n_workers: Number of worker threads to use.
            chunk_size: Number of samples assigned to each chunk.

        Returns:
            A tuple ``(normalized, unnormalized, residuals)`` where the first two
            arrays have shape (n_samples, n_cell_types) and ``residuals`` has shape
            (n_samples,).

        Raises:
            ValueError: If ``chunk_size`` is not strictly positive.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        chunks = [
            X[start : start + chunk_size] for start in range(0, X.shape[0], chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._predict_chunk_sequential, chunks),
                    total=len(chunks),
                    desc="Predicting with NNLS in parallel",
                )
            )

        mixture_prop_pred = np.vstack([res[0] for res in results])
        mixture_prop_unnorm_pred = np.vstack([res[1] for res in results])
        residuals = np.concatenate([res[2] for res in results])
        return mixture_prop_pred, mixture_prop_unnorm_pred, residuals

    def _predict_sequential(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples sequentially.

        Args:
            X: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.

        Returns:
            A tuple ``(normalized, unnormalized, residuals)`` where the first two
            arrays have shape (n_samples, n_cell_types) and ``residuals`` has shape
            (n_samples,).
        """
        n_samples = X.shape[0]

        # prepare_predictions for each sample
        mixture_prop_pred = np.zeros((n_samples, self.n_cell_types_))
        mixture_prop_unnorm_pred = np.zeros((n_samples, self.n_cell_types_))
        residuals = np.zeros(n_samples)

        # predict the mixture proportions for each sample
        for i, sample in tqdm(
            enumerate(X), total=n_samples, desc="Predicting with NNLS sequentially"
        ):
            mixture_prop_pred[i], mixture_prop_unnorm_pred[i], residuals[i] = (
                self.predict_single_sample(sample)
            )
        return mixture_prop_pred, mixture_prop_unnorm_pred, residuals

    def predict(
        self, X: np.ndarray, n_workers: int = 1, chunk_size: int = 100
    ) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples.

        Sequential (n_workers=1) prediction is the default and the fastest.

        Args:
            X: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.
            n_workers: Number of worker threads to use. Values greater than 1 enable
                chunk-based parallel prediction.
            chunk_size: Number of samples per chunk when ``n_workers > 1``.

        Returns:
            A tuple ``(normalized, unnormalized, residuals)`` where the first two
            arrays have shape (n_samples, n_cell_types) and ``residuals`` has shape
            (n_samples,).
        """
        if n_workers > 1:
            return self._predict_parallel(X, n_workers, chunk_size=chunk_size)
        else:
            return self._predict_sequential(X)


class PSLSDeconvolver(AbstractLSDeconvolver):
    """Probability Simplex Least Squares Deconvolver (PSLS).

    This is a variant of NNLS where we add the additional constraint
    that the mixture proportions must sum to 1, in addition to being non-negative.

    It is highly advised to use the solver_type "cvxpy" for this deconvolver,
    as the PGD solver is slower and less accurate.
    """

    def __init__(self, solver_type="cvxpy"):
        """Initialize the deconvolver state used across repeated fits."""
        assert solver_type in [
            "cvxpy",
            "pgd",
        ], f"Unsupported solver_type: {solver_type}"
        self._cvxpy_fit_generation_ = -1
        self.solver_type = solver_type

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the deconvolver from pure reference predictions.

        Args:
            X: Array of shape (n_cell_types, n_features) containing one flattened
                reference prediction vector per cell type.
            y: Array of shape (n_cell_types,) containing the integer label associated
                with each row of X. Labels must be unique and span
                0, ..., n_cell_types - 1.

        Returns:
            The fitted PSLSDeconvolver instance.
        """
        # call AbstractLSDeconvolver.fit to store the reference prediction matrix and related state
        super().fit(X, y)

        if self.solver_type == "pgd":
            # precomputed matrices for the PGD algorithm
            self._pgd_ptp_ = (
                self.reference_prediction_matrix_.T @ self.reference_prediction_matrix_
            )
            self._pgd_xtp_multiplier_ = self.reference_prediction_matrix_
            self._pgd_step_size_ = 1.0 / np.linalg.norm(self._pgd_ptp_, ord=2)
        elif self.solver_type == "cvxpy":
            # cvxpy problem state for each worker thread
            # (to avoid rebuilding the problem for each worker)
            self._cvxpy_thread_cache_ = threading.local()
            self._cvxpy_fit_generation_ = self._cvxpy_fit_generation_ + 1
        else:
            raise ValueError(f"Unsupported solver_type: {self.solver_type}")

        return self

    def __getstate__(self):
        """Exclude the unpicklable ``threading.local`` CVXPY cache."""
        state = self.__dict__.copy()
        state.pop("_cvxpy_thread_cache_", None)
        return state

    def __setstate__(self, state):
        """Restore state and create a fresh thread-local CVXPY cache."""
        self.__dict__.update(state)
        if self.solver_type == "cvxpy":
            self._cvxpy_thread_cache_ = threading.local()

    def predict_single_sample_pgd(
        self, x: np.ndarray, max_iter=1000, tol=1e-5, verbose=False
    ):
        """
        Predict mixture proportions for one sample using projected gradient descent.

        This solves the probability-simplex least-squares problem
        ``min 0.5 * ||P w - x||^2`` subject to ``w >= 0`` and ``sum(w) = 1``.

        Args:
            x: Array of shape (n_features,) containing the sample prediction vector.
            max_iter: Maximum number of PGD iterations.
            tol: Infinity-norm convergence tolerance between successive iterates.
            verbose: Whether to print convergence information.

        Returns:
            Array of shape (n_cell_types,) containing the estimated mixture
            proportions.
        """
        assert x.shape == (self.n_features_,)
        w = np.full(self.n_cell_types_, 1.0 / self.n_cell_types_)
        ptm = x @ self._pgd_xtp_multiplier_

        for i in range(max_iter):
            w_old = w.copy()

            # 1. Gradient Step: w_next = w - eta * grad
            # grad = P^T(Pw - M) = PtP @ w - PtM
            grad = self._pgd_ptp_ @ w - ptm
            v = w - self._pgd_step_size_ * grad

            # 2. Projection Step
            w = _project_on_simplex(v)

            # Check convergence
            if np.linalg.norm(w - w_old, ord=np.inf) < tol:
                if verbose:
                    print(f"Converged in {i} iterations.")
                return w

        print("Warning: Reached max iterations without full convergence.")
        return w

    def predict_batch_pgd(
        self, X: np.ndarray, max_iter=2000, tol=1e-5, verbose=False
    ) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples using projected gradient
        descent.

        Args:
            X: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.
            max_iter: Maximum number of PGD iterations.
            tol: Stopping tolerance on the maximum absolute update across the batch.
            verbose: Whether to print convergence information.

        Returns:
            Array of shape (n_samples, n_cell_types) containing the estimated
            mixture proportions.

        Raises:
            ValueError: If X is not two-dimensional or has the wrong feature count.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features_}), got {X.shape}"
            )

        n_samples = X.shape[0]
        w = np.full(
            (n_samples, self.n_cell_types_),
            1.0 / self.n_cell_types_,
        )
        ptm = X @ self._pgd_xtp_multiplier_

        for iteration in range(max_iter):
            w_old = w.copy()
            grad = w @ self._pgd_ptp_ - ptm
            v = w - self._pgd_step_size_ * grad
            w = _project_rows_on_simplex(v)

            if np.max(np.abs(w - w_old)) < tol:
                if verbose:
                    print(f"Converged in {iteration} iterations.")
                return w

        print(f"Warning: Reached max iterations {max_iter} without full convergence.")
        return w

    def _build_cvxpy_problem(self):
        """
        Build the reusable CVXPY problem for single-sample PSLS inference.

        Returns:
            A tuple ``(x_param, w, problem)`` containing the input parameter,
            optimization variable, and compiled CVXPY problem object.
        """
        x_param = cp.Parameter(self.n_features_)
        w = cp.Variable(self.n_cell_types_)
        objective = cp.Minimize(
            cp.sum_squares(self.reference_prediction_matrix_ @ w - x_param)
        )
        constraints = [w >= 0, cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        return x_param, w, problem

    def _get_cvxpy_worker_problem(self):
        """
        Get the thread-local CVXPY problem associated with the current fit.

        The problem is cached per worker thread and rebuilt when the model is fit
        again so that solver state does not leak across incompatible shapes.

        Returns:
            A tuple ``(x_param, w, problem)`` for the current worker thread.
        """
        thread_cache = self._cvxpy_thread_cache_
        cache_is_missing = not hasattr(thread_cache, "fit_generation")
        cache_is_stale = (
            not cache_is_missing
            and thread_cache.fit_generation != self._cvxpy_fit_generation_
        )

        if cache_is_missing or cache_is_stale:
            x_param, w, problem = self._build_cvxpy_problem()
            thread_cache.x_param = x_param
            thread_cache.w = w
            thread_cache.problem = problem
            thread_cache.fit_generation = self._cvxpy_fit_generation_

        return thread_cache.x_param, thread_cache.w, thread_cache.problem

    def predict_single_sample_cvxpy(self, x: np.ndarray) -> np.ndarray:
        """
        Predict mixture proportions for one sample using the CVXPY formulation.

        Args:
            x: Array of shape (n_features,) containing the sample prediction vector.

        Returns:
            Array of shape (n_cell_types,) containing the estimated mixture
            proportions.
        """
        assert x.shape == (self.n_features_,)
        x_param, w, problem = self._get_cvxpy_worker_problem()
        x_param.value = x
        problem.solve(solver=cp.OSQP, warm_start=True)
        return w.value

    def predict_single_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the mixture proportions for a single sample using the specified solver_type.

        Args:
            x: Array of shape (n_features,) containing the sample prediction vector.

        Returns:
            Array of shape (n_cell_types,) containing the estimated mixture
            proportions.
        """
        if self.solver_type == "cvxpy":
            return self.predict_single_sample_cvxpy(x)
        elif self.solver_type == "pgd":
            return self.predict_single_sample_pgd(x)
        else:
            raise ValueError(f"Unsupported solver_type: {self.solver_type}")

    def _predict_sequential(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples sequentially.

        Args:
            X: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.

        Returns:
            Array of shape (n_samples, n_cell_types) containing the estimated
            mixture proportions.
        """
        n_samples = X.shape[0]

        # prepare_predictions for each sample
        mixture_prop_pred = np.zeros((n_samples, self.n_cell_types_))

        # predict the mixture proportions for each sample
        for i, sample in tqdm(
            enumerate(X), total=n_samples, desc="Predicting with PSLS sequentially"
        ):
            mixture_prop_pred[i] = self.predict_single_sample(sample)
        return mixture_prop_pred

    def _predict_chunk_sequential(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Predict mixture proportions for one chunk of samples sequentially.

        Args:
            X_chunk: Array of shape (chunk_size, n_features) containing the samples
                assigned to a worker.

        Returns:
            Array of shape (chunk_size, n_cell_types) containing the estimated
            mixture proportions for the chunk.
        """
        if self.solver_type == "pgd":
            return self.predict_batch_pgd(X_chunk)

        n_samples = X_chunk.shape[0]

        # prepare_predictions for each sample
        mixture_prop_pred = np.zeros((n_samples, self.n_cell_types_))

        # predict the mixture proportions for each sample
        for i, sample in enumerate(X_chunk):
            mixture_prop_pred[i] = self.predict_single_sample(sample)
        return mixture_prop_pred

    def _predict_parallel(
        self, x: np.ndarray, n_workers: int = 1, chunk_size=100
    ) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples in parallel.

        The input batch is split into chunks, and each worker processes its chunk
        sequentially. For the ``"pgd"`` path, chunk execution delegates to the batched
        PGD implementation.

        Args:
            x: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.
            n_workers: Number of worker threads to use.
            chunk_size: Number of samples assigned to each chunk.

        Returns:
            Array of shape (n_samples, n_cell_types) containing the estimated
            mixture proportions.

        Raises:
            ValueError: If ``chunk_size`` is not strictly positive.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        chunks = [
            x[start : start + chunk_size] for start in range(0, x.shape[0], chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda chunk: self._predict_chunk_sequential(chunk),
                        chunks,
                    ),
                    total=len(chunks),
                    desc="Predicting with PSLS in parallel",
                )
            )
        mixture_prop_pred = np.vstack(results)
        return mixture_prop_pred

    def predict(self, X: np.ndarray, n_workers: int = 1, chunk_size=100) -> np.ndarray:
        """
        Predict mixture proportions for a batch of samples.

        The fastest tested setup is n_workers>1 (but not too large) with solver_type="cvxpy"
        and a moderate chunk_size (e.g. 100).

        Args:
            X: Array of shape (n_samples, n_features) containing the samples to
                deconvolve.
            n_workers: Number of worker threads to use. Values greater than 1 enable
                chunk-based parallel prediction.
            chunk_size: Number of samples per chunk when ``n_workers > 1``.

        Returns:
            Array of shape (n_samples, n_cell_types) containing the estimated
            mixture proportions.
        """
        if n_workers > 1:
            return self._predict_parallel(X, n_workers=n_workers, chunk_size=chunk_size)
        else:
            return self._predict_sequential(X)


def _project_on_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projects a vector v onto the probability simplex in O(c log c) time.

    The projection is given by the formula:
        w_i = max(v_i - theta, 0)
    where theta is chosen such that sum(w) = 1.

    Args:
        v: A vector of shape (c,) containing the input values to be projected.

    Returns:
        A vector of shape (c,) containing the projected values, which are non-negative and sum to 1.
    """
    c = v.shape[0]
    # Sort v in descending order
    u = np.sort(v)[::-1]
    # Calculate the cumulative sum of sorted elements
    cssv = np.cumsum(u)
    # Find the indices where the threshold condition is met
    ind = np.arange(1, c + 1)
    cond = u - (cssv - 1.0) / ind > 0
    # rho is the largest index that satisfies the condition
    rho = ind[cond][-1]
    # Calculate the threshold theta
    theta = (cssv[rho - 1] - 1.0) / rho
    # Project: w_i = max(v_i - theta, 0)
    return np.maximum(v - theta, 0)


def _project_rows_on_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projects each row of v onto the probability simplex.

    Args:
        v: A matrix of shape (n_samples, c) containing the input values to be projected.

    Returns:
        A matrix of shape (n_samples, c) containing the projected values, where each row
        is non-negative and sums to 1.
    """
    if v.ndim != 2:
        raise ValueError("v must be a 2D array of shape (n_samples, n_cell_types)")

    n_samples, c = v.shape
    # Sort v in descending order
    u = np.sort(v, axis=1)[:, ::-1]
    # Calculate the cumulative sum of sorted elements
    cssv = np.cumsum(u, axis=1)
    # Find the indices where the threshold condition is met
    ind = np.arange(1, c + 1)
    cond = u - (cssv - 1.0) / ind > 0
    # rho is the largest index that satisfies the condition
    rho = cond.sum(axis=1) - 1
    # Calculate the threshold theta
    theta = (cssv[np.arange(n_samples), rho] - 1.0) / (rho + 1)
    # Project: w_i = max(v_i - theta, 0)
    return np.maximum(v - theta[:, None], 0)
