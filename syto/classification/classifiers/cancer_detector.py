"""CancerDetector-style Bayesian classifier for cell-type deconvolution.

Implements the read-level classification approach described in the CancerDetector
paper (Li et al., 2018). For each marker (DMR) and class (cell type), a Beta
distribution is fitted to the observed methylation rates. At inference time,
read-level likelihoods are computed from these Beta parameters and combined with
class priors via Bayes' theorem to produce posterior class probabilities.
"""

from typing import Union
from pathlib import Path
import logging
import time

import joblib
import numpy as np
import pickle
import pandas as pd
from scipy import stats
from scipy.special import beta as beta_func  # pylint: disable=no-name-in-module
from tqdm import tqdm

from syto.classification.classifiers.abstract_read_classifier import (
    AbstractReadClassifier,
)
from syto.classification.evaluation import compute_metrics

_module_logger = logging.getLogger(__name__)


def _fit_beta_worker(args):
    """Worker function that unpacks arguments and delegates to
    :meth:`CancerDetectorClassifier.fit_single_beta_distribution`.

    Designed to be used as a map target for parallel or sequential beta fitting.

    Args:
        args: Tuple of (marker_idx, class_idx, n_meth_cpgs, n_unmeth_cpgs,
            eps_beta_fit).

    Returns:
        Tuple of (marker_idx, class_idx, eta, rho, bayesian_est_0,
        bayesian_est_1, insufficient_data).
    """
    marker_idx, class_idx, n_meth_cpgs, n_unmeth_cpgs, eps_beta_fit = args
    result = CancerDetectorClassifier.fit_single_beta_distribution(
        n_meth_cpgs, n_unmeth_cpgs, eps_beta_fit
    )
    return (marker_idx, class_idx, *result)


class CancerDetectorClassifier(AbstractReadClassifier):
    """Bayesian classifier that models per-marker methylation rates as Beta
    distributions for each cell type.

    During fitting, a Beta distribution is estimated for every (marker, class)
    pair from the observed methylation / unmethylation CpG counts. At inference
    time, likelihoods are derived from these Beta parameters and multiplied by
    class priors to obtain posterior probabilities via Bayes' theorem.

    Attributes:
        param_eta_matrix: Array of shape (n_markers, n_classes) holding the
            first Beta shape parameter for each marker-class pair.
        param_rho_matrix: Array of shape (n_markers, n_classes) holding the
            second Beta shape parameter for each marker-class pair.
        markers: Sorted list of unique marker labels seen during fitting.
        classes: Sorted list of unique class labels seen during fitting.
        class_priors: Array of shape (n_classes,) with the prior probability
            of each class.
        is_fitted: Whether :meth:`fit` has been called successfully.
    """

    def __init__(self):
        """Initialise an unfitted CancerDetectorClassifier with default
        (``None``) attributes."""
        self.param_eta_matrix = None
        self.param_rho_matrix = None
        self.eps_beta_fit = None
        self.markers = None
        self.classes = None
        self.marker_to_idx = None
        self.class_to_idx = None
        self.class_prior_type = None
        self.class_priors = None
        self.n_classes = None
        self.n_markers = None
        self.is_fitted = False
        self.mask_bayesian_estimation_0 = None
        self.mask_bayesian_estimation_1 = None
        self.mask_insufficient_data = None
        self.history: list = []

    @staticmethod
    def fit_single_beta_distribution(
        n_meth_cpgs: np.ndarray, n_unmeth_cpgs: np.ndarray, eps_beta_fit: float
    ) -> tuple[float, float, bool, bool, bool]:
        """Fit a Beta distribution to the methylation rates of a single
        marker-class pair.

        Handles edge cases where all rates are 0 or 1 (Bayesian estimation
        with a uniform prior) and where fewer than 3 reads are available
        (returns the uniform Beta(1, 1)).

        Args:
            n_meth_cpgs: 1-D array of methylated CpG counts per read.
            n_unmeth_cpgs: 1-D array of unmethylated CpG counts per read.
            eps_beta_fit: Small epsilon used to clip methylation rates away
                from 0 and 1 before MLE fitting.

        Returns:
            Tuple of (eta, rho, bayesian_est_0, bayesian_est_1,
            insufficient_data) where *eta* and *rho* are the fitted Beta
            shape parameters and the boolean flags indicate which fallback
            strategy (if any) was used.
        """
        n_cpgs = n_meth_cpgs + n_unmeth_cpgs
        meth_rates = n_meth_cpgs / np.clip(n_cpgs, 1, None)
        n_reads = len(n_meth_cpgs)
        bayesian_est_0 = False
        bayesian_est_1 = False
        insufficient_data = False

        if n_reads < 3:
            # not enough data to fit a beta distribution, return default parameters
            # (eta=1, rho=1: uniform distribution)
            insufficient_data = True
            eta = 1
            rho = 1
        elif np.all(meth_rates == 0):
            # all meth rates are zero, set the parameters using
            # bayes estimation with a uniform prior (alpha=1, beta=1)
            # (the scipy MLE would not converge)
            eta = 1
            rho = 1 + n_cpgs.sum()
            bayesian_est_0 = True
        elif np.all(meth_rates == 1):
            # all meth rates are one, set the parameters using
            # bayes estimation with a uniform prior (alpha=1, beta=1)
            # (the scipy MLE would not converge)
            eta = 1 + n_cpgs.sum()
            rho = 1
            bayesian_est_1 = True
        else:
            # fit a beta distribution to the meth rates
            # we clip the meth rates at 0+eps and 1-eps to avoid issues during fitting
            clipped_meth_rates = np.clip(meth_rates, eps_beta_fit, 1 - eps_beta_fit)
            eta, rho, _, _ = stats.beta.fit(clipped_meth_rates, floc=0, fscale=1)
        return eta, rho, bayesian_est_0, bayesian_est_1, insufficient_data

    def fit(
        self,
        train_data: pd.DataFrame,
        col_n_meth_cpgs: str = "M",
        col_n_unmeth_cpgs: str = "U",
        col_label: str = "original_label",
        col_marker_label: str = "dmr_ctype_label",
        eps_beta_fit: float = 1e-2,
        class_prior_type: str = "uniform",
        val_data: pd.DataFrame = None,
    ) -> "CancerDetectorClassifier":
        """Fit Beta distributions for every (marker, class) pair and compute
        class priors.

        Args:
            train_data: DataFrame where each row corresponds to a single read
                overlapping a marker region.
            col_n_meth_cpgs: Column name for the number of methylated CpGs.
            col_n_unmeth_cpgs: Column name for the number of unmethylated CpGs.
            col_label: Column name for the class / cell-type label.
            col_marker_label: Column name for the marker (DMR) label.
            eps_beta_fit: Epsilon for clipping methylation rates before MLE
                fitting (see :meth:`fit_single_beta_distribution`).
            class_prior_type: Strategy for computing class priors. Either
                ``"uniform"`` (equal priors) or ``"train_freq"`` (proportional
                to class frequency in the training data).
            val_data: Optional validation DataFrame with the same schema as
                ``train_data``. When provided, validation metrics are computed
                and stored in :attr:`history`.
        """
        # argument checks
        assert (
            col_n_meth_cpgs in train_data.columns
        ), f"Column {col_n_meth_cpgs} not found in train_data"
        assert (
            col_n_unmeth_cpgs in train_data.columns
        ), f"Column {col_n_unmeth_cpgs} not found in train_data"
        assert (
            col_label in train_data.columns
        ), f"Column {col_label} not found in train_data"
        assert (
            col_marker_label in train_data.columns
        ), f"Column {col_marker_label} not found in train_data"
        assert class_prior_type in ["uniform", "train_freq"]

        t_start = time.time()

        self.eps_beta_fit = eps_beta_fit
        self.class_prior_type = class_prior_type
        self.markers = np.array(sorted(train_data[col_marker_label].unique()))
        self.classes = np.array(sorted(train_data[col_label].unique()))
        self.marker_to_idx = {m: i for i, m in enumerate(self.markers)}
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_markers = len(self.markers)
        self.n_classes = len(self.classes)

        self.param_eta_matrix = np.ones((self.n_markers, self.n_classes))
        self.param_rho_matrix = np.ones((self.n_markers, self.n_classes))
        # this mask will be used to identify marker-class combinations for which the beta
        # distribution parameters were estimated using bayesian estimation
        # due to methylation rates being all zero
        self.mask_bayesian_estimation_0 = np.zeros(
            (self.n_markers, self.n_classes), dtype=bool
        )
        # this mask will be used to identify marker-class combinations for which the beta
        # distribution parameters were estimated using bayesian estimation
        # due to methylation rates being all one
        self.mask_bayesian_estimation_1 = np.zeros(
            (self.n_markers, self.n_classes), dtype=bool
        )
        # this mask will be used to identify marker-class combinations for which the beta
        # distribution parameters were kept as the default (eta=1, rho=1) due to insufficient data to fit a beta distribution
        self.mask_insufficient_data = np.zeros(
            (self.n_markers, self.n_classes), dtype=bool
        )

        # pre-group data in a single pass instead of M×C DataFrame scans
        grouped = train_data.groupby([col_marker_label, col_label])
        tasks = []
        for (marker_label, class_label), group in grouped:
            tasks.append(
                (
                    self.marker_to_idx[marker_label],
                    self.class_to_idx[class_label],
                    group[col_n_meth_cpgs].to_numpy(),
                    group[col_n_unmeth_cpgs].to_numpy(),
                    eps_beta_fit,
                )
            )
        results = [_fit_beta_worker(t) for t in tasks]

        for marker_idx, class_idx, eta, rho, be0, be1, insuf in results:
            self.param_eta_matrix[marker_idx, class_idx] = eta
            self.param_rho_matrix[marker_idx, class_idx] = rho
            self.mask_bayesian_estimation_0[marker_idx, class_idx] = be0
            self.mask_bayesian_estimation_1[marker_idx, class_idx] = be1
            self.mask_insufficient_data[marker_idx, class_idx] = insuf

        # compute the class priors
        if self.class_prior_type == "uniform":
            self.class_priors = np.ones(self.n_classes) / self.n_classes
        elif self.class_prior_type == "train_freq":
            class_counts = train_data[col_label].value_counts().sort_index().to_numpy()
            self.class_priors = class_counts / class_counts.sum()

        self.is_fitted = True

        elapsed = time.time() - t_start

        fit_record: dict = {
            "n_markers": self.n_markers,
            "n_classes": self.n_classes,
            "class_prior_type": self.class_prior_type,
            "elapsed_time": elapsed,
            "n_insufficient_data": int(self.mask_insufficient_data.sum()),
            "n_bayesian_est_0": int(self.mask_bayesian_estimation_0.sum()),
            "n_bayesian_est_1": int(self.mask_bayesian_estimation_1.sum()),
        }

        train_proba = self.predict_proba(
            train_data,
            col_n_meth_cpgs=col_n_meth_cpgs,
            col_n_unmeth_cpgs=col_n_unmeth_cpgs,
            col_marker_label=col_marker_label,
        )
        train_labels = train_data[col_label].to_numpy()
        fit_record.update(self._compute_fit_metrics(train_labels, train_proba, "train"))

        if val_data is not None:
            val_proba = self.predict_proba(
                val_data,
                col_n_meth_cpgs=col_n_meth_cpgs,
                col_n_unmeth_cpgs=col_n_unmeth_cpgs,
                col_marker_label=col_marker_label,
            )
            val_labels = val_data[col_label].to_numpy()
            fit_record.update(self._compute_fit_metrics(val_labels, val_proba, "val"))

        self.history.append(fit_record)

        _module_logger.info(
            "Fitted CancerDetectorClassifier: %d markers, %d classes, "
            "prior=%s, elapsed=%.2fs. "
            "Train: acc=%.4f f1=%.4f%s",
            self.n_markers,
            self.n_classes,
            self.class_prior_type,
            elapsed,
            fit_record.get("train_accuracy", float("nan")),
            fit_record.get("train_f1", float("nan")),
            (
                " Val: acc=%.4f f1=%.4f"
                % (
                    fit_record.get("val_accuracy", float("nan")),
                    fit_record.get("val_f1", float("nan")),
                )
                if val_data is not None
                else ""
            ),
        )

        return self

    def _compute_fit_metrics(
        self,
        labels: np.ndarray,
        predictions_proba: np.ndarray,
        prefix: str,
    ) -> dict:
        """Compute sklearn classification metrics from fit-time predictions.

        Args:
            labels: 1-D integer array of ground-truth class labels.
            predictions_proba: Array of shape (n_reads, n_classes) with
                posterior probabilities from :meth:`predict_proba`.
            prefix: Prefix prepended to each metric key (e.g. ``"train"``
                or ``"val"``).

        Returns:
            Dict with keys like ``{prefix}_accuracy``, ``{prefix}_f1``, etc.
        """
        # Map original class labels (which may be strings or non-contiguous ints)
        # to the 0-based integer indices used by predict_proba column ordering.
        integer_labels = np.array([self.class_to_idx[lbl] for lbl in labels])
        metrics = compute_metrics((predictions_proba, integer_labels))
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def _compute_likelihood_single_read(
        self, n_meth: int, n_unmeth: int, marker_label: str
    ) -> np.ndarray:
        """Compute the likelihood of a single read across all classes.

        Uses the Beta-Binomial likelihood formula derived from the fitted
        Beta parameters for the given marker.

        Args:
            n_meth: Number of methylated CpGs in the read.
            n_unmeth: Number of unmethylated CpGs in the read.
            marker_label: The marker (DMR) label for this read.

        Returns:
            1-D array of shape (n_classes,) with the likelihood of the read
            under each class's Beta distribution. Returns uniform likelihoods
            if the read has zero CpGs.
        """
        n_cpgs = n_meth + n_unmeth
        if n_cpgs == 0:
            # if there are no CpGs, we cannot compute a likelihood, return uniform likelihoods
            return np.ones(self.n_classes) / self.n_classes

        marker_idx = self.marker_to_idx[marker_label]
        eta = self.param_eta_matrix[marker_idx, :]
        rho = self.param_rho_matrix[marker_idx, :]
        likelihoods = (
            beta_func(1 + eta, rho) ** n_meth * beta_func(eta, 1 + rho) ** n_unmeth
        ) / beta_func(eta, rho) ** n_cpgs
        return likelihoods

    def compute_likelihood_bulk(
        self,
        n_meth_array: np.ndarray,
        n_unmeth_array: np.ndarray,
        marker_labels: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute likelihoods for a batch of reads.

        Iterates over all reads and delegates to
        :meth:`_compute_likelihood_single_read`.

        Args:
            n_meth_array: 1-D array of methylated CpG counts, one per read.
            n_unmeth_array: 1-D array of unmethylated CpG counts, one per read.
            marker_labels: 1-D array of marker (DMR) labels, one per read.
            verbose: If ``True``, display a progress bar.

        Returns:
            Array of shape (n_reads, n_classes) with the likelihood of each
            read under each class.
        """
        n_reads = len(n_meth_array)
        likelihoods = np.ones((n_reads, self.n_classes))
        for i in tqdm(range(n_reads), disable=not verbose):
            marker_label = marker_labels[i]
            n_meth = n_meth_array[i]
            n_unmeth = n_unmeth_array[i]
            likelihoods[i] = self._compute_likelihood_single_read(
                n_meth, n_unmeth, marker_label
            )
        return likelihoods

    def _predict_proba_from_likelihoods(self, likelihoods: np.ndarray) -> np.ndarray:
        """Convert raw likelihoods into posterior class probabilities using
        Bayes' theorem.

        Args:
            likelihoods: Array of shape (n_reads, n_classes) containing the
                likelihood of each read under each class's Beta distribution.

        Returns:
            Array of shape (n_reads, n_classes) with the posterior probability
            of each class for every read.

        Raises:
            ValueError: If the model has not been fitted or if the likelihoods
                array has an unexpected number of columns.
        """
        # check arguments
        if self.class_priors is None:
            raise ValueError("Class priors not set. Fit the model first.")
        if likelihoods.ndim == 1:
            likelihoods = likelihoods.reshape(1, -1)
        if likelihoods.shape[1] != self.n_classes:
            raise ValueError(
                f"Likelihoods should have shape (n_reads, n_classes), but got {likelihoods.shape}"
            )

        # compute the posterior probabilities using Bayes' theorem
        posterior_numerators = likelihoods * self.class_priors.reshape(
            1, -1
        )  # shape (n_reads, n_classes)
        posterior_denominators = posterior_numerators.sum(
            axis=1, keepdims=True
        )  # shape (n_reads, 1)
        posterior_probas = (
            posterior_numerators / posterior_denominators
        )  # shape (n_reads, n_classes)
        return posterior_probas

    def predict_proba(
        self,
        test_data: pd.DataFrame,
        col_n_meth_cpgs: str = "M",
        col_n_unmeth_cpgs: str = "U",
        col_marker_label: str = "dmr_ctype_label",
        return_likelihoods: bool = False,
        verbose: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Predict posterior class probabilities for reads in a test DataFrame.

        Args:
            test_data: DataFrame where each row is a read overlapping a marker.
            col_n_meth_cpgs: Column name for the number of methylated CpGs.
            col_n_unmeth_cpgs: Column name for the number of unmethylated CpGs.
            col_marker_label: Column name for the marker (DMR) label.
            return_likelihoods: If ``True``, also return the raw likelihoods.
            verbose: If ``True``, display a progress bar during likelihood
                computation.

        Returns:
            If ``return_likelihoods`` is ``False``, an array of shape
            (n_reads, n_classes) with posterior probabilities. Otherwise a
            tuple of (posterior_probabilities, likelihoods).
        """
        n_meth_array = test_data[col_n_meth_cpgs].to_numpy()
        n_unmeth_array = test_data[col_n_unmeth_cpgs].to_numpy()
        marker_labels = test_data[col_marker_label].to_numpy()
        likelihoods = self.compute_likelihood_bulk(
            n_meth_array, n_unmeth_array, marker_labels, verbose=verbose
        )
        posterior_probas = self._predict_proba_from_likelihoods(likelihoods)

        if return_likelihoods:
            return posterior_probas, likelihoods
        return posterior_probas

    def save(self, path: str) -> None:
        """Save the fitted model parameters to a .pkl file.

        Args:
            path: Path to the .pkl file where the model parameters will be saved.
        """
        file_extension = Path(path).suffix

        if file_extension == ".pkl":
            _module_logger.warning(
                "Saving as .pkl is deprecated and is left only for backward compatibility. "
                "Please switch to .joblib."
            )
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "param_eta_matrix": self.param_eta_matrix,
                        "param_rho_matrix": self.param_rho_matrix,
                        "eps_beta_fit": self.eps_beta_fit,
                        "markers": self.markers,
                        "classes": self.classes,
                        "marker_to_idx": self.marker_to_idx,
                        "class_to_idx": self.class_to_idx,
                        "class_priors": self.class_priors,
                        "class_prior_type": self.class_prior_type,
                        "n_classes": self.n_classes,
                        "n_markers": self.n_markers,
                        "mask_bayesian_estimation_0": self.mask_bayesian_estimation_0,
                        "mask_bayesian_estimation_1": self.mask_bayesian_estimation_1,
                        "mask_insufficient_data": self.mask_insufficient_data,
                    },
                    f,
                )
        elif file_extension == ".joblib":
            joblib.dump(value=self, filename=path)
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Must be .pkl or .joblib."
            )

    @classmethod
    def load(cls, path: str, **kwargs) -> "CancerDetectorClassifier":
        """Load model parameters from a .pkl file.

        Args:
            path: Path to the .pkl file from which to load the model parameters.

        Returns:
            An instance of ``CancerDetectorClassifier`` with the loaded parameters.
        """

        file_extension = Path(path).suffix

        if file_extension == ".pkl":
            _module_logger.warning(
                "Loading as .pkl is deprecated and is left only for backward compatibility. "
                "Please switch to .joblib."
            )
            with open(path, "rb") as f:
                data = pickle.load(f)
            instance = cls()
            instance.param_eta_matrix = data["param_eta_matrix"]
            instance.param_rho_matrix = data["param_rho_matrix"]
            instance.eps_beta_fit = data["eps_beta_fit"]
            instance.markers = data["markers"]
            instance.classes = data["classes"]
            instance.class_priors = data["class_priors"]
            instance.class_prior_type = data["class_prior_type"]
            instance.n_classes = data["n_classes"]
            instance.n_markers = data["n_markers"]
            instance.mask_bayesian_estimation_0 = data["mask_bayesian_estimation_0"]
            instance.mask_bayesian_estimation_1 = data["mask_bayesian_estimation_1"]
            instance.mask_insufficient_data = data["mask_insufficient_data"]
            instance.marker_to_idx = data["marker_to_idx"]
            instance.class_to_idx = data["class_to_idx"]
            instance.is_fitted = True
        elif file_extension == ".joblib":
            instance = joblib.load(path)
            if not isinstance(instance, cls):
                raise ValueError(f"Loaded object is not a {cls.__name__} instance")
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Must be .pkl or .joblib."
            )
        _module_logger.info("CancerDetector model loaded with checkpoint: %s", path)
        _module_logger.info("CancerDetector model loaded: %s", path)
        return instance

    def __str__(self) -> str:
        """String representation of the model, showing key attributes."""
        return (
            f"CancerDetectorClassifier(n_markers={self.n_markers}, "
            f"n_classes={self.n_classes}, "
            f"class_prior_type='{self.class_prior_type}', "
            f"is_fitted={self.is_fitted})"
        )

    def predict_split(self, split_df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict class probabilities for a DataFrame split.

        Args:
            split_df: Input DataFrame with read-level data.
            **kwargs: Additional keyword arguments. Can include:
                - grg_label_column: Column name for the Genomic Region Group (GRG) label
        """
        if not self.is_fitted:
            raise ValueError(
                "CancerDetectorClassifier must be fitted before prediction."
            )

        if not "grg_label_column" in kwargs:
            _module_logger.warning(
                "grg_label_column not provided in kwargs. Defaulting to 'dmr_ctype_label'."
            )
            grg_label_column = "dmr_ctype_label"
        else:
            grg_label_column = kwargs["grg_label_column"]

        probabilities = self.predict_proba(
            test_data=split_df,
            col_n_meth_cpgs="M",
            col_n_unmeth_cpgs="U",
            col_marker_label=grg_label_column,
            return_likelihoods=False,
            verbose=False,
        )
        pred_cols = [f"prediction_{i}" for i in range(self.n_classes)]
        pred_df = pd.DataFrame(probabilities, columns=pred_cols)
        result = split_df.copy()
        for col in pred_cols:
            result[col] = pred_df[col].values
        return result
