import pandas as pd
from scipy import stats
import numpy as np


class CancerDetectorClassifier:

    def __init__(self):
        self.is_fitted = False

    def fit(
        self,
        train_data: pd.DataFrame,
        col_n_meth_cpgs: str = "M",
        col_n_unmeth_cpgs: str = "U",
        col_label: str = "original_label",
        col_marker_label: str = "dmr_ctype_label",
        eps_beta_fit: float = 1e-2,
    ):
        """"""
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

        self.eps_beta_fit = eps_beta_fit
        self.col_n_meth_cpgs = col_n_meth_cpgs
        self.col_n_unmeth_cpgs = col_n_unmeth_cpgs
        self.col_label = col_label
        self.col_marker_label = col_marker_label

        # compute the number of reads in each marker
        self.cnts_in_marker = (
            train_data.groupby(col_marker_label)
            .agg(n_reads=(col_marker_label, "size"))
            .reset_index()
        )

        self.n_markers = train_data[col_marker_label].nunique()
        self.n_classes = train_data[col_label].nunique()

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

        # fit beta distribution for each marker and class
        for marker_label in train_data[col_marker_label].unique():
            for class_label in train_data[col_label].unique():
                subset = train_data[
                    (train_data[col_marker_label] == marker_label)
                    & (train_data[col_label] == class_label)
                ]
                n_reads = len(subset)
                n_meth_cpgs = subset[col_n_meth_cpgs].to_numpy()
                n_unmeth_cpgs = subset[col_n_unmeth_cpgs].to_numpy()
                n_cpgs = n_meth_cpgs + n_unmeth_cpgs
                meth_rates = n_meth_cpgs / np.clip(
                    n_cpgs, 1, None
                )  # avoid division by zero

                if n_reads < 3:
                    # not enough data to fit a beta distribution, skip this marker-class combination
                    # the default parameters (eta=1, rho=1: uniform distribution) will be used
                    self.mask_insufficient_data[marker_label, class_label] = True
                    continue
                elif np.all(meth_rates == 0):
                    # all meth rates are zero, set the parameters using
                    # bayes estimation with a uniform prior (alpha=1, beta=1)
                    # (the scipy MLE would not converge)
                    eta = 1
                    rho = 1 + n_cpgs.sum()
                    self.mask_bayesian_estimation_0[marker_label, class_label] = True
                elif np.all(meth_rates == 1):
                    # all meth rates are one, set the parameters using
                    # bayes estimation with a uniform prior (alpha=1, beta=1)
                    # (the scipy MLE would not converge)
                    eta = 1 + n_cpgs.sum()
                    rho = 1
                    self.mask_bayesian_estimation_1[marker_label, class_label] = True
                else:
                    # fit a beta distribution to the meth rates
                    # we clip the meth rates at 0+eps and 1-eps to avoid issues during fitting
                    clipped_meth_rates = np.clip(
                        meth_rates, self.eps_beta_fit, 1 - self.eps_beta_fit
                    )
                    eta, rho, _, _ = stats.beta.fit(
                        clipped_meth_rates, floc=0, fscale=1
                    )
                self.param_eta_matrix[marker_label, class_label] = eta
                self.param_rho_matrix[marker_label, class_label] = rho

        self.is_fitted = True

    def compute_likelihood(self, n_meth, n_unmeth):
        # TODO