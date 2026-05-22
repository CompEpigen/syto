"""Test suite for :mod:`syto.modelling.classifiers.cancer_detector`.

Covers every public and private method of ``CancerDetectorClassifier`` as well
as the module-level ``_fit_beta_worker`` helper.  Each test class targets one
logical unit; edge-case branches (insufficient data, all-zero / all-one
methylation rates, 1-D likelihood input, …) are exercised explicitly.
"""

import logging
import sys
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.special import beta as beta_func  # pylint: disable=no-name-in-module

from syto.modelling.classifiers.cancer_detector import (
    CancerDetectorClassifier,
    _fit_beta_worker,
)

# Configure logging to display INFO and above messages in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def _make_train_df(
    markers=("m1", "m2"),
    classes=("A", "B"),
    n_reads_per_group=10,
    col_m="M",
    col_u="U",
    col_label="original_label",
    col_marker="dmr_ctype_label",
    rng=None,
):
    """Build a small synthetic training DataFrame.

    Each (marker, class) combination gets *n_reads_per_group* rows with random
    CpG counts.  Column names can be customised to test the classifier's
    column-name forwarding logic.
    """
    if rng is None:
        rng = np.random.RandomState(42)  # pylint: disable=no-member
    rows = []
    for marker in markers:
        for cls in classes:
            for _ in range(n_reads_per_group):
                n_cpgs = rng.randint(5, 20)
                n_meth = rng.randint(0, n_cpgs + 1)
                rows.append(
                    {
                        col_m: n_meth,
                        col_u: n_cpgs - n_meth,
                        col_label: cls,
                        col_marker: marker,
                    }
                )
    return pd.DataFrame(rows)


def _fitted_classifier(train_df=None, **fit_kwargs):
    """Return a ``(classifier, train_df)`` tuple ready for inference tests.

    If *train_df* is ``None`` a default DataFrame is generated via
    :func:`_make_train_df`.  Extra keyword arguments are forwarded to
    :meth:`CancerDetectorClassifier.fit`.
    """
    if train_df is None:
        train_df = _make_train_df()
    clf = CancerDetectorClassifier()
    clf.fit(train_df, **fit_kwargs)
    return clf, train_df


# ---------------------------------------------------------------------------
# Tests for fit_single_beta_distribution
# ---------------------------------------------------------------------------
class TestFitSingleBetaDistribution(unittest.TestCase):
    """Tests for CancerDetectorClassifier.fit_single_beta_distribution."""

    eps = 1e-2

    def test_insufficient_data_fewer_than_3_reads(self):
        """With only 2 reads the method should fall back to Beta(1,1)."""
        n_meth = np.array([3, 2])
        n_unmeth = np.array([2, 3])
        eta, rho, be0, be1, insuf = (
            CancerDetectorClassifier.fit_single_beta_distribution(
                n_meth, n_unmeth, self.eps
            )
        )
        self.assertEqual(eta, 1)
        self.assertEqual(rho, 1)
        self.assertTrue(insuf)
        self.assertFalse(be0)
        self.assertFalse(be1)

    def test_insufficient_data_zero_reads(self):
        """Empty arrays (0 reads) should also trigger the insufficient-data path."""
        eta, rho, _, _, insuf = CancerDetectorClassifier.fit_single_beta_distribution(
            np.array([]), np.array([]), self.eps
        )
        self.assertTrue(insuf)
        self.assertEqual(eta, 1)
        self.assertEqual(rho, 1)

    def test_all_meth_rates_zero(self):
        """All-zero rates trigger Bayesian estimation: eta=1, rho=1+sum(n_cpgs)."""
        n_meth = np.array([0, 0, 0, 0])
        n_unmeth = np.array([5, 6, 7, 8])
        eta, rho, be0, be1, insuf = (
            CancerDetectorClassifier.fit_single_beta_distribution(
                n_meth, n_unmeth, self.eps
            )
        )
        self.assertEqual(eta, 1)
        self.assertEqual(rho, 1 + (n_meth + n_unmeth).sum())
        self.assertTrue(be0)
        self.assertFalse(be1)
        self.assertFalse(insuf)

    def test_all_meth_rates_one(self):
        """All-one rates trigger Bayesian estimation: eta=1+sum(n_cpgs), rho=1."""
        n_meth = np.array([5, 6, 7, 8])
        n_unmeth = np.array([0, 0, 0, 0])
        eta, rho, be0, be1, insuf = (
            CancerDetectorClassifier.fit_single_beta_distribution(
                n_meth, n_unmeth, self.eps
            )
        )
        self.assertEqual(eta, 1 + (n_meth + n_unmeth).sum())
        self.assertEqual(rho, 1)
        self.assertFalse(be0)
        self.assertTrue(be1)
        self.assertFalse(insuf)

    def test_normal_mle_fitting(self):
        """Mixed rates should produce a standard MLE fit with no fallback flags."""
        rng = np.random.RandomState(0)  # pylint: disable=no-member
        n_meth = rng.randint(1, 10, size=50)
        n_unmeth = rng.randint(1, 10, size=50)
        eta, rho, be0, be1, insuf = (
            CancerDetectorClassifier.fit_single_beta_distribution(
                n_meth, n_unmeth, self.eps
            )
        )
        self.assertGreater(eta, 0)
        self.assertGreater(rho, 0)
        self.assertFalse(be0)
        self.assertFalse(be1)
        self.assertFalse(insuf)


# ---------------------------------------------------------------------------
# Tests for _fit_beta_worker
# ---------------------------------------------------------------------------
class TestFitBetaWorker(unittest.TestCase):
    """Tests for the module-level ``_fit_beta_worker`` helper."""

    def test_worker_returns_correct_tuple(self):
        """The worker should pass through marker/class indices and return a 7-element tuple."""
        n_meth = np.array([0, 0, 0, 0])
        n_unmeth = np.array([5, 6, 7, 8])
        result = _fit_beta_worker((3, 1, n_meth, n_unmeth, 1e-2))
        self.assertEqual(result[0], 3)  # marker_idx
        self.assertEqual(result[1], 1)  # class_idx
        self.assertEqual(len(result), 7)


# ---------------------------------------------------------------------------
# Tests for fit
# ---------------------------------------------------------------------------
class TestFit(unittest.TestCase):
    """Tests for :meth:`CancerDetectorClassifier.fit`."""

    def test_fit_sets_attributes(self):
        """After fitting, all matrix attributes should exist with correct shapes."""
        clf, _ = _fitted_classifier()
        self.assertTrue(clf.is_fitted)
        self.assertEqual(clf.n_markers, 2)
        self.assertEqual(clf.n_classes, 2)
        self.assertEqual(clf.param_eta_matrix.shape, (2, 2))
        self.assertEqual(clf.param_rho_matrix.shape, (2, 2))
        self.assertEqual(clf.mask_bayesian_estimation_0.shape, (2, 2))
        self.assertEqual(clf.mask_bayesian_estimation_1.shape, (2, 2))
        self.assertEqual(clf.mask_insufficient_data.shape, (2, 2))

    def test_fit_returns_self(self):
        """``fit()`` should return the classifier instance for chaining."""
        clf = CancerDetectorClassifier()
        result = clf.fit(_make_train_df())
        self.assertIs(result, clf)

    def test_uniform_priors(self):
        """Uniform priors should be 1/n_classes for every class."""
        clf, _ = _fitted_classifier(class_prior_type="uniform")
        np.testing.assert_array_almost_equal(clf.class_priors, [0.5, 0.5])

    def test_train_freq_priors(self):
        """Train-frequency priors should be proportional to class counts."""
        df = _make_train_df()
        # Double class A rows to make priors unequal
        extra = df[df["original_label"] == "A"]
        df = pd.concat([df, extra], ignore_index=True)
        clf = CancerDetectorClassifier()
        clf.fit(df, class_prior_type="train_freq")
        # A should have higher prior; use np.where since classes is now an ndarray
        idx_a = int(np.where(clf.classes == "A")[0][0])
        idx_b = int(np.where(clf.classes == "B")[0][0])
        self.assertGreater(clf.class_priors[idx_a], clf.class_priors[idx_b])
        np.testing.assert_almost_equal(clf.class_priors.sum(), 1.0)

    def test_fit_sorted_markers_and_classes(self):
        """Markers and classes should be stored in sorted order."""
        df = _make_train_df(markers=("z_marker", "a_marker"), classes=("Z", "A"))
        clf, _ = _fitted_classifier(train_df=df)
        np.testing.assert_array_equal(clf.markers, ["a_marker", "z_marker"])
        np.testing.assert_array_equal(clf.classes, ["A", "Z"])

    def test_fit_missing_column_raises(self):
        """Passing a non-existent column name should raise ``AssertionError``."""
        df = _make_train_df()
        clf = CancerDetectorClassifier()
        with self.assertRaises(AssertionError):
            clf.fit(df, col_n_meth_cpgs="MISSING")
        with self.assertRaises(AssertionError):
            clf.fit(df, col_n_unmeth_cpgs="MISSING")
        with self.assertRaises(AssertionError):
            clf.fit(df, col_label="MISSING")
        with self.assertRaises(AssertionError):
            clf.fit(df, col_marker_label="MISSING")

    def test_fit_invalid_prior_type_raises(self):
        """An unsupported ``class_prior_type`` should raise ``AssertionError``."""
        df = _make_train_df()
        clf = CancerDetectorClassifier()
        with self.assertRaises(AssertionError):
            clf.fit(df, class_prior_type="invalid")

    def test_fit_custom_column_names(self):
        """Non-default column names should be correctly forwarded."""
        df = _make_train_df(
            col_m="meth", col_u="unmeth", col_label="lbl", col_marker="mk"
        )
        clf = CancerDetectorClassifier()
        clf.fit(
            df,
            col_n_meth_cpgs="meth",
            col_n_unmeth_cpgs="unmeth",
            col_label="lbl",
            col_marker_label="mk",
        )
        self.assertTrue(clf.is_fitted)

    def test_fit_with_insufficient_data_marker_class(self):
        """A marker-class pair with < 3 reads should be flagged as insufficient."""
        df = _make_train_df(n_reads_per_group=2)
        clf, _ = _fitted_classifier(train_df=df)
        self.assertTrue(clf.mask_insufficient_data.all())


# ---------------------------------------------------------------------------
# Tests for _compute_likelihood_single_read
# ---------------------------------------------------------------------------
class TestComputeLikelihoodSingleRead(unittest.TestCase):
    """Tests for :meth:`CancerDetectorClassifier._compute_likelihood_single_read`."""

    # pylint: disable=protected-access

    def setUp(self):
        """Fit a default classifier for reuse across tests."""
        self.clf, _ = _fitted_classifier()

    def test_zero_cpgs_returns_uniform(self):
        """A read with 0 CpGs cannot inform the likelihood; expect uniform."""
        lik = self.clf._compute_likelihood_single_read(0, 0, self.clf.markers[0])
        np.testing.assert_array_almost_equal(
            lik, np.ones(self.clf.n_classes) / self.clf.n_classes
        )

    def test_positive_cpgs_returns_positive_likelihoods(self):
        """With real CpG counts the likelihood vector should be strictly positive."""
        lik = self.clf._compute_likelihood_single_read(3, 5, self.clf.markers[0])
        self.assertEqual(lik.shape, (self.clf.n_classes,))
        self.assertTrue(np.all(lik > 0))

    def test_likelihood_formula_matches_manual(self):
        """Verify the beta-binomial likelihood against a hand-computed value.

        Re-implements the formula
        ``B(1+η,ρ)^M · B(η,1+ρ)^U / B(η,ρ)^(M+U)`` outside the classifier
        and checks elementwise equality.
        """
        clf = self.clf
        marker_idx = 0
        # Grab the fitted Beta parameters for the first marker across all classes
        eta = clf.param_eta_matrix[marker_idx, :]
        rho = clf.param_rho_matrix[marker_idx, :]
        n_meth, n_unmeth = 4, 6
        expected = (
            beta_func(1 + eta, rho) ** n_meth * beta_func(eta, 1 + rho) ** n_unmeth
        ) / beta_func(eta, rho) ** (n_meth + n_unmeth)
        actual = clf._compute_likelihood_single_read(
            n_meth, n_unmeth, clf.markers[marker_idx]
        )
        np.testing.assert_array_almost_equal(actual, expected)


# ---------------------------------------------------------------------------
# Tests for compute_likelihood_bulk
# ---------------------------------------------------------------------------
class TestComputeLikelihoodBulk(unittest.TestCase):
    """Tests for :meth:`CancerDetectorClassifier.compute_likelihood_bulk`."""

    def setUp(self):
        """Fit a default classifier for reuse across tests."""
        self.clf, self.train_df = _fitted_classifier()

    def test_output_shape(self):
        """Output should be (n_reads, n_classes)."""
        n_meth = np.array([3, 0, 5])
        n_unmeth = np.array([5, 0, 3])
        markers = np.array([self.clf.markers[0]] * 3)
        lik = self.clf.compute_likelihood_bulk(n_meth, n_unmeth, markers)
        self.assertEqual(lik.shape, (3, self.clf.n_classes))

    def test_consistent_with_single_read(self):
        """Bulk computation must match calling the single-read method row by row."""
        marker = self.clf.markers[0]
        n_meth = np.array([2, 4])
        n_unmeth = np.array([6, 3])
        markers = np.array([marker, marker])
        bulk = self.clf.compute_likelihood_bulk(n_meth, n_unmeth, markers)
        for i in range(2):
            # pylint: disable=protected-access
            single = self.clf._compute_likelihood_single_read(
                n_meth[i], n_unmeth[i], markers[i]
            )
            np.testing.assert_array_almost_equal(bulk[i], single)

    def test_verbose_flag_works(self):
        """Setting ``verbose=True`` should not raise (exercises the tqdm path)."""
        n_meth = np.array([1])
        n_unmeth = np.array([1])
        markers = np.array([self.clf.markers[0]])
        self.clf.compute_likelihood_bulk(n_meth, n_unmeth, markers, verbose=True)


# ---------------------------------------------------------------------------
# Tests for _predict_proba_from_likelihoods
# ---------------------------------------------------------------------------
class TestPredictProbaFromLikelihoods(unittest.TestCase):
    """Tests for :meth:`CancerDetectorClassifier._predict_proba_from_likelihoods`."""

    # pylint: disable=protected-access

    def setUp(self):
        """Fit a default classifier for reuse across tests."""
        self.clf, _ = _fitted_classifier()

    def test_probabilities_sum_to_one(self):
        """Posterior probabilities must sum to 1 for every read."""
        lik = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        proba = self.clf._predict_proba_from_likelihoods(lik)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(3))

    def test_1d_input_is_reshaped(self):
        """A 1-D likelihood vector should be auto-reshaped to (1, n_classes)."""
        lik = np.array([0.3, 0.7])
        proba = self.clf._predict_proba_from_likelihoods(lik)
        self.assertEqual(proba.shape, (1, 2))
        np.testing.assert_almost_equal(proba.sum(), 1.0)

    def test_uniform_likelihoods_with_uniform_priors(self):
        """Equal likelihoods + uniform priors → equal posteriors."""
        lik = np.array([[1.0, 1.0]])
        proba = self.clf._predict_proba_from_likelihoods(lik)
        np.testing.assert_array_almost_equal(proba, [[0.5, 0.5]])

    def test_raises_if_not_fitted(self):
        """Calling on an unfitted classifier should raise ``ValueError``."""
        clf = CancerDetectorClassifier()
        with self.assertRaises(ValueError, msg="Class priors not set"):
            clf._predict_proba_from_likelihoods(np.array([[0.5, 0.5]]))

    def test_raises_on_wrong_shape(self):
        """A likelihood matrix with the wrong number of columns should raise."""
        lik = np.array([[0.1, 0.2, 0.3]])  # 3 columns but only 2 classes
        with self.assertRaises(ValueError):
            self.clf._predict_proba_from_likelihoods(lik)

    def test_bayes_theorem_correctness(self):
        """With non-uniform priors, posterior should shift toward the more likely class."""
        df = _make_train_df()
        # Triple the "A" rows so the train-frequency prior for A is ~3×B
        extra = df[df["original_label"] == "A"]
        df = pd.concat([df, extra, extra], ignore_index=True)
        clf = CancerDetectorClassifier()
        clf.fit(df, class_prior_type="train_freq")

        # With equal likelihoods, posterior = prior, so P(A) > P(B)
        lik = np.array([[1.0, 1.0]])
        proba = clf._predict_proba_from_likelihoods(lik)
        idx_a = int(np.where(clf.classes == "A")[0][0])
        idx_b = int(np.where(clf.classes == "B")[0][0])
        self.assertGreater(proba[0, idx_a], proba[0, idx_b])


# ---------------------------------------------------------------------------
# Tests for predict_proba (end-to-end)
# ---------------------------------------------------------------------------
class TestPredictProba(unittest.TestCase):
    """End-to-end tests for :meth:`CancerDetectorClassifier.predict_proba`."""

    def setUp(self):
        """Fit a default classifier and keep the training DataFrame for inference."""
        self.clf, self.train_df = _fitted_classifier()

    def test_output_shape(self):
        """Output array should be (n_reads, n_classes)."""
        proba = self.clf.predict_proba(self.train_df)
        self.assertEqual(proba.shape, (len(self.train_df), self.clf.n_classes))

    def test_probabilities_sum_to_one(self):
        """All posterior rows must sum to 1."""
        proba = self.clf.predict_proba(self.train_df)
        np.testing.assert_array_almost_equal(
            proba.sum(axis=1), np.ones(len(self.train_df))
        )

    def test_return_likelihoods(self):
        """``return_likelihoods=True`` should yield a (proba, lik) tuple."""
        result = self.clf.predict_proba(self.train_df, return_likelihoods=True)
        self.assertIsInstance(result, tuple)
        proba, lik = result
        self.assertEqual(proba.shape, lik.shape)

    def test_return_likelihoods_false(self):
        """``return_likelihoods=False`` should return a plain ndarray."""
        result = self.clf.predict_proba(self.train_df, return_likelihoods=False)
        self.assertIsInstance(result, np.ndarray)

    def test_custom_column_names(self):
        """Non-default column names should propagate through fit → predict."""
        df = _make_train_df(col_m="meth", col_u="unmeth", col_marker="mk")
        clf = CancerDetectorClassifier()
        clf.fit(
            df,
            col_n_meth_cpgs="meth",
            col_n_unmeth_cpgs="unmeth",
            col_marker_label="mk",
        )
        proba = clf.predict_proba(
            df,
            col_n_meth_cpgs="meth",
            col_n_unmeth_cpgs="unmeth",
            col_marker_label="mk",
        )
        self.assertEqual(proba.shape[0], len(df))


# ---------------------------------------------------------------------------
# Tests for save / load
# ---------------------------------------------------------------------------
class TestSaveLoad(unittest.TestCase):
    """Tests for :meth:`CancerDetectorClassifier.save` and
    :meth:`CancerDetectorClassifier.load`.

    Both serialisation formats (.pkl for backward compatibility and .joblib as
    the preferred format) are exercised.  ``load`` is a classmethod and always
    returns a fresh instance.
    """

    def setUp(self):
        """Fit a classifier and create a temp directory for serialised files."""
        self.clf, self.train_df = _fitted_classifier()
        self.tmp_dir = tempfile.mkdtemp(prefix="cancer_det_test_")
        self.pkl_path = os.path.join(self.tmp_dir, "model.pkl")
        self.joblib_path = os.path.join(self.tmp_dir, "model.joblib")

    def tearDown(self):
        """Remove any files created during the test, then the temp directory."""
        for path in (self.pkl_path, self.joblib_path):
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(self.tmp_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_attributes_equal(self, loaded, original):
        """Check that all serialised attributes are restored correctly."""
        self.assertTrue(loaded.is_fitted)
        self.assertEqual(loaded.n_markers, original.n_markers)
        self.assertEqual(loaded.n_classes, original.n_classes)
        self.assertEqual(loaded.eps_beta_fit, original.eps_beta_fit)
        self.assertEqual(loaded.class_prior_type, original.class_prior_type)

        np.testing.assert_array_equal(
            loaded.param_eta_matrix, original.param_eta_matrix
        )
        np.testing.assert_array_equal(
            loaded.param_rho_matrix, original.param_rho_matrix
        )
        np.testing.assert_array_equal(loaded.class_priors, original.class_priors)
        np.testing.assert_array_equal(loaded.markers, original.markers)
        np.testing.assert_array_equal(loaded.classes, original.classes)
        np.testing.assert_array_equal(
            loaded.mask_bayesian_estimation_0, original.mask_bayesian_estimation_0
        )
        np.testing.assert_array_equal(
            loaded.mask_bayesian_estimation_1, original.mask_bayesian_estimation_1
        )
        np.testing.assert_array_equal(
            loaded.mask_insufficient_data, original.mask_insufficient_data
        )
        self.assertEqual(loaded.marker_to_idx, original.marker_to_idx)
        self.assertEqual(loaded.class_to_idx, original.class_to_idx)

    def _make_train_freq_clf(self):
        """Return a ``(df, clf)`` pair fitted with ``train_freq`` priors."""
        df = _make_train_df()
        extra = df[df["original_label"] == "A"]
        df = pd.concat([df, extra], ignore_index=True)
        clf = CancerDetectorClassifier()
        clf.fit(df, class_prior_type="train_freq")
        return df, clf

    # ------------------------------------------------------------------
    # .pkl tests
    # ------------------------------------------------------------------

    def test_pkl_save_creates_file(self):
        """``save()`` with a .pkl path should create a non-empty file on disk."""
        self.clf.save(self.pkl_path)
        self.assertTrue(os.path.isfile(self.pkl_path))
        self.assertGreater(os.path.getsize(self.pkl_path), 0)

    def test_pkl_load_returns_classifier_instance(self):
        """``load()`` from .pkl should return a ``CancerDetectorClassifier``."""
        self.clf.save(self.pkl_path)
        loaded = CancerDetectorClassifier.load(self.pkl_path)
        self.assertIsInstance(loaded, CancerDetectorClassifier)

    def test_pkl_load_restores_all_attributes(self):
        """Every attribute serialised via .pkl must be restored by ``load()``."""
        self.clf.save(self.pkl_path)
        loaded = CancerDetectorClassifier.load(self.pkl_path)
        self._assert_attributes_equal(loaded, self.clf)

    def test_pkl_loaded_model_predicts_identically(self):
        """Predictions from a .pkl-loaded model must match the original exactly."""
        self.clf.save(self.pkl_path)
        loaded = CancerDetectorClassifier.load(self.pkl_path)
        np.testing.assert_array_equal(
            self.clf.predict_proba(self.train_df),
            loaded.predict_proba(self.train_df),
        )

    def test_pkl_roundtrip_with_train_freq_priors(self):
        """Save/load roundtrip via .pkl should preserve ``train_freq`` priors."""
        _, clf = self._make_train_freq_clf()
        clf.save(self.pkl_path)
        loaded = CancerDetectorClassifier.load(self.pkl_path)
        np.testing.assert_array_equal(loaded.class_priors, clf.class_priors)
        self.assertEqual(loaded.class_prior_type, "train_freq")

    # ------------------------------------------------------------------
    # .joblib tests
    # ------------------------------------------------------------------

    def test_joblib_save_creates_file(self):
        """``save()`` with a .joblib path should create a non-empty file on disk."""
        self.clf.save(self.joblib_path)
        self.assertTrue(os.path.isfile(self.joblib_path))
        self.assertGreater(os.path.getsize(self.joblib_path), 0)

    def test_joblib_load_returns_classifier_instance(self):
        """``load()`` from .joblib should return a ``CancerDetectorClassifier``."""
        self.clf.save(self.joblib_path)
        loaded = CancerDetectorClassifier.load(self.joblib_path)
        self.assertIsInstance(loaded, CancerDetectorClassifier)

    def test_joblib_load_restores_all_attributes(self):
        """Every attribute serialised via .joblib must be restored by ``load()``."""
        self.clf.save(self.joblib_path)
        loaded = CancerDetectorClassifier.load(self.joblib_path)
        self._assert_attributes_equal(loaded, self.clf)

    def test_joblib_loaded_model_predicts_identically(self):
        """Predictions from a .joblib-loaded model must match the original exactly."""
        self.clf.save(self.joblib_path)
        loaded = CancerDetectorClassifier.load(self.joblib_path)
        np.testing.assert_array_equal(
            self.clf.predict_proba(self.train_df),
            loaded.predict_proba(self.train_df),
        )

    def test_joblib_roundtrip_with_train_freq_priors(self):
        """Save/load roundtrip via .joblib should preserve ``train_freq`` priors."""
        _, clf = self._make_train_freq_clf()
        clf.save(self.joblib_path)
        loaded = CancerDetectorClassifier.load(self.joblib_path)
        np.testing.assert_array_equal(loaded.class_priors, clf.class_priors)
        self.assertEqual(loaded.class_prior_type, "train_freq")

    # ------------------------------------------------------------------
    # Cross-format consistency
    # ------------------------------------------------------------------

    def test_pkl_and_joblib_predict_identically(self):
        """Models loaded from .pkl and .joblib should produce the same predictions."""
        self.clf.save(self.pkl_path)
        self.clf.save(self.joblib_path)
        pkl_loaded = CancerDetectorClassifier.load(self.pkl_path)
        joblib_loaded = CancerDetectorClassifier.load(self.joblib_path)
        np.testing.assert_array_equal(
            pkl_loaded.predict_proba(self.train_df),
            joblib_loaded.predict_proba(self.train_df),
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_unsupported_extension_raises_on_save(self):
        """``save()`` with an unsupported extension should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.clf.save(os.path.join(self.tmp_dir, "model.txt"))

    def test_unsupported_extension_raises_on_load(self):
        """``load()`` with an unsupported extension should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            CancerDetectorClassifier.load(os.path.join(self.tmp_dir, "model.txt"))


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------
class TestInit(unittest.TestCase):
    """Tests for :meth:`CancerDetectorClassifier.__init__`."""

    def test_default_attributes(self):
        """A freshly constructed classifier should be unfitted with ``None`` attributes."""
        clf = CancerDetectorClassifier()
        self.assertFalse(clf.is_fitted)
        self.assertIsNone(clf.param_eta_matrix)
        self.assertIsNone(clf.param_rho_matrix)
        self.assertIsNone(clf.markers)
        self.assertIsNone(clf.classes)
        self.assertIsNone(clf.class_priors)
        self.assertIsNone(clf.n_classes)
        self.assertIsNone(clf.n_markers)


if __name__ == "__main__":
    unittest.main()
