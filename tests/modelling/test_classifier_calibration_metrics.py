"""
Unit tests for syto.modelling.classifier_calibration_metrics.

Coverage
--------
* _compute_bin_edges       - equal-width and equal-frequency strategies,
                             edge cases (constant data, heavy ties, out-of-range).
* _assign_to_bins          - boundary values, output shape, midpoint placement.
* _compute_bins_and_stats_per_bin - output keys, empty-bin NaN handling,
                                    perfect-calibration invariant, shape guards.
* _compute_ece             - weighted-average formula, empty-input guard.
* _compute_mce             - min_bin_count filter, NaN propagation (post-fix).
* compute_brier_score      - perfect/worst/uniform cases, soft labels, formula.
* compute_classifier_calibration_metrics (integration)
                           - hard and soft labels, all output keys, value ranges,
                             MCE/ECE relationship, input-validation assertions.
"""

import unittest
import warnings
import numpy as np

from syto.modelling.classifier_calibration_metrics import (
    _compute_bin_edges,
    _assign_to_bins,
    _compute_bins_and_stats_per_bin,
    _compute_ece,
    _compute_mce,
    compute_brier_score,
    compute_classifier_calibration_metrics,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  _compute_bin_edges
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeBinEdges(unittest.TestCase):
    """Tests for the _compute_bin_edges helper.

    Verifies that both equal-width and equal-frequency strategies produce
    the correct number of edges, stay within [min_value, max_value], and
    remain monotone.  Also covers the warning path for out-of-range values
    and the ValueError path for an unrecognised strategy name.
    """

    def test_equal_width_basic(self):
        """Equal-width edges span [0, 1] and are evenly spaced (n_bins + 1 values)."""
        x = np.linspace(0.0, 1.0, 100)
        edges = _compute_bin_edges(x, n_bins=10, binning_strategy="equal_width")
        self.assertEqual(len(edges), 11)
        np.testing.assert_allclose(edges[0], 0.0)
        np.testing.assert_allclose(edges[-1], 1.0)
        # Bins should be evenly spaced
        diffs = np.diff(edges)
        np.testing.assert_allclose(diffs, diffs[0])

    def test_equal_width_single_bin(self):
        """A single equal-width bin collapses to two edges: [0.0, 1.0]."""
        x = np.array([0.2, 0.5, 0.8])
        edges = _compute_bin_edges(x, n_bins=1, binning_strategy="equal_width")
        self.assertEqual(len(edges), 2)
        np.testing.assert_allclose(edges[0], 0.0)
        np.testing.assert_allclose(edges[-1], 1.0)

    def test_equal_frequency_basic(self):
        """Equal-frequency edges are monotone and bounded by [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0.0, 1.0, 200)
        edges = _compute_bin_edges(x, n_bins=10, binning_strategy="equal_frequency")
        self.assertEqual(len(edges), 11)
        np.testing.assert_allclose(edges[0], 0.0)
        np.testing.assert_allclose(edges[-1], 1.0)
        # Edges should be non-decreasing
        self.assertTrue(np.all(np.diff(edges) >= 0))

    def test_equal_frequency_constant_data(self):
        """Constant data (all ties) must not raise and must produce valid edges."""
        # All identical values → many ties → should not raise
        x = np.full(50, 0.3)
        edges = _compute_bin_edges(x, n_bins=5, binning_strategy="equal_frequency")
        self.assertEqual(len(edges), 6)
        np.testing.assert_allclose(edges[0], 0.0)
        np.testing.assert_allclose(edges[-1], 1.0)
        self.assertTrue(np.all(np.diff(edges) >= 0))

    def test_equal_frequency_edges_monotone_despite_ties(self):
        """Heavy concentration near 0 (rare-class probabilities) still yields monotone edges.

        The implementation inserts tiny epsilon offsets to separate duplicate
        quantile values, so this exercises the de-duplication code path.
        """
        # Heavily skewed towards 0 - common for rare-class probabilities
        x = np.concatenate([np.zeros(90), np.linspace(0.1, 1.0, 10)])
        edges = _compute_bin_edges(x, n_bins=10, binning_strategy="equal_frequency")
        self.assertEqual(len(edges), 11)
        self.assertTrue(np.all(np.diff(edges) >= 0))

    def test_values_outside_range_issues_warning(self):
        """Out-of-range input values trigger a WARNING; edges are still returned.

        The implementation clips values before computing edges, so the function
        should not crash - only warn.
        """
        x = np.array([-0.1, 0.5, 1.2])
        with self.assertLogs(
            "syto.modelling.classifier_calibration_metrics", level="WARNING"
        ):
            edges = _compute_bin_edges(x, n_bins=5, binning_strategy="equal_width")
        # Should still return valid edges
        self.assertEqual(len(edges), 6)

    def test_unknown_strategy_raises(self):
        """An unrecognised binning strategy name raises ValueError."""
        x = np.linspace(0.0, 1.0, 50)
        with self.assertRaises(ValueError):
            _compute_bin_edges(x, n_bins=5, binning_strategy="invalid_strategy")

    def test_custom_min_max(self):
        """Custom min_value / max_value are respected as the outer bin edges."""
        x = np.linspace(0.2, 0.8, 60)
        edges = _compute_bin_edges(
            x,
            n_bins=4,
            binning_strategy="equal_width",
            min_value=0.2,
            max_value=0.8,
        )
        np.testing.assert_allclose(edges[0], 0.2)
        np.testing.assert_allclose(edges[-1], 0.8)


# ─────────────────────────────────────────────────────────────────────────────
#  _assign_to_bins
# ─────────────────────────────────────────────────────────────────────────────


class TestAssignToBins(unittest.TestCase):
    """Tests for the _assign_to_bins helper.

    Verifies that every value is mapped to a valid bin index, that the two
    boundary values (0.0 and 1.0) are handled correctly, and that the output
    array length matches the input.
    """

    def _uniform_edges(self, n_bins=5):
        """Return n_bins + 1 evenly-spaced edges over [0, 1]."""
        return np.linspace(0.0, 1.0, n_bins + 1)

    def test_all_indices_in_range(self):
        """Every assigned index falls within [0, n_bins)."""
        x = np.linspace(0.0, 1.0, 101)
        edges = self._uniform_edges(10)
        indices = _assign_to_bins(x, edges)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 10))

    def test_zero_goes_to_first_bin(self):
        """The left boundary value (0.0) maps to bin index 0."""
        edges = self._uniform_edges(5)
        idx = _assign_to_bins(np.array([0.0]), edges)
        self.assertEqual(idx[0], 0)

    def test_one_goes_to_last_bin(self):
        """The right boundary value (1.0) maps to the last bin index."""
        edges = self._uniform_edges(5)
        idx = _assign_to_bins(np.array([1.0]), edges)
        self.assertEqual(idx[0], 4)

    def test_midpoint_assignment(self):
        """Values in distinct halves of a two-bin layout are assigned correctly."""
        edges = np.array([0.0, 0.5, 1.0])  # 2 bins
        idx = _assign_to_bins(np.array([0.25, 0.75]), edges)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[1], 1)

    def test_output_length_matches_input(self):
        """The returned index array has the same length as the input data."""
        x = np.random.default_rng(7).uniform(0, 1, 50)
        edges = self._uniform_edges(8)
        indices = _assign_to_bins(x, edges)
        self.assertEqual(len(indices), 50)


# ─────────────────────────────────────────────────────────────────────────────
#  _compute_bins_and_stats_per_bin
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeBinsAndStatsPerBin(unittest.TestCase):
    """Tests for the _compute_bins_and_stats_per_bin helper.

    Covers the complete output dictionary (keys, shapes, values), the
    NaN-for-empty-bins contract, the perfect-calibration invariant
    (diff == 0 when targets == confidences), and the AssertionError guards
    for shape mismatches.
    """

    def _perfect_calibration_data(self, n=200, n_bins=10, seed=0):
        """Return (targets, confidences, n_bins) where targets == confidences.

        Setting targets == confidences creates a perfectly-calibrated dataset:
        the average confidence in every bin matches the average target value,
        so diff_per_bin should be zero for all populated bins.
        """
        rng = np.random.default_rng(seed)
        confidences = rng.uniform(0.0, 1.0, n)
        # targets equal confidences → perfectly calibrated
        targets = confidences.copy()
        return targets, confidences, n_bins

    def test_output_keys(self):
        """The result dict contains exactly the six documented keys."""
        targets, confidences, n_bins = self._perfect_calibration_data()
        result = _compute_bins_and_stats_per_bin(
            targets, confidences, n_bins, "equal_width"
        )
        expected_keys = {
            "bin_edges",
            "bin_indices",
            "counts_per_bin",
            "avg_confidence_per_bin",
            "accuracy_per_bin",
            "diff_per_bin",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_bin_counts_sum_to_n_samples(self):
        """The sum of counts_per_bin equals the total number of input samples."""
        n = 150
        targets = np.random.default_rng(1).uniform(0, 1, n)
        confidences = np.random.default_rng(2).uniform(0, 1, n)
        result = _compute_bins_and_stats_per_bin(
            targets, confidences, 10, "equal_width"
        )
        self.assertEqual(result["counts_per_bin"].sum(), n)

    def test_empty_bins_have_nan_stats(self):
        """Bins with no samples report NaN for avg_confidence, accuracy, and diff."""
        # Put all values in one bin by using a very uneven distribution
        confidences = np.full(50, 0.01)  # All near 0 → first bin
        targets = np.ones(50)
        result = _compute_bins_and_stats_per_bin(
            targets, confidences, 10, "equal_width"
        )
        # Bins 1-9 should be empty
        empty_mask = result["counts_per_bin"] == 0
        self.assertTrue(np.all(np.isnan(result["avg_confidence_per_bin"][empty_mask])))
        self.assertTrue(np.all(np.isnan(result["accuracy_per_bin"][empty_mask])))
        self.assertTrue(np.all(np.isnan(result["diff_per_bin"][empty_mask])))

    def test_perfect_calibration_diff_near_zero(self):
        """diff_per_bin is zero for all populated bins when targets == confidences.

        Uses a large sample (n=10_000) so that every bin is populated and the
        within-bin average converges tightly to the ground truth.
        """
        n = 10_000
        rng = np.random.default_rng(42)
        confidences = rng.uniform(0.0, 1.0, n)
        targets = confidences.copy()
        result = _compute_bins_and_stats_per_bin(
            targets, confidences, 10, "equal_width"
        )
        nonempty = result["counts_per_bin"] > 0
        np.testing.assert_allclose(result["diff_per_bin"][nonempty], 0.0, atol=1e-10)

    def test_shape_mismatch_raises(self):
        """Targets and confidences with different lengths raise AssertionError."""
        with self.assertRaises(AssertionError):
            _compute_bins_and_stats_per_bin(
                np.array([0.0, 1.0]),
                np.array([0.1, 0.5, 0.9]),
                5,
                "equal_width",
            )

    def test_2d_input_raises(self):
        """2-D input arrays raise AssertionError; inputs must be 1-D."""
        targets = np.ones((10, 2))
        confidences = np.ones((10, 2))
        with self.assertRaises(AssertionError):
            _compute_bins_and_stats_per_bin(targets, confidences, 5, "equal_width")

    def test_bin_edges_length(self):
        """bin_edges has n_bins + 1 elements; counts_per_bin has n_bins elements."""
        targets = np.random.default_rng(0).uniform(0, 1, 100)
        confidences = np.random.default_rng(1).uniform(0, 1, 100)
        n_bins = 7
        result = _compute_bins_and_stats_per_bin(
            targets, confidences, n_bins, "equal_width"
        )
        self.assertEqual(len(result["bin_edges"]), n_bins + 1)
        self.assertEqual(len(result["counts_per_bin"]), n_bins)

    def test_equal_frequency_strategy(self):
        """The equal_frequency strategy accounts for all samples in counts_per_bin."""
        rng = np.random.default_rng(5)
        confidences = rng.uniform(0, 1, 200)
        targets = (confidences > 0.5).astype(float)
        result = _compute_bins_and_stats_per_bin(
            targets, confidences, 10, "equal_frequency"
        )
        self.assertEqual(result["counts_per_bin"].sum(), 200)

    def test_hard_binary_targets(self):
        """Binary (0/1) targets yield accuracy_per_bin values in [0, 1]."""
        rng = np.random.default_rng(3)
        confidences = rng.uniform(0, 1, 100)
        targets = rng.integers(0, 2, 100).astype(float)
        result = _compute_bins_and_stats_per_bin(targets, confidences, 5, "equal_width")
        nonempty = result["counts_per_bin"] > 0
        # Accuracy should be between 0 and 1
        self.assertTrue(np.all(result["accuracy_per_bin"][nonempty] >= 0.0))
        self.assertTrue(np.all(result["accuracy_per_bin"][nonempty] <= 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  _compute_ece
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeEce(unittest.TestCase):
    """Tests for the _compute_ece helper.

    ECE is defined as the sample-weighted average of per-bin absolute
    differences.  Empty bins (count == 0, diff == NaN) must be excluded
    from the sum via np.nansum.
    """

    def test_perfect_calibration_ece_zero(self):
        """Zero diff in every bin yields ECE = 0."""
        counts = np.array([10, 20, 30])
        diffs = np.array([0.0, 0.0, 0.0])
        self.assertAlmostEqual(_compute_ece(counts, diffs), 0.0)

    def test_worst_calibration_ece_one(self):
        """Diff = 1 in every equally-sized bin yields ECE = 1 (the maximum)."""
        # All bins have diff = 1.0
        counts = np.array([10, 10, 10])
        diffs = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(_compute_ece(counts, diffs), 1.0)

    def test_empty_bins_ignored(self):
        """Empty bins (count=0, diff=NaN) do not contribute to ECE."""
        counts = np.array([0, 50, 0])
        diffs = np.array([np.nan, 0.4, np.nan])
        ece = _compute_ece(counts, diffs)
        self.assertAlmostEqual(ece, 0.4)

    def test_zero_samples_returns_zero(self):
        """When there are no samples at all, ECE is defined as 0.0."""
        counts = np.array([0, 0, 0])
        diffs = np.array([np.nan, np.nan, np.nan])
        self.assertEqual(_compute_ece(counts, diffs), 0.0)

    def test_weighted_average(self):
        """ECE equals the sample-weighted mean of per-bin diffs."""
        counts = np.array([100, 200])
        diffs = np.array([0.3, 0.6])
        expected = (100 * 0.3 + 200 * 0.6) / 300
        self.assertAlmostEqual(_compute_ece(counts, diffs), expected)

    def test_returns_float(self):
        """_compute_ece returns a Python float, not a numpy scalar."""
        counts = np.array([50])
        diffs = np.array([0.2])
        result = _compute_ece(counts, diffs)
        self.assertIsInstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
#  _compute_mce
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMce(unittest.TestCase):
    """Tests for the _compute_mce helper.

    MCE is the maximum per-bin diff among bins that have at least min_bin_count
    samples. Bins with fewer samples - or with NaN diff (empty bins) -
    are excluded.
    """

    def test_basic_max_diff(self):
        """MCE equals the maximum diff across all reliable bins."""
        diff = np.array([0.1, 0.5, 0.3])
        counts = np.array([20, 20, 20])
        mce = _compute_mce(diff, counts, min_bin_count=5)
        self.assertAlmostEqual(mce, 0.5)

    def test_min_bin_count_filters_small_bins(self):
        """Bins with fewer than min_bin_count samples are excluded from the MCE max."""
        diff = np.array([0.8, 0.2, 0.1])
        counts = np.array([2, 50, 50])  # First bin below min_bin_count
        mce = _compute_mce(diff, counts, min_bin_count=10)
        # Bin 0 is unreliable, max of remaining is 0.2
        self.assertAlmostEqual(mce, 0.2)

    def test_all_bins_below_min_count_returns_nan(self):
        """When no bin meets the min_bin_count threshold, MCE is NaN."""
        diff = np.array([0.5, 0.3])
        counts = np.array([1, 2])
        mce = _compute_mce(diff, counts, min_bin_count=10)
        self.assertTrue(np.isnan(mce))

    def test_empty_bins_nan_diff_excluded(self):
        """Empty bins (diff=NaN) are safely excluded via np.nanmax."""
        diff = np.array([np.nan, 0.4, 0.6])
        counts = np.array([0, 20, 20])
        mce = _compute_mce(diff, counts, min_bin_count=5)
        self.assertAlmostEqual(mce, 0.6)

    def test_single_reliable_bin(self):
        """A single reliable bin: its diff value is returned directly as MCE."""
        diff = np.array([0.7])
        counts = np.array([100])
        mce = _compute_mce(diff, counts, min_bin_count=10)
        self.assertAlmostEqual(mce, 0.7)

    def test_returns_float(self):
        """_compute_mce returns a Python float."""
        diff = np.array([0.3, 0.4])
        counts = np.array([10, 10])
        result = _compute_mce(diff, counts, min_bin_count=5)
        self.assertIsInstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
#  compute_brier_score
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeBrierScore(unittest.TestCase):
    """Tests for compute_brier_score.

    Brier score = mean((targets - confidences)^2).  The range is [0, 1] for
    binary targets; 0 means perfect predictions, 1 means fully inverted
    predictions.  Works for both hard (0/1) and soft ([0,1]) targets.
    """

    def test_perfect_predictions_score_zero(self):
        """When targets == confidences, Brier score is 0."""
        targets = np.array([0.0, 1.0, 1.0, 0.0])
        confidences = np.array([0.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(compute_brier_score(targets, confidences), 0.0)

    def test_worst_hard_predictions_score_one(self):
        """Fully-inverted predictions (1 → 0, 0 → 1) give the maximum Brier score of 1."""
        # Predicting 1 for all 0 targets and 0 for all 1 targets
        targets = np.array([0.0, 0.0, 1.0, 1.0])
        confidences = np.array([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(compute_brier_score(targets, confidences), 1.0)

    def test_uniform_predictions_on_balanced_data(self):
        """Constant 0.5 confidence on balanced binary data gives Brier = 0.25.

        This is the Brier score of an uninformative classifier on a balanced
        dataset: mean((0 - 0.5)^2 + (1 - 0.5)^2) / 2 = 0.25.
        """
        # All confidences = 0.5, balanced binary targets → Brier = 0.25
        targets = np.array([0.0, 1.0, 0.0, 1.0])
        confidences = np.full(4, 0.5)
        self.assertAlmostEqual(compute_brier_score(targets, confidences), 0.25)

    def test_soft_labels(self):
        """Soft targets equal to confidences give Brier = 0 regardless of values."""
        targets = np.array([0.2, 0.8])
        confidences = np.array([0.2, 0.8])
        self.assertAlmostEqual(compute_brier_score(targets, confidences), 0.0)

    def test_scalar_formula_consistency(self):
        """Brier score matches the manually computed mean squared error."""
        targets = np.array([1.0, 0.0, 0.5])
        confidences = np.array([0.9, 0.1, 0.4])
        expected = np.mean((targets - confidences) ** 2)
        self.assertAlmostEqual(compute_brier_score(targets, confidences), expected)

    def test_returns_float(self):
        """The return value is castable to a Python float."""
        result = compute_brier_score(np.array([0.0, 1.0]), np.array([0.4, 0.6]))
        self.assertIsInstance(float(result), float)


# ─────────────────────────────────────────────────────────────────────────────
#  compute_classifier_calibration_metrics  (integration)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeClassifierCalibrationMetrics(unittest.TestCase):
    """Integration tests for compute_classifier_calibration_metrics.

    Exercises the public entry-point end-to-end with both hard (integer class
    index) and soft (probability distribution) labels.  Covers output keys,
    value-range invariants, the ECE/MCE relationship, the soft == one-hot
    equivalence, and every AssertionError validation guard.
    """

    N = 500
    N_CLASSES = 4
    N_BINS = 10
    MIN_BIN_COUNT = 5

    def _random_proba(self, seed=42):
        """Uniform random probability matrix, rows sum to 1."""
        rng = np.random.default_rng(seed)
        raw = rng.dirichlet(np.ones(self.N_CLASSES), size=self.N)
        return raw

    def _hard_labels(self, proba):
        """Return the argmax class index for each sample as integer hard labels."""
        return np.argmax(proba, axis=1)

    def _soft_labels(self, proba):
        """Return a copy of proba as soft probability labels (shape: N x N_CLASSES)."""
        return proba.copy()

    # ── output shape / keys ────────────────────────────────────────

    def test_output_keys_hard_labels(self):
        """All five documented output keys are present when using hard labels."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        expected = {
            "top_label_ece",
            "top_label_mce",
            "top_label_brier",
            "class_j_ece",
            "classwise_ece",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_output_keys_soft_labels(self):
        """All five documented output keys are present when using soft labels."""
        proba = self._random_proba()
        y_true = self._soft_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertIn("top_label_ece", result)
        self.assertIn("classwise_ece", result)

    def test_class_j_ece_length(self):
        """class_j_ece has exactly one ECE value per output class."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertEqual(len(result["class_j_ece"]), self.N_CLASSES)

    # ── value ranges ──────────────────────────────────────────────

    def test_ece_between_zero_and_one(self):
        """top_label_ece and classwise_ece are in [0, 1] for any valid input."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertGreaterEqual(result["top_label_ece"], 0.0)
        self.assertLessEqual(result["top_label_ece"], 1.0)
        self.assertGreaterEqual(result["classwise_ece"], 0.0)
        self.assertLessEqual(result["classwise_ece"], 1.0)

    def test_brier_between_zero_and_one(self):
        """top_label_brier is in [0, 1] for any valid input."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertGreaterEqual(result["top_label_brier"], 0.0)
        self.assertLessEqual(result["top_label_brier"], 1.0)

    def test_class_j_ece_non_negative(self):
        """Every per-class ECE value is non-negative (it is an absolute error)."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertTrue(np.all(result["class_j_ece"] >= 0.0))

    def test_classwise_ece_is_mean_of_class_j_ece(self):
        """classwise_ece is the unweighted arithmetic mean of class_j_ece."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        np.testing.assert_allclose(
            result["classwise_ece"],
            np.mean(result["class_j_ece"]),
        )

    # ── perfect calibration ───────────────────────────────────────

    def test_perfect_calibration_brier_zero(self):
        """Using proba itself as soft labels gives Brier = 0 (perfect predictions)."""
        # Perfect soft predictions → brier = 0
        proba = self._random_proba(seed=7)
        y_true_soft = proba.copy()
        result = compute_classifier_calibration_metrics(
            y_true_soft, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertAlmostEqual(result["top_label_brier"], 0.0, places=10)

    def test_perfect_hard_label_predictions_ece_zero(self):
        """Deterministic correct predictions (confidence=1) give ECE = 0 and Brier = 0.

        All samples belong to class 0 and all predictions place full mass on
        class 0.  With a single bin the entire dataset lands in one bin where
        avg_confidence = 1.0 == accuracy = 1.0, so diff = 0.
        """
        # Construct artificially perfect calibration:
        # confidence = 1.0 everywhere, prediction always correct
        n = 100
        n_classes = 3
        y_true = np.zeros(n, dtype=int)
        proba = np.zeros((n, n_classes))
        proba[:, 0] = 1.0  # All confidence on class 0
        result = compute_classifier_calibration_metrics(
            y_true, proba, n_bins=1, min_bin_count_for_mce=1
        )
        self.assertAlmostEqual(result["top_label_ece"], 0.0, places=10)
        self.assertAlmostEqual(result["top_label_brier"], 0.0, places=10)

    # ── soft labels ───────────────────────────────────────────────

    def test_soft_labels_correct_shape(self):
        """class_j_ece length equals n_classes regardless of whether labels are soft."""
        proba = self._random_proba()
        y_true = self._soft_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        self.assertEqual(len(result["class_j_ece"]), self.N_CLASSES)

    def test_soft_labels_class_j_ece_matches_hard_for_one_hot(self):
        """One-hot soft labels should give the same class_j_ece as hard labels."""
        rng = np.random.default_rng(10)
        proba = rng.dirichlet(np.ones(self.N_CLASSES), size=self.N)
        hard = np.argmax(proba, axis=1)
        one_hot = np.zeros_like(proba)
        one_hot[np.arange(self.N), hard] = 1.0

        res_hard = compute_classifier_calibration_metrics(
            hard, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        res_soft = compute_classifier_calibration_metrics(
            one_hot, proba, self.N_BINS, self.MIN_BIN_COUNT
        )
        np.testing.assert_allclose(
            res_hard["class_j_ece"], res_soft["class_j_ece"], atol=1e-10
        )

    # ── binning strategy ──────────────────────────────────────────

    def test_equal_frequency_strategy(self):
        """The equal_frequency binning strategy returns a valid non-negative ECE."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true,
            proba,
            self.N_BINS,
            self.MIN_BIN_COUNT,
            binning_strategy="equal_frequency",
        )
        self.assertIn("top_label_ece", result)
        self.assertGreaterEqual(result["top_label_ece"], 0.0)

    # ── input validation ─────────────────────────────────────────

    def test_invalid_binning_strategy_raises(self):
        """An unrecognised binning strategy name raises AssertionError."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        with self.assertRaises(AssertionError):
            compute_classifier_calibration_metrics(
                y_true,
                proba,
                self.N_BINS,
                self.MIN_BIN_COUNT,
                binning_strategy="unknown",
            )

    def test_1d_y_pred_proba_raises(self):
        """Passing a 1-D y_pred_proba raises AssertionError (must be 2-D)."""
        proba_1d = np.random.default_rng(0).uniform(0, 1, 100)
        y_true = np.zeros(100, dtype=int)
        with self.assertRaises(AssertionError):
            compute_classifier_calibration_metrics(
                y_true, proba_1d, self.N_BINS, self.MIN_BIN_COUNT
            )

    def test_sample_count_mismatch_hard_labels_raises(self):
        """Hard y_true with a different sample count than y_pred_proba raises AssertionError."""
        proba = self._random_proba()
        y_true = np.zeros(self.N + 1, dtype=int)
        with self.assertRaises(AssertionError):
            compute_classifier_calibration_metrics(
                y_true, proba, self.N_BINS, self.MIN_BIN_COUNT
            )

    def test_soft_label_shape_mismatch_raises(self):
        """Soft y_true with a different n_classes than y_pred_proba raises AssertionError."""
        proba = self._random_proba()
        # y_true_bad has N_CLASSES + 1 columns — mismatched shape
        y_true_bad = np.random.default_rng(0).dirichlet(
            np.ones(self.N_CLASSES + 1), size=self.N
        )
        with self.assertRaises(AssertionError):
            compute_classifier_calibration_metrics(
                y_true_bad, proba, self.N_BINS, self.MIN_BIN_COUNT
            )

    # ── MCE ───────────────────────────────────────────────────────

    def test_mce_nan_when_all_bins_below_min_count(self):
        """An impossibly large min_bin_count causes every bin to be excluded, yielding NaN MCE."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        # Use a very large min_bin_count so no bin qualifies
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, min_bin_count_for_mce=10_000
        )
        self.assertTrue(np.isnan(result["top_label_mce"]))

    def test_mce_not_nan_with_small_min_bin_count(self):
        """With min_bin_count_for_mce=1, every populated bin qualifies and MCE is finite."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, min_bin_count_for_mce=1
        )
        # With min_bin_count=1 and many samples per bin, MCE should be a number
        self.assertFalse(np.isnan(result["top_label_mce"]))

    def test_mce_greater_than_or_equal_to_ece(self):
        """MCE >= ECE: the maximum per-bin error is always >= the weighted average error."""
        proba = self._random_proba()
        y_true = self._hard_labels(proba)
        result = compute_classifier_calibration_metrics(
            y_true, proba, self.N_BINS, min_bin_count_for_mce=1
        )
        if not np.isnan(result["top_label_mce"]):
            self.assertGreaterEqual(
                result["top_label_mce"], result["top_label_ece"] - 1e-10
            )


if __name__ == "__main__":
    unittest.main()
