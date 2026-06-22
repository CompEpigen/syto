import unittest
import numpy as np
import torch

from syto.classification.evaluation import (
    calculate_metric_with_sklearn,
    preprocess_logits_for_prediction,
    compute_metrics,
    compute_metrics_soft_labels,
    make_compute_metrics,
)


import unittest
import numpy as np
import sklearn.metrics


class TestCalculateMetricWithSklearn(unittest.TestCase):
    """Test suite for calculate_metric_with_sklearn."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1.0, f1=1.0, and average_precision=1.0."""
        # Using binary labels (0, 1) to satisfy average_precision_score
        labels = np.array([0, 1, 0, 0, 1, 1])
        predictions = np.array([0, 1, 0, 0, 1, 1])
        predictions_proba = np.array([0.1, 0.9, 0.2, 0.15, 0.85, 0.95])

        metrics = calculate_metric_with_sklearn(predictions_proba, predictions, labels)

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["matthews_correlation"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)

    def test_padding_exclusion(self):
        """Labels with -100 should be excluded from metric calculation."""
        labels = np.array([0, 1, -100, 1])
        predictions = np.array([0, 1, 0, 1])  # prediction at index 2 should be ignored
        # The proba at index 2 (0.9) will be ignored by the valid_mask
        predictions_proba = np.array([0.2, 0.8, 0.9, 0.85])

        metrics = calculate_metric_with_sklearn(predictions_proba, predictions, labels)

        # Only indices 0, 1, 3 are used -> all correct
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)

    def test_imperfect_predictions(self):
        """Metrics should reflect imperfect predictions."""
        labels = np.array([0, 1, 1])
        predictions = np.array([0, 0, 1])  # 1 wrong
        predictions_proba = np.array([0.2, 0.4, 0.8])

        metrics = calculate_metric_with_sklearn(predictions_proba, predictions, labels)

        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0, places=5)
        self.assertTrue("average_precision" in metrics)


class TestPreprocessLogitsForPrediction(unittest.TestCase):
    """Test suite for preprocess_logits_for_prediction."""

    def test_multiclass_returns_softmax(self):
        """Multi-class (>=2) logits should return softmax probabilities."""
        logits = torch.randn(3, 5)
        result = preprocess_logits_for_prediction(logits, None)
        expected = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.allclose(result, expected))
        # Probabilities should sum to 1
        sums = result.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(3), atol=1e-5))

    def test_3d_logits_reshaped(self):
        """3D logits (B, S, C) should be reshaped to (B*S, C)."""
        logits = torch.randn(2, 3, 5)
        result = preprocess_logits_for_prediction(logits, None)
        self.assertEqual(result.shape, (6, 5))

    def test_tuple_logits_unpacked(self):
        """Tuple input should unpack the first element."""
        logits = torch.randn(3, 5)
        extra = torch.randn(3, 5)
        result = preprocess_logits_for_prediction((logits, extra), None)
        expected = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.allclose(result, expected))


class TestComputeMetrics(unittest.TestCase):
    """Test suite for compute_metrics (backward-compat alias)."""

    def test_multiclass_hard_labels(self):
        """Standard multiclass predictions should produce valid metrics."""
        # logits where argmax matches labels
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        labels = np.array([0, 1, 2])
        metrics = compute_metrics((logits, labels))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_soft_label_input_converts_to_hard(self):
        """2D soft labels should be converted to hard labels via argmax."""
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        soft_labels = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        metrics = compute_metrics((logits, soft_labels))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_binary_two_outputs(self):
        """Binary classification with 2-class logits should use argmax."""
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])
        labels = np.array([0, 1])
        metrics = compute_metrics((logits, labels))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)


class TestComputeMetricsSoftLabels(unittest.TestCase):
    """Test suite for compute_metrics_soft_labels (backward-compat alias)."""

    def test_confident_predictions_match_labels(self):
        """When predictions and labels are both confident, they should match."""
        # Very confident logits
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        soft_labels = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        )
        metrics = compute_metrics_soft_labels((logits, soft_labels))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_low_confidence_assigned_background_class(self):
        """Predictions and labels below threshold should be assigned background class."""
        num_classes = 3
        # Flat logits → softmax will be ~uniform, max_prob ≈ 0.33 < 0.5
        logits = np.zeros((1, num_classes))
        # Flat labels too
        flat_labels = np.ones((1, num_classes)) / num_classes

        metrics = compute_metrics_soft_labels((logits, flat_labels))
        # Both prediction and label should be background → correct match
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_padded_rows_excluded(self):
        """Rows with negative sum (HuggingFace padding) should be treated as -100."""
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        # Second row is padded (sum < 0)
        soft_labels = np.array([[0.9, 0.05, 0.05], [-100.0, -100.0, -100.0]])
        metrics = compute_metrics_soft_labels((logits, soft_labels))
        # Only first sample is used → accuracy=1.0
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_hard_labels_fallback(self):
        """If 1D labels are passed, they should be used as-is."""
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        hard_labels = np.array([0, 1])
        metrics = compute_metrics_soft_labels((logits, hard_labels))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)


class TestMakeComputeMetrics(unittest.TestCase):
    """Test suite for the make_compute_metrics factory."""

    def test_factory_no_args_matches_compute_metrics(self):
        """make_compute_metrics() should behave identically to compute_metrics."""
        fn = make_compute_metrics()
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        labels = np.array([0, 1, 2])
        m1 = fn((logits, labels))
        m2 = compute_metrics((logits, labels))
        for key in m1:
            self.assertAlmostEqual(
                m1[key], m2[key], places=10, msg=f"Mismatch on {key}"
            )

    def test_factory_with_threshold_matches_soft_labels(self):
        """make_compute_metrics(threshold) should behave identically to compute_metrics_soft_labels."""
        fn = make_compute_metrics(background_threshold_func=lambda n: 0.5)
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        soft_labels = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        )
        m1 = fn((logits, soft_labels))
        m2 = compute_metrics_soft_labels((logits, soft_labels))
        for key in m1:
            self.assertAlmostEqual(
                m1[key], m2[key], places=10, msg=f"Mismatch on {key}"
            )

    def test_custom_threshold_func(self):
        """Custom threshold function should be respected."""
        logits = np.array([[10.0, 0.0, 0.0]])  # max prob ~1.0 after softmax
        soft_labels = np.array([[0.9, 0.05, 0.05]])

        # Very high threshold → label becomes background since 0.9 < 0.99
        fn = make_compute_metrics(background_threshold_func=lambda n: 0.99)
        metrics = fn((logits, soft_labels))
        # Softmax of [10,0,0] → max ~0.9998 > 0.99, still passes
        # Label max = 0.9 < 0.99, so label becomes background
        # prediction = class 0, label = background → they differ
        self.assertAlmostEqual(metrics["accuracy"], 0.0)

    def test_append_background_probability_in_ap(self):
        """With background threshold, AP should include synthesized background probability
        even when no label exceeds n_prob_classes (the old heuristic's failure case)."""
        # 3 classes, all labels are within [0, 2] — old heuristic would NOT append background
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        soft_labels = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        )

        fn = make_compute_metrics(background_threshold_func=lambda n: 0.5)
        metrics = fn((logits, soft_labels))
        # Should not raise and should produce valid AP
        self.assertFalse(np.isnan(metrics["average_precision"]))

    def test_background_threshold_with_hard_labels_fallback(self):
        """1D hard labels should work in background mode too."""
        fn = make_compute_metrics(background_threshold_func=lambda n: 0.5)
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        hard_labels = np.array([0, 1])
        metrics = fn((logits, hard_labels))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
