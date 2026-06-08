import unittest

import numpy as np
import pandas as pd

from syto.data.labelers.smoothed_hard_with_background import (
    SmoothedHardWithBackgroundLabeler,
)


class TestComputeLabels(unittest.TestCase):
    """Tests for SmoothedHardWithBackgroundLabeler.compute_labels."""

    def setUp(self):
        """Create a labeler and shared parameters for the tests."""
        self.labeler = SmoothedHardWithBackgroundLabeler()
        self.num_original_classes = 3  # 4 columns total (0,1,2 + background=3)
        self.epsilon = 0.1

    def test_output_vector_length(self):
        """Each smoothed label has num_original_classes + 1 entries."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        res = self.labeler.compute_labels(
            df, self.num_original_classes, epsilon=self.epsilon
        )
        self.assertEqual(
            len(res["smoothed_label"].iloc[0]), self.num_original_classes + 1
        )

    def test_smoothed_labels_sum_to_one(self):
        """Every smoothed label is a valid distribution (sums to 1)."""
        df = pd.DataFrame({"original_label": [0, 1, 2], "dmr_ctype_label": [0, 9, 2]})
        res = self.labeler.compute_labels(
            df, self.num_original_classes, epsilon=self.epsilon
        )
        sums = res["smoothed_label"].apply(np.sum)
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)

    def test_on_target_peak_on_original_class(self):
        """On-target reads peak on their original class with smoothed mass elsewhere."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        res = self.labeler.compute_labels(
            df, self.num_original_classes, epsilon=self.epsilon
        )
        vec = np.asarray(res["smoothed_label"].iloc[0])
        # peak: (1 - eps) + eps / (K+1); others: eps / (K+1)
        n = self.num_original_classes + 1
        expected_peak = (1 - self.epsilon) + self.epsilon / n
        expected_off = self.epsilon / n
        self.assertEqual(int(np.argmax(vec)), 0)
        self.assertAlmostEqual(vec[0], expected_peak)
        self.assertAlmostEqual(vec[1], expected_off)
        self.assertAlmostEqual(vec[2], expected_off)
        self.assertAlmostEqual(vec[3], expected_off)

    def test_off_target_peak_on_background(self):
        """Off-target reads peak on the background class (last index)."""
        df = pd.DataFrame({"original_label": [1], "dmr_ctype_label": [2]})
        res = self.labeler.compute_labels(
            df, self.num_original_classes, epsilon=self.epsilon
        )
        vec = np.asarray(res["smoothed_label"].iloc[0])
        self.assertEqual(int(np.argmax(vec)), self.num_original_classes)

    def test_zero_epsilon_is_one_hot(self):
        """With epsilon=0 the smoothed label collapses to a one-hot vector."""
        df = pd.DataFrame({"original_label": [2], "dmr_ctype_label": [2]})
        res = self.labeler.compute_labels(df, self.num_original_classes, epsilon=0.0)
        vec = np.asarray(res["smoothed_label"].iloc[0])
        np.testing.assert_allclose(vec, [0, 0, 1, 0], atol=1e-9)

    def test_default_epsilon(self):
        """The default epsilon (0.1) is used when not provided."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        res = self.labeler.compute_labels(df, self.num_original_classes)
        vec = np.asarray(res["smoothed_label"].iloc[0])
        n = self.num_original_classes + 1
        self.assertAlmostEqual(vec[0], (1 - 0.1) + 0.1 / n)

    def test_custom_column_names(self):
        """The configured class/GRG column names are honored."""
        df = pd.DataFrame({"ctype": [1], "grg": [0]})
        res = self.labeler.compute_labels(
            df,
            self.num_original_classes,
            epsilon=self.epsilon,
            class_label_column="ctype",
            grg_class_label_column="grg",
        )
        vec = np.asarray(res["smoothed_label"].iloc[0])
        # off-target -> peaks on background class
        self.assertEqual(int(np.argmax(vec)), self.num_original_classes)

    def test_input_not_modified(self):
        """compute_labels does not mutate the caller's DataFrame."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        original_cols = list(df.columns)
        self.labeler.compute_labels(df, self.num_original_classes)
        self.assertEqual(list(df.columns), original_cols)

    def test_original_columns_preserved(self):
        """Original columns are kept alongside the new 'smoothed_label' column."""
        df = pd.DataFrame(
            {"original_label": [0], "dmr_ctype_label": [0], "extra": ["x"]}
        )
        res = self.labeler.compute_labels(df, self.num_original_classes)
        self.assertIn("original_label", res.columns)
        self.assertIn("dmr_ctype_label", res.columns)
        self.assertIn("extra", res.columns)
        self.assertIn("smoothed_label", res.columns)


if __name__ == "__main__":
    unittest.main()
