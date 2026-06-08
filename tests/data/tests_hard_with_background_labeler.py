import unittest

import pandas as pd

from syto.data.labelers.hard_with_background_labeler import HardWithBackgroundLabeler


class TestComputeLabels(unittest.TestCase):
    """Tests for HardWithBackgroundLabeler.compute_labels."""

    def setUp(self):
        """Create a labeler shared across the tests."""
        self.labeler = HardWithBackgroundLabeler()
        self.num_original_classes = 3  # background class index is therefore 3

    def test_on_target_keeps_original_label(self):
        """On-target reads keep their original class label."""
        df = pd.DataFrame({"original_label": [0, 2], "dmr_ctype_label": [0, 2]})
        res = self.labeler.compute_labels(df, self.num_original_classes)
        self.assertEqual(list(res["label"]), [0, 2])

    def test_off_target_assigned_background(self):
        """Off-target reads are assigned to the background class index."""
        df = pd.DataFrame({"original_label": [0, 1], "dmr_ctype_label": [2, 2]})
        res = self.labeler.compute_labels(df, self.num_original_classes)
        # background class == num_original_classes == 3
        self.assertEqual(list(res["label"]), [3, 3])

    def test_mixed_reads(self):
        """On-target reads keep their label while off-target go to background."""
        df = pd.DataFrame({"original_label": [0, 1, 2], "dmr_ctype_label": [0, 9, 8]})
        res = self.labeler.compute_labels(df, self.num_original_classes)
        self.assertEqual(list(res["label"]), [0, 3, 3])

    def test_custom_column_names(self):
        """The configured class/GRG column names are honored."""
        df = pd.DataFrame({"ctype": [1, 2], "grg": [1, 0]})
        res = self.labeler.compute_labels(
            df,
            self.num_original_classes,
            class_label_column="ctype",
            grg_class_label_column="grg",
        )
        self.assertEqual(list(res["label"]), [1, 3])

    def test_background_class_index_scales_with_num_classes(self):
        """The background index equals num_original_classes for any value."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [5]})
        res = self.labeler.compute_labels(df, num_original_classes=10)
        self.assertEqual(res["label"].iloc[0], 10)

    def test_input_not_modified(self):
        """compute_labels does not mutate the caller's DataFrame."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        original_cols = list(df.columns)
        self.labeler.compute_labels(df, self.num_original_classes)
        self.assertEqual(list(df.columns), original_cols)

    def test_original_columns_preserved(self):
        """Original columns are kept alongside the new 'label' column."""
        df = pd.DataFrame(
            {"original_label": [0], "dmr_ctype_label": [0], "extra": ["x"]}
        )
        res = self.labeler.compute_labels(df, self.num_original_classes)
        self.assertIn("original_label", res.columns)
        self.assertIn("dmr_ctype_label", res.columns)
        self.assertIn("extra", res.columns)
        self.assertIn("label", res.columns)


if __name__ == "__main__":
    unittest.main()
