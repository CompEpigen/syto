import unittest

import pandas as pd

from syto.data.labelers.hard_binary_labeler import HardBinaryLabeler


class TestComputeLabels(unittest.TestCase):
    """Tests for HardBinaryLabeler.compute_labels."""

    def setUp(self):
        """Create a labeler shared across the tests."""
        self.labeler = HardBinaryLabeler()

    def test_on_target_reads_labeled_one(self):
        """Reads whose class matches the GRG class get label 1."""
        df = pd.DataFrame({"original_label": [0, 2], "dmr_ctype_label": [0, 2]})
        res = self.labeler.compute_labels(df)
        self.assertEqual(list(res["label"]), [1, 1])

    def test_off_target_reads_labeled_zero(self):
        """Reads whose class differs from the GRG class get label 0."""
        df = pd.DataFrame({"original_label": [0, 1], "dmr_ctype_label": [3, 2]})
        res = self.labeler.compute_labels(df)
        self.assertEqual(list(res["label"]), [0, 0])

    def test_mixed_reads(self):
        """A mix of on-target and off-target reads is labeled correctly."""
        df = pd.DataFrame(
            {"original_label": [0, 1, 2, 3], "dmr_ctype_label": [0, 9, 2, 8]}
        )
        res = self.labeler.compute_labels(df)
        self.assertEqual(list(res["label"]), [1, 0, 1, 0])

    def test_custom_column_names(self):
        """The configured class/GRG column names are honored."""
        df = pd.DataFrame({"ctype": [5, 6], "grg": [5, 7]})
        res = self.labeler.compute_labels(
            df, class_label_column="ctype", grg_class_label_column="grg"
        )
        self.assertEqual(list(res["label"]), [1, 0])

    def test_label_column_is_int(self):
        """The produced 'label' column has integer dtype."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        res = self.labeler.compute_labels(df)
        self.assertTrue(pd.api.types.is_integer_dtype(res["label"]))

    def test_input_not_modified(self):
        """compute_labels does not mutate the caller's DataFrame."""
        df = pd.DataFrame({"original_label": [0], "dmr_ctype_label": [0]})
        original_cols = list(df.columns)
        self.labeler.compute_labels(df)
        self.assertEqual(list(df.columns), original_cols)

    def test_original_columns_preserved(self):
        """Original columns are kept alongside the new 'label' column."""
        df = pd.DataFrame(
            {"original_label": [0], "dmr_ctype_label": [0], "extra": ["x"]}
        )
        res = self.labeler.compute_labels(df)
        self.assertIn("original_label", res.columns)
        self.assertIn("dmr_ctype_label", res.columns)
        self.assertIn("extra", res.columns)
        self.assertIn("label", res.columns)


if __name__ == "__main__":
    unittest.main()
