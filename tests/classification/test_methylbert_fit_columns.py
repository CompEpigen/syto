"""MethylBert must declare every column its data preparation reads."""

import unittest
from unittest.mock import patch


class TestMethylBertRequiredFitColumns(unittest.TestCase):
    def _classifier(self):
        from syto.classification.classifiers.methylbert import MethylBert

        # __init__ loads a foundation model; only required_fit_columns is under
        # test, so build the object without running it.
        clf = MethylBert.__new__(MethylBert)
        clf.num_grg_labels = 50
        return clf

    def test_ctype_column_is_declared_alongside_a_custom_grg_column(self):
        """prepare_methylbert_list() reads dmr_ctype_label whatever grg_label_column is.

        With grg_label_column pointing elsewhere (a block cluster, say), the
        columnar loader would otherwise project the ctype column away and the
        fit would die during data preparation.
        """
        cols = self._classifier().required_fit_columns(
            {"training": {"grg_label_column": "cluster_label"}}
        )
        self.assertIn("cluster_label", cols)
        self.assertIn("dmr_ctype_label", cols)
        for col in ("input_ids", "methylation_ids", "original_label"):
            self.assertIn(col, cols)

    def test_default_grg_column_is_not_duplicated(self):
        cols = self._classifier().required_fit_columns({"training": {}})
        self.assertEqual(cols.count("dmr_ctype_label"), 2)  # same name, deduped later
        from syto.classification.fit_data import resolve_fit_columns

        resolved = resolve_fit_columns(cols, cols + ["split"])
        self.assertEqual(resolved.count("dmr_ctype_label"), 1)


if __name__ == "__main__":
    unittest.main()
