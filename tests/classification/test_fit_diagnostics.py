import unittest

import numpy as np

from syto.classification.fit_diagnostics import (
    data_list_column,
    on_target_scores,
)


class TestDataListColumn(unittest.TestCase):
    def _table(self):
        # Mirrors the prepare_methylbert_list layout: header row, then data rows.
        return [
            ["dna_seq", "grg_ctype", "on_target_mask"],
            ["AAA CCC", 2, True],
            ["GGG TTT", 0, False],
            ["CCC AAA", 1, True],
        ]

    def test_extracts_named_column_in_row_order(self):
        got = data_list_column(self._table(), "grg_ctype")
        np.testing.assert_array_equal(got, np.array([2, 0, 1]))

    def test_extracts_boolean_column(self):
        got = data_list_column(self._table(), "on_target_mask")
        np.testing.assert_array_equal(got, np.array([True, False, True]))

    def test_raises_on_unknown_column(self):
        with self.assertRaises(ValueError) as ctx:
            data_list_column(self._table(), "nope")
        self.assertIn("nope", str(ctx.exception))

    def test_empty_table_yields_empty_array(self):
        got = data_list_column([["dna_seq", "grg_ctype"]], "grg_ctype")
        self.assertEqual(len(got), 0)


class TestOnTargetScores(unittest.TestCase):
    def test_picks_dmr_label_probability_for_on_target_rows_only(self):
        # 4 chunks x 3 classes.
        predictions = np.array(
            [
                [0.10, 0.70, 0.20],  # on target, dmr label 1 -> 0.70
                [0.60, 0.30, 0.10],  # off target -> excluded
                [0.05, 0.15, 0.80],  # on target, dmr label 2 -> 0.80
                [0.90, 0.05, 0.05],  # on target, dmr label 0 -> 0.90
            ]
        )
        dmr_labels = np.array([1, 0, 2, 0])
        on_target_mask = np.array([True, False, True, True])

        got = on_target_scores(predictions, dmr_labels, on_target_mask)

        np.testing.assert_allclose(got, np.array([0.70, 0.80, 0.90]))

    def test_returns_empty_when_nothing_is_on_target(self):
        predictions = np.array([[0.4, 0.6], [0.7, 0.3]])
        got = on_target_scores(
            predictions, np.array([0, 1]), np.array([False, False])
        )
        self.assertEqual(len(got), 0)

    def test_accepts_integer_mask(self):
        # prepare_methylbert_list may hand back 0/1 rather than bool.
        predictions = np.array([[0.4, 0.6], [0.7, 0.3]])
        got = on_target_scores(predictions, np.array([1, 0]), np.array([0, 1]))
        np.testing.assert_allclose(got, np.array([0.7]))
