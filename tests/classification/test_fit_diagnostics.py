import tempfile
import unittest
from pathlib import Path

import numpy as np

from syto.classification.fit_diagnostics import (
    OnTargetScoreRecorder,
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


class _FakeDataset:
    """Stand-in for MethylBertFinetuneDataset: identity is all that matters."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class TestOnTargetScoreRecorder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.plot_dir = Path(self._tmp.name) / "on_target_score_plots"
        self.recorder = OnTargetScoreRecorder(self.plot_dir)
        self.dataset = _FakeDataset(4)
        self.predictions = np.array(
            [
                [0.10, 0.70, 0.20],
                [0.60, 0.30, 0.10],
                [0.05, 0.15, 0.80],
                [0.90, 0.05, 0.05],
            ]
        )
        self.recorder.register(
            self.dataset,
            dmr_labels=np.array([1, 0, 2, 0]),
            on_target_mask=np.array([True, False, True, True]),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_plot_dir_on_construction(self):
        self.assertTrue(self.plot_dir.is_dir())

    def test_writes_png_named_by_prefix_and_step(self):
        path = self.recorder.on_eval(
            self.dataset, self.predictions, step=200, prefix="eval"
        )
        self.assertEqual(path, self.plot_dir / "eval_step_0000200.png")
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_train_prefix_writes_a_separate_file(self):
        self.recorder.on_eval(self.dataset, self.predictions, step=7, prefix="eval")
        path = self.recorder.on_eval(
            self.dataset, self.predictions, step=7, prefix="train"
        )
        self.assertEqual(path, self.plot_dir / "train_step_0000007.png")
        self.assertTrue((self.plot_dir / "eval_step_0000007.png").exists())

    def test_unwraps_tuple_predictions(self):
        path = self.recorder.on_eval(
            self.dataset, (self.predictions, None), step=1, prefix="eval"
        )
        self.assertIsNotNone(path)

    def test_no_op_for_unregistered_dataset(self):
        other = _FakeDataset(4)
        with self.assertLogs("syto.classification.fit_diagnostics", "WARNING"):
            result = self.recorder.on_eval(
                other, self.predictions, step=1, prefix="eval"
            )
        self.assertIsNone(result)
        self.assertEqual(list(self.plot_dir.iterdir()), [])

    def test_no_op_when_predictions_are_none(self):
        with self.assertLogs("syto.classification.fit_diagnostics", "WARNING"):
            result = self.recorder.on_eval(self.dataset, None, step=1, prefix="eval")
        self.assertIsNone(result)
        self.assertEqual(list(self.plot_dir.iterdir()), [])

    def test_no_op_on_length_mismatch(self):
        with self.assertLogs("syto.classification.fit_diagnostics", "WARNING"):
            result = self.recorder.on_eval(
                self.dataset, self.predictions[:2], step=1, prefix="eval"
            )
        self.assertIsNone(result)
        self.assertEqual(list(self.plot_dir.iterdir()), [])

    def test_no_op_when_no_on_target_rows(self):
        dataset = _FakeDataset(2)
        self.recorder.register(
            dataset,
            dmr_labels=np.array([0, 1]),
            on_target_mask=np.array([False, False]),
        )
        with self.assertLogs("syto.classification.fit_diagnostics", "WARNING"):
            result = self.recorder.on_eval(
                dataset, np.array([[0.4, 0.6], [0.7, 0.3]]), step=1, prefix="eval"
            )
        self.assertIsNone(result)
        self.assertEqual(list(self.plot_dir.iterdir()), [])

    def test_same_dataset_object_registered_once_serves_both_prefixes(self):
        # fit_classificaton falls back to val_dataset is train_dataset when
        # val_df is None; a single registration must still work.
        self.recorder.register(
            self.dataset,
            dmr_labels=np.array([1, 0, 2, 0]),
            on_target_mask=np.array([True, False, True, True]),
        )
        self.assertIsNotNone(
            self.recorder.on_eval(
                self.dataset, self.predictions, step=3, prefix="train"
            )
        )
