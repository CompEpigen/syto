"""Train-split de-duplication in the classifier fitting pipeline."""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from syto.app.classifier_fit_pipeline import ClassifierFittingPipeline
from syto.classification.fit_data import deduplicate_reads, load_columnar_split


def _bucket(directory, bucket, rows):
    d = Path(directory) / "final" / f"region_bucket={bucket}"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "part-0.parquet", index=False)


def _rows(name, start, seq, pattern, split, n, label=(0.5, 0.5)):
    return {
        "name": [name] * n,
        "read_start": [start] * n,
        "input_ids": [seq] * n,
        "methylation_ids": [pattern] * n,
        "dmr_ctype_label": [0] * n,
        "soft_label": [list(label)] * n,
        "split": [split] * n,
    }


def _concat(*parts):
    return {k: sum((p[k] for p in parts), []) for k in parts[0]}


def _dataset(directory):
    """Two buckets; train reads repeat, valid reads repeat too."""
    _bucket(
        directory,
        0,
        _concat(
            _rows("r1", 10, "ACGTACGT", "01010101", "train", 5),
            _rows("r1", 12, "ACGTACGT", "01010101", "train", 2),  # other start
            _rows("r1", 10, "ACGTACGT", "22122222", "train", 3),  # short pattern
            _rows("r1", 10, "ACGTACGT", "01010101", "valid", 4),
        ),
    )
    _bucket(
        directory,
        1,
        _concat(
            _rows("r2", 50, "TTCGTTCG", "11110000", "train", 6, label=(0.9, 0.1)),
            _rows("r2", 50, "TTCGTTCG", "11110000", "valid", 2, label=(0.9, 0.1)),
        ),
    )


def _config(directory, **overrides):
    cfg = {
        "model": {"architecture": "dismir", "num_labels": 2},
        "data_path": str(directory),
        "datasets": "final",
        "label_column": "soft_label",
        "output_dir": str(Path(directory) / "out"),
        "mlflow": {"enabled": False},
        "training": {},
    }
    cfg.update(overrides)
    return cfg


class TestDeduplicateReads(unittest.TestCase):
    def test_collapses_groups_and_counts_them(self):
        df = pd.DataFrame({"k": ["a", "b", "a", "a"], "v": [1, 2, 1, 1]})
        out = deduplicate_reads(df, ["k"])
        self.assertEqual(out["k"].tolist(), ["a", "b"])
        self.assertEqual(out["read_count"].tolist(), [3, 1])

    def test_existing_counts_are_summed(self):
        df = pd.DataFrame({"k": ["a", "a", "b"], "read_count": [2, 5, 1]})
        out = deduplicate_reads(df, ["k"])
        self.assertEqual(out["read_count"].tolist(), [7, 1])

    def test_missing_key_is_reported(self):
        with self.assertRaises(ValueError):
            deduplicate_reads(pd.DataFrame({"k": [1]}), ["k", "absent"])


class TestPerFragmentLoading(unittest.TestCase):
    def test_transform_runs_once_per_part_file(self):
        with tempfile.TemporaryDirectory() as d:
            _dataset(d)
            sizes = []

            def record(part):
                sizes.append(len(part))
                return part.head(1)

            df = load_columnar_split(
                Path(d) / "final",
                "train",
                declared_columns=["input_ids"],
                per_fragment=record,
            )
        self.assertEqual(sorted(sizes), [6, 10])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["input_ids"])

    def test_kept_list_cells_do_not_pin_the_part_buffer(self):
        with tempfile.TemporaryDirectory() as d:
            _dataset(d)
            df = load_columnar_split(
                Path(d) / "final",
                "train",
                declared_columns=["soft_label"],
                per_fragment=lambda part: part.head(1),
            )
        for cell in df["soft_label"]:
            self.assertTrue(cell.flags.owndata)
        self.assertEqual([list(c) for c in df["soft_label"]], [[0.5, 0.5], [0.9, 0.1]])


class TestTrainDeduplication(unittest.TestCase):
    def _run(self, supports_read_counts=True, **overrides):
        with tempfile.TemporaryDirectory() as d:
            _dataset(d)
            pipeline = ClassifierFittingPipeline(
                _config(d, **overrides), logging.getLogger("t")
            )
            classifier = MagicMock()
            classifier.supports_read_counts = supports_read_counts
            classifier.required_fit_columns.return_value = [
                "input_ids",
                "methylation_ids",
                "dmr_ctype_label",
            ]
            with patch(
                "syto.app.classifier_fit_pipeline.read_classifier_factory",
                return_value=classifier,
            ):
                pipeline.run()
            return classifier.fit_classificaton.call_args.kwargs

    def test_train_is_collapsed_with_exact_counts(self):
        train = self._run(deduplicate_train=True)["train_df"]
        self.assertEqual(len(train), 4)
        self.assertEqual(int(train["read_count"].sum()), 16)
        self.assertEqual(sorted(train["read_count"]), [2, 3, 5, 6])

    def test_valid_is_left_read_for_read(self):
        val = self._run(deduplicate_train=True)["val_df"]
        self.assertEqual(len(val), 6)
        self.assertNotIn("read_count", val.columns)

    def test_key_only_columns_are_dropped(self):
        train = self._run(deduplicate_train=True)["train_df"]
        self.assertEqual(
            sorted(train.columns),
            ["dmr_ctype_label", "input_ids", "label", "methylation_ids", "read_count"],
        )

    def test_pattern_filter_applies_before_counting(self):
        train = self._run(deduplicate_train=True, min_pattern_length=4)["train_df"]
        self.assertEqual(sorted(train["read_count"]), [2, 5, 6])

    def test_custom_keys_merge_more(self):
        train = self._run(
            deduplicate_train=True, dedup_keys=["name", "input_ids", "methylation_ids"]
        )["train_df"]
        self.assertEqual(sorted(train["read_count"]), [3, 6, 7])

    def test_disabled_by_default(self):
        train = self._run()["train_df"]
        self.assertEqual(len(train), 16)
        self.assertNotIn("read_count", train.columns)

    def test_classifier_without_count_support_is_refused(self):
        with self.assertRaises(ValueError):
            self._run(supports_read_counts=False, deduplicate_train=True)

    def test_valid_is_collapsed_when_requested(self):
        kwargs = self._run(deduplicate_valid=True)
        self.assertEqual(sorted(kwargs["val_df"]["read_count"]), [2, 4])
        self.assertNotIn("read_count", kwargs["train_df"].columns)

    def test_valid_dedup_alone_still_needs_count_support(self):
        with self.assertRaises(ValueError):
            self._run(supports_read_counts=False, deduplicate_valid=True)

    def test_valid_subsample_draws_reads_not_rows(self):
        val = self._run(deduplicate_valid=True, valid_subsample=3, seed=1)["val_df"]
        self.assertEqual(int(val["read_count"].sum()), 3)
        self.assertTrue((val["read_count"] > 0).all())

    def test_valid_subsample_above_read_total_is_a_no_op(self):
        val = self._run(deduplicate_valid=True, valid_subsample=6)["val_df"]
        self.assertEqual(sorted(val["read_count"]), [2, 4])

    def test_valid_subsample_is_deterministic_for_a_seed(self):
        first = self._run(deduplicate_valid=True, valid_subsample=3, seed=2)
        second = self._run(deduplicate_valid=True, valid_subsample=3, seed=2)
        pd.testing.assert_frame_equal(first["val_df"], second["val_df"])


if __name__ == "__main__":
    unittest.main()
