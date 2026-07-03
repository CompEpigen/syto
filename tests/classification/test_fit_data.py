import unittest
from pathlib import Path
import tempfile

import pandas as pd

from syto.classification.fit_data import (
    detect_format,
    resolve_fit_columns,
    load_legacy_split,
    apply_label_rename,
    load_columnar_split,
)


class TestDetectFormat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_override_wins(self):
        self.assertEqual(detect_format(self.dir, override="legacy"), "legacy")
        self.assertEqual(detect_format(self.dir, override="columnar"), "columnar")

    def test_bad_override_raises(self):
        with self.assertRaises(ValueError):
            detect_format(self.dir, override="nonsense")

    def test_partitions_are_columnar(self):
        (self.dir / "region_bucket=0").mkdir()
        pd.DataFrame({"split": ["train"]}).to_parquet(
            self.dir / "region_bucket=0" / "part-0.parquet"
        )
        self.assertEqual(detect_format(self.dir), "columnar")

    def test_train_file_is_legacy(self):
        pd.DataFrame({"a": [1]}).to_parquet(self.dir / "train.parquet")
        self.assertEqual(detect_format(self.dir), "legacy")

    def test_schema_peek_columnar(self):
        pd.DataFrame({"split": ["train"], "x": [1]}).to_parquet(
            self.dir / "part.parquet"
        )
        self.assertEqual(detect_format(self.dir), "columnar")

    def test_undetectable_raises(self):
        with self.assertRaises(ValueError):
            detect_format(self.dir)


class TestResolveFitColumns(unittest.TestCase):
    def test_alias_resolution_and_dedup(self):
        available = ["seq", "methylation_ids", "dmr_ctype_label", "split"]
        # "input_ids" resolves to alias "seq"; duplicates collapse
        out = resolve_fit_columns(
            ["input_ids", "methylation_ids", "input_ids"], available
        )
        self.assertEqual(out, ["seq", "methylation_ids"])

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            resolve_fit_columns(["input_ids", "nope"], ["seq"])


class TestLoadLegacySplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parquet(self):
        pd.DataFrame({"a": [1, 2]}).to_parquet(self.dir / "train.parquet")
        df = load_legacy_split(self.dir, "train")
        self.assertEqual(len(df), 2)

    def test_missing_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_legacy_split(self.dir, "valid")


class TestApplyLabelRename(unittest.TestCase):
    def test_rename_to_soft_label(self):
        df = pd.DataFrame({"soft_label_pooled": [[0.1, 0.9]]})
        out = apply_label_rename(df, "soft_label_pooled", soft_labels=True)
        self.assertIn("soft_label", out.columns)
        self.assertNotIn("soft_label_pooled", out.columns)

    def test_rename_to_label(self):
        df = pd.DataFrame({"original_label": [3]})
        out = apply_label_rename(df, "original_label", soft_labels=False)
        self.assertIn("label", out.columns)

    def test_noop_when_already_canonical(self):
        df = pd.DataFrame({"label": [1]})
        out = apply_label_rename(df, "label", soft_labels=False)
        self.assertEqual(list(out.columns), ["label"])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x": [1]})
        with self.assertRaises(ValueError):
            apply_label_rename(df, "soft_label_pooled", soft_labels=True)


class TestLoadColumnarSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Two region_bucket partitions, each with train + valid rows and two
        # label columns (only one of which we will request).
        for bucket in (0, 1):
            part = self.dir / f"region_bucket={bucket}"
            part.mkdir()
            pd.DataFrame(
                {
                    "input_ids": [f"seqA{bucket}", f"seqB{bucket}"],
                    "methylation_ids": ["0101", "1100"],
                    "dmr_ctype_label": [bucket, bucket],
                    "soft_label_pooled": [[0.2, 0.8], [0.7, 0.3]],
                    "soft_label_other": [[0.9, 0.1], [0.4, 0.6]],
                    "split": ["train", "valid"],
                    "region_bucket": [bucket, bucket],
                }
            ).to_parquet(part / "part-0.parquet")

    def tearDown(self):
        self.tmp.cleanup()

    def test_projection_and_split_filter(self):
        df = load_columnar_split(
            self.dir,
            "train",
            declared_columns=[
                "input_ids",
                "methylation_ids",
                "dmr_ctype_label",
                "soft_label_pooled",
            ],
        )
        # Only requested columns (split dropped, soft_label_other pruned).
        self.assertEqual(
            set(df.columns),
            {"input_ids", "methylation_ids", "dmr_ctype_label", "soft_label_pooled"},
        )
        # Only train rows: one per bucket.
        self.assertEqual(len(df), 2)
        self.assertTrue(all(s.startswith("seqA") for s in df["input_ids"]))

    def test_empty_when_split_absent(self):
        df = load_columnar_split(
            self.dir,
            "test",
            declared_columns=["input_ids", "methylation_ids"],
        )
        self.assertTrue(df.empty)
