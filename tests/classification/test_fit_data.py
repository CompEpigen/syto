import unittest
from pathlib import Path
import tempfile

import pandas as pd

from syto.classification.fit_data import (
    detect_format,
    resolve_fit_columns,
    load_legacy_split,
    apply_label_rename,
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
