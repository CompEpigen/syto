import tempfile
import unittest
from pathlib import Path

import pandas as pd

from syto.data.pseudobulk_input import (
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    build_declared_columns,
    ensure_ncpgs,
    load_raw_splits,
)


class TestBuildDeclaredColumns(unittest.TestCase):
    def test_required_plus_classifier_plus_present_optionals(self):
        available = [
            "original_label",
            "dmr_ctype_label",
            "chr",
            "read_start",
            "read_end",
            "methylation_ids",
            "input_ids",
            "NCPGS",
            "soft_label_other",
        ]
        declared = build_declared_columns(
            ["input_ids", "methylation_ids", "dmr_ctype_label"], available
        )
        # Required + classifier feature cols are present, de-duplicated.
        for col in REQUIRED_COLUMNS + ["input_ids"]:
            self.assertIn(col, declared)
        self.assertEqual(len(declared), len(set(declared)))
        # Present optional (NCPGS) included; absent optionals (file/name) not.
        self.assertIn("NCPGS", declared)
        self.assertNotIn("file", declared)
        # Non-declared dataset columns are not pulled in.
        self.assertNotIn("soft_label_other", declared)

    def test_order_is_required_then_classifier_then_optional(self):
        available = ["methylation_ids", "dmr_ctype", "input_ids"] + REQUIRED_COLUMNS
        declared = build_declared_columns(["input_ids"], available)
        self.assertEqual(declared[: len(REQUIRED_COLUMNS)], REQUIRED_COLUMNS)
        self.assertIn("input_ids", declared)
        self.assertTrue(declared.index("input_ids") < declared.index("dmr_ctype"))

    def test_absent_optional_excluded(self):
        available = list(REQUIRED_COLUMNS)  # no optionals at all
        declared = build_declared_columns([], available)
        for opt in OPTIONAL_COLUMNS:
            self.assertNotIn(opt, declared)


class TestEnsureNcpgs(unittest.TestCase):
    def test_derives_when_absent(self):
        df = pd.DataFrame({"methylation_ids": ["0101", "1", "0011"]})
        out = ensure_ncpgs(df)
        self.assertEqual(list(out["NCPGS"]), [4, 1, 4])

    def test_noop_when_present(self):
        df = pd.DataFrame({"methylation_ids": ["0101"], "NCPGS": [99]})
        out = ensure_ncpgs(df)
        self.assertEqual(list(out["NCPGS"]), [99])

    def test_resolves_pattern_alias(self):
        df = pd.DataFrame({"pattern": ["0101", "1"]})
        out = ensure_ncpgs(df)
        self.assertEqual(list(out["NCPGS"]), [4, 1])

    def test_missing_pattern_raises(self):
        df = pd.DataFrame({"x": [1]})
        with self.assertRaises(ValueError):
            ensure_ncpgs(df)


class TestLoadRawSplitsColumnar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        part = self.dir / "region_bucket=0"
        part.mkdir()
        # Two splits, an extra label column to prune, and NO NCPGS column.
        pd.DataFrame(
            {
                "input_ids": ["seqT", "seqV"],
                "methylation_ids": ["0101", "1"],
                "dmr_ctype_label": [3, 3],
                "original_label": [3, 5],
                "chr": ["chr1", "chr1"],
                "read_start": [100, 200],
                "read_end": [150, 250],
                "soft_label_other": [[0.1, 0.9], [0.5, 0.5]],
                "split": ["train", "valid"],
                "region_bucket": [0, 0],
            }
        ).to_parquet(part / "part-0.parquet")

    def tearDown(self):
        self.tmp.cleanup()

    def test_projects_filters_and_derives_ncpgs(self):
        splits = load_raw_splits(
            self.dir,
            ["train", "valid"],
            classifier_required_columns=[
                "input_ids",
                "methylation_ids",
                "dmr_ctype_label",
            ],
        )
        self.assertEqual(set(splits), {"train", "valid"})
        train = splits["train"]
        # Only the train row.
        self.assertEqual(len(train), 1)
        self.assertEqual(train["input_ids"].iloc[0], "seqT")
        # Unrequested label column pruned; region_bucket/split not present.
        self.assertNotIn("soft_label_other", train.columns)
        self.assertNotIn("split", train.columns)
        # NCPGS derived from the methylation pattern length.
        self.assertEqual(train["NCPGS"].iloc[0], 4)
        self.assertEqual(splits["valid"]["NCPGS"].iloc[0], 1)

    def test_min_pattern_length_filter(self):
        splits = load_raw_splits(
            self.dir,
            ["valid"],
            classifier_required_columns=["input_ids", "methylation_ids"],
            min_pattern_length=2,
        )
        # valid row has a single-CpG pattern ("1") -> dropped.
        self.assertEqual(len(splits["valid"]), 0)

    def test_missing_required_column_raises(self):
        bad = Path(self.tmp.name) / "bad"
        (bad / "region_bucket=0").mkdir(parents=True)
        pd.DataFrame({"methylation_ids": ["01"], "split": ["train"]}).to_parquet(
            bad / "region_bucket=0" / "part-0.parquet"
        )
        with self.assertRaises(ValueError):
            load_raw_splits(
                bad, ["train"], classifier_required_columns=["methylation_ids"]
            )


class TestLoadRawSplitsLegacy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        pd.DataFrame(
            {
                "input_ids": ["a", "b"],
                "methylation_ids": ["0101", "01"],
                "original_label": [1, 2],
            }
        ).to_parquet(self.dir / "train.parquet")

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_whole_file_and_derives_ncpgs(self):
        splits = load_raw_splits(
            self.dir, ["train"], classifier_required_columns=["input_ids"]
        )
        train = splits["train"]
        # Legacy layout: no projection, all source columns kept.
        self.assertIn("original_label", train.columns)
        self.assertEqual(list(train["NCPGS"]), [4, 2])
