import unittest

import pandas as pd

from syto.data.pseudobulk_input import (
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    build_declared_columns,
    ensure_ncpgs,
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
