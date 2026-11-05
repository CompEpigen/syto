import unittest
import tempfile
from pathlib import Path

import pandas as pd

from syto.data.dataset_build.filters import (
    pattern_length,
    filter_by_pattern_length,
    filter_by_atlas_regions,
    load_region_names,
    staged_counts,
)


class TestPatternLength(unittest.TestCase):
    def test_counts_only_zero_and_one(self):
        self.assertEqual(pattern_length("0101"), 4)
        self.assertEqual(pattern_length("01.2x01"), 4)
        self.assertEqual(pattern_length(""), 0)


class TestFilterByPatternLength(unittest.TestCase):
    def test_keeps_reads_at_or_above_threshold(self):
        df = pd.DataFrame({"pat": ["01", "0101", "0"], "x": [1, 2, 3]})
        out = filter_by_pattern_length(df, "pat", 2)
        self.assertEqual(list(out["x"]), [1, 2])


class TestFilterByAtlasRegions(unittest.TestCase):
    def test_keeps_only_named_regions(self):
        df = pd.DataFrame({"name": ["a", "b", "c"], "x": [1, 2, 3]})
        out = filter_by_atlas_regions(df, {"a", "c"})
        self.assertEqual(list(out["x"]), [1, 3])


class TestLoadRegionNames(unittest.TestCase):
    def test_reads_name_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "atlas.tsv"
            pd.DataFrame({"name": ["r1", "r2"], "chr": ["chr1", "chr1"]}).to_csv(
                p, sep="\t", index=False
            )
            self.assertEqual(load_region_names(str(p)), {"r1", "r2"})


class TestStagedCounts(unittest.TestCase):
    def _write_bucket(self, staged_dir, bucket, df):
        d = Path(staged_dir) / f"region_bucket={bucket}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "s.parquet", index=False)

    def test_recomputes_counts_after_filters(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_bucket(
                d,
                0,
                pd.DataFrame(
                    {
                        "file": ["f0", "f0", "f0"],
                        "original_label": [0, 0, 1],
                        "name": ["r1", "r2", "r1"],
                        "pat": ["0101", "01", "0101"],
                    }
                ),
            )
            out = staged_counts(
                d,
                region_names={"r1"},
                min_pattern_length=3,
                pattern_column="pat",
            )
            got = {(r.file, r.original_label): r.n_reads for r in out.itertuples()}
            self.assertEqual(got, {("f0", 0): 1, ("f0", 1): 1})


class TestLabelRemap(unittest.TestCase):
    def test_maps_codes_by_cell_type_name(self):
        from syto.data.dataset_build.filters import subset_label_remap

        source = {"0": "A", "1": "B", "2": "C", "3": "D"}
        target = {"0": "B", "1": "D"}
        self.assertEqual(subset_label_remap(source, target), {1: 0, 3: 1})

    def test_rejects_cell_types_missing_from_source(self):
        from syto.data.dataset_build.filters import subset_label_remap

        with self.assertRaisesRegex(ValueError, "Z"):
            subset_label_remap({"0": "A"}, {"0": "A", "1": "Z"})

    def test_rejects_non_contiguous_target_codes(self):
        from syto.data.dataset_build.filters import subset_label_remap

        with self.assertRaisesRegex(ValueError, "0..1"):
            subset_label_remap({"0": "A", "1": "B"}, {"0": "A", "2": "B"})


class TestFilterAndRemapLabels(unittest.TestCase):
    def test_drops_reads_outside_subset_and_reindexes(self):
        from syto.data.dataset_build.filters import filter_and_remap_labels

        df = pd.DataFrame(
            {
                "original_label": [1, 3, 1, 2, 3],
                "dmr_ctype_label": [3, 1, 2, 1, 3],
                "x": [0, 1, 2, 3, 4],
            }
        )
        out = filter_and_remap_labels(df, {1: 0, 3: 1})
        # row 2: region target 2 not kept; row 3: sample label 2 not kept
        self.assertEqual(list(out["x"]), [0, 1, 4])
        self.assertEqual(list(out["original_label"]), [0, 1, 1])
        self.assertEqual(list(out["dmr_ctype_label"]), [1, 0, 1])


class TestStagedCountsLabelRemap(unittest.TestCase):
    def test_counts_use_remapped_labels(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / "region_bucket=0"
            b.mkdir(parents=True)
            pd.DataFrame(
                {
                    "file": ["f0", "f0", "f1", "f1"],
                    "original_label": [3, 3, 2, 3],
                    "dmr_ctype_label": [3, 2, 3, 3],
                    "name": ["r1"] * 4,
                    "pat": ["0101"] * 4,
                }
            ).to_parquet(b / "s.parquet", index=False)
            out = staged_counts(d, pattern_column="pat", label_remap={3: 0})
            got = {(r.file, r.original_label): r.n_reads for r in out.itertuples()}
            self.assertEqual(got, {("f0", 0): 1, ("f1", 0): 1})
