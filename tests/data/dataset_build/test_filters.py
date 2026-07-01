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
