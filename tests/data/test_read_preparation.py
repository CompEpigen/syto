import tempfile
import unittest

import pandas as pd

from syto.data.read_preparation import (
    _trim_reads_by_atlas_overlap,
    prepare_splits_for_pseudobulk,
)


def _atlas_df():
    """Two non-adjacent atlas regions on chr1, covering 0-based positions
    10-19 (region1) and 21-30 (region2), with a one-base gap at position 20."""
    return pd.DataFrame(
        {
            "chr": ["chr1", "chr1"],
            "start": [11, 22],
            "end": [20, 31],
            "name": ["chr1:11-20", "chr1:22-31"],
            "target": ["ctype_a", "ctype_b"],
        }
    )


class TestTrimReadsByAtlasOverlap(unittest.TestCase):
    """Tests for read_preparation._trim_reads_by_atlas_overlap."""

    def _trim(self, df):
        atlas = (
            _atlas_df()
            .sort_values(by=["chr", "start", "end"])
            .reset_index(drop=True)
        )
        df_sorted = df.sort_values(
            by=["chromosome", "read_start", "read_end"]
        ).reset_index(drop=True)
        return _trim_reads_by_atlas_overlap(
            df_sorted, atlas, seq_column="seq", methylation_pattern_column="pattern"
        )

    def test_fully_contained_read_is_unchanged(self):
        """A read fully inside an atlas region keeps its coordinates and sequence."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1"],
                "read_name": ["r_contained"],
                "read_start": [12],
                "read_end": [18],
                "seq": ["A" * 7],
                "pattern": ["1010101"],
            }
        )

        res = self._trim(df)

        self.assertEqual(len(res), 1)
        row = res.iloc[0]
        self.assertEqual(row["read_start"], 12)
        self.assertEqual(row["read_end"], 18)
        self.assertEqual(row["seq"], "A" * 7)
        self.assertEqual(row["pattern"], "1010101")

    def test_read_extending_past_region_start_is_clipped(self):
        """A read starting before the atlas region is clipped to the region's span."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1"],
                "read_name": ["r_left"],
                "read_start": [5],
                "read_end": [20],
                "seq": ["ABCDEFGHIJKLMNOP"],  # 16 chars, positions 5..20
                "pattern": ["0123456789012345"],
            }
        )

        res = self._trim(df)

        self.assertEqual(len(res), 1)
        row = res.iloc[0]
        self.assertEqual(row["read_start"], 10)
        self.assertEqual(row["read_end"], 19)
        self.assertEqual(row["seq"], "FGHIJKLMNOP")
        self.assertEqual(row["pattern"], "56789012345")

    def test_read_overlapping_two_regions_is_duplicated(self):
        """A read spanning two atlas regions produces one clipped row per region."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1"],
                "read_name": ["r_multi"],
                "read_start": [15],
                "read_end": [25],
                "seq": ["ABCDEFGHIJK"],  # 11 chars, positions 15..25
                "pattern": ["01234567890"],
                "original_label": [3],
            }
        )

        res = self._trim(df)

        self.assertEqual(len(res), 2)
        self.assertTrue((res["read_name"] == "r_multi").all())
        self.assertTrue((res["original_label"] == 3).all())

        region1 = res[res["read_start"] == 15].iloc[0]
        self.assertEqual(region1["read_end"], 19)
        self.assertEqual(region1["seq"], "ABCDEF")
        self.assertEqual(region1["pattern"], "012345")

        region2 = res[res["read_start"] == 21].iloc[0]
        self.assertEqual(region2["read_end"], 25)
        self.assertEqual(region2["seq"], "GHIJK")
        self.assertEqual(region2["pattern"], "67890")

    def test_reads_without_any_overlap_are_dropped(self):
        """Reads that don't overlap any atlas region are excluded from the result."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr2"],
                "read_name": ["r_far", "r_other_chrom"],
                "read_start": [100, 5],
                "read_end": [110, 10],
                "seq": ["A" * 11, "C" * 6],
                "pattern": ["1" * 11, "0" * 6],
            }
        )

        res = self._trim(df)

        self.assertEqual(len(res), 0)


class TestPrepareSplitsForPseudobulk(unittest.TestCase):
    """Tests for prepare_splits_for_pseudobulk."""

    def _write_atlas(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, newline=""
        )
        _atlas_df().to_csv(f.name, sep="\t", index=False)
        return f.name

    def _splits(self):
        return {
            "train": pd.DataFrame(
                {
                    "chr": ["chr1", "chr1", "chr2"],
                    "read_name": ["r_contained", "r_left", "r_dropped"],
                    "trimmed_start": [12, 5, 100],
                    "trimmed_end": [18, 20, 110],
                    "seq": ["A" * 7, "ABCDEFGHIJKLMNOP", "C" * 6],
                    "pattern": ["1010101", "0123456789012345", "0" * 6],
                    "original_label": [0, 1, 2],
                    "dmr_ctype_label": [0, 1, 2],
                }
            )
        }

    def test_atlas_path_required_when_trimming_enabled(self):
        """trim_by_atlas_regions_overlap=True (the default) requires atlas_path."""
        with self.assertRaises(ValueError):
            prepare_splits_for_pseudobulk(self._splits())

    def test_trimming_disabled_keeps_all_reads_untouched(self):
        """trim_by_atlas_regions_overlap=False skips atlas trimming entirely."""
        result = prepare_splits_for_pseudobulk(
            self._splits(), trim_by_atlas_regions_overlap=False
        )

        df = result["train"]
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["read_name"]), {"r_contained", "r_left", "r_dropped"})
        # Derived columns are still added.
        self.assertIn("is_cell_informative_region", df.columns)
        self.assertIn("chromosome", df.columns)
        self.assertEqual(df.loc[df["read_name"] == "r_left", "read_start"].iloc[0], 5)

    def test_trimming_enabled_clips_and_drops_reads(self):
        """trim_by_atlas_regions_overlap=True clips reads to atlas regions and drops non-overlapping ones."""
        atlas_path = self._write_atlas()

        result = prepare_splits_for_pseudobulk(self._splits(), atlas_path=atlas_path)

        df = result["train"]
        self.assertEqual(set(df["read_name"]), {"r_contained", "r_left"})

        left = df[df["read_name"] == "r_left"].iloc[0]
        self.assertEqual(left["read_start"], 10)
        self.assertEqual(left["read_end"], 19)
        self.assertEqual(left["seq"], "FGHIJKLMNOP")

        contained = df[df["read_name"] == "r_contained"].iloc[0]
        self.assertEqual(contained["read_start"], 12)
        self.assertEqual(contained["read_end"], 18)
        self.assertEqual(contained["seq"], "A" * 7)


if __name__ == "__main__":
    unittest.main()
