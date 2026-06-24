import unittest
import pandas as pd
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.data.dataset_build.buckets import build_region_index
from syto.data.dataset_build.stage import stage_dataframe

LABELS = {"0": "Adipocytes", "1": "Gallbladder"}


def _atlas_df():
    """Two regions, all 39 expected cell-type columns present (zeros)."""
    base = pd.DataFrame({
        "chr": ["chr1", "chr1"],
        "start": [1001, 2001],
        "end": [1010, 2010],
        "startCpG": [10, 20],
        "endCpG": [12, 22],
        "target": ["Adipocytes", "Gallbladder"],
        "name": ["chr1:1001-1010", "chr1:2001-2010"],
        "direction": ["U", "U"],
    })
    for c in UXMMethylationAtlas.EXPECTED_CTYPE_COLUMNS:
        base[c] = 0.0
    return base


class TestStageDataframe(unittest.TestCase):
    def setUp(self):
        self.atlas = UXMMethylationAtlas(
            atlas_name="test", reference_genome="hg38", atlas_df=_atlas_df()
        )
        self.region_index = build_region_index(self.atlas.atlas, n_buckets=2)
        # One read fully inside region chr1:1001-1010 (0-based 1000..1009).
        self.reads = pd.DataFrame({
            "ref_name": ["chr1"],
            "ref_pos": [1000],
            "original_seq": ["ACGTACGTAC"],   # len 10 -> read_end 1009
            "methyl_seq": ["2210122100"],
            "ctype": ["Adipocytes"],
        })

    def test_staged_has_region_bucket_file_and_labels(self):
        staged, counts = stage_dataframe(
            self.reads, self.atlas, self.region_index, LABELS, sample_id="S1"
        )
        self.assertGreaterEqual(len(staged), 1)
        for col in ["name", "region_bucket", "file", "original_label", "dmr_ctype_label"]:
            self.assertIn(col, staged.columns)
        self.assertTrue((staged["file"] == "S1").all())

    def test_counts_shape(self):
        _, counts = stage_dataframe(
            self.reads, self.atlas, self.region_index, LABELS, sample_id="S1"
        )
        self.assertListEqual(sorted(counts.columns), ["file", "n_reads", "original_label"])
        self.assertEqual(counts["n_reads"].sum(), 1)

    def test_trim_pins_coordinate_semantics(self):
        # Read spans 0-based 998..1007; region is 0-based 1000..1009.
        # Expect trimmed read_start == 1000 and pattern sliced from offset 2.
        reads = pd.DataFrame({
            "ref_name": ["chr1"], "ref_pos": [998],
            "original_seq": ["AACCGGTTAC"], "methyl_seq": ["0122100212"],
            "ctype": ["Adipocytes"],
        })
        staged, _ = stage_dataframe(reads, self.atlas, self.region_index, LABELS, sample_id="S1")
        row = staged[staged["name"] == "chr1:1001-1010"].iloc[0]
        self.assertEqual(row["read_start"], 1000)
        self.assertEqual(row["methylation_ids"], "22100212")  # original[2:] (offset 2)
