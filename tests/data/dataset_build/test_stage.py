import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.data.dataset_build import stage
from syto.data.dataset_build.buckets import build_region_index
from syto.data.dataset_build.stage import stage_dataframe

LABELS = {"0": "Adipocytes", "1": "Gallbladder"}


def _atlas_df():
    """Two regions, all 39 expected cell-type columns present (zeros)."""
    base = pd.DataFrame(
        {
            "chr": ["chr1", "chr1"],
            "start": [1001, 2001],
            "end": [1010, 2010],
            "startCpG": [10, 20],
            "endCpG": [12, 22],
            "target": ["Adipocytes", "Gallbladder"],
            "name": ["chr1:1001-1010", "chr1:2001-2010"],
            "direction": ["U", "U"],
        }
    )
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
        self.reads = pd.DataFrame(
            {
                "ref_name": ["chr1"],
                "ref_pos": [1000],
                "original_seq": ["ACGTACGTAC"],  # len 10 -> read_end 1009
                "methyl_seq": ["2210122100"],
                "ctype": ["Adipocytes"],
            }
        )

    def test_staged_has_region_bucket_file_and_labels(self):
        staged, counts = stage_dataframe(
            self.reads, self.atlas, self.region_index, LABELS, sample_id="S1"
        )
        self.assertGreaterEqual(len(staged), 1)
        for col in [
            "name",
            "region_bucket",
            "file",
            "original_label",
            "dmr_ctype_label",
        ]:
            self.assertIn(col, staged.columns)
        self.assertTrue((staged["file"] == "S1").all())

    def test_counts_shape(self):
        _, counts = stage_dataframe(
            self.reads, self.atlas, self.region_index, LABELS, sample_id="S1"
        )
        self.assertListEqual(
            sorted(counts.columns), ["file", "n_reads", "original_label"]
        )
        self.assertEqual(counts["n_reads"].sum(), 1)

    def test_trim_pins_coordinate_semantics(self):
        # Read spans 0-based 998..1007; region is 0-based 1000..1009.
        # Expect trimmed read_start == 1000 and pattern sliced from offset 2.
        reads = pd.DataFrame(
            {
                "ref_name": ["chr1"],
                "ref_pos": [998],
                "original_seq": ["AACCGGTTAC"],
                "methyl_seq": ["0122100212"],
                "ctype": ["Adipocytes"],
            }
        )
        staged, _ = stage_dataframe(
            reads, self.atlas, self.region_index, LABELS, sample_id="S1"
        )
        row = staged[staged["name"] == "chr1:1001-1010"].iloc[0]
        self.assertEqual(row["read_start"], 1000)
        self.assertEqual(row["methylation_ids"], "22100212")  # original[2:] (offset 2)


class _FakeBam:
    def __init__(self, references):
        self.references = references

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestStageBamFile(unittest.TestCase):
    INTERVALS = [("chr1", 0, 3010)]
    BAM_READS = pd.DataFrame(
        {
            "read_name": ["r1"],
            "chromosome": ["1"],
            "read_start": [1000],
            "read_end": [1010],
            "seq": ["ACGTACGTAC"],
            "methylation_encoding": ["2210122100"],
        }
    )

    def setUp(self):
        self.atlas = UXMMethylationAtlas(
            atlas_name="test", reference_genome="hg38", atlas_df=_atlas_df()
        )
        self.region_index = build_region_index(self.atlas.atlas, n_buckets=2)
        self.tmp = tempfile.TemporaryDirectory()
        self.staged_dir = Path(self.tmp.name) / "staged"
        self.counts_dir = Path(self.tmp.name) / "counts"

    def tearDown(self):
        self.tmp.cleanup()

    def _stage(self, references, reads):
        with patch.object(stage, "pysam") as pysam_mock, patch.object(
            stage, "read_bam_regions", return_value=reads
        ) as read_mock:
            pysam_mock.AlignmentFile.return_value = _FakeBam(references)
            stats = stage.stage_bam_file(
                "/data/S1_merged.mdup.bam",
                "Gallbladder",
                self.atlas,
                self.region_index,
                LABELS,
                self.staged_dir,
                self.counts_dir,
                bam_params={"bam_path": None},
                intervals=self.INTERVALS,
            )
        return stats, read_mock

    def test_unprefixed_bam_fetches_bare_names_and_stages_prefixed_reads(self):
        stats, read_mock = self._stage(("1", "2"), self.BAM_READS)

        params, intervals = read_mock.call_args.args
        self.assertEqual(params["bam_path"], "/data/S1_merged.mdup.bam")
        self.assertEqual(intervals, [("1", 0, 3010)])

        self.assertEqual(stats["sample"], "S1_merged.mdup")
        self.assertEqual(stats["n_in"], 1)
        shards = list(self.staged_dir.rglob("S1_merged.mdup.parquet"))
        self.assertEqual(len(shards), 1)
        staged = pd.read_parquet(shards[0])
        self.assertEqual(list(staged["chromosome"]), ["chr1"])
        self.assertEqual(list(staged["original_label"]), [1])
        counts = pd.read_parquet(self.counts_dir / "S1_merged.mdup.parquet")
        self.assertEqual(counts["n_reads"].tolist(), [1])

    def test_prefixed_bam_keeps_interval_names(self):
        reads = self.BAM_READS.assign(chromosome=["chr1"])
        _, read_mock = self._stage(("chr1", "chr2"), reads)
        self.assertEqual(read_mock.call_args.args[1], self.INTERVALS)

    def test_empty_bam_writes_nothing(self):
        stats, _ = self._stage(("1",), None)
        self.assertEqual(stats["n_in"], 0)
        self.assertFalse(self.staged_dir.exists())
        self.assertFalse(self.counts_dir.exists())
