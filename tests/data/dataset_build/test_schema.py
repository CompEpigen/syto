import unittest
import pandas as pd
from syto.data.dataset_build.schema import (
    adapt_recovered_reads,
    bam_reads_to_recovered,
    RECOVERED_READS_REQUIRED,
)

LABELS = {"0": "Adipocytes", "1": "Endothel"}


class TestAdaptRecoveredReads(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ref_name": ["chr1", "chr1"],
                "ref_pos": [1000, 2000],
                "original_seq": ["ACGTACG", "ACG"],
                "methyl_seq": ["2210122", "210"],
                "ctype": ["Adipocytes", "UnknownType"],
                "dmr_label": [0, 1],  # extra column, ignored
            }
        )

    def test_renames_and_derives_columns(self):
        out = adapt_recovered_reads(self.df, LABELS)
        row = out.iloc[0]
        self.assertEqual(row["chromosome"], "chr1")
        self.assertEqual(row["read_start"], 1000)
        self.assertEqual(row["read_end"], 1006)  # 1000 + 7 - 1
        self.assertEqual(row["input_ids"], "ACGTACG")
        self.assertEqual(row["methylation_ids"], "2210122")
        self.assertEqual(row["original_label"], 0)

    def test_drops_rows_with_unknown_ctype(self):
        out = adapt_recovered_reads(self.df, LABELS)
        self.assertEqual(len(out), 1)

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            adapt_recovered_reads(self.df.drop(columns=["ref_pos"]), LABELS)


class TestBamReadsToRecovered(unittest.TestCase):
    def test_maps_bam_columns_and_labels_every_read(self):
        bam_reads = pd.DataFrame(
            {
                "read_name": ["a", "b"],
                "chromosome": ["9", "X"],
                "read_start": [1000, 2000],
                "read_end": [1004, 2003],  # exclusive; recomputed downstream
                "seq": ["ACGTA", "CGCG"],
                "methylation_encoding": ["21222", "1202"],
            }
        )
        recovered = bam_reads_to_recovered(
            bam_reads, "Medulloblastoma", chrom_prefix="chr"
        )
        self.assertTrue(RECOVERED_READS_REQUIRED.issubset(recovered.columns))
        self.assertEqual(list(recovered["ref_name"]), ["chr9", "chrX"])

        out = adapt_recovered_reads(recovered, {"0": "Medulloblastoma"})
        self.assertEqual(list(out["read_end"]), [1004, 2003])  # inclusive
        self.assertEqual(list(out["methylation_ids"]), ["21222", "1202"])
        self.assertEqual(list(out["original_label"]), [0, 0])
