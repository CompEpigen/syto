import unittest
import pandas as pd
from syto.data.dataset_build.schema import adapt_recovered_reads, RECOVERED_READS_REQUIRED

LABELS = {"0": "Adipocytes", "1": "Endothel"}


class TestAdaptRecoveredReads(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "ref_name": ["chr1", "chr1"],
            "ref_pos": [1000, 2000],
            "original_seq": ["ACGTACG", "ACG"],
            "methyl_seq": ["2210122", "210"],
            "ctype": ["Adipocytes", "UnknownType"],
            "dmr_label": [0, 1],  # extra column, ignored
        })

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
