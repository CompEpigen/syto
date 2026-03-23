import unittest
import pandas as pd
from parameterized import parameterized
from methyldl.data.utils import count_unique_positions, split_long_reads


class TestCountUniquePositions(unittest.TestCase):
    """Test suite for count_unique_positions function."""

    def test_single_read_single_chromosome(self):
        """Test with a single read on one chromosome."""
        df = pd.DataFrame(
            {"chromosome": ["chr1"], "read_start": [100], "read_end": [200]}
        )
        result = count_unique_positions(df)
        self.assertEqual(result, 101)  # 200 - 100 + 1

    def test_non_overlapping_reads(self):
        """Test with non-overlapping reads on same chromosome."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1", "chr1"],
                "read_start": [100, 300, 500],
                "read_end": [200, 400, 600],
            }
        )
        result = count_unique_positions(df)
        # (200-100+1) + (400-300+1) + (600-500+1) = 101 + 101 + 101 = 303
        self.assertEqual(result, 303)

    def test_overlapping_reads(self):
        """Test with overlapping reads on same chromosome."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1"],
                "read_start": [100, 150],
                "read_end": [200, 250],
            }
        )
        result = count_unique_positions(df)
        # Merged interval: 100-250 = 151 positions
        self.assertEqual(result, 151)

    def test_completely_overlapping_reads(self):
        """Test with one read completely inside another."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1"],
                "read_start": [100, 120],
                "read_end": [200, 180],
            }
        )
        result = count_unique_positions(df)
        # Merged interval: 100-200 = 101 positions
        self.assertEqual(result, 101)

    def test_multiple_chromosomes(self):
        """Test with reads on multiple chromosomes."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1", "chr2", "chr2"],
                "read_start": [100, 150, 100, 300],
                "read_end": [200, 250, 200, 400],
            }
        )
        result = count_unique_positions(df)
        # chr1: 100-250 = 151, chr2: (100-200) + (300-400) = 101 + 101 = 202
        # Total: 151 + 202 = 353
        self.assertEqual(result, 353)

    def test_adjacent_reads(self):
        """Test with adjacent reads (touching but not overlapping)."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1"],
                "read_start": [100, 201],
                "read_end": [200, 300],
            }
        )
        result = count_unique_positions(df)
        # Two separate intervals: (200-100+1) + (300-201+1) = 101 + 100 = 201
        self.assertEqual(result, 201)

    def test_reads_touching_at_boundary(self):
        """Test with reads that touch at exactly one position."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1"],
                "read_start": [100, 200],
                "read_end": [200, 300],
            }
        )
        result = count_unique_positions(df)
        # Merged: 100-300 = 201 positions
        self.assertEqual(result, 201)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"chromosome": [], "read_start": [], "read_end": []})
        result = count_unique_positions(df)
        self.assertEqual(result, 0)

    def test_complex_overlapping_pattern(self):
        """Test with complex overlapping pattern."""
        df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1", "chr1", "chr1"],
                "read_start": [100, 150, 180, 250],
                "read_end": [200, 220, 240, 300],
            }
        )
        result = count_unique_positions(df)
        # Merged intervals: 100-240, 250-300
        # (240-100+1) + (300-250+1) = 141 + 51 = 192
        self.assertEqual(result, 192)


class TestSplitLongReads(unittest.TestCase):
    """Test suite for split_long_reads function."""

    def test_read_within_max_length(self):
        """Test with a read that doesn't need splitting."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCG"],
                "methylation_ids": ["2212"],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["read_name"], "read1")
        self.assertEqual(result.iloc[0]["input_ids"], "ATCG")
        self.assertEqual(result.iloc[0]["methylation_ids"], "2212")
        self.assertEqual(result.iloc[0]["num_cpgs"], 1)  # '0' and '1'

    def test_read_exactly_max_length(self):
        """Test with a read exactly at max length."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCGATCG"],
                "methylation_ids": ["01201220"],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=8)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["input_ids"], "ATCGATCG")

    def test_read_needs_splitting_into_two_chunks(self):
        """Test splitting a read into two chunks."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCGATCGATCG"],
                "methylation_ids": ["012012012012"],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=6)

        self.assertEqual(len(result), 2)
        # First chunk
        self.assertEqual(result.iloc[0]["input_ids"], "ATCGAT")
        self.assertEqual(result.iloc[0]["methylation_ids"], "012012")
        self.assertEqual(result.iloc[0]["num_cpgs"], 4)  # Four '0' or '1'
        # Second chunk
        self.assertEqual(result.iloc[1]["input_ids"], "CGATCG")
        self.assertEqual(result.iloc[1]["methylation_ids"], "012012")
        self.assertEqual(result.iloc[1]["num_cpgs"], 4)

    def test_read_needs_splitting_into_multiple_chunks(self):
        """Test splitting a read into multiple chunks."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["A" * 25],
                "methylation_ids": ["0" * 25],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [1],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        self.assertEqual(len(result), 3)  # 25 / 10 = 3 chunks
        self.assertEqual(result.iloc[0]["input_ids"], "A" * 10)
        self.assertEqual(result.iloc[1]["input_ids"], "A" * 10)
        self.assertEqual(result.iloc[2]["input_ids"], "A" * 5)

    def test_multiple_reads_some_split_some_not(self):
        """Test with multiple reads, some requiring splitting."""
        df = pd.DataFrame(
            {
                "read_name": ["read1", "read2", "read3"],
                "input_ids": ["ATCG", "ATCGATCGATCG", "GGCC"],
                "methylation_ids": ["0120", "012012012012", "0102"],
                "chromosome": ["chr1", "chr1", "chr2"],
                "original_file": ["file1.bam", "file1.bam", "file2.bam"],
                "label": [0, 1, 0],
            }
        )
        result = split_long_reads(df, max_read_length=6)

        # read1: 1 chunk, read2: 2 chunks, read3: 1 chunk = 4 total
        self.assertEqual(len(result), 4)

        # Verify read names are preserved
        read_names = result["read_name"].tolist()
        self.assertEqual(read_names.count("read1"), 1)
        self.assertEqual(read_names.count("read2"), 2)
        self.assertEqual(read_names.count("read3"), 1)

    def test_cpg_counting_only_0_and_1(self):
        """Test that CpG counting only counts '0' and '1'."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCG"],
                "methylation_ids": ["0122"],  # 0, 1, 2, 2
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        # Should count only '0' and '1', not '2'
        self.assertEqual(result.iloc[0]["num_cpgs"], 2)

    def test_cpg_counting_no_cpgs(self):
        """Test CpG counting when there are no CpGs."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCG"],
                "methylation_ids": ["2222"],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        self.assertEqual(result.iloc[0]["num_cpgs"], 0)

    def test_cpg_counting_all_cpgs(self):
        """Test CpG counting when all positions are CpGs."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCG"],
                "methylation_ids": ["0101"],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        self.assertEqual(result.iloc[0]["num_cpgs"], 4)

    def test_preserved_metadata_after_split(self):
        """Test that metadata is preserved in split reads."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["ATCGATCGATCG"],
                "methylation_ids": ["012012012012"],
                "chromosome": ["chr5"],
                "original_file": ["sample_data.bam"],
                "label": [1],
            }
        )
        result = split_long_reads(df, max_read_length=6)

        # Check that all chunks have the same metadata
        for _, row in result.iterrows():
            self.assertEqual(row["read_name"], "read1")
            self.assertEqual(row["chromosome"], "chr5")
            self.assertEqual(row["original_file"], "sample_data.bam")
            self.assertEqual(row["label"], 1)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            {
                "read_name": [],
                "input_ids": [],
                "methylation_ids": [],
                "chromosome": [],
                "original_file": [],
                "label": [],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        self.assertEqual(len(result), 0)
        self.assertTrue(result.empty)

    @parameterized.expand(
        [
            ("short_max_length", 5),
            ("medium_max_length", 50),
            ("long_max_length", 500),
        ]
    )
    def test_different_max_lengths(self, name, max_length):
        """Test with different max_read_length values."""
        sequence = "A" * 100
        methylation = "0" * 100

        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": [sequence],
                "methylation_ids": [methylation],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )

        result = split_long_reads(df, max_read_length=max_length)

        # Calculate expected number of chunks
        expected_chunks = (100 + max_length - 1) // max_length  # Ceiling division
        self.assertEqual(len(result), expected_chunks)

        # Verify total length is preserved
        total_length = sum(len(row["input_ids"]) for _, row in result.iterrows())
        self.assertEqual(total_length, 100)

    def test_chunk_sizes_correct(self):
        """Test that chunk sizes are correct and don't exceed max_length."""
        df = pd.DataFrame(
            {
                "read_name": ["read1"],
                "input_ids": ["A" * 23],
                "methylation_ids": ["0" * 23],
                "chromosome": ["chr1"],
                "original_file": ["file1.bam"],
                "label": [0],
            }
        )
        result = split_long_reads(df, max_read_length=10)

        # Should create 3 chunks: 10, 10, 3
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result.iloc[0]["input_ids"]), 10)
        self.assertEqual(len(result.iloc[1]["input_ids"]), 10)
        self.assertEqual(len(result.iloc[2]["input_ids"]), 3)

        # Verify no chunk exceeds max_length
        for _, row in result.iterrows():
            self.assertLessEqual(len(row["input_ids"]), 10)


if __name__ == "__main__":
    unittest.main()
