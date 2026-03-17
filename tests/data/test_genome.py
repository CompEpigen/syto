import io
import unittest
from unittest.mock import patch

import pandas as pd
from parameterized import parameterized
from methyldl.data.sequencing.genome import (
    add_reference_cpg_counts,
    generate_kmer_str_with_overlap,
    get_alter_of_dna_sequence,
    collapse_methylation,
    process_chunk,
)


class FakeReferenceFasta:
    """Minimal stand-in for pysam.FastaFile used by reference-count tests."""

    def __init__(self, sequences=None, failing_regions=None):
        """Store reference sequences and any fetch calls that should fail."""
        self.sequences = sequences or {}
        self.failing_regions = set(failing_regions or [])
        self.fetch_calls = []
        self.closed = False

    def fetch(self, chrom, start, end):
        """Return the requested slice or raise to mimic pysam lookup failures."""
        self.fetch_calls.append((chrom, start, end))
        if (chrom, start, end) in self.failing_regions:
            raise RuntimeError("Synthetic fetch failure")
        if chrom not in self.sequences:
            raise KeyError(chrom)
        return self.sequences[chrom][start:end]

    def close(self):
        """Record that the reference handle was closed."""
        self.closed = True


class TestGenerateKmerStrWithOverlap(unittest.TestCase):
    """Test suite for generate_kmer_str_with_overlap function."""

    def test_basic_3mer(self):
        """Test basic 3-mer generation."""
        sequence = "ATCGATCG"
        result = generate_kmer_str_with_overlap(sequence, kmer=3)
        expected = "ATC TCG CGA GAT ATC TCG"
        self.assertEqual(result, expected)

    def test_basic_2mer(self):
        """Test basic 2-mer generation."""
        sequence = "ATCG"
        result = generate_kmer_str_with_overlap(sequence, kmer=2)
        expected = "AT TC CG"
        self.assertEqual(result, expected)

    def test_basic_4mer(self):
        """Test basic 4-mer generation."""
        sequence = "ATCGATCG"
        result = generate_kmer_str_with_overlap(sequence, kmer=4)
        expected = "ATCG TCGA CGAT GATC ATCG"
        self.assertEqual(result, expected)

    def test_short_sequence(self):
        """Test with sequence shorter than k."""
        sequence = "AT"
        result = generate_kmer_str_with_overlap(sequence, kmer=3)
        self.assertEqual(result, "")

    def test_exact_kmer_length(self):
        """Test with sequence exactly k length."""
        sequence = "ATC"
        result = generate_kmer_str_with_overlap(sequence, kmer=3)
        expected = "ATC"
        self.assertEqual(result, expected)

    def test_single_base(self):
        """Test with single base and k=1."""
        sequence = "A"
        result = generate_kmer_str_with_overlap(sequence, kmer=1)
        expected = "A"
        self.assertEqual(result, expected)

    def test_overlap_verification(self):
        """Test that k-mers properly overlap."""
        sequence = "AAATTTCCC"
        result = generate_kmer_str_with_overlap(sequence, kmer=3)
        kmers = result.split()

        # Verify overlap: last 2 chars of kmer[i] = first 2 chars of kmer[i+1]
        for i in range(len(kmers) - 1):
            self.assertEqual(kmers[i][1:], kmers[i + 1][:-1])

    @parameterized.expand(
        [
            ("k1", 1),
            ("k2", 2),
            ("k3", 3),
            ("k4", 4),
            ("k5", 5),
        ]
    )
    def test_different_k_values(self, name, k):
        """Test with different k values."""
        sequence = "ATCGATCGATCG"
        result = generate_kmer_str_with_overlap(sequence, kmer=k)

        if len(sequence) >= k:
            kmers = result.split()
            # Number of k-mers should be len(sequence) - k + 1
            expected_count = len(sequence) - k + 1
            self.assertEqual(len(kmers), expected_count)

            # Each k-mer should be exactly k bases long
            for kmer in kmers:
                self.assertEqual(len(kmer), k)


class TestGetAlterOfDnaSequence(unittest.TestCase):
    """Test suite for get_alter_of_dna_sequence function."""

    def test_basic_complement(self):
        """Test basic DNA complement."""
        sequence = "ATCG"
        result = get_alter_of_dna_sequence(sequence)
        expected = "TAGC"
        self.assertEqual(result, expected)

    def test_all_adenine(self):
        """Test sequence of all A's."""
        sequence = "AAAA"
        result = get_alter_of_dna_sequence(sequence)
        expected = "TTTT"
        self.assertEqual(result, expected)

    def test_all_thymine(self):
        """Test sequence of all T's."""
        sequence = "TTTT"
        result = get_alter_of_dna_sequence(sequence)
        expected = "AAAA"
        self.assertEqual(result, expected)

    def test_all_cytosine(self):
        """Test sequence of all C's."""
        sequence = "CCCC"
        result = get_alter_of_dna_sequence(sequence)
        expected = "GGGG"
        self.assertEqual(result, expected)

    def test_all_guanine(self):
        """Test sequence of all G's."""
        sequence = "GGGG"
        result = get_alter_of_dna_sequence(sequence)
        expected = "CCCC"
        self.assertEqual(result, expected)

    def test_double_complement(self):
        """Test that complementing twice returns original."""
        sequence = "ATCGATCG"
        complement = get_alter_of_dna_sequence(sequence)
        double_complement = get_alter_of_dna_sequence(complement)
        self.assertEqual(sequence, double_complement)

    def test_empty_sequence(self):
        """Test with empty sequence."""
        sequence = ""
        result = get_alter_of_dna_sequence(sequence)
        self.assertEqual(result, "")

    def test_single_base(self):
        """Test with single base."""
        for base, complement in [("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")]:
            with self.subTest(base=base):
                result = get_alter_of_dna_sequence(base)
                self.assertEqual(result, complement)

    def test_palindromic_sequence(self):
        """Test palindromic sequence that's its own complement."""
        # Note: No DNA sequence is its own complement, but we can test
        # sequences that are palindromes of their complement
        sequence = "ATCGAT"
        result = get_alter_of_dna_sequence(sequence)
        expected = "TAGCTA"
        self.assertEqual(result, expected)


class TestCollapseMethylation(unittest.TestCase):
    """Test suite for collapse_methylation function."""

    def test_methylated_present(self):
        """Test with methylated cytosine present."""
        seq = [1, 2, 2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 1)

    def test_unmethylated_present(self):
        """Test with unmethylated cytosine present."""
        seq = [0, 2, 2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 0)

    def test_methylated_priority_over_unmethylated(self):
        """Test that methylated (1) has priority over unmethylated (0)."""
        seq = [0, 1, 2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 1)

    def test_all_other(self):
        """Test with all 'other' values."""
        seq = [2, 2, 2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 2)

    def test_empty_sequence(self):
        """Test with empty sequence."""
        seq = []
        result = collapse_methylation(seq)
        self.assertEqual(result, 2)

    def test_single_methylated(self):
        """Test with single methylated value."""
        seq = [1]
        result = collapse_methylation(seq)
        self.assertEqual(result, 1)

    def test_single_unmethylated(self):
        """Test with single unmethylated value."""
        seq = [0]
        result = collapse_methylation(seq)
        self.assertEqual(result, 0)

    def test_single_other(self):
        """Test with single 'other' value."""
        seq = [2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 2)

    def test_multiple_methylated(self):
        """Test with multiple methylated values."""
        seq = [1, 1, 1, 2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 1)

    def test_multiple_unmethylated(self):
        """Test with multiple unmethylated values."""
        seq = [0, 0, 0, 2]
        result = collapse_methylation(seq)
        self.assertEqual(result, 0)


class TestProcessChunk(unittest.TestCase):
    """Test suite for process_chunk function."""

    def test_process_single_chromosome(self):
        """Test processing a single valid chromosome."""
        chunk = [">CHR1", "ATCGATCGATCG"]
        result = process_chunk(chunk, k=3, seq_len=12, valid_chromosomes=["CHR1"])

        self.assertEqual(len(result), 1)
        self.assertIn("ATC", result[0])

    def test_skip_invalid_chromosome(self):
        """Test that invalid chromosomes are skipped."""
        chunk = [">CHRMT", "ATCGATCGATCG"]  # Mitochondrial, not in valid list
        result = process_chunk(chunk, k=3, seq_len=12, valid_chromosomes=["CHR1"])

        self.assertEqual(len(result), 0)

    def test_skip_sequences_with_n(self):
        """Test that sequences with N are skipped."""
        chunk = [">CHR1", "ATCNGATCG"]
        result = process_chunk(chunk, k=3, seq_len=9, valid_chromosomes=["CHR1"])

        self.assertEqual(len(result), 0)

    def test_multiple_sequences_from_long_line(self):
        """Test splitting long sequence into multiple seq_len chunks."""
        chunk = [">CHR1", "ATCGATCGATCGATCGATCGATCG"]  # 24 bases
        result = process_chunk(chunk, k=3, seq_len=12, valid_chromosomes=["CHR1"])

        self.assertEqual(len(result), 2)  # 24 / 12 = 2 chunks

    def test_concatenate_multiple_lines(self):
        """Test that multiple lines are concatenated before processing."""
        chunk = [">CHR1", "ATCGAT", "CGATCG"]
        result = process_chunk(chunk, k=3, seq_len=12, valid_chromosomes=["CHR1"])

        self.assertEqual(len(result), 1)

    def test_uppercase_conversion(self):
        """Test that lowercase sequences are converted to uppercase."""
        chunk = [">CHR1", "atcgatcgatcg"]
        result = process_chunk(chunk, k=3, seq_len=12, valid_chromosomes=["CHR1"])

        self.assertEqual(len(result), 1)
        # Check that result contains uppercase k-mers
        self.assertTrue(all(c.isupper() for kmer in result[0].split() for c in kmer))

    def test_remainder_handling(self):
        """Test that sequences shorter than seq_len are not lost."""
        chunk = [">CHR1", "ATCGATCGATCGATCG"]  # 16 bases
        result = process_chunk(chunk, k=3, seq_len=12, valid_chromosomes=["CHR1"])

        # Should get 1 chunk of 12, remainder of 4 is too short
        self.assertEqual(len(result), 1)


class TestAddReferenceCpgCounts(unittest.TestCase):
    """Test suite for add_reference_cpg_counts function."""

    def test_returns_empty_dataframe_without_opening_reference(self):
        """Test that an empty input is returned immediately without opening FASTA."""
        empty_df = pd.DataFrame(
            columns=[
                "chromosome",
                "read_start",
                "read_end",
                "methylated_cpgs",
                "unmethylated_cpgs",
            ]
        )

        with patch("methyldl.data.sequencing.genome.pysam.FastaFile") as fasta_cls:
            result = add_reference_cpg_counts(empty_df, "unused.fa", verbose=True)

        self.assertIs(result, empty_df)
        fasta_cls.assert_not_called()

    def test_adds_reference_counts_and_verbose_summary(self):
        """Test that counts and verbose totals are added for a populated DataFrame."""
        df = pd.DataFrame(
            [
                {
                    "chromosome": "chr1",
                    "read_start": 1,
                    "read_end": 6,
                    "methylated_cpgs": 1,
                    "unmethylated_cpgs": 0,
                },
                {
                    "chromosome": "chr2",
                    "read_start": 1,
                    "read_end": 4,
                    "methylated_cpgs": 0,
                    "unmethylated_cpgs": 1,
                },
            ]
        )
        fake_ref = FakeReferenceFasta(
            sequences={
                "chr1": "AACGTCGAACG",
                "chr2": "TTCGAA",
            }
        )

        with patch(
            "methyldl.data.sequencing.genome.pysam.FastaFile", return_value=fake_ref
        ):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = add_reference_cpg_counts(df, "reference.fa", verbose=True)

        self.assertListEqual(result["ref_cpg_count"].tolist(), [2, 1])
        self.assertListEqual(result["unknown_cpgs"].tolist(), [1, 0])
        self.assertTrue(fake_ref.closed)
        self.assertNotIn("ref_cpg_count", df.columns)
        self.assertNotIn("unknown_cpgs", df.columns)

        # The implementation intentionally widens the fetch window with end + 1,
        # so the test locks in the current boundary behavior the user chose to keep.
        self.assertEqual(fake_ref.fetch_calls, [("chr1", 1, 7), ("chr2", 1, 5)])

        output = stdout.getvalue()
        self.assertIn("Counting reference CpGs for 2 fragments...", output)
        self.assertIn("Reference CpG counts - Total: 3, M: 1, U: 1, X: 1", output)

    def test_clips_negative_unknown_counts_when_called_cpgs_exceed_reference(self):
        """Test that negative unknown counts are clipped back to zero."""
        df = pd.DataFrame(
            [
                {
                    "chromosome": "chr1",
                    "read_start": 0,
                    "read_end": 3,
                    "methylated_cpgs": 2,
                    "unmethylated_cpgs": 1,
                }
            ]
        )
        fake_ref = FakeReferenceFasta(sequences={"chr1": "ACGAAA"})

        with patch(
            "methyldl.data.sequencing.genome.pysam.FastaFile", return_value=fake_ref
        ):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = add_reference_cpg_counts(df, "reference.fa", verbose=False)

        self.assertEqual(result.loc[0, "ref_cpg_count"], 1)
        self.assertEqual(result.loc[0, "unknown_cpgs"], 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(fake_ref.closed)

    def test_uses_zero_reference_count_when_lookup_fails(self):
        """Test that lookup failures fall back to zero reference CpGs."""
        df = pd.DataFrame(
            [
                {
                    "chromosome": "chr_missing",
                    "read_start": 0,
                    "read_end": 3,
                    "methylated_cpgs": 0,
                    "unmethylated_cpgs": 0,
                }
            ]
        )
        fake_ref = FakeReferenceFasta(sequences={"chr1": "ACGAAA"})

        with patch(
            "methyldl.data.sequencing.genome.pysam.FastaFile", return_value=fake_ref
        ):
            result = add_reference_cpg_counts(df, "reference.fa", verbose=False)

        # A missing chromosome makes the fake raise, exercising the helper's
        # broad exception path without depending on a real FASTA/index pair.
        self.assertEqual(result.loc[0, "ref_cpg_count"], 0)
        self.assertEqual(result.loc[0, "unknown_cpgs"], 0)
        self.assertEqual(fake_ref.fetch_calls, [("chr_missing", 0, 4)])
        self.assertTrue(fake_ref.closed)


if __name__ == "__main__":
    unittest.main()
