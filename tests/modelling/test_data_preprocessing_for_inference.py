"""Tests for inference-time preprocessing helpers."""

import unittest
from unittest.mock import patch

import pandas as pd

from syto.classification.data_preprocessing_for_inference import (
    chunk_tokens,
    generate_valid_tokens,
    prepare_methylbert_list_inference,
)


class TestChunkTokens(unittest.TestCase):
    """Tests for the token chunking helper."""

    def test_returns_original_tokens_when_sequence_is_shorter_than_window(self):
        """Returns the input as a single chunk when the read is shorter than the window."""
        tokens = ["a", "b", "c"]

        chunks = list(chunk_tokens(tokens, window_size=5, stride=2))

        self.assertEqual(chunks, [tokens])

    def test_returns_original_tokens_when_sequence_matches_window_size(self):
        """Returns one chunk when the read length matches the window size exactly."""
        tokens = ["a", "b", "c", "d"]

        chunks = list(chunk_tokens(tokens, window_size=4, stride=2))

        self.assertEqual(chunks, [tokens])

    def test_yields_sliding_windows_without_tail_when_last_window_already_reaches_end(
        self,
    ):
        """Avoids duplicating the final window when the sliding loop already lands on the tail."""
        tokens = list(range(9))

        chunks = list(chunk_tokens(tokens, window_size=5, stride=2))

        self.assertEqual(chunks, [tokens[0:5], tokens[2:7], tokens[4:9]])

    def test_yields_tail_window_when_stride_skips_the_last_possible_start(self):
        """Emits the final tail window when the last valid start is missed by the stride."""
        tokens = list(range(8))

        chunks = list(chunk_tokens(tokens, window_size=5, stride=2))

        # This is the regression case: the last possible window starts at index 3,
        # but the stride visits only starts 0 and 2, so an explicit tail chunk is required.
        self.assertEqual(chunks, [tokens[0:5], tokens[2:7], tokens[3:8]])


class TestGenerateValidTokens(unittest.TestCase):
    """Tests for k-mer generation with methylation labels."""

    def test_yields_only_kmers_without_n_bases(self):
        """Skips windows containing N and keeps the methylation code at the center base."""
        read_data = {
            "input_ids": "ATNCGA",
            "methylation_ids": "012210",
        }

        tokens = list(generate_valid_tokens(read_data, k=3))

        self.assertEqual(tokens, [["CGA", "1"]])

    def test_returns_empty_iterator_when_no_valid_kmers_exist(self):
        """Produces no output when every candidate k-mer contains N."""
        read_data = {
            "input_ids": "NNNN",
            "methylation_ids": "0123",
        }

        tokens = list(generate_valid_tokens(read_data, k=3))

        self.assertEqual(tokens, [])

    def test_uses_requested_kmer_size(self):
        """Supports k-mer lengths other than the default size of three."""
        read_data = {
            "input_ids": "ATCG",
            "methylation_ids": "0123",
        }

        tokens = list(generate_valid_tokens(read_data, k=2))

        self.assertEqual(tokens, [["AT", "1"], ["TC", "2"], ["CG", "3"]])


class TestPrepareMethylbertListInference(unittest.TestCase):
    """Tests for MethylBERT inference table generation."""

    def test_builds_header_and_skips_reads_without_valid_tokens(self):
        """Keeps the header row and ignores reads that become empty after N filtering."""
        results_df = pd.DataFrame(
            [
                {
                    "read_name": "read-empty",
                    "input_ids": "NNNN",
                    "methylation_ids": "0123",
                    "dmr_ctype_label": "tumor",
                    "dmr_label": "dmr-a",
                }
            ]
        )

        prepared = prepare_methylbert_list_inference(
            results_df, dmr_label_column="dmr_label", seq_length=3, stride=1
        )

        self.assertEqual(
            prepared,
            [
                [
                    "dna_seq",
                    "methyl_seq",
                    "dmr_ctype",
                    "dmr_label",
                    "ctype",
                    "original_label",
                    "read_name",
                    "ncpgs_marked",
                ]
            ],
        )

    def test_creates_one_output_row_per_chunk_with_metadata_propagation(self):
        """Splits valid tokens into chunks and propagates DMR metadata to each output row."""
        results_df = pd.DataFrame(
            [
                {
                    "read_name": "read-1",
                    "input_ids": "ATCGAT",
                    "methylation_ids": "010101",
                    "dmr_ctype_label": "tumor",
                    "dmr_label": "dmr-a",
                }
            ]
        )

        prepared = prepare_methylbert_list_inference(
            results_df, dmr_label_column="dmr_label", seq_length=2, stride=1
        )

        self.assertEqual(len(prepared), 4)
        self.assertEqual(
            prepared[0],
            [
                "dna_seq",
                "methyl_seq",
                "dmr_ctype",
                "dmr_label",
                "ctype",
                "original_label",
                "read_name",
                "ncpgs_marked",
            ],
        )
        self.assertEqual(
            prepared[1:],
            [
                ["ATC TCG", "10", "tumor", "dmr-a", 0, 0, "read-1", 2],
                ["TCG CGA", "01", "tumor", "dmr-a", 0, 0, "read-1", 2],
                ["CGA GAT", "10", "tumor", "dmr-a", 0, 0, "read-1", 2],
            ],
        )

    def test_counts_only_zero_and_one_as_marked_cpgs(self):
        """Excludes other methylation symbols from the ncpgs_marked tally."""
        results_df = pd.DataFrame(
            [
                {
                    "read_name": "read-2",
                    "input_ids": "ATCGA",
                    "methylation_ids": "12010",
                    "dmr_ctype_label": "normal",
                    "dmr_label": "dmr-b",
                }
            ]
        )

        prepared = prepare_methylbert_list_inference(
            results_df, dmr_label_column="dmr_label", seq_length=10, stride=5
        )

        self.assertEqual(prepared[1][-1], 2)


if __name__ == "__main__":
    unittest.main()
