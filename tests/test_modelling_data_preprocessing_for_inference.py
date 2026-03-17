"""Tests for inference-time preprocessing helpers."""

import unittest
from unittest.mock import patch

import pandas as pd

from methyldl.modelling.data_preprocessing_for_inference import (
    chunk_read_data,
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
                ["ATC TCG", "10", "tumor", "dmr-a", 39, 39, "read-1", 2],
                ["TCG CGA", "01", "tumor", "dmr-a", 39, 39, "read-1", 2],
                ["CGA GAT", "10", "tumor", "dmr-a", 39, 39, "read-1", 2],
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


class TestChunkReadData(unittest.TestCase):
    """Tests for read chunking and DMR clipping."""

    def test_returns_no_chunks_when_no_dmrs_overlap(self):
        """Drops chunks entirely when the overlap helper returns no DMRs."""
        dmr_trees = {"chr1": object()}

        with patch(
            "methyldl.modelling.data_preprocessing_for_inference.get_overlapping_dmrs",
            return_value=[],
        ) as mock_get_overlaps:
            chunks = chunk_read_data(
                seq="ACGTAC",
                methylation_encoding="010101",
                cpg_positions=[101, 103],
                meth_states=[1, 0],
                read_start=100,
                read_end=106,
                chrom="chr1",
                chunk_size=6,
                dmr_trees=dmr_trees,
                strict=False,
            )

        self.assertEqual(chunks, [])
        mock_get_overlaps.assert_called_once_with(100, 106, "chr1", dmr_trees, False)

    def test_creates_chunk_with_clipped_sequence_and_cpg_statistics(self):
        """Builds a result row for an overlapping DMR and clips sequence data to the overlap."""
        overlap = [
            {
                "name": "dmr-a",
                "type": "tumor",
                "dmr_start": 102,
                "dmr_end": 105,
                "overlap_bp": 3,
                "overlap_pct": 100.0,
            }
        ]

        with patch(
            "methyldl.modelling.data_preprocessing_for_inference.get_overlapping_dmrs",
            return_value=overlap,
        ):
            chunks = chunk_read_data(
                seq="ACGTAC",
                methylation_encoding="012210",
                cpg_positions=[101, 102, 104, 105],
                meth_states=[1, 0, 1, 0],
                read_start=100,
                read_end=106,
                chrom="chr7",
                chunk_size=6,
                dmr_trees={},
                strict=True,
            )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["seq"], "ACGTAC")
        self.assertEqual(chunk["methylation_encoding"], "012210")
        self.assertEqual(chunk["chunk_start"], 100)
        self.assertEqual(chunk["chunk_end"], 106)
        self.assertEqual(chunk["chunk_length"], 6)
        self.assertEqual(chunk["total_cpgs"], 4)
        self.assertEqual(chunk["methylated_cpgs"], 2)
        self.assertEqual(chunk["unmethylated_cpgs"], 2)
        self.assertEqual(chunk["methylation_rate"], 0.5)
        self.assertEqual(chunk["seq_clipped"], "GTA")
        self.assertEqual(chunk["methylation_clipped"], "221")
        self.assertEqual(chunk["clip_start"], 102)
        self.assertEqual(chunk["clip_end"], 105)
        self.assertEqual(chunk["clip_length"], 3)
        self.assertEqual(chunk["clipped_total"], 2)
        self.assertEqual(chunk["clipped_methylated"], 1)
        self.assertEqual(chunk["clipped_unmethylated"], 1)
        self.assertEqual(chunk["clipped_meth_rate"], 0.5)
        self.assertTrue(chunk["overlaps_dmr"])
        self.assertEqual(chunk["dmr_label"], "dmr-a")
        self.assertEqual(chunk["dmr_type"], "tumor")
        self.assertEqual(chunk["dmr_start"], 102)
        self.assertEqual(chunk["dmr_end"], 105)
        self.assertEqual(chunk["overlap_bp"], 3)
        self.assertEqual(chunk["overlap_pct"], 100.0)

    def test_creates_one_output_per_overlapping_dmr(self):
        """Duplicates the chunk once per overlapping DMR while preserving DMR-specific metadata."""
        overlaps = [
            {
                "name": "dmr-a",
                "type": "tumor",
                "dmr_start": 100,
                "dmr_end": 104,
                "overlap_bp": 4,
                "overlap_pct": 100.0,
            },
            {
                "name": "dmr-b",
                "type": "normal",
                "dmr_start": 101,
                "dmr_end": 103,
                "overlap_bp": 2,
                "overlap_pct": 50.0,
            },
        ]

        with patch(
            "methyldl.modelling.data_preprocessing_for_inference.get_overlapping_dmrs",
            return_value=overlaps,
        ):
            chunks = chunk_read_data(
                seq="ACGT",
                methylation_encoding="0101",
                cpg_positions=[100, 101, 102, 103],
                meth_states=[0, 1, 0, 1],
                read_start=100,
                read_end=104,
                chrom="chr2",
                chunk_size=4,
                dmr_trees={},
                strict=False,
            )

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk["dmr_label"] for chunk in chunks], ["dmr-a", "dmr-b"])
        self.assertEqual([chunk["seq_clipped"] for chunk in chunks], ["ACGT", "CG"])

    def test_processes_multiple_chunks_and_handles_zero_cpg_regions(self):
        """Evaluates per-chunk overlap data independently across a multi-chunk read."""

        def overlap_side_effect(chunk_start, chunk_end, chrom, dmr_trees, strict):
            """Return different DMR matches for each genomic chunk to cover both branches."""
            if (chunk_start, chunk_end) == (100, 104):
                return [
                    {
                        "name": "dmr-left",
                        "type": "tumor",
                        "dmr_start": 101,
                        "dmr_end": 103,
                        "overlap_bp": 2,
                        "overlap_pct": 50.0,
                    }
                ]
            if (chunk_start, chunk_end) == (104, 108):
                return [
                    {
                        "name": "dmr-right",
                        "type": "normal",
                        "dmr_start": 104,
                        "dmr_end": 108,
                        "overlap_bp": 4,
                        "overlap_pct": 100.0,
                    }
                ]
            return []

        with patch(
            "methyldl.modelling.data_preprocessing_for_inference.get_overlapping_dmrs",
            side_effect=overlap_side_effect,
        ):
            chunks = chunk_read_data(
                seq="ACGTTGCA",
                methylation_encoding="01012222",
                cpg_positions=[101, 102],
                meth_states=[1, 0],
                read_start=100,
                read_end=108,
                chrom="chr3",
                chunk_size=4,
                dmr_trees={},
                strict=False,
            )

        self.assertEqual(len(chunks), 2)

        left_chunk, right_chunk = chunks

        # The left chunk contains two CpGs, but only the clipped subregion should contribute
        # to the UXM-specific clipped counters.
        self.assertEqual(left_chunk["chunk_offset_in_read"], 0)
        self.assertEqual(left_chunk["seq_clipped"], "CG")
        self.assertEqual(left_chunk["clipped_total"], 2)
        self.assertEqual(left_chunk["clipped_methylated"], 1)
        self.assertEqual(left_chunk["clipped_unmethylated"], 1)

        # The right chunk has no CpGs at all, which should drive both overall and clipped
        # methylation rates to zero without raising a division-by-zero error.
        self.assertEqual(right_chunk["chunk_offset_in_read"], 4)
        self.assertEqual(right_chunk["total_cpgs"], 0)
        self.assertEqual(right_chunk["methylation_rate"], 0.0)
        self.assertEqual(right_chunk["clipped_total"], 0)
        self.assertEqual(right_chunk["clipped_meth_rate"], 0.0)
        self.assertEqual(right_chunk["seq_clipped"], "TGCA")


if __name__ == "__main__":
    unittest.main()
