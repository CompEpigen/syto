"""Tests for bam_processing utilities and orchestration paths."""

import unittest
from unittest.mock import patch

import pandas as pd

from methyldl.data.sequencing import bam_processing as bp


class FakeRead:
    """Minimal stand-in for a pysam read used by bam_processing tests."""

    def __init__(
        self,
        query_name="read1",
        reference_name="chr1",
        reference_start=100,
        reference_end=106,
        mapping_quality=60,
        flag=3,
        is_reverse=False,
        is_unmapped=False,
        tags=None,
        forward_sequence="ACGCGT",
        query_sequence=None,
        query_alignment_sequence=None,
        cigartuples=None,
    ):
        """Store the attributes and tag accessors used by the production code."""
        self.query_name = query_name
        self.reference_name = reference_name
        self.reference_start = reference_start
        self.reference_end = reference_end
        self.mapping_quality = mapping_quality
        self.flag = flag
        self.is_reverse = is_reverse
        self.is_unmapped = is_unmapped
        self.tags = tags or {}
        self._forward_sequence = forward_sequence
        self.query_sequence = (
            query_sequence if query_sequence is not None else forward_sequence
        )
        self.query_alignment_sequence = (
            query_alignment_sequence
            if query_alignment_sequence is not None
            else forward_sequence
        )
        self.cigartuples = cigartuples

    def get_tag(self, tag_name):
        """Return a stored tag value or raise KeyError like pysam does."""
        if tag_name not in self.tags:
            raise KeyError(tag_name)
        return self.tags[tag_name]

    def get_forward_sequence(self):
        """Return the forward-strand sequence expected by ONT processing."""
        return self._forward_sequence


class FakeAlignmentFile:
    """Context-manager stub for pysam.AlignmentFile."""

    def __init__(self, reads=None, references=None, lengths=None, fetch_reads=None):
        """Capture iterable reads and fetch results for a fake BAM handle."""
        self._reads = list(reads or [])
        self.references = tuple(references or [])
        self.lengths = tuple(lengths or [])
        self._fetch_reads = list(fetch_reads or [])

    def __enter__(self):
        """Enter the fake context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the fake context manager without suppressing exceptions."""
        return False

    def __iter__(self):
        """Iterate over all stored reads."""
        return iter(self._reads)

    def fetch(self, chrom, start, end):
        """Return the configured reads for a genomic fetch call."""
        return iter(self._fetch_reads)


class FakeFastaFile:
    """Context-free stub for pysam.FastaFile."""

    def __init__(self, sequence_by_region=None, should_raise=False):
        """Store lookup behavior and whether fetch should fail."""
        self.sequence_by_region = sequence_by_region or {}
        self.should_raise = should_raise
        self.closed = False

    def fetch(self, chrom, start, end):
        """Return a configured reference slice or raise to simulate I/O errors."""
        if self.should_raise:
            raise RuntimeError("fetch failed")
        return self.sequence_by_region[(chrom, start, end)]

    def close(self):
        """Record that the fasta handle was closed."""
        self.closed = True


class FakePool:
    """Simple multiprocessing.Pool stand-in for deterministic tests."""

    def __init__(self, n_jobs, map_result):
        """Store the worker count and precomputed map result."""
        self.n_jobs = n_jobs
        self.map_result = map_result
        self.received_func = None
        self.received_tasks = None

    def __enter__(self):
        """Enter the fake pool context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the fake pool context without suppressing exceptions."""
        return False

    def map(self, func, tasks):
        """Record the dispatched work and return the configured result."""
        self.received_func = func
        self.received_tasks = list(tasks)
        return self.map_result


def build_read(**kwargs):
    """Create a fake read with sensible defaults that tests can override."""
    return FakeRead(**kwargs)


class TestDetectBamDataType(unittest.TestCase):
    """Test BAM type auto-detection based on ML tag prevalence."""

    def test_detects_ont_when_majority_of_mapped_reads_have_ml(self):
        """Return ONT when more than half of sampled mapped reads expose ML tags."""
        reads = [
            build_read(tags={"ML": [100]}),
            build_read(query_name="read2", tags={"ML": [120]}),
            build_read(query_name="read3", tags={}),
        ]

        with patch.object(
            bp.pysam, "AlignmentFile", return_value=FakeAlignmentFile(reads=reads)
        ):
            result = bp.detect_bam_data_type("fake.bam", sample_size=3)

        self.assertEqual(result, "ont")

    def test_detects_wgbs_when_half_or_fewer_reads_have_ml(self):
        """Return WGBS when the ML-tag ratio does not cross the ONT threshold."""
        reads = [
            build_read(tags={"ML": [100]}),
            build_read(query_name="read2", tags={}),
        ]

        with patch.object(
            bp.pysam, "AlignmentFile", return_value=FakeAlignmentFile(reads=reads)
        ):
            result = bp.detect_bam_data_type("fake.bam", sample_size=2)

        self.assertEqual(result, "wgbs")

    def test_raises_when_no_mapped_reads_are_available(self):
        """Reject BAM files that contain no mapped reads in the inspected sample."""
        reads = [
            build_read(is_unmapped=True),
            build_read(query_name="read2", is_unmapped=True),
        ]

        with patch.object(
            bp.pysam, "AlignmentFile", return_value=FakeAlignmentFile(reads=reads)
        ):
            with self.assertRaisesRegex(ValueError, "No mapped reads found"):
                bp.detect_bam_data_type("fake.bam")


class TestParseMmTag(unittest.TestCase):
    """Test MM-tag parsing for mixed modification groups and offsets."""

    def test_returns_none_for_empty_tag(self):
        """Return None when no MM tag is present."""
        self.assertIsNone(bp.parse_mm_tag(""))
        self.assertIsNone(bp.parse_mm_tag(None))

    def test_parses_multiple_modification_groups_and_cpg_offsets(self):
        """Extract the CpG modification index, offset, and counts from a mixed MM tag."""
        mm_info = bp.parse_mm_tag("A+a,10,5;C+m?,2,1,4;G+g,5;")

        self.assertEqual(
            mm_info["modifications"],
            [("A", "a", [10, 5]), ("C", "m?", [2, 1, 4]), ("G", "g", [5])],
        )
        self.assertEqual(mm_info["cpg_mod_index"], 1)
        self.assertEqual(mm_info["cpg_ml_offset"], 2)
        self.assertEqual(mm_info["cpg_count"], 3)
        self.assertEqual(mm_info["total_mods"], 6)

    def test_skips_invalid_groups_and_reports_missing_cpg_modification(self):
        """Ignore malformed groups and leave the CpG index unset when no C+m group exists."""
        mm_info = bp.parse_mm_tag("not-a-group;C+h,1,2;A+a,3")

        self.assertEqual(
            mm_info["modifications"], [("C", "h", [1, 2]), ("A", "a", [3])]
        )
        self.assertEqual(mm_info["cpg_mod_index"], -1)
        self.assertEqual(mm_info["cpg_ml_offset"], 0)
        self.assertEqual(mm_info["cpg_count"], 0)
        self.assertEqual(mm_info["total_mods"], 3)


class TestCpgScans(unittest.TestCase):
    """Test low-level CpG scanning helpers for ONT and WGBS reads."""

    def test_cpg_scan_classifies_methylated_unmethylated_and_missing_calls(self):
        """Map ML values to CpG states and mark uncovered CpGs as missing."""
        positions, states = bp.cpg_scan(b"ACGCGT", [200], tr=122)

        self.assertEqual(positions, [1, 3])
        self.assertEqual(states, [1, 2])

        positions, states = bp.cpg_scan(b"ACGCGT", [200, 100], tr=122)
        self.assertEqual(positions, [1, 3])
        self.assertEqual(states, [1, 0])

    def test_cpg_scan_wgbs_forward_handles_all_state_types(self):
        """Classify forward-strand WGBS CpGs as methylated, unmethylated, or ambiguous."""
        positions, states = bp.cpg_scan_wgbs_forward(b"ACGCG", b"ACT")

        self.assertEqual(positions, [1, 3])
        self.assertEqual(states, [1, 2])

        positions, states = bp.cpg_scan_wgbs_forward(b"ACGCG", b"ATTTT")
        self.assertEqual(states, [0, 0])

    def test_cpg_scan_wgbs_reverse_handles_all_state_types(self):
        """Classify reverse-strand WGBS CpGs using the base paired to the reference G."""
        positions, states = bp.cpg_scan_wgbs_reverse(b"ACGCG", b"AGG")

        self.assertEqual(positions, [1, 3])
        self.assertEqual(states, [1, 2])

        positions, states = bp.cpg_scan_wgbs_reverse(b"ACGCG", b"AAAAA")
        self.assertEqual(states, [0, 0])


class TestExtractCpgMlValues(unittest.TestCase):
    """Test extraction of CpG-specific ML values from mixed modification arrays."""

    def test_returns_empty_array_when_no_cpg_modification_is_present(self):
        """Return an empty array for missing MM information or non-CpG modifications."""
        self.assertEqual(bp.extract_cpg_ml_values([1, 2], None).tolist(), [])
        self.assertEqual(
            bp.extract_cpg_ml_values(
                [1, 2], {"cpg_mod_index": -1, "cpg_ml_offset": 0, "cpg_count": 0}
            ).tolist(),
            [],
        )

    def test_extracts_full_or_truncated_cpg_slice(self):
        """Slice CpG ML values and gracefully truncate when the ML array is shorter than expected."""
        mm_info = {"cpg_mod_index": 1, "cpg_ml_offset": 2, "cpg_count": 3}

        self.assertEqual(
            bp.extract_cpg_ml_values([10, 11, 12, 13, 14], mm_info).tolist(),
            [12, 13, 14],
        )
        self.assertEqual(bp.extract_cpg_ml_values([10, 11, 12], mm_info).tolist(), [12])


class TestProcessSingleReadOnt(unittest.TestCase):
    """Test ONT-specific read processing, filtering, and methylation encoding."""

    def test_filters_reads_by_chromosome_flags_and_mapping_quality(self):
        """Skip reads that fail the chromosome, SAM flag, or MAPQ filters."""
        read = build_read()

        self.assertEqual(
            bp.process_single_read(read, "ont", interesting_chromosomes=["chr2"]),
            [],
        )
        self.assertEqual(bp.process_single_read(build_read(flag=1), "ont"), [])
        self.assertEqual(bp.process_single_read(build_read(flag=1796), "ont"), [])
        self.assertEqual(
            bp.process_single_read(build_read(mapping_quality=5), "ont"), []
        )

    def test_returns_empty_when_required_ont_tags_or_cpg_calls_are_missing(self):
        """Skip ONT reads without MM or ML tags, without C+m groups, or without enough CpGs."""
        self.assertEqual(bp.process_single_read(build_read(tags={}), "ont"), [])

        no_cpg_read = build_read(tags={"ML": [200], "MM": "A+a,0"})
        self.assertEqual(bp.process_single_read(no_cpg_read, "ont"), [])

        no_ml_values = build_read(tags={"ML": [], "MM": "C+m,0"})
        self.assertEqual(bp.process_single_read(no_ml_values, "ont"), [])

        too_few_cpgs = build_read(
            forward_sequence="ACGTAA",
            tags={"ML": [200], "MM": "C+m,0"},
        )
        self.assertEqual(bp.process_single_read(too_few_cpgs, "ont", min_cpgs=2), [])

    def test_returns_processed_ont_read_with_expected_statistics(self):
        """Build the read-level methylation payload for a valid ONT read."""
        read = build_read(
            query_name="ont-read",
            reference_start=100,
            reference_end=106,
            forward_sequence="ACGCGT",
            tags={"ML": [200, 100], "MM": "C+m,0,0"},
        )

        result = bp.process_single_read(read, "ont")

        self.assertEqual(len(result), 1)
        read_data = result[0]
        self.assertEqual(read_data["read_name"], "ont-read")
        self.assertEqual(read_data["cpg_positions"], [101, 103])
        self.assertEqual(read_data["meth_states"], [1, 0])
        self.assertEqual(read_data["methylation_encoding"], "212022")
        self.assertEqual(read_data["total_cpgs"], 2)
        self.assertEqual(read_data["methylated_cpgs"], 1)
        self.assertEqual(read_data["unmethylated_cpgs"], 1)
        self.assertAlmostEqual(read_data["methylation_rate"], 0.5)
        self.assertEqual(read_data["read_length"], 6)


class TestProcessSingleReadWgbs(unittest.TestCase):
    """Test WGBS read processing for trimming, scanning, and error handling."""

    def test_requires_reference_for_wgbs_processing(self):
        """Raise an error when WGBS processing is requested without a reference genome."""
        with self.assertRaisesRegex(ValueError, "Reference genome"):
            bp.process_single_read(build_read(), "wgbs", ref_fasta=None)

    def test_returns_empty_when_reference_fetch_or_cigar_cleaning_fails(self):
        """Skip WGBS reads when reference retrieval fails or the read cannot be aligned to reference coordinates."""
        read = build_read()

        self.assertEqual(
            bp.process_single_read(
                read,
                "wgbs",
                ref_fasta=FakeFastaFile(should_raise=True),
            ),
            [],
        )

        ref_fasta = FakeFastaFile({("chr1", 100, 107): "ACGTAAA"})
        with patch.object(bp, "clean_cigar_sequence", return_value=None):
            self.assertEqual(
                bp.process_single_read(read, "wgbs", ref_fasta=ref_fasta), []
            )

    def test_trims_non_cg_lookahead_base_before_forward_scan(self):
        """Drop the extra fetched base when it does not preserve a terminal CpG."""
        read = build_read(reference_end=104, is_reverse=False)
        ref_fasta = FakeFastaFile({("chr1", 100, 105): "AACCA"})

        with patch.object(
            bp, "clean_cigar_sequence", return_value="TTTT"
        ) as clean_mock, patch.object(
            bp, "cpg_scan_wgbs_forward", return_value=([1], [0])
        ) as scan_mock:
            result = bp.process_single_read(read, "wgbs", ref_fasta=ref_fasta)

        clean_mock.assert_called_once_with(read)
        # The extra fetched base is just lookahead, so the test asserts that it is removed
        # before scan dispatch when the suffix is not a terminal CpG.
        self.assertEqual(scan_mock.call_args.args[0], b"AACC")
        self.assertEqual(scan_mock.call_args.args[1], b"TTTT")
        self.assertEqual(result[0]["seq"], "AACC")
        self.assertEqual(result[0]["methylation_encoding"], "2022")

    def test_keeps_cg_lookahead_base_and_uses_reverse_scanner(self):
        """Retain the extra fetched base when it exposes a terminal CpG on reverse reads."""
        read = build_read(reference_end=104, is_reverse=True)
        ref_fasta = FakeFastaFile({("chr1", 100, 105): "AACG"})

        with patch.object(
            bp, "clean_cigar_sequence", return_value="GGGG"
        ), patch.object(
            bp, "cpg_scan_wgbs_reverse", return_value=([2], [1])
        ) as scan_mock:
            result = bp.process_single_read(read, "wgbs", ref_fasta=ref_fasta)

        self.assertEqual(scan_mock.call_args.args[0], b"AACG")
        self.assertEqual(result[0]["seq"], "AACG")
        self.assertEqual(result[0]["methylation_encoding"], "2212")
        self.assertEqual(result[0]["meth_states"], [1])
        self.assertTrue(result[0]["is_reverse"])

    def test_applies_min_cpg_filter_after_wgbs_scan(self):
        """Skip WGBS reads that do not yield enough CpG calls after scanning."""
        read = build_read(reference_end=104, is_reverse=False)
        ref_fasta = FakeFastaFile({("chr1", 100, 105): "AACCA"})

        with patch.object(
            bp, "clean_cigar_sequence", return_value="TTTT"
        ), patch.object(bp, "cpg_scan_wgbs_forward", return_value=([1], [1])):
            result = bp.process_single_read(
                read, "wgbs", ref_fasta=ref_fasta, min_cpgs=2
            )

        self.assertEqual(result, [])


class TestProcessTabularChunk(unittest.TestCase):
    """Test chunk-level BAM processing and chunk boundary handling."""

    def test_requires_reference_path_for_wgbs_chunks(self):
        """Raise an error when a WGBS chunk is requested without a reference FASTA path."""
        args = (
            "chr1",
            0,
            100,
            "fake.bam",
            ["chr1"],
            False,
            122,
            "wgbs",
            None,
            10,
            3,
            1796,
            1,
        )

        with self.assertRaisesRegex(ValueError, "Reference genome path"):
            bp.process_tabular_chunk(args)

    def test_processes_only_reads_starting_inside_the_chunk(self):
        """Skip overlapping reads that started in a previous chunk and close the FASTA handle afterward."""
        reads = [
            build_read(query_name="skip", reference_start=95),
            build_read(query_name="keep1", reference_start=100),
            build_read(query_name="keep2", reference_start=150),
            build_read(query_name="skip-end", reference_start=200),
        ]
        fake_bam = FakeAlignmentFile(fetch_reads=reads)
        fake_fasta = FakeFastaFile()

        args = (
            "chr1",
            100,
            200,
            "fake.bam",
            ["chr1"],
            False,
            122,
            "wgbs",
            "ref.fa",
            10,
            3,
            1796,
            1,
        )

        with patch.object(
            bp.pysam, "AlignmentFile", return_value=fake_bam
        ), patch.object(bp.pysam, "FastaFile", return_value=fake_fasta), patch.object(
            bp,
            "process_single_read",
            side_effect=lambda **kwargs: [{"read_name": kwargs["read"].query_name}],
        ) as process_mock:
            result = bp.process_tabular_chunk(args)

        # bam.fetch returns overlapping reads, so the chunk code must gate on reference_start
        # to avoid double-counting the same fragment in neighboring chunks.
        self.assertEqual(result, [{"read_name": "keep1"}, {"read_name": "keep2"}])
        self.assertEqual(
            [call.kwargs["read"].query_name for call in process_mock.call_args_list],
            ["keep1", "keep2"],
        )
        self.assertTrue(fake_fasta.closed)


class TestProcessBamWithChunking(unittest.TestCase):
    """Test BAM-level orchestration, task generation, and optional pair merging."""

    def test_requires_reference_path_for_wgbs_bam_processing(self):
        """Reject WGBS processing when no reference genome path is supplied."""
        with self.assertRaisesRegex(ValueError, "Reference genome path"):
            bp.process_bam_with_chunking(
                bam_path="fake.bam",
                chromosomes=["chr1"],
                data_type="wgbs",
                reference_path=None,
            )

    def test_auto_detects_type_generates_tasks_and_merges_pairs_serially(self):
        """Auto-detect the BAM type, process chunks serially, and merge paired reads when requested."""
        fake_bam = FakeAlignmentFile(references=["chr1"], lengths=[250])
        chunk_results = [
            [{"read_name": "r1", "value": 1}],
            [{"read_name": "r2", "value": 2}],
            [{"read_name": "r3", "value": 3}],
        ]
        merged_df = pd.DataFrame([{"read_name": "merged", "value": 99}])

        with patch.object(
            bp, "detect_bam_data_type", return_value="ont"
        ) as detect_mock, patch.object(
            bp.pysam, "AlignmentFile", return_value=fake_bam
        ), patch.object(
            bp, "process_tabular_chunk", side_effect=chunk_results
        ) as chunk_mock, patch.object(
            bp, "merge_paired_reads", return_value=merged_df
        ) as merge_mock:
            result = bp.process_bam_with_chunking(
                bam_path="fake.bam",
                chromosomes=["chr1", "chr-missing"],
                n_jobs=1,
                chunk_size_genomic=100,
                data_type=None,
                merge_pairs=True,
            )

        detect_mock.assert_called_once_with("fake.bam")
        self.assertEqual(chunk_mock.call_count, 3)
        self.assertEqual(len(chunk_mock.call_args_list[0].args[0]), 13)
        merge_mock.assert_called_once()
        self.assertTrue(result.equals(merged_df))

    def test_uses_pool_for_parallel_chunk_processing_without_merging(self):
        """Dispatch chunk tasks through multiprocessing when more than one worker is requested."""
        fake_bam = FakeAlignmentFile(references=["chr1"], lengths=[150])
        pool_instance = FakePool(2, [[{"read_name": "r1"}], [{"read_name": "r2"}]])

        def pool_factory(n_jobs, *args, **kwargs):
            """Return the prebuilt fake pool so the test can inspect dispatched tasks."""
            self.assertEqual(n_jobs, 2)
            return pool_instance

        with patch("multiprocessing.pool.Pool", side_effect=pool_factory), patch.object(
            bp.pysam, "AlignmentFile", return_value=fake_bam
        ), patch.object(bp, "merge_paired_reads") as merge_mock:
            result = bp.process_bam_with_chunking(
                bam_path="fake.bam",
                chromosomes=["chr1"],
                n_jobs=2,
                chunk_size_genomic=100,
                data_type="ont",
                merge_pairs=False,
            )

        # The important observable behavior is that the parallel path returns the flattened
        # chunk results unchanged when pair merging is disabled.
        merge_mock.assert_not_called()
        self.assertEqual(result["read_name"].tolist(), ["r1", "r2"])


class TestCleanCigarSequence(unittest.TestCase):
    """Test CIGAR-driven read-to-reference sequence normalization."""

    def test_returns_alignment_sequence_when_cigar_is_missing(self):
        """Fall back to the aligned query sequence when no CIGAR is available."""
        read = build_read(query_alignment_sequence="ALIGN", cigartuples=None)
        self.assertEqual(bp.clean_cigar_sequence(read), "ALIGN")

    def test_handles_match_insertion_deletion_skip_clips_and_mismatch_ops(self):
        """Normalize the query sequence across the supported CIGAR operations."""
        read = build_read(
            query_sequence="abCDefGHI",
            cigartuples=[
                (4, 2),
                (0, 2),
                (1, 2),
                (2, 1),
                (3, 1),
                (5, 1),
                (7, 2),
                (8, 1),
            ],
        )

        self.assertEqual(bp.clean_cigar_sequence(read), "CDNNGHI")


class TestMergePairedReads(unittest.TestCase):
    """Test dataframe-level mate grouping for singleton, pair, and multiplet cases."""

    def test_returns_empty_dataframe_unchanged(self):
        """Leave empty dataframes untouched."""
        df = pd.DataFrame(columns=["read_name"])
        result = bp.merge_paired_reads(df, verbose=False)
        self.assertTrue(result.equals(df))

    def test_groups_by_read_name_and_dmr_label_when_present(self):
        """Merge exact mate pairs per DMR while preserving singletons and multiplets."""
        df = pd.DataFrame(
            [
                {
                    "read_name": "single",
                    "dmr_label": "A",
                    "read_start": 10,
                    "payload": "single",
                },
                {
                    "read_name": "pair",
                    "dmr_label": "A",
                    "read_start": 20,
                    "payload": "mate2",
                },
                {
                    "read_name": "pair",
                    "dmr_label": "A",
                    "read_start": 10,
                    "payload": "mate1",
                },
                {
                    "read_name": "multi",
                    "dmr_label": "A",
                    "read_start": 1,
                    "payload": "m1",
                },
                {
                    "read_name": "multi",
                    "dmr_label": "A",
                    "read_start": 2,
                    "payload": "m2",
                },
                {
                    "read_name": "multi",
                    "dmr_label": "A",
                    "read_start": 3,
                    "payload": "m3",
                },
            ]
        )

        with patch.object(
            bp, "_merge_mate_pair", return_value={"read_name": "pair-merged"}
        ) as merge_mock:
            result = bp.merge_paired_reads(df, verbose=False)

        self.assertEqual(len(result), 5)
        self.assertIn("pair-merged", result["read_name"].tolist())
        self.assertEqual(merge_mock.call_args.args[0]["payload"], "mate1")
        self.assertEqual(merge_mock.call_args.args[1]["payload"], "mate2")

    def test_groups_only_by_read_name_when_dmr_columns_are_absent(self):
        """Fallback to read-name grouping when no DMR annotation columns exist."""
        df = pd.DataFrame(
            [
                {"read_name": "pair", "read_start": 8, "payload": "mate2"},
                {"read_name": "pair", "read_start": 5, "payload": "mate1"},
            ]
        )

        with patch.object(
            bp, "_merge_mate_pair", return_value={"read_name": "pair-merged"}
        ) as merge_mock:
            result = bp.merge_paired_reads(df, verbose=False)

        self.assertEqual(result["read_name"].tolist(), ["pair-merged"])
        self.assertEqual(merge_mock.call_args.args[0]["payload"], "mate1")
        self.assertEqual(merge_mock.call_args.args[1]["payload"], "mate2")


class TestMergeMatePair(unittest.TestCase):
    """Test fragment-level mate merging and clipped-region metadata handling."""

    def test_merges_fragment_and_clipped_metadata_when_clipped_sequences_exist(self):
        """Use merged fragment coordinates and recomputed clipped statistics when clipped data is available."""
        mate1 = pd.Series(
            {
                "read_name": "pair",
                "chromosome": "chr1",
                "read_start": 100,
                "read_end": 105,
                "seq": "AAAAA",
                "methylation_encoding": "10202",
                "seq_clipped": "AAA",
                "methylation_clipped": "101",
                "clip_start": 101,
                "mapping_quality": 40,
                "is_reverse": False,
                "data_type": "wgbs",
                "overlaps_dmr": True,
                "dmr_label": "dmr-1",
                "dmr_type": "hyper",
                "dmr_start": 101,
                "dmr_end": 108,
            }
        )
        mate2 = pd.Series(
            {
                "read_name": "pair",
                "chromosome": "chr1",
                "read_start": 103,
                "read_end": 108,
                "seq": "CCCCC",
                "methylation_encoding": "00111",
                "seq_clipped": "CCC",
                "methylation_clipped": "001",
                "clip_start": 103,
                "mapping_quality": 35,
                "is_reverse": True,
                "data_type": "wgbs",
                "overlaps_dmr": False,
                "dmr_label": "dmr-1",
                "dmr_type": "hyper",
                "dmr_start": 101,
                "dmr_end": 108,
            }
        )

        with patch.object(
            bp,
            "_merge_methylation_encodings",
            side_effect=[
                (
                    {"seq": "AAACCCCC", "methylation_encoding": "10200111"},
                    {
                        "total": 5,
                        "methylated": 3,
                        "unmethylated": 2,
                        "methylation_rate": 0.6,
                    },
                ),
                (
                    {"seq": "AACCC", "methylation_encoding": "10101"},
                    {
                        "total": 4,
                        "methylated": 3,
                        "unmethylated": 1,
                        "methylation_rate": 0.75,
                    },
                ),
            ],
        ) as merge_mock:
            merged = bp._merge_mate_pair(mate1, mate2)

        self.assertEqual(merge_mock.call_count, 2)
        self.assertEqual(merged["read_start"], 100)
        self.assertEqual(merged["read_end"], 108)
        self.assertEqual(merged["read_length"], 8)
        self.assertEqual(merged["seq"], "AAACCCCC")
        self.assertEqual(merged["methylation_encoding"], "10200111")
        self.assertEqual(merged["clipped_total"], 4)
        self.assertEqual(merged["clipped_methylated"], 3)
        self.assertEqual(merged["clipped_unmethylated"], 1)
        self.assertAlmostEqual(merged["clipped_meth_rate"], 0.75)
        self.assertEqual(merged["mapping_quality"], 35)
        self.assertTrue(merged["is_merged_pair"])

    def test_falls_back_to_precomputed_clipped_counts_when_clipped_sequences_are_missing(
        self,
    ):
        """Use stored clipped counts and legacy DMR keys when clipped sequences are unavailable."""
        mate1 = pd.Series(
            {
                "read_name": "pair",
                "chromosome": "chr1",
                "read_start": 10,
                "read_end": 15,
                "seq": "AAAAA",
                "methylation_encoding": "11111",
                "mapping_quality": 30,
                "is_reverse": False,
                "data_type": "ont",
                "overlaps_dmr": False,
                "dmr_labels": "legacy-dmr",
                "dmr_types": "hypo",
                "clipped_methylated": 1,
                "clipped_unmethylated": 2,
            }
        )
        mate2 = pd.Series(
            {
                "read_name": "pair",
                "chromosome": "chr1",
                "read_start": 16,
                "read_end": 20,
                "seq": "CCCC",
                "methylation_encoding": "0000",
                "mapping_quality": 20,
                "is_reverse": True,
                "data_type": "ont",
                "overlaps_dmr": True,
                "clipped_methylated": 2,
                "clipped_unmethylated": 1,
            }
        )

        with patch.object(
            bp,
            "_merge_methylation_encodings",
            return_value=(
                {"seq": "AAAAANCCCC", "methylation_encoding": "1111120000"},
                {
                    "total": 9,
                    "methylated": 5,
                    "unmethylated": 4,
                    "methylation_rate": 5 / 9,
                },
            ),
        ):
            merged = bp._merge_mate_pair(mate1, mate2)

        self.assertEqual(merged["dmr_label"], "legacy-dmr")
        self.assertEqual(merged["dmr_type"], "hypo")
        self.assertTrue(merged["overlaps_dmr"])
        self.assertEqual(merged["clipped_total"], 6)
        self.assertEqual(merged["seq_clipped"], "")
        self.assertEqual(merged["methylation_clipped"], "")
        self.assertAlmostEqual(merged["clipped_meth_rate"], 0.5)


class TestMergeMethylationEncodings(unittest.TestCase):
    """Test consensus merging of per-base methylation encodings across mates."""

    def test_applies_consensus_rules_and_computes_summary_stats(self):
        """Resolve overlap conflicts exactly as the production merge logic specifies."""
        merged_encoding, stats = bp._merge_methylation_encodings(
            seq1="AAAA",
            enc1="1021",
            start1=10,
            seq2="CCCC",
            enc2="1210",
            start2=10,
            frag_start=10,
            frag_end=14,
        )

        # Position 10: both mates agree on '1'.
        # Position 11: mate2 is unknown, so mate1's '0' survives.
        # Position 12: mate1 is unknown, so mate2's '1' is taken.
        # Position 13: mates disagree, so the merged state becomes unknown.
        self.assertEqual(
            merged_encoding, {"seq": "CCCC", "methylation_encoding": "1012"}
        )
        self.assertEqual(stats["methylated"], 2)
        self.assertEqual(stats["unmethylated"], 1)
        self.assertEqual(stats["total"], 3)
        self.assertAlmostEqual(stats["methylation_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
