import unittest

from syto.data.sequencing.dmr_overlap_analysis import (
    _empty_overlap_result,
    get_overlapping_dmrs,
    analyze_read_dmr_overlap,
)


class FakeInterval:
    """Minimal interval object used to mimic the intervaltree API in tests."""

    def __init__(self, begin, end, name, dmr_type):
        """Store interval coordinates and the DMR metadata expected by the module."""
        self.begin = begin
        self.end = end
        self.data = {"name": name, "type": dmr_type}


class FakeIntervalTree:
    """Sliceable fake tree that returns either true overlaps or a forced response."""

    def __init__(self, intervals=None, forced_response=None):
        """Configure the tree with intervals or a static response for edge-case branches."""
        self.intervals = list(intervals or [])
        self.forced_response = (
            list(forced_response) if forced_response is not None else None
        )

    def __getitem__(self, key):
        """Return intervals overlapping the requested slice."""
        if not isinstance(key, slice):
            raise TypeError("FakeIntervalTree expects slicing syntax")

        if self.forced_response is not None:
            # This lets the tests reach defensive branches such as denominator == 0,
            # which a real interval query would rarely expose.
            return list(self.forced_response)

        query_start = key.start
        query_end = key.stop
        return [
            interval
            for interval in self.intervals
            if interval.begin < query_end and interval.end > query_start
        ]


class TestEmptyOverlapResult(unittest.TestCase):
    """Tests for the helper that builds the empty overlap payload."""

    def test_returns_expected_empty_payload(self):
        """Returns the exact empty schema used when no DMR overlap is found."""
        self.assertEqual(
            _empty_overlap_result(),
            {
                "overlaps_dmr": False,
                "num_dmrs_overlapped": 0,
                "dmr_labels": "",
                "dmr_types": "",
                "overlap_bps": "",
                "overlap_pcts": "",
            },
        )


class TestGetOverlappingDmrs(unittest.TestCase):
    """Tests for collecting per-DMR overlap details."""

    def test_returns_empty_list_when_chromosome_is_missing(self):
        """Returns no overlaps when the chromosome is absent from the tree mapping."""
        result = get_overlapping_dmrs(
            read_start=100,
            read_end=110,
            chrom="chr2",
            dmr_trees={"chr1": FakeIntervalTree()},
        )

        self.assertEqual(result, [])

    def test_returns_empty_list_when_query_finds_no_overlaps(self):
        """Returns no overlaps when the queried tree slice is empty."""
        result = get_overlapping_dmrs(
            read_start=100,
            read_end=110,
            chrom="chr1",
            dmr_trees={"chr1": FakeIntervalTree()},
        )

        self.assertEqual(result, [])

    def test_filters_out_partial_matches_in_strict_mode(self):
        """Drops overlaps unless the full read is contained inside the DMR."""
        tree = FakeIntervalTree(
            [
                FakeInterval(95, 105, "partial-left", "tumor"),
                FakeInterval(105, 115, "partial-right", "normal"),
            ]
        )

        result = get_overlapping_dmrs(
            read_start=100,
            read_end=110,
            chrom="chr1",
            dmr_trees={"chr1": tree},
            strict=True,
        )

        self.assertEqual(result, [])

    def test_keeps_only_fully_containing_dmr_in_strict_mode(self):
        """Keeps the single DMR that fully contains the read in strict mode."""
        tree = FakeIntervalTree(
            [
                FakeInterval(95, 115, "container", "tumor"),
                FakeInterval(108, 120, "partial-right", "normal"),
            ]
        )

        result = get_overlapping_dmrs(
            read_start=100,
            read_end=110,
            chrom="chr1",
            dmr_trees={"chr1": tree},
            strict=True,
        )

        # Strict mode converts the result to a set internally, so this test keeps only
        # one surviving interval and avoids coupling the assertion to iteration order.
        self.assertEqual(
            result,
            [
                {
                    "name": "container",
                    "type": "tumor",
                    "dmr_start": 95,
                    "dmr_end": 115,
                    "overlap_bp": 10,
                    "overlap_pct": 100.0,
                }
            ],
        )

    def test_returns_overlap_details_for_each_matching_dmr(self):
        """Computes overlap sizes and percentages for every matching DMR."""
        tree = FakeIntervalTree(
            [
                FakeInterval(10, 20, 101, 7),
                FakeInterval(15, 25, "dmr-b", "normal"),
            ]
        )

        result = get_overlapping_dmrs(
            read_start=12,
            read_end=18,
            chrom="chr1",
            dmr_trees={"chr1": tree},
        )

        self.assertEqual(
            result,
            [
                {
                    "name": "101",
                    "type": "7",
                    "dmr_start": 10,
                    "dmr_end": 20,
                    "overlap_bp": 6,
                    "overlap_pct": 100.0,
                },
                {
                    "name": "dmr-b",
                    "type": "normal",
                    "dmr_start": 15,
                    "dmr_end": 25,
                    "overlap_bp": 3,
                    "overlap_pct": 50.0,
                },
            ],
        )

    def test_returns_zero_percent_when_shorter_length_is_zero(self):
        """Falls back to 0.0 percent when the shorter length used as denominator is zero."""
        tree = FakeIntervalTree(
            forced_response=[FakeInterval(20, 30, "zero-case", "tumor")]
        )

        result = get_overlapping_dmrs(
            read_start=25,
            read_end=25,
            chrom="chr1",
            dmr_trees={"chr1": tree},
        )

        self.assertEqual(result[0]["overlap_bp"], 0)
        self.assertEqual(result[0]["overlap_pct"], 0.0)


class TestAnalyzeReadDmrOverlap(unittest.TestCase):
    """Tests for the aggregated read-level DMR overlap summary."""

    def test_returns_empty_payload_when_chromosome_is_missing(self):
        """Returns the empty payload when the chromosome is absent."""
        result = analyze_read_dmr_overlap(
            read_start=100,
            read_end=110,
            chrom="chr2",
            dmr_trees={"chr1": FakeIntervalTree()},
        )

        self.assertEqual(result, _empty_overlap_result())

    def test_returns_empty_payload_when_query_finds_no_overlaps(self):
        """Returns the empty payload when the queried tree slice is empty."""
        result = analyze_read_dmr_overlap(
            read_start=100,
            read_end=110,
            chrom="chr1",
            dmr_trees={"chr1": FakeIntervalTree()},
        )

        self.assertEqual(result, _empty_overlap_result())

    def test_returns_empty_payload_when_strict_filter_removes_all_overlaps(self):
        """Returns the empty payload when strict mode rejects partial overlaps."""
        tree = FakeIntervalTree(
            [
                FakeInterval(95, 105, "partial-left", "tumor"),
                FakeInterval(105, 115, "partial-right", "normal"),
            ]
        )

        result = analyze_read_dmr_overlap(
            read_start=100,
            read_end=110,
            chrom="chr1",
            dmr_trees={"chr1": tree},
            strict=True,
        )

        self.assertEqual(result, _empty_overlap_result())

    def test_aggregates_multiple_overlaps_into_comma_separated_fields(self):
        """Aggregates labels, types, base-pair overlaps, and percentages across DMRs."""
        tree = FakeIntervalTree(
            [
                FakeInterval(10, 20, "dmr-a", "tumor"),
                FakeInterval(15, 25, "dmr-b", "normal"),
            ]
        )

        result = analyze_read_dmr_overlap(
            read_start=12,
            read_end=18,
            chrom="chr1",
            dmr_trees={"chr1": tree},
        )

        self.assertEqual(
            result,
            {
                "overlaps_dmr": True,
                "num_dmrs_overlapped": 2,
                "dmr_labels": "dmr-a,dmr-b",
                "dmr_types": "tumor,normal",
                "overlap_bps": "6,3",
                "overlap_pcts": "100.0,50.0",
            },
        )

    def test_strict_mode_can_return_a_successful_aggregated_result(self):
        """Aggregates a strict-mode overlap when the read is fully contained in a DMR."""
        tree = FakeIntervalTree(
            [
                FakeInterval(95, 115, "container", "tumor"),
                FakeInterval(108, 120, "partial-right", "normal"),
            ]
        )

        result = analyze_read_dmr_overlap(
            read_start=100,
            read_end=110,
            chrom="chr1",
            dmr_trees={"chr1": tree},
            strict=True,
        )

        # Only one DMR survives the strict containment filter, which avoids depending
        # on the nondeterministic ordering introduced by the internal set conversion.
        self.assertEqual(
            result,
            {
                "overlaps_dmr": True,
                "num_dmrs_overlapped": 1,
                "dmr_labels": "container",
                "dmr_types": "tumor",
                "overlap_bps": "10",
                "overlap_pcts": "100.0",
            },
        )

    def test_formats_zero_percent_when_shorter_length_is_zero(self):
        """Formats the defensive zero-denominator branch as 0.0 in the summary payload."""
        tree = FakeIntervalTree(
            forced_response=[FakeInterval(20, 30, "zero-case", "tumor")]
        )

        result = analyze_read_dmr_overlap(
            read_start=25,
            read_end=25,
            chrom="chr1",
            dmr_trees={"chr1": tree},
        )

        self.assertEqual(
            result,
            {
                "overlaps_dmr": True,
                "num_dmrs_overlapped": 1,
                "dmr_labels": "zero-case",
                "dmr_types": "tumor",
                "overlap_bps": "0",
                "overlap_pcts": "0.0",
            },
        )
