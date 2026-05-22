import unittest
from parameterized import parameterized
import pandas as pd

from methyldl.data.optimal_subsequences import (
    select_optimal_subsequence,
    select_optimal_subsequence_vectorized,
    select_optimal_subsequence_rolling_window,
    extract_optimal_subsequence_dataframe,
    extract_optimal_subsequence_dataframe_chunked,
)


class TestSelectOptimalSubsequence(unittest.TestCase):
    """Tests for baseline optimal subsequence selection."""

    def test_invalid_inputs_raise(self):
        """Reject mismatched sequence lengths and unknown criteria."""
        with self.assertRaises(ValueError):
            select_optimal_subsequence("AAAA", "222", 2)

        with self.assertRaises(ValueError):
            select_optimal_subsequence("AAAA", "2222", 2, selection_criteria="bad")

    def test_counts_mode_selects_window_with_max_labeled(self):
        """Counts mode should pick the window with most labeled positions."""
        dna = "AATTTTAAAA"
        methyl = "2200112222"

        selected_dna, selected_methyl, labeled_count, entropy_score = (
            select_optimal_subsequence(
                dna,
                methyl,
                target_read_length=4,
                selection_criteria="counts",
                stride=1,
            )
        )

        self.assertEqual(selected_dna, "TTTT")
        self.assertEqual(selected_methyl, "0011")
        self.assertEqual(labeled_count, 4)
        self.assertEqual(entropy_score, float("inf"))

    def test_entropy_mode_prefers_low_entropy_window(self):
        """Entropy mode should prefer homogeneous labeled windows."""
        dna = "TTTTAAAA"
        methyl = "11110022"

        selected_dna, selected_methyl, labeled_count, entropy_score = (
            select_optimal_subsequence(
                dna,
                methyl,
                target_read_length=4,
                selection_criteria="entropy",
                min_labeled_cpgs=2,
                stride=1,
            )
        )

        self.assertEqual(selected_dna, "TTTT")
        self.assertEqual(selected_methyl, "1111")
        self.assertEqual(labeled_count, 4)
        self.assertEqual(entropy_score, 0.0)

    def test_sequence_shorter_than_target_returns_original(self):
        """Short reads should be returned unchanged."""
        dna = "AAAA"
        methyl = "0011"

        selected_dna, selected_methyl, labeled_count, entropy_score = (
            select_optimal_subsequence(dna, methyl, target_read_length=10)
        )

        self.assertEqual(selected_dna, dna)
        self.assertEqual(selected_methyl, methyl)
        self.assertEqual(labeled_count, 4)
        self.assertEqual(entropy_score, float("inf"))


class TestSelectOptimalSubsequenceVectorized(unittest.TestCase):
    """Tests for vectorized optimal subsequence selection."""

    def test_invalid_inputs_raise(self):
        """Reject mismatched sequence lengths and unknown criteria."""
        with self.assertRaises(ValueError):
            select_optimal_subsequence_vectorized("AAAA", "222", 2)

        with self.assertRaises(ValueError):
            select_optimal_subsequence_vectorized(
                "AAAA", "2222", 2, selection_criteria="bad"
            )

    def test_vectorized_matches_baseline_counts_mode(self):
        """Vectorized implementation should match baseline in counts mode."""
        dna = "AAAAAAAAAA"
        methyl = "2200112222"

        baseline = select_optimal_subsequence(
            dna, methyl, target_read_length=4, selection_criteria="counts", stride=1
        )
        vectorized = select_optimal_subsequence_vectorized(
            dna, methyl, target_read_length=4, selection_criteria="counts", stride=1
        )

        self.assertEqual(vectorized, baseline)

    def test_vectorized_matches_baseline_entropy_mode(self):
        """Vectorized implementation should match baseline in entropy mode."""
        dna = "TTTTAAAA"
        methyl = "11110022"

        baseline = select_optimal_subsequence(
            dna,
            methyl,
            target_read_length=4,
            selection_criteria="entropy",
            min_labeled_cpgs=2,
            stride=1,
        )
        vectorized = select_optimal_subsequence_vectorized(
            dna,
            methyl,
            target_read_length=4,
            selection_criteria="entropy",
            min_labeled_cpgs=2,
            stride=1,
        )

        self.assertEqual(vectorized, baseline)

    def test_short_sequence_computes_stats(self):
        """Short-read path should return expected entropy statistics."""
        dna = "AAAA"
        methyl = "0011"

        selected_dna, selected_methyl, labeled_count, entropy_score = (
            select_optimal_subsequence_vectorized(
                dna, methyl, target_read_length=10, min_labeled_cpgs=2
            )
        )

        self.assertEqual(selected_dna, dna)
        self.assertEqual(selected_methyl, methyl)
        self.assertEqual(labeled_count, 4)
        self.assertAlmostEqual(entropy_score, 1.0)


class TestSelectOptimalSubsequenceRollingWindow(unittest.TestCase):
    """Tests for rolling-window optimized subsequence selection."""

    def test_invalid_criteria_raises(self):
        """Reject unsupported selection criteria."""
        with self.assertRaises(ValueError):
            select_optimal_subsequence_rolling_window(
                "AAAA", "2222", 2, selection_criteria="bad"
            )

    @parameterized.expand([("counts"), ("entropy")])
    def test_short_sequence_delegates_to_vectorized(self, selection_criteria):
        """Short-read path should match vectorized implementation output."""
        dna = "AAAA"
        methyl = "0011"

        rolling = select_optimal_subsequence_rolling_window(
            dna,
            methyl,
            target_read_length=10,
            min_labeled_cpgs=2,
            selection_criteria=selection_criteria,
        )
        vectorized = select_optimal_subsequence_vectorized(
            dna,
            methyl,
            target_read_length=10,
            min_labeled_cpgs=2,
            selection_criteria=selection_criteria,
        )

        self.assertEqual(rolling, vectorized)

    @parameterized.expand(
        [
            (
                "counts_middle_window",
                "counts",
                "AATTTTAAAA",
                "2200112222",
                4,
                5,
                "TTTT",
                "0011",
                4,
                float("inf"),
            ),
            (
                "counts_last_window",
                "counts",
                "AAAACCCC",
                "22220011",
                4,
                5,
                "CCCC",
                "0011",
                4,
                float("inf"),
            ),
            (
                "counts_first_window",
                "counts",
                "GGGGTTTT",
                "11002222",
                4,
                5,
                "GGGG",
                "1100",
                4,
                float("inf"),
            ),
            (
                "entropy_homogeneous_first",
                "entropy",
                "TTTTAAAA",
                "11110022",
                4,
                2,
                "TTTT",
                "1111",
                4,
                0.0,
            ),
            (
                "entropy_homogeneous_last",
                "entropy",
                "AAAATTTT",
                "22221111",
                4,
                2,
                "TTTT",
                "1111",
                4,
                0.0,
            ),
        ]
    )
    def test_selects_expected_window(
        self,
        _,
        mode,
        dna,
        methyl,
        target_read_length,
        min_labeled_cpgs,
        expected_dna,
        expected_methyl,
        expected_labeled_count,
        expected_entropy_score,
    ):
        """Rolling window selection should return the expected subsequence for each scenario."""

        rolling = select_optimal_subsequence_rolling_window(
            dna,
            methyl,
            target_read_length=target_read_length,
            selection_criteria=mode,
            min_labeled_cpgs=min_labeled_cpgs,
            stride=1,
        )

        self.assertEqual(rolling[0], expected_dna)
        self.assertEqual(rolling[1], expected_methyl)
        self.assertEqual(rolling[2], expected_labeled_count)
        if expected_entropy_score == float("inf"):
            self.assertEqual(rolling[3], expected_entropy_score)
        else:
            self.assertAlmostEqual(rolling[3], expected_entropy_score)


class TestExtractOptimalSubsequenceDataframe(unittest.TestCase):
    """Tests for dataframe-wide optimal subsequence extraction."""

    def test_extract_dataframe_adds_expected_columns(self):
        """Extraction should append computed selection columns."""
        df = pd.DataFrame(
            {
                "input_ids": ["AAAAAA", "CCCCCC"],
                "methylation_ids": ["220011", "222200"],
                "sample_id": ["s1", "s2"],
            }
        )

        out = extract_optimal_subsequence_dataframe(
            df, target_read_length=4, selection_criteria="counts", stride=1
        )

        self.assertEqual(len(out), 2)
        self.assertIn("selected_input_ids", out.columns)
        self.assertIn("selected_methylation_ids", out.columns)
        self.assertIn("labeled_cpg_count", out.columns)
        self.assertIn("entropy_score", out.columns)
        self.assertIn("sample_id", out.columns)

    def test_extract_dataframe_skips_invalid_rows(self):
        """Rows that raise processing errors should be skipped."""
        df = pd.DataFrame(
            {
                "input_ids": ["AAAAAA", "CCCC"],
                "methylation_ids": ["220011", "22222"],
                "sample_id": ["s1", "bad"],
            }
        )

        out = extract_optimal_subsequence_dataframe(
            df, target_read_length=4, selection_criteria="counts", stride=1
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["sample_id"], "s1")

    def test_extract_dataframe_short_sequence_computes_stats(self):
        """Short sequences should return expected entropy statistics."""
        df = pd.DataFrame(
            {
                "input_ids": ["AAAA", "CCCC"],
                "methylation_ids": ["0011", "2222"],
                "sample_id": ["s1", "s2"],
            }
        )

        out = extract_optimal_subsequence_dataframe(
            df,
            target_read_length=10,
            selection_criteria="entropy",
            min_labeled_cpgs=2,
            stride=1,
        )

        print(out)

        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["selected_input_ids"], "AAAA")
        self.assertEqual(out.iloc[0]["selected_methylation_ids"], "0011")
        self.assertEqual(out.iloc[0]["labeled_cpg_count"], 4)
        self.assertEqual(out.iloc[0]["entropy_score"], 1.0)
        self.assertEqual(out.iloc[1]["selected_input_ids"], "CCCC")
        self.assertEqual(out.iloc[1]["selected_methylation_ids"], "2222")
        self.assertEqual(out.iloc[1]["labeled_cpg_count"], 0)
        self.assertEqual(out.iloc[1]["entropy_score"], float("inf"))


class TestExtractOptimalSubsequenceDataframeChunked(unittest.TestCase):
    """Tests for chunked dataframe optimal subsequence extraction."""

    def test_chunked_extract_dataframe_adds_expected_columns(self):
        """Chunked extraction should produce computed output columns."""
        df = pd.DataFrame(
            {
                "input_ids": ["AAAAAA", "CCCCCC"],
                "methylation_ids": ["220011", "222200"],
                "sample_id": ["s1", "s2"],
            }
        )

        out = extract_optimal_subsequence_dataframe_chunked(
            df,
            target_read_length=4,
            selection_criteria="counts",
            stride=1,
            chunk_size=1,
        )

        self.assertEqual(len(out), 2)
        self.assertIn("selected_input_ids", out.columns)
        self.assertIn("selected_methylation_ids", out.columns)
        self.assertIn("labeled_cpg_count", out.columns)
        self.assertIn("entropy_score", out.columns)
        self.assertIn("sample_id", out.columns)
        self.assertNotIn("input_ids", out.columns)
        self.assertNotIn("methylation_ids", out.columns)

    def test_chunked_extract_skips_invalid_rows(self):
        """Chunked processing should skip rows that trigger errors."""
        df = pd.DataFrame(
            {
                "input_ids": ["AAAAAA", "CCCC"],
                "methylation_ids": ["220011", "22222"],
                "sample_id": ["s1", "bad"],
            }
        )

        out = extract_optimal_subsequence_dataframe_chunked(
            df,
            target_read_length=4,
            selection_criteria="counts",
            stride=1,
            chunk_size=1,
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["sample_id"], "s1")


if __name__ == "__main__":
    unittest.main()
