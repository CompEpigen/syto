import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from baselines.deconvolution.celfieish.celfieish import (
    CelFiEISHDeconvolver,
    METHYLATED,
    NOVAL,
    UNMETHYLATED,
    _CelfieISHModel,
)

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------


def _make_pure_signal_model():
    """2 cell types, 1 region, 4 CpGs.

    ct1 = high methylation probability (0.9); ct2 = low (0.1).
    All reads are fully methylated → EM should assign most weight to ct1.
    """
    n_reads, n_cpgs = 6, 4
    mixture = [np.full((n_reads, n_cpgs), METHYLATED, dtype=np.int8)]
    beta = [np.array([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])]
    return _CelfieISHModel(mixture, beta)


def _make_atlas_mock(region_name="r1", cpg_lookup=None, n_cpgs=3):
    atlas = MagicMock()
    atlas.__contains__ = MagicMock(return_value=True)
    atlas.get_cpg_lookup.return_value = cpg_lookup or {10: 0, 11: 1, 12: 2}
    atlas.get_n_cpgs.return_value = n_cpgs
    return atlas


def _make_reads(region_name="r1", patterns=("111", "000"), read_start=10):
    return pd.DataFrame(
        {
            "name": [region_name] * len(patterns),
            "read_start": [read_start] * len(patterns),
            "pattern": list(patterns),
        }
    )


# ---------------------------------------------------------------------------
# _CelfieISHModel
# ---------------------------------------------------------------------------


class TestCelfieISHModelTwoStep(unittest.TestCase):
    """Tests for _CelfieISHModel.two_step()."""

    def test_returns_alpha_summing_to_one(self):
        model = _make_pure_signal_model()
        alpha, n_iter = model.two_step()
        self.assertEqual(alpha.shape, (2,))
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)
        self.assertIsInstance(n_iter, (int, np.integer))

    def test_recovers_dominant_cell_type(self):
        """All-methylated reads against clear reference → ct1 dominates."""
        np.random.seed(0)
        model = _make_pure_signal_model()
        alpha, _ = model.two_step()
        self.assertGreater(float(alpha[0]), 0.7)

    def test_convergence_stops_early(self):
        """EM should converge well before the max iteration limit."""
        model = _make_pure_signal_model()
        model.num_iterations = 200
        _, n_iter = model.two_step()
        self.assertLess(n_iter, 200)


class TestCelfieISHModelCheckpoints(unittest.TestCase):
    """Tests for _CelfieISHModel.run_with_checkpoints()."""

    def test_returns_correct_number_of_snapshots(self):
        model = _make_pure_signal_model()
        checkpoints = [2, 5, 10]
        results = model.run_with_checkpoints(checkpoints)
        self.assertEqual(len(results), 3)

    def test_snapshot_iteration_numbers_match(self):
        model = _make_pure_signal_model()
        checkpoints = [1, 3, 7]
        iters = [n for n, _ in model.run_with_checkpoints(checkpoints)]
        self.assertEqual(iters, checkpoints)

    def test_snapshot_proportions_sum_to_one(self):
        model = _make_pure_signal_model()
        for _, alpha in model.run_with_checkpoints([5, 10]):
            self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)


# ---------------------------------------------------------------------------
# CelFiEISHDeconvolver.build_input
# ---------------------------------------------------------------------------


class TestBuildCelfieishInput(unittest.TestCase):
    """Tests for CelFiEISHDeconvolver.build_input (previously build_celfieish_input)."""

    def _make_deconvolver(self, in_atlas=True, n_cpgs=3):
        atlas = _make_atlas_mock(n_cpgs=n_cpgs)
        atlas.__contains__ = MagicMock(return_value=in_atlas)
        atlas.ref_cells = ["ct1", "ct2"]
        atlas.get_beta_for_regions.return_value = []
        return CelFiEISHDeconvolver(atlas)

    def test_returns_correct_keys(self):
        d = self._make_deconvolver()
        result = d.build_input(_make_reads())
        self.assertIn("matrices", result)
        self.assertIn("region_names", result)

    def test_matrix_shape(self):
        """Matrix should be (n_reads, n_cpgs) per region."""
        d = self._make_deconvolver(n_cpgs=3)
        result = d.build_input(_make_reads(patterns=("111", "000")))
        self.assertEqual(len(result["matrices"]), 1)
        self.assertEqual(result["matrices"][0].shape, (2, 3))

    def test_methylated_reads_encoded_correctly(self):
        d = self._make_deconvolver(n_cpgs=3)
        result = d.build_input(_make_reads(patterns=("111",)))
        self.assertTrue(np.all(result["matrices"][0] == METHYLATED))

    def test_unmethylated_reads_encoded_correctly(self):
        d = self._make_deconvolver(n_cpgs=3)
        result = d.build_input(_make_reads(patterns=("000",)))
        self.assertTrue(np.all(result["matrices"][0] == UNMETHYLATED))

    def test_skips_region_not_in_atlas(self):
        d = self._make_deconvolver(in_atlas=False)
        result = d.build_input(_make_reads())
        self.assertEqual(result["region_names"], [])
        self.assertEqual(result["matrices"], [])


# ---------------------------------------------------------------------------
# CelFiEISHDeconvolver
# ---------------------------------------------------------------------------


def _make_deconvolver_with_mock_atlas():
    atlas = _make_atlas_mock(n_cpgs=3)
    atlas.ref_cells = ["ct1", "ct2"]
    beta = [np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])]
    atlas.get_beta_for_regions.return_value = beta
    return CelFiEISHDeconvolver(atlas)


def _make_prepared_reads():
    return pd.DataFrame(
        {
            "name": ["r1", "r1", "r1"],
            "read_start": [10, 10, 10],
            "pattern": ["111", "111", "111"],
            "chromosome": ["chr1", "chr1", "chr1"],
            "read_end": [13, 13, 13],
        }
    )


class TestCelFiEISHDeconvolver(unittest.TestCase):
    """Integration tests for CelFiEISHDeconvolver.deconvolute_reads."""

    def test_returns_proportions_of_correct_length(self):
        d = _make_deconvolver_with_mock_atlas()
        result = d.deconvolute_reads(
            _make_prepared_reads(), {"ct1": 0, "ct2": 1}, prepare=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_proportions_sum_to_one(self):
        d = _make_deconvolver_with_mock_atlas()
        result = d.deconvolute_reads(
            _make_prepared_reads(), {"ct1": 0, "ct2": 1}, prepare=False
        )
        self.assertAlmostEqual(sum(result), 1.0, places=4)

    def test_no_region_overlap_returns_none(self):
        """When no reads land in atlas regions, deconvolution returns None."""
        atlas = _make_atlas_mock()
        atlas.__contains__ = MagicMock(return_value=False)
        atlas.ref_cells = ["ct1", "ct2"]
        atlas.get_beta_for_regions.return_value = []
        d = CelFiEISHDeconvolver(atlas)
        result = d.deconvolute_reads(
            _make_prepared_reads(), {"ct1": 0, "ct2": 1}, prepare=False
        )
        self.assertIsNone(result)

    def test_checkpoints_return_list_of_tuples(self):
        atlas = _make_atlas_mock(n_cpgs=3)
        atlas.ref_cells = ["ct1", "ct2"]
        atlas.get_beta_for_regions.return_value = [
            np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
        ]
        d = CelFiEISHDeconvolver(atlas, em_checkpoints=[2, 5])
        result = d.deconvolute_reads(
            _make_prepared_reads(), {"ct1": 0, "ct2": 1}, prepare=False
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for n_steps, proportions in result:
            self.assertIn(n_steps, [2, 5])
            self.assertAlmostEqual(sum(proportions), 1.0, places=4)
