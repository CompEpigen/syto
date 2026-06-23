import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from baselines.deconvolution.celfie.celfie import (
    CelFiEDeconvolver,
    _CelfieModel,
)

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------


def _make_pure_signal_model():
    """2 cell types, 1 region, 4 CpGs.

    ct1 = fully methylated reference; ct2 = fully unmethylated.
    Sample is fully methylated → EM should assign most weight to ct1.
    """
    x_meth = [np.array([[4.0, 4.0, 4.0, 4.0]])]
    x_cov = [np.array([[4.0, 4.0, 4.0, 4.0]])]
    y = [np.array([[8.0, 8.0, 8.0, 8.0], [0.0, 0.0, 0.0, 0.0]])]
    y_cov = [np.array([[8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]])]
    return _CelfieModel(x_meth, x_cov, y, y_cov)


def _make_atlas_mock(region_name="r1", cpg_lookup=None, n_cpgs=3):
    """Atlas mock that works with build_celfie_input."""
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
# _CelfieModel
# ---------------------------------------------------------------------------


class TestCelfieModelFit(unittest.TestCase):
    """Tests for _CelfieModel.fit()."""

    def test_returns_1d_array_summing_to_one(self):
        model = _make_pure_signal_model()
        alpha = model.fit(num_iterations=50, convergence_criteria=0.01)
        self.assertEqual(alpha.ndim, 1)
        self.assertEqual(alpha.shape[0], 2)
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)

    def test_recovers_dominant_cell_type(self):
        """Fully methylated sample against a clear reference → ct1 dominates."""
        model = _make_pure_signal_model()
        np.random.seed(42)
        alpha = model.fit(
            num_iterations=100, convergence_criteria=1e-5, random_restarts=3
        )
        self.assertGreater(float(alpha[0]), 0.7)

    def test_random_restarts_dont_corrupt_reference(self):
        """Each restart should operate on a fresh copy of y / y_depths."""
        model = _make_pure_signal_model()
        y_before = model._y.copy()
        model.fit(num_iterations=20, random_restarts=5)
        np.testing.assert_array_equal(model._y, y_before)

    def test_fit_with_checkpoints_count(self):
        model = _make_pure_signal_model()
        checkpoints = [3, 7, 10]
        results = model.fit_with_checkpoints(checkpoints)
        self.assertEqual(len(results), 3)
        iters = [n for n, _ in results]
        self.assertEqual(iters, checkpoints)

    def test_fit_with_checkpoints_proportions_sum_to_one(self):
        model = _make_pure_signal_model()
        for _, alpha in model.fit_with_checkpoints([5, 10]):
            self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)


# ---------------------------------------------------------------------------
# CelFiEDeconvolver.build_input
# ---------------------------------------------------------------------------


class TestBuildCelfieInput(unittest.TestCase):
    """Tests for CelFiEDeconvolver.build_input (previously build_celfie_input)."""

    def _make_deconvolver(self, in_atlas=True, n_cpgs=3):
        atlas = _make_atlas_mock(n_cpgs=n_cpgs)
        atlas.__contains__ = MagicMock(return_value=in_atlas)
        atlas.ref_cells = ["ct1", "ct2"]
        atlas.get_meth_cov_for_regions.return_value = ([], [])
        return CelFiEDeconvolver(atlas)

    def test_returns_correct_keys(self):
        d = self._make_deconvolver()
        result = d.build_input(_make_reads())
        self.assertIn("x_meth", result)
        self.assertIn("x_cov", result)
        self.assertIn("region_names", result)

    def test_meth_and_cov_shapes(self):
        d = self._make_deconvolver(n_cpgs=3)
        result = d.build_input(_make_reads(patterns=("111", "000")))
        self.assertEqual(len(result["x_meth"]), 1)
        self.assertEqual(result["x_meth"][0].shape, (1, 3))
        self.assertEqual(result["x_cov"][0].shape, (1, 3))

    def test_methylated_counts_correct(self):
        """Two reads: '111' + '000' → x_meth=[1,1,1], x_cov=[2,2,2]."""
        d = self._make_deconvolver(n_cpgs=3)
        result = d.build_input(_make_reads(patterns=("111", "000")))
        np.testing.assert_array_almost_equal(result["x_meth"][0], [[1.0, 1.0, 1.0]])
        np.testing.assert_array_almost_equal(result["x_cov"][0], [[2.0, 2.0, 2.0]])

    def test_skips_region_not_in_atlas(self):
        d = self._make_deconvolver(in_atlas=False)
        result = d.build_input(_make_reads())
        self.assertEqual(result["region_names"], [])
        self.assertEqual(result["x_meth"], [])


# ---------------------------------------------------------------------------
# CelFiEDeconvolver
# ---------------------------------------------------------------------------


def _make_deconvolver_with_mock_atlas():
    """CelFiEDeconvolver backed by a mock atlas that returns useful arrays."""
    atlas = _make_atlas_mock(n_cpgs=3)
    atlas.ref_cells = ["ct1", "ct2"]
    y = np.array([[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]])
    y_cov = np.array([[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]])
    atlas.get_meth_cov_for_regions.return_value = ([y], [y_cov])
    return CelFiEDeconvolver(atlas)


def _make_prepared_reads():
    return pd.DataFrame(
        {
            "name": ["r1", "r1"],
            "read_start": [10, 10],
            "pattern": ["111", "111"],
            "chromosome": ["chr1", "chr1"],
            "read_end": [13, 13],
        }
    )


class TestCelFiEDeconvolver(unittest.TestCase):
    """Integration tests for CelFiEDeconvolver.deconvolute_reads."""

    def test_returns_proportions_of_correct_length(self):
        d = _make_deconvolver_with_mock_atlas()
        labels = {"ct1": 0, "ct2": 1}
        result = d.deconvolute_reads(_make_prepared_reads(), labels, prepare=False)
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
        atlas.get_meth_cov_for_regions.return_value = ([], [])
        d = CelFiEDeconvolver(atlas)
        result = d.deconvolute_reads(
            _make_prepared_reads(), {"ct1": 0, "ct2": 1}, prepare=False
        )
        self.assertIsNone(result)

    def test_checkpoints_return_list_of_tuples(self):
        atlas = _make_atlas_mock(n_cpgs=3)
        atlas.ref_cells = ["ct1", "ct2"]
        y = np.array([[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]])
        y_cov = np.array([[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]])
        atlas.get_meth_cov_for_regions.return_value = ([y], [y_cov])
        d = CelFiEDeconvolver(atlas, em_checkpoints=[2, 5])
        result = d.deconvolute_reads(
            _make_prepared_reads(), {"ct1": 0, "ct2": 1}, prepare=False
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for n_steps, proportions in result:
            self.assertIn(n_steps, [2, 5])
            self.assertAlmostEqual(sum(proportions), 1.0, places=4)
