import unittest

import numpy as np

from baselines.deconvolution.epidish.epidish import _do_rpc, _normalize_nonneg


class TestNormalizeNonneg(unittest.TestCase):
    def test_clips_and_normalizes(self):
        out = _normalize_nonneg(np.array([-1.0, 1.0, 3.0]))
        np.testing.assert_allclose(out, [0.0, 0.25, 0.75])
        self.assertAlmostEqual(float(out.sum()), 1.0)

    def test_uniform_fallback_when_all_nonpositive(self):
        out = _normalize_nonneg(np.array([-1.0, -2.0]))
        np.testing.assert_allclose(out, [0.5, 0.5])


class TestDoRPC(unittest.TestCase):
    def _synthetic(self, seed=0, n=200):
        rng = np.random.default_rng(seed)
        ref = rng.uniform(0.0, 1.0, size=(n, 3))
        true = np.array([0.5, 0.3, 0.2])
        mixture = ref @ true + rng.normal(0.0, 0.01, size=n)
        return mixture, ref, true

    def test_recovers_fractions(self):
        mixture, ref, true = self._synthetic()
        frac = _do_rpc(mixture, ref)
        self.assertEqual(frac.shape, (3,))
        self.assertAlmostEqual(float(frac.sum()), 1.0, places=6)
        np.testing.assert_allclose(frac, true, atol=0.05)

    def test_robust_to_outliers(self):
        mixture, ref, true = self._synthetic()
        rng = np.random.default_rng(1)
        idx = rng.choice(len(mixture), size=20, replace=False)
        mixture[idx] += 5.0  # gross outliers
        frac = _do_rpc(mixture, ref)
        np.testing.assert_allclose(frac, true, atol=0.1)


from baselines.deconvolution.epidish.epidish import _do_cbs


class TestDoCBS(unittest.TestCase):
    def _synthetic(self, seed=0, n=200):
        rng = np.random.default_rng(seed)
        ref = rng.uniform(0.0, 1.0, size=(n, 3))
        true = np.array([0.6, 0.3, 0.1])
        mixture = ref @ true + rng.normal(0.0, 0.01, size=n)
        return mixture, ref, true

    def test_returns_valid_simplex(self):
        mixture, ref, _ = self._synthetic()
        frac = _do_cbs(mixture, ref)
        self.assertEqual(frac.shape, (3,))
        self.assertAlmostEqual(float(frac.sum()), 1.0, places=6)
        self.assertTrue((frac >= 0).all())

    def test_identifies_dominant_cell_type(self):
        mixture, ref, true = self._synthetic()
        frac = _do_cbs(mixture, ref)
        self.assertEqual(int(frac.argmax()), int(true.argmax()))
        np.testing.assert_allclose(frac, true, atol=0.2)


from baselines.deconvolution.epidish.epidish import _do_cp


class TestDoCP(unittest.TestCase):
    def _synthetic(self, seed=3, n=150):
        rng = np.random.default_rng(seed)
        ref = rng.uniform(0.0, 1.0, size=(n, 3))
        true = np.array([0.5, 0.3, 0.2])
        mixture = ref @ true  # noise-free: QP recovers exactly
        return mixture, ref, true

    def test_inequality_recovers_fractions(self):
        mixture, ref, true = self._synthetic()
        frac = _do_cp(mixture, ref, constraint="inequality")
        self.assertEqual(frac.shape, (3,))
        self.assertTrue((frac >= -1e-8).all())
        np.testing.assert_allclose(frac, true, atol=1e-3)

    def test_equality_sums_to_one(self):
        mixture, ref, true = self._synthetic()
        frac = _do_cp(mixture, ref, constraint="equality")
        self.assertAlmostEqual(float(frac.sum()), 1.0, places=6)
        np.testing.assert_allclose(frac, true, atol=1e-3)


from unittest.mock import MagicMock

import pandas as pd

from baselines.deconvolution.epidish.epidish import EpiDishDeconvolver


def _make_atlas_mock(beta, cpg_lookup=None, n_cpgs=3, ref_cells=("ct1", "ct2"),
                     in_atlas=True):
    atlas = MagicMock()
    atlas.__contains__ = MagicMock(return_value=in_atlas)
    atlas.get_cpg_lookup.return_value = cpg_lookup or {10: 0, 11: 1, 12: 2}
    atlas.get_n_cpgs.return_value = n_cpgs
    atlas.get_beta_for_regions.return_value = beta
    atlas.ref_cells = list(ref_cells)
    return atlas


def _reads(patterns, region="r1", read_start=10):
    return pd.DataFrame(
        {
            "name": [region] * len(patterns),
            "read_start": [read_start] * len(patterns),
            "pattern": list(patterns),
        }
    )


class TestBuildInput(unittest.TestCase):
    def test_mixture_is_per_cpg_meth_fraction(self):
        beta = [np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])]  # (T=2, 3)
        d = EpiDishDeconvolver(_make_atlas_mock(beta))
        out = d.build_input(_reads(("111", "000")))
        # col j: one '1' + one '0' -> 1/2
        np.testing.assert_allclose(out["mixture"], [0.5, 0.5, 0.5])
        self.assertEqual(out["ref"].shape, (3, 2))
        np.testing.assert_allclose(out["ref"][:, 0], [0.9, 0.9, 0.9])
        self.assertEqual(out["ref_cells"], ["ct1", "ct2"])

    def test_drops_cpg_with_nan_reference(self):
        beta = [np.array([[0.9, np.nan, 0.9], [0.1, 0.2, 0.1]])]  # NaN at col 1
        d = EpiDishDeconvolver(_make_atlas_mock(beta))
        out = d.build_input(_reads(("111",)))
        self.assertEqual(out["mixture"].shape, (2,))  # col 1 dropped
        self.assertEqual(out["ref"].shape, (2, 2))

    def test_drops_cpg_with_no_mixture_coverage(self):
        # pattern '212' -> only middle CpG covered; cols 0 and 2 have no data
        beta = [np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])]
        d = EpiDishDeconvolver(_make_atlas_mock(beta))
        out = d.build_input(_reads(("212",)))
        self.assertEqual(out["mixture"].shape, (1,))
        np.testing.assert_allclose(out["mixture"], [1.0])

    def test_returns_none_when_no_region_in_atlas(self):
        d = EpiDishDeconvolver(_make_atlas_mock([], in_atlas=False))
        self.assertIsNone(d.build_input(_reads(("111",))))

    def test_rejects_invalid_method(self):
        with self.assertRaises(ValueError):
            EpiDishDeconvolver(_make_atlas_mock([]), method="BOGUS")
