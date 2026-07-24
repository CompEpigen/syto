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
