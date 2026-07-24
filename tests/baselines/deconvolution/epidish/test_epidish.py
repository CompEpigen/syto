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
