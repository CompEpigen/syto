"""Equal-size k-means of the regions of one target."""

import unittest

import numpy as np

from syto.data.atlases.region_clustering import BalancedRegionClusterer


def _features(n, n_profiles=1, n_features=4, seed=0):
    """``n`` rows drawn around ``n_profiles`` distinct centres (row i -> i % n_profiles)."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 1, size=(n_profiles, n_features))
    return centres[np.arange(n) % n_profiles] + rng.normal(0, 0.01, (n, n_features))


class TestBalancedRegionClusterer(unittest.TestCase):
    def _cluster(self, X, size=5, seed=0):
        return BalancedRegionClusterer(group_size=size, seed=seed).fit_predict(X)

    def test_cluster_sizes_are_balanced_and_capped(self):
        for n, expected in [(50, [5] * 10), (23, [4, 4, 5, 5, 5]), (4, [4])]:
            ids = self._cluster(_features(n))
            self.assertEqual(sorted(np.bincount(ids)), expected, msg=f"n={n}")

    def test_ids_are_contiguous_from_zero(self):
        ids = self._cluster(_features(23))
        self.assertEqual(ids.dtype.kind, "i")
        self.assertEqual(set(ids), set(range(5)))

    def test_single_cluster_when_rows_fit_one_group(self):
        np.testing.assert_array_equal(
            self._cluster(_features(5)), np.zeros(5, dtype=int)
        )

    def test_recovers_well_separated_profiles(self):
        # 4 tight profile groups of 5 rows each.
        ids = self._cluster(_features(20, n_profiles=4))
        true_group = np.arange(20) % 4
        for g in range(4):
            self.assertEqual(len(set(ids[true_group == g])), 1)

    def test_is_deterministic_for_a_seed(self):
        X = _features(40, seed=3)
        np.testing.assert_array_equal(
            self._cluster(X, seed=1), self._cluster(X, seed=1)
        )

    def test_rejects_non_positive_group_size(self):
        with self.assertRaises(ValueError):
            BalancedRegionClusterer(group_size=0)

    def test_rejects_missing_values(self):
        X = _features(10)
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self._cluster(X)


if __name__ == "__main__":
    unittest.main()
