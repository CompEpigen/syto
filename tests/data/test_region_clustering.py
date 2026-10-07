"""Equal-size clustering of atlas regions within each target cell type."""

import unittest

import numpy as np
import pandas as pd

from syto.data.atlases.region_clustering import BalancedRegionClusterer

CELL_TYPES = ["A", "B", "C", "D"]


def _atlas(sizes, seed=0, n_profiles=None):
    """Regions per target drawn around ``n_profiles`` distinct profile centres."""
    rng = np.random.default_rng(seed)
    rows = []
    for target, n in sizes.items():
        k = n_profiles or 1
        centres = rng.uniform(0, 1, size=(k, len(CELL_TYPES)))
        for i in range(n):
            values = np.clip(centres[i % k] + rng.normal(0, 0.01, len(CELL_TYPES)), 0, 1)
            rows.append({"name": f"{target}{i}", "target": target, **dict(zip(CELL_TYPES, values))})
    return pd.DataFrame(rows)


class TestBalancedRegionClusterer(unittest.TestCase):
    def _cluster(self, atlas, size=5, seed=0):
        return BalancedRegionClusterer(group_size=size, seed=seed).fit_predict(atlas, CELL_TYPES)

    def test_cluster_sizes_are_balanced_and_capped(self):
        atlas = _atlas({"A": 50, "B": 23, "C": 4})
        sizes = self._cluster(atlas).value_counts()
        a = sizes[sizes.index.str.startswith("A__")]
        b = sizes[sizes.index.str.startswith("B__")]
        c = sizes[sizes.index.str.startswith("C__")]
        self.assertEqual(sorted(a), [5] * 10)
        self.assertEqual(sorted(b), [4, 4, 5, 5, 5])  # ceil(23/5) = 5 clusters
        self.assertEqual(sorted(c), [4])

    def test_clusters_stay_within_their_target(self):
        atlas = _atlas({"A": 20, "B": 20})
        labels = self._cluster(atlas)
        self.assertTrue((labels.str.split("__").str[0] == atlas["target"]).all())

    def test_recovers_well_separated_profiles(self):
        # 4 tight profile groups of 5 regions each within one target.
        atlas = _atlas({"A": 20}, n_profiles=4)
        labels = self._cluster(atlas)
        true_group = np.arange(20) % 4
        for g in range(4):
            self.assertEqual(labels[true_group == g].nunique(), 1)

    def test_is_deterministic_for_a_seed(self):
        atlas = _atlas({"A": 40, "B": 17}, seed=3)
        pd.testing.assert_series_equal(self._cluster(atlas, seed=1), self._cluster(atlas, seed=1))

    def test_missing_values_are_imputed(self):
        atlas = _atlas({"A": 20}, n_profiles=4)
        atlas.loc[[0, 5], "B"] = np.nan
        atlas.loc[:, "D"] = np.nan  # a column missing for the whole target
        labels = self._cluster(atlas)
        self.assertFalse(labels.isna().any())
        self.assertEqual(sorted(labels.value_counts()), [5] * 4)

    def test_names_are_ordered_by_target_column(self):
        atlas = _atlas({"A": 10})
        atlas["A"] = np.r_[np.full(5, 0.9), np.full(5, 0.1)]
        labels = self._cluster(atlas)
        self.assertTrue((labels[:5] == "A__0").all())  # highest target value first
        self.assertTrue((labels[5:] == "A__1").all())

    def test_rejects_non_positive_group_size(self):
        with self.assertRaises(ValueError):
            BalancedRegionClusterer(group_size=0)


if __name__ == "__main__":
    unittest.main()
