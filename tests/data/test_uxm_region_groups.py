"""UXMMethylationAtlas.with_region_groups: per-target GR groups written as atlas columns."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from syto.data.atlases.region_clustering import BalancedRegionClusterer
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.data.dataset_build.filters import load_region_annotations

CELL_TYPES = ["A", "B", "C", "D"]


def _atlas_df(sizes, n_profiles=1, seed=0):
    """Regions per target drawn around ``n_profiles`` distinct profile centres."""
    rng = np.random.default_rng(seed)
    rows = []
    for target, n in sizes.items():
        centres = rng.uniform(0, 1, size=(n_profiles, len(CELL_TYPES)))
        for i in range(n):
            values = np.clip(
                centres[i % n_profiles] + rng.normal(0, 0.01, len(CELL_TYPES)), 0, 1
            )
            start = 1000 * (len(rows) + 1)
            rows.append(
                {
                    "chr": "chr1",
                    "start": start,
                    "end": start + 100,
                    "startCpG": len(rows) * 10,
                    "endCpG": len(rows) * 10 + 5,
                    "target": target,
                    "name": f"{target}{i}",
                    "direction": "U",
                    **dict(zip(CELL_TYPES, values)),
                }
            )
    return pd.DataFrame(rows)


def _atlas(df):
    return UXMMethylationAtlas("t", "hg38", atlas_df=df, cell_types=CELL_TYPES)


def _group(atlas, size=5, **kwargs):
    return atlas.with_region_groups(
        BalancedRegionClusterer(group_size=size, seed=0), "grg", **kwargs
    )


class TestWithRegionGroups(unittest.TestCase):
    def test_groups_are_balanced_within_each_target(self):
        grouped = _group(_atlas(_atlas_df({"A": 50, "B": 23, "C": 4}))).atlas
        self.assertTrue(
            (grouped["grg"].str.split("__").str[0] == grouped["target"]).all()
        )
        sizes = grouped.groupby("target")["grg"].value_counts()
        self.assertEqual(sorted(sizes["A"]), [5] * 10)
        self.assertEqual(sorted(sizes["B"]), [4, 4, 5, 5, 5])
        self.assertEqual(sorted(sizes["C"]), [4])

    def test_names_are_ordered_by_the_targets_own_column(self):
        df = _atlas_df({"A": 10})
        df["A"] = np.r_[np.full(5, 0.9), np.full(5, 0.1)]
        grouped = _group(_atlas(df)).atlas.set_index("name").loc[df["name"]]
        self.assertTrue((grouped["grg"].iloc[:5] == "A__0").all())  # highest first
        self.assertTrue((grouped["grg"].iloc[5:] == "A__1").all())

    def test_ordering_survives_missing_values_in_the_targets_own_column(self):
        # A single gap in the own column must not make the group's mean NaN and
        # push it last (most U250 targets have such gaps).
        df = _atlas_df({"A": 10})
        df["A"] = np.r_[np.full(5, 0.9), np.full(5, 0.1)]
        df["B"] = np.r_[
            np.zeros(5), np.ones(5)
        ]  # keeps row 0 grouped once A is imputed
        df.loc[0, "A"] = np.nan
        grouped = _group(_atlas(df)).atlas.set_index("name").loc[df["name"]]
        self.assertTrue((grouped["grg"].iloc[:5] == "A__0").all())
        self.assertTrue((grouped["grg"].iloc[5:] == "A__1").all())

    def test_missing_values_are_imputed(self):
        df = _atlas_df({"A": 20}, n_profiles=4)
        df.loc[[0, 5], "B"] = np.nan
        df["D"] = np.nan  # a column missing everywhere
        grouped = _group(_atlas(df)).atlas
        self.assertFalse(grouped["grg"].isna().any())
        self.assertEqual(sorted(grouped["grg"].value_counts()), [5] * 4)

    def test_rejects_atlas_without_cell_type_values(self):
        df = _atlas_df({"A": 10})
        df[CELL_TYPES] = np.nan  # regions only, as from_block_table leaves them
        with self.assertRaisesRegex(ValueError, "from_reads"):
            _group(_atlas(df))

    def test_region_shared_by_two_targets_gets_a_group_per_row(self):
        df = _atlas_df({"A": 10, "B": 10})
        shared = df.loc[df["target"] == "A"].iloc[[0]].assign(target="B")
        df = pd.concat([df, shared], ignore_index=True)
        grouped = _group(_atlas(df)).atlas
        rows = grouped[grouped["name"] == shared["name"].iloc[0]]
        self.assertEqual(sorted(rows["grg"].str[0]), ["A", "B"])

    def test_returns_a_new_atlas_and_leaves_the_original_alone(self):
        atlas = _atlas(_atlas_df({"A": 10, "B": 10}))
        grouped = _group(atlas)
        self.assertNotIn("grg", atlas.atlas.columns)
        self.assertIsInstance(grouped, UXMMethylationAtlas)
        self.assertEqual(grouped.ref_cells, atlas.ref_cells)
        self.assertEqual(grouped.atlas_name, atlas.atlas_name)
        pd.testing.assert_frame_equal(grouped.atlas.drop(columns="grg"), atlas.atlas)

    def test_label_column_matches_the_dataset_build_encoding(self):
        atlas = _atlas(_atlas_df({"A": 30, "B": 12}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atlas.tsv"
            grouped = _group(atlas, label_column="grg_label", output_path=str(path))
            saved = pd.read_csv(path, sep="\t")
            encoded, mapping = load_region_annotations(str(path), {"grg": "grg_label"})
        self.assertEqual(grouped.atlas["grg_label"].dtype.kind, "i")
        np.testing.assert_array_equal(saved["grg_label"], encoded["grg_label"])
        self.assertEqual(
            {v: k for k, v in mapping["grg_label"].items()},
            dict(zip(saved["grg"], saved["grg_label"])),
        )

    def test_output_reloads_as_an_atlas(self):
        atlas = _atlas(_atlas_df({"A": 10, "B": 10}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atlas.tsv"
            _group(atlas, output_path=str(path))
            reloaded = UXMMethylationAtlas(
                "t", "hg38", atlas_path=str(path), cell_types=CELL_TYPES
            )
        self.assertIn("grg", reloaded.atlas.columns)


if __name__ == "__main__":
    unittest.main()
