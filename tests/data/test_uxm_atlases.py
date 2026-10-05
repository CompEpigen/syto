"""Tests for UXMMethylationAtlas custom cell types and block/marker import."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from syto.data.atlases.uxm_atlases import UXMMethylationAtlas

BED_HEADER = (
    "#chr\tstart\tend\tstartCpG\tendCpG\ttarget\tregion\tlenCpG\tbp\t"
    "tg_mean\tbg_mean\tdelta_means\tdelta_quants\tdelta_maxmin\tttest\tdirection\n"
)


def _write_bed(directory, group, rows):
    path = Path(directory) / f"Markers.{group}.bed"
    lines = [
        f"{c}\t{s}\t{e}\t{sc}\t{ec}\t{group}\t{c}:{s}-{e}\t5CpGs\t{e - s}bp\t"
        f"{tg}\t{bg}\t0.9\t0.9\t0.9\t0.001\t{d}\n"
        for c, s, e, sc, ec, tg, bg, d in rows
    ]
    path.write_text(BED_HEADER + "".join(lines))
    return str(path)


def _region_df(targets):
    return pd.DataFrame(
        {
            "chr": ["chr1"] * len(targets),
            "start": [100 * (i + 1) for i in range(len(targets))],
            "end": [100 * (i + 1) + 50 for i in range(len(targets))],
            "startCpG": [10 * (i + 1) for i in range(len(targets))],
            "endCpG": [10 * (i + 1) + 5 for i in range(len(targets))],
            "target": targets,
            "name": [f"r{i}" for i in range(len(targets))],
            "direction": ["U"] * len(targets),
        }
    )


class TestCustomCellTypes(unittest.TestCase):
    def test_default_still_requires_loyfer_columns(self):
        df = _region_df(["GroupA"])
        df["GroupA"] = 0.5
        with self.assertRaisesRegex(ValueError, "missing expected cell type"):
            UXMMethylationAtlas("t", "hg19", atlas_df=df)

    def test_custom_cell_types_drive_validation_ref_cells_and_rows(self):
        df = _region_df(["GroupA", "GroupB", "Other"])
        df["GroupA"] = 0.1
        df["GroupB"] = 0.9
        atlas = UXMMethylationAtlas(
            "t", "hg19", atlas_df=df, cell_types=["GroupA", "GroupB"]
        )
        self.assertEqual(atlas.ref_cells, ["GroupA", "GroupB"])
        # Rows targeting a group outside cell_types are dropped, as for Loyfer.
        self.assertEqual(sorted(atlas.atlas["target"]), ["GroupA", "GroupB"])

        with self.assertRaisesRegex(ValueError, "GroupC"):
            UXMMethylationAtlas(
                "t", "hg19", atlas_df=df, cell_types=["GroupA", "GroupC"]
            )


class TestFromBlockTableMarkers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = [
            _write_bed(
                self.tmp.name,
                "Medulloblastoma",
                [("9", 1000, 1200, 10, 15, 0.1, 0.9, "U")],
            ),
            _write_bed(
                self.tmp.name,
                "Healthy CSF",
                [
                    ("22", 500, 600, 20, 26, 0.2, 0.8, "U"),
                    ("X", 700, 800, 30, 35, 0.95, 0.05, "M"),
                ],
            ),
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_concatenates_marker_files_into_regions(self):
        output = Path(self.tmp.name) / "atlas.tsv"
        atlas = UXMMethylationAtlas.from_block_table(
            self.paths, "dkfz", "hg19", output_path=str(output)
        )
        df = atlas.atlas.set_index("name")

        self.assertEqual(atlas.ref_cells, ["Healthy CSF", "Medulloblastoma"])
        self.assertEqual(
            sorted(df.index), ["chr22:500-600", "chr9:1000-1200", "chrX:700-800"]
        )
        self.assertEqual(sorted(df["chr"]), ["chr22", "chr9", "chrX"])
        self.assertEqual(df.loc["chrX:700-800", "direction"], "M")
        self.assertNotIn("region", df.columns)
        # Fractions are left for from_reads; find_markers stats ride along.
        self.assertTrue(df[["Healthy CSF", "Medulloblastoma"]].isna().all().all())
        self.assertAlmostEqual(df.loc["chr9:1000-1200", "tg_mean"], 0.1)

        reloaded = UXMMethylationAtlas(
            "dkfz",
            "hg19",
            atlas_path=str(output),
            cell_types=["Healthy CSF", "Medulloblastoma"],
        )
        self.assertEqual(len(reloaded.atlas), 3)

    def test_wgbstools_markers_alias_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            atlas = UXMMethylationAtlas.from_wgbstools_markers(
                self.paths, "dkfz", "hg19"
            )
        self.assertEqual(len(atlas.atlas), 3)


class TestFromBlockTable(unittest.TestCase):
    """Block tables define regions only; fractions come from reads."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.blocks_path = Path(self.tmp.name) / "blocks.tsv"
        # Post-processed block table: target and direction are per-cluster
        # proxies held in custom columns.
        pd.DataFrame(
            {
                "chr": ["1", "2"],
                "start": [100, 500],
                "end": [200, 600],
                "startCpG": [10, 30],
                "endCpG": [15, 36],
                "name": ["1:100-200", "2:500-600"],
                "cluster": [7, 9],
                "cluster_target": ["GroupA", "GroupB"],
                "cluster_direction": ["hypo", "hyper"],
                "cluster_target_score": [1.5, -0.2],
            }
        ).to_csv(self.blocks_path, sep="\t", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _block_atlas(self):
        return UXMMethylationAtlas.from_block_table(
            str(self.blocks_path),
            "blocks",
            "hg19",
            target_column="cluster_target",
            direction_column="cluster_direction",
        )

    def test_defines_regions_with_empty_cell_type_columns(self):
        atlas = self._block_atlas()
        df = atlas.atlas.set_index("name")

        self.assertEqual(atlas.ref_cells, ["GroupA", "GroupB"])
        self.assertEqual(sorted(df.index), ["chr1:100-200", "chr2:500-600"])
        self.assertEqual(list(df["target"]), ["GroupA", "GroupB"])
        self.assertEqual(list(df["direction"]), ["U", "M"])
        self.assertTrue(df[["GroupA", "GroupB"]].isna().all().all())

        # Extra block columns survive so finalize can filter on them later.
        self.assertEqual(list(df["cluster"]), [7, 9])
        self.assertIn("cluster_target_score", df.columns)
        self.assertNotIn("cluster_target", df.columns)

    def test_from_reads_fills_fractions_and_keeps_block_columns(self):
        reads = pd.DataFrame(
            {
                "name": ["chr1:100-200"] * 4 + ["chr2:500-600"] * 3,
                "original_label": [0, 0, 0, 1, 0, 1, 1],
                "pattern": [
                    "0000",
                    "0000",
                    "1111",
                    "1111",
                    "1111",
                    "0000",
                    "0101",
                ],
            }
        )
        atlas = UXMMethylationAtlas.from_reads(
            reads,
            "blocks",
            "hg19",
            labels_dict={0: "GroupA", 1: "GroupB"},
            markers=self._block_atlas(),
        )
        df = atlas.atlas.set_index("name")

        # U block: fraction of U reads; M block: fraction of M reads.
        self.assertAlmostEqual(df.loc["chr1:100-200", "GroupA"], 0.667)
        self.assertAlmostEqual(df.loc["chr1:100-200", "GroupB"], 0.0)
        self.assertAlmostEqual(df.loc["chr2:500-600", "GroupA"], 1.0)
        self.assertAlmostEqual(df.loc["chr2:500-600", "GroupB"], 0.0)

        self.assertEqual(list(df["cluster"]), [7, 9])
        self.assertIn("cluster_target_score", df.columns)
        self.assertEqual(list(df.columns).count("GroupA"), 1)

    def test_rejects_unknown_direction(self):
        bad = pd.read_csv(self.blocks_path, sep="\t")
        bad["cluster_direction"] = ["sideways", "hyper"]
        bad.to_csv(self.blocks_path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "sideways"):
            self._block_atlas()


if __name__ == "__main__":
    unittest.main()
