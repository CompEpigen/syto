"""Tests for UXMMethylationAtlas custom cell types and wgbstools marker import."""

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


class TestFromWgbstoolsMarkers(unittest.TestCase):
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

    def test_builds_prefixed_atlas_with_group_columns(self):
        output = Path(self.tmp.name) / "atlas.tsv"
        atlas = UXMMethylationAtlas.from_wgbstools_markers(
            self.paths, "dkfz", "hg19", output_path=str(output)
        )
        df = atlas.atlas.set_index("name")

        self.assertEqual(atlas.ref_cells, ["Healthy CSF", "Medulloblastoma"])
        self.assertEqual(
            sorted(df.index), ["chr22:500-600", "chr9:1000-1200", "chrX:700-800"]
        )
        self.assertEqual(sorted(df["chr"]), ["chr22", "chr9", "chrX"])

        # U marker: own group gets 1 - tg_mean, the rest 1 - bg_mean.
        self.assertAlmostEqual(df.loc["chr9:1000-1200", "Medulloblastoma"], 0.9)
        self.assertAlmostEqual(df.loc["chr9:1000-1200", "Healthy CSF"], 0.1)
        # M marker: means are kept as methylated fractions.
        self.assertAlmostEqual(df.loc["chrX:700-800", "Healthy CSF"], 0.95)
        self.assertAlmostEqual(df.loc["chrX:700-800", "Medulloblastoma"], 0.05)

        reloaded = UXMMethylationAtlas(
            "dkfz",
            "hg19",
            atlas_path=str(output),
            cell_types=["Healthy CSF", "Medulloblastoma"],
        )
        self.assertEqual(len(reloaded.atlas), 3)


if __name__ == "__main__":
    unittest.main()
