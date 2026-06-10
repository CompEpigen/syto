import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from baselines.uxm import uxm


def _make_valid_atlas_df():
    return pd.DataFrame(
        {
            "chr": ["chr1", "chr1"],
            "start": [10, 30],
            "end": [20, 40],
            "startCpG": [1, 2],
            "endCpG": [3, 4],
            "target": ["ct1", "ct2"],
            "name": ["chr1:10-20", "chr1:30-40"],
            "direction": ["U", "M"],
            "ct1": [1.0, 0.0],
            "ct2": [0.0, 1.0],
        }
    )


def _base_decon_atlas():
    return pd.DataFrame(
        {
            "name": ["m1", "m2"],
            "direction": ["U", "U"],
            "ct1": [1.0, 0.0],
            "ct2": [0.0, 1.0],
        }
    )


def _prepare_reads_atlas():
    return pd.DataFrame(
        {
            "name": ["region1", "region2"],
            "chr": ["chr1", "chr1"],
            "start": [11, 31],
            "end": [21, 41],
            "target": ["ctype_a", "ctype_b"],
        }
    )


def _prepare_reads_df():
    return pd.DataFrame(
        {
            "chromosome": ["chr1", "chr1", "chr1", "chr1", "chr2"],
            "read_name": ["r1", "r2", "r3", "r4", "r5"],
            "read_start": [10, 30, 33, 60, 5],
            "read_end": [20, 40, 36, 80, 10],
            "methylation_encoding": [
                "11111111111",
                "00000000011",
                "0011",
                "0011",
                "000",
            ],
            "seq": ["A" * 11, "C" * 11, "G" * 4, "G" * 4, "T" * 3],
            "original_label": [9, 8, 5, 7, 6],
            "label": [1, 2, 5, 3, 4],
        }
    )


class TestValidateRefTissues(unittest.TestCase):
    """Tests for uxm.validate_ref_tissues."""

    def test_validate_ref_tissues_valid(self):
        """Validate that no exception is raised for existing tissue columns."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        uxm.validate_ref_tissues(df, ["a", "b"])

    def test_validate_ref_tissues_invalid_raises_system_exit(self):
        """Validate that missing tissue columns trigger SystemExit."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with self.assertRaises(SystemExit):
            uxm.validate_ref_tissues(df, ["a", "c"])


class TestValidateFile(unittest.TestCase):
    """Tests for uxm.validate_file."""

    def test_validate_file_existing(self):
        """Return the same path when the input file exists."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("ok")
            fpath = f.name

        try:
            out = uxm.validate_file(fpath)
            self.assertEqual(out, fpath)
        finally:
            if os.path.exists(fpath):
                os.unlink(fpath)

    def test_validate_file_missing_raises_system_exit(self):
        """Raise SystemExit when validating a non-existent file path."""
        with self.assertRaises(SystemExit):
            uxm.validate_file("/tmp/definitely_missing_file_uxm_test.tsv")


class TestLoadAtlas(unittest.TestCase):
    """Tests for uxm.load_atlas."""

    def setUp(self):
        """Create a valid atlas file path available to each test method."""
        self._tmp_atlas = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".tsv"
        )
        self._tmp_atlas.close()
        _make_valid_atlas_df().to_csv(self._tmp_atlas.name, sep="\t", index=False)
        self.atlas_path = self._tmp_atlas.name

    def tearDown(self):
        """Remove atlas temp file created for the current test method."""
        if hasattr(self, "atlas_path") and os.path.exists(self.atlas_path):
            os.unlink(self.atlas_path)

    def test_load_atlas_valid(self):
        """Load a valid atlas and return expected reference cell columns."""
        atlas, ref_cells = uxm.load_atlas(self.atlas_path)
        self.assertEqual(ref_cells, ["ct1", "ct2"])
        self.assertEqual(len(atlas), 2)
        self.assertIn("target", atlas.columns)

    def test_load_atlas_ignore(self):
        """Drop ignored tissue columns and matching target rows during load."""
        atlas, ref_cells = uxm.load_atlas(self.atlas_path, ignore=["ct2"])
        self.assertEqual(ref_cells, ["ct1"])
        self.assertNotIn("ct2", atlas.columns)
        self.assertTrue((atlas["target"] != "ct2").all())

    def test_load_atlas_include(self):
        """Keep only included tissue rows/columns during atlas loading."""
        atlas, ref_cells = uxm.load_atlas(self.atlas_path, include=["ct1"])
        self.assertEqual(ref_cells, ["ct1"])
        self.assertTrue((atlas["target"] == "ct1").all())
        self.assertEqual(list(atlas.columns[:8]) + ["ct1"], list(atlas.columns))

    def test_load_atlas_too_few_columns_raises(self):
        """Raise SystemExit when atlas has fewer than the required columns."""
        bad = pd.DataFrame(
            {
                "name": ["chr1:10-20"],
                "direction": ["U"],
                "chr": ["chr1"],
                "start": [10],
                "end": [20],
                "strand": ["U"],
                "target": ["ct1"],
            }
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as f:
            bad.to_csv(f.name, sep="\t", index=False)
            atlas_path = f.name

        try:
            with self.assertRaises(SystemExit):
                uxm.load_atlas(atlas_path)
        finally:
            if os.path.exists(atlas_path):
                os.unlink(atlas_path)

    def test_load_atlas_invalid_name_prefix_raises(self):
        """
        Raise SystemExit when any atlas name does not start with 'chr'.
        """
        df = _make_valid_atlas_df()
        df.loc[0, "name"] = "1:10-20"
        df.to_csv(self.atlas_path, sep="\t", index=False)

        with self.assertRaises(SystemExit):
            uxm.load_atlas(self.atlas_path)


class TestDeconSingleSamp(unittest.TestCase):
    """Tests for uxm.decon_single_samp."""

    def test_decon_single_samp_happy_path(self):
        """Recover normalized mixture proportions for a well-formed sample."""
        atlas = _base_decon_atlas()
        samp = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [0.8, 0.2]}
        )
        counts = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [1, 1]}
        )

        mixture = uxm.decon_single_samp(samp, atlas, counts, verbose=True)
        self.assertEqual(len(mixture), 2)
        self.assertAlmostEqual(float(np.sum(mixture)), 1.0, places=8)
        self.assertAlmostEqual(float(mixture[0]), 0.8, places=6)
        self.assertAlmostEqual(float(mixture[1]), 0.2, places=6)

    def test_decon_single_samp_happy_path_non_uniform_counts(self):
        """
        Recover normalized mixture proportions for a well-formed sample
        with non-uniform counts.
        """
        atlas = _base_decon_atlas()
        samp = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [0.8, 0.2]}
        )
        counts = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [45, 5]}
        )

        mixture = uxm.decon_single_samp(samp, atlas, counts, verbose=False)
        self.assertEqual(len(mixture), 2)
        self.assertAlmostEqual(float(np.sum(mixture)), 1.0, places=8)
        self.assertAlmostEqual(float(mixture[0]), 0.8, places=6)
        self.assertAlmostEqual(float(mixture[1]), 0.2, places=6)

    def test_decon_single_samp_empty_merge_returns_nan_tuple(self):
        """Return NaN tuple when sample markers do not overlap the atlas."""
        atlas = _base_decon_atlas()
        samp = pd.DataFrame({"name": ["x"], "direction": ["U"], "s1": [1.0]})
        counts = pd.DataFrame({"name": ["x"], "direction": ["U"], "s1": [1]})

        out = uxm.decon_single_samp(samp, atlas, counts, verbose=False)
        self.assertEqual(len(out), 2)
        self.assertTrue(np.isnan(out[0]))
        self.assertTrue(np.isnan(out[1]))

    def test_decon_single_samp_merge_error_returns_none_tuple(self):
        """Return (None, None) when merge size indicates invalid atlas/sample merge."""
        atlas = _base_decon_atlas()
        samp = pd.DataFrame(
            {
                "name": ["m1", "m1", "m2"],
                "direction": ["U", "U", "U"],
                "s1": [5.0, 5.0, 1.0],
            }
        )
        counts = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [1, 1]}
        )

        out = uxm.decon_single_samp(samp, atlas, counts, verbose=False)
        self.assertEqual(out, (None, None))


class TestUxmDeconvolution(unittest.TestCase):
    """Tests for uxm.uxm_deconvolution."""

    def test_uxm_deconvolution_multiple_samples(self):
        """Run deconvolution for multiple samples and preserve sample-specific mixes."""
        atlas = _base_decon_atlas()
        ref_cells = ["ct1", "ct2"]
        sf = pd.DataFrame(
            {
                "name": ["m1", "m2"],
                "direction": ["U", "U"],
                "s1": [0.8, 0.2],
                "s2": [0.1, 0.9],
            }
        )
        counts = pd.DataFrame(
            {
                "name": ["m1", "m2"],
                "direction": ["U", "U"],
                "s1": [5, 2],
                "s2": [1, 1],
            }
        )

        arr = uxm.uxm_deconvolution(
            atlas, ref_cells, sf, counts, sample_names=["s1", "s2"]
        )
        self.assertEqual(len(arr), 2)
        self.assertAlmostEqual(float(arr[0][0]), 0.8, places=6)
        self.assertAlmostEqual(float(arr[0][1]), 0.2, places=6)
        self.assertAlmostEqual(float(arr[1][0]), 0.1, places=6)
        self.assertAlmostEqual(float(arr[1][1]), 0.9, places=6)


class TestRearangeUxmDeconvolutionResults(unittest.TestCase):
    """Tests for uxm.rearange_uxm_deconvolution_results."""

    def test_rearange_uxm_deconvolution_results(self):
        """Reorder proportions to align with label index order from 0 to 38."""
        labels_dict_reversed = {f"ct{i}": i for i in range(39)}
        # Build a non-trivial order to ensure reordering is actually tested.
        perm = list(range(38, -1, -1))
        ref_cells = [f"ct{i}" for i in perm]
        uxm_proportions = [i / 100.0 for i in perm]

        out = uxm.rearange_uxm_deconvolution_results(
            labels_dict_reversed=labels_dict_reversed,
            uxm_proportions=uxm_proportions,
            ref_cells=ref_cells,
        )

        self.assertEqual(len(out), 39)
        self.assertEqual(out, [i / 100.0 for i in range(39)])


class TestPrepareReadsForUXM(unittest.TestCase):
    """Tests for uxm.prepare_reads_for_uxm.
    These tests assume a particular specification for the function since
    no docstring was provided at the time of writing. If the function's behavior or
      expected output format changes, these tests may need to be updated accordingly.

    In particular, the tests assume that the function expects reads_data to be sorted
    by chromosome and then by read_start.
    """

    def test_prepare_reads_for_uxm_basic(self):
        """Prepare overlapping reads and compute expected UXM classification fields."""
        reads_data = _prepare_reads_df()
        atlas = _prepare_reads_atlas()
        labels_dict = {0: "ctype_a", 1: "ctype_b"}
        cell_type_match_dict = {"ctype_b": "ctype_b"}

        res = uxm.prepare_reads_for_uxm(
            reads_data=reads_data,
            atlas=atlas,
            labels_dict=labels_dict,
            cell_type_match_dict=cell_type_match_dict,
            debug=False,
        )

        self.assertEqual(len(res), 3)
        self.assertIn("record_M", res.columns)
        self.assertIn("record_U", res.columns)
        self.assertIn("record_X", res.columns)
        self.assertIn("dmr_ctype_label", res.columns)

        r1 = res[res["read_name"] == "r1"].iloc[0]
        self.assertEqual(r1["record_M"], 1)
        self.assertEqual(r1["record_U"], 0)
        self.assertEqual(r1["record_X"], 0)
        self.assertEqual(r1["dmr_ctype_label"], 0)

        r2 = res[res["read_name"] == "r2"].iloc[0]
        self.assertEqual(r2["record_M"], 0)
        self.assertEqual(r2["record_U"], 1)
        self.assertEqual(r2["record_X"], 0)
        self.assertEqual(r2["dmr_ctype_label"], 1)

        r3 = res[res["read_name"] == "r3"].iloc[0]
        self.assertEqual(r3["record_M"], 0)
        self.assertEqual(r3["record_U"], 0)
        self.assertEqual(r3["record_X"], 1)
        self.assertEqual(r3["dmr_ctype_label"], 1)

        # Reads that do not overlap atlas regions should be excluded.
        self.assertNotIn("r4", set(res["read_name"]))
        self.assertNotIn("r5", set(res["read_name"]))

    def test_prepare_reads_for_uxm_debug_columns_present(self):
        """Include verbose trimming/debug columns when debug mode is enabled."""
        reads_data = _prepare_reads_df()
        atlas = _prepare_reads_atlas()
        labels_dict = {0: "ctype_a", 1: "ctype_b"}

        res = uxm.prepare_reads_for_uxm(
            reads_data=reads_data,
            atlas=atlas,
            labels_dict=labels_dict,
            cell_type_match_dict={},
            debug=True,
        )

        self.assertIn("record_start", res.columns)
        self.assertIn("record_end", res.columns)
        self.assertIn("offset_start", res.columns)
        self.assertIn("offset_end", res.columns)
        self.assertIn("original_pattern", res.columns)
        self.assertIn("original_seq", res.columns)

    def test_prepare_reads_for_uxm_drops_unmapped_cell_types(self):
        """Drop rows whose atlas cell types cannot be mapped to label IDs."""
        reads_data = _prepare_reads_df()
        atlas = _prepare_reads_atlas().copy()
        atlas.loc[:, "target"] = ["unknown_a", "unknown_b"]

        res = uxm.prepare_reads_for_uxm(
            reads_data=reads_data,
            atlas=atlas,
            labels_dict={0: "ctype_a"},
            cell_type_match_dict={},
            debug=False,
        )

        self.assertEqual(len(res), 0)


if __name__ == "__main__":
    unittest.main()
