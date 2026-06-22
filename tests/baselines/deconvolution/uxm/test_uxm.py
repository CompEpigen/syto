import unittest
from unittest import mock

import numpy as np
import pandas as pd

from baselines.deconvolution.uxm import uxm
from baselines.deconvolution.uxm.uxm import UXMDeconvolver


def _base_decon_atlas():
    return pd.DataFrame(
        {
            "name": ["m1", "m2"],
            "direction": ["U", "U"],
            "ct1": [1.0, 0.0],
            "ct2": [0.0, 1.0],
        }
    )


def _make_deconvolver():
    """Minimal UXMDeconvolver whose atlas is never accessed by NNLS helpers."""
    mock_atlas = mock.MagicMock()
    mock_atlas.ref_cells = ["ct1", "ct2"]
    return UXMDeconvolver(mock_atlas)


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


class TestDeconSingleSamp(unittest.TestCase):
    """Tests for UXMDeconvolver.deconvolute_single_sample."""

    def setUp(self):
        self.d = _make_deconvolver()

    def test_decon_single_samp_happy_path(self):
        """Recover normalized mixture proportions for a well-formed sample."""
        atlas = _base_decon_atlas()
        samp = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [0.8, 0.2]}
        )
        counts = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [1, 1]}
        )

        mixture = self.d.deconvolute_single_sample(samp, atlas, counts, verbose=True)
        self.assertEqual(len(mixture), 2)
        self.assertAlmostEqual(float(np.sum(mixture)), 1.0, places=8)
        self.assertAlmostEqual(float(mixture[0]), 0.8, places=6)
        self.assertAlmostEqual(float(mixture[1]), 0.2, places=6)

    def test_decon_single_samp_happy_path_non_uniform_counts(self):
        """Recover normalized mixture proportions with non-uniform counts."""
        atlas = _base_decon_atlas()
        samp = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [0.8, 0.2]}
        )
        counts = pd.DataFrame(
            {"name": ["m1", "m2"], "direction": ["U", "U"], "s1": [45, 5]}
        )

        mixture = self.d.deconvolute_single_sample(samp, atlas, counts, verbose=False)
        self.assertEqual(len(mixture), 2)
        self.assertAlmostEqual(float(np.sum(mixture)), 1.0, places=8)
        self.assertAlmostEqual(float(mixture[0]), 0.8, places=6)
        self.assertAlmostEqual(float(mixture[1]), 0.2, places=6)

    def test_decon_single_samp_empty_merge_returns_nan_tuple(self):
        """Return NaN tuple when sample markers do not overlap the atlas."""
        atlas = _base_decon_atlas()
        samp = pd.DataFrame({"name": ["x"], "direction": ["U"], "s1": [1.0]})
        counts = pd.DataFrame({"name": ["x"], "direction": ["U"], "s1": [1]})

        out = self.d.deconvolute_single_sample(samp, atlas, counts, verbose=False)
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

        out = self.d.deconvolute_single_sample(samp, atlas, counts, verbose=False)
        self.assertEqual(out, (None, None))


class TestUxmDeconvolution(unittest.TestCase):
    """Tests for UXMDeconvolver.deconvolute_multiple_samples."""

    def setUp(self):
        self.d = _make_deconvolver()

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

        arr = self.d.deconvolute_multiple_samples(
            atlas, ref_cells, sf, counts, sample_names=["s1", "s2"]
        )
        self.assertEqual(len(arr), 2)
        self.assertAlmostEqual(float(arr[0][0]), 0.8, places=6)
        self.assertAlmostEqual(float(arr[0][1]), 0.2, places=6)
        self.assertAlmostEqual(float(arr[1][0]), 0.1, places=6)
        self.assertAlmostEqual(float(arr[1][1]), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
