import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from App.pseudobulk_deconvolution_pipeline import _run_baseline_on_reads
from baselines.deconvolution.epidish.epidish import EpiDishDeconvolver


def _mock_atlas(beta, cpg_lookup=None, n_cpgs=5, ref_cells=("ct1", "ct2")):
    atlas = MagicMock()
    atlas.__contains__ = MagicMock(return_value=True)
    atlas.get_cpg_lookup.return_value = cpg_lookup or {
        10: 0,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
    }
    atlas.get_n_cpgs.return_value = n_cpgs
    atlas.get_beta_for_regions.return_value = beta
    atlas.ref_cells = list(ref_cells)
    return atlas


def _reads():
    pats = ["11001", "11001", "11001"]
    return pd.DataFrame(
        {
            "name": ["r1"] * len(pats),
            "read_start": [10] * len(pats),
            "pattern": pats,
            "chromosome": ["chr1"] * len(pats),
            "read_end": [15] * len(pats),
        }
    )


class TestEpidishInPseudobulkWorker(unittest.TestCase):
    """EpiDISH runs through the pseudobulk pipeline's unified worker helper."""

    def _state(self):
        return {
            "labels_dict_reversed": {"ct1": 0, "ct2": 1},
            "labels_dict": {0: "ct1", 1: "ct2"},
            "n_labels": 2,
        }

    def test_run_baseline_on_reads_packages_epidish_results(self):
        beta = [np.array([[0.9, 0.85, 0.2, 0.15, 0.5], [0.1, 0.2, 0.8, 0.75, 0.45]])]
        dec = EpiDishDeconvolver(_mock_atlas(beta), method="CP")
        dec.name = "epidish_cp"  # as set by the pipeline loader

        out = _run_baseline_on_reads(
            _reads(),
            dec,
            self._state(),
            pb_index=0,
            target_proportions=np.array([0.7, 0.3]),
        )

        # Output is keyed by the (method-disambiguated) result label.
        self.assertIn("epidish_cp", out)
        rows = out["epidish_cp"]
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["cell_type"] for r in rows}, {"ct1", "ct2"})
        total = sum(r["epidish_cp"] for r in rows)
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_rpc_default_label(self):
        beta = [np.array([[0.9, 0.85, 0.2, 0.15, 0.5], [0.1, 0.2, 0.8, 0.75, 0.45]])]
        dec = EpiDishDeconvolver(_mock_atlas(beta), method="RPC")
        dec.name = "epidish"

        out = _run_baseline_on_reads(
            _reads(),
            dec,
            self._state(),
            pb_index=3,
            target_proportions=np.array([0.7, 0.3]),
        )
        self.assertIn("epidish", out)
        self.assertEqual(out["epidish"][0]["pb_index"], 3)


if __name__ == "__main__":
    unittest.main()
