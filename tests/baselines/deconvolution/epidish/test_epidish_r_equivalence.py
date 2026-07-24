"""Regression test: our EpiDISH port reproduces R EpiDISH's estimates.

The reference numbers in ``fixtures/r_epidish_results.csv`` were produced by the
R EpiDISH package (v4.5.x, Bioconductor) via ``epidish(beta.m, ref.m, ...)`` on
the exact synthetic matrices in ``fixtures/ref.csv`` / ``fixtures/beta.csv``.
They are frozen here so this test needs no R at run time — it guards that future
changes to ``_do_rpc`` / ``_do_cbs`` / ``_do_cp`` still match the reference
implementation.

Observed agreement (max abs diff over 20 cells per method):
    RPC 2.3e-04 · CP 3.3e-04 · CBS 2.7e-04
RPC uses a numpy Huber-IWLS reimplementation of MASS::rlm (chosen over
statsmodels for ~20x throughput); it tracks R to <1e-3 rather than the ~1e-6 of
the statsmodels fit. CP/CBS differ only by convex-solver / libsvm numerics.
Tolerances below are set comfortably above the observed differences.
"""

import os
import unittest

import numpy as np
import pandas as pd

from baselines.deconvolution.epidish.epidish import _do_cbs, _do_cp, _do_rpc

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")

# Per-method absolute tolerance (element-wise). RPC ports the same IWLS Huber
# estimator as MASS::rlm and matches to floating point; CP/CBS differ only by
# convex-solver / libsvm numerics.
_TOL = {"RPC": 1e-3, "CBS": 1e-3, "CP": 1e-3}


def _estimate(method: str, beta: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if method == "RPC":
        return _do_rpc(beta, ref)
    if method == "CBS":
        return _do_cbs(beta, ref)
    return _do_cp(beta, ref, constraint="inequality")


class TestEpiDishREquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref_df = pd.read_csv(os.path.join(_FIX, "ref.csv"), index_col=0)
        beta_df = pd.read_csv(os.path.join(_FIX, "beta.csv"), index_col=0)
        cls.ref = ref_df.to_numpy()  # (N, T)
        cls.beta = beta_df.to_numpy()  # (N, S)
        cls.ct_names = list(ref_df.columns)  # CT1..CT4
        cls.samp_names = list(beta_df.columns)  # S1..S5
        cls.r = pd.read_csv(os.path.join(_FIX, "r_epidish_results.csv"))

    def test_matches_r_epidish(self):
        # Build our port's estimates in the same tidy long form as the R output.
        py_rows = []
        for method in ("RPC", "CBS", "CP"):
            for k, samp in enumerate(self.samp_names):
                frac = _estimate(method, self.beta[:, k], self.ref)
                for j, ct in enumerate(self.ct_names):
                    py_rows.append((method, samp, ct, float(frac[j])))
        py = pd.DataFrame(py_rows, columns=["method", "sample", "cell_type", "python"])

        merged = py.merge(self.r, on=["method", "sample", "cell_type"])
        self.assertEqual(len(merged), 60, "expected 3 methods x 5 samples x 4 CTs")

        for method, g in merged.groupby("method"):
            max_abs = float((g["python"] - g["r"]).abs().max())
            self.assertLess(
                max_abs,
                _TOL[method],
                msg=f"{method}: max abs diff {max_abs:.2e} exceeds tol {_TOL[method]:.0e}",
            )
