"""
EpiDISH reference-based deconvolution — Python port.

Ports the RPC, CBS, and CP estimators from the EpiDISH R package
(https://github.com/sjczheng/EpiDISH, GPL-2). See LICENSE.md.
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT
from statsmodels.robust.robust_linear_model import RLM


def _normalize_nonneg(coef: np.ndarray) -> np.ndarray:
    """Clip negatives to 0 and normalize to sum 1 (uniform if all <= 0)."""
    coef = np.asarray(coef, dtype=float).copy()
    coef[coef < 0] = 0.0
    total = coef.sum()
    if total <= 0:
        return np.full(coef.shape, 1.0 / coef.size)
    return coef / total


def _do_rpc(mixture: np.ndarray, ref: np.ndarray, maxit: int = 50) -> np.ndarray:
    """Robust Partial Correlations (EpiDISH RPC).

    Robust linear regression (Huber M-estimation via IWLS) of the mixture
    beta-vector on the reference centroid columns; intercept dropped, negative
    coefficients clipped, result normalized. Mirrors MASS::rlm defaults.
    """
    X = sm.add_constant(np.asarray(ref, dtype=float))  # intercept is column 0
    res = RLM(np.asarray(mixture, dtype=float), X, M=HuberT()).fit(maxiter=maxit)
    coef = np.asarray(res.params)[1:]  # drop intercept (R: coef[2:(N+1)])
    return _normalize_nonneg(coef)
