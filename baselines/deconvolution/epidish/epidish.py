"""
EpiDISH reference-based deconvolution — Python port.

Ports the RPC, CBS, and CP estimators from the EpiDISH R package
(https://github.com/sjczheng/EpiDISH, GPL-2). See LICENSE.md.
"""

import cvxpy as cp
import numpy as np
import statsmodels.api as sm
from sklearn.svm import NuSVR
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


def _do_cbs(mixture: np.ndarray, ref: np.ndarray, nu_v=(0.25, 0.5, 0.75)) -> np.ndarray:
    """CIBERSORT (EpiDISH CBS): linear nu-SVR over candidate nu values.

    Reproduces e1071's ``scale=TRUE`` by z-scoring feature columns and the
    response before fitting. For each nu, coefficients (== t(coefs) %*% SV) are
    clipped and normalized; the nu minimizing reconstruction RMSE in the
    original beta space is selected.
    """
    mixture = np.asarray(mixture, dtype=float)
    ref = np.asarray(ref, dtype=float)

    ref_mean = ref.mean(axis=0)
    ref_std = ref.std(axis=0)
    ref_std[ref_std == 0] = 1.0
    ref_s = (ref - ref_mean) / ref_std

    y_std = mixture.std()
    y_std = y_std if y_std > 0 else 1.0
    y_s = (mixture - mixture.mean()) / y_std

    best_coef = None
    best_rmse = np.inf
    for nu in nu_v:
        model = NuSVR(kernel="linear", nu=nu).fit(ref_s, y_s)
        coef = _normalize_nonneg(model.coef_.ravel())
        rmse = float(np.sqrt(np.mean((mixture - ref @ coef) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_coef = coef
    return best_coef


def _do_cp(mixture: np.ndarray, ref: np.ndarray, constraint: str = "inequality") -> np.ndarray:
    """Constrained Projection (EpiDISH CP, Houseman): quadratic program.

    Minimizes ||ref @ w - mixture||^2 subject to w >= 0 and either sum(w) <= 1
    (``inequality``) or sum(w) == 1 (``equality``). EpiDISH's inequality mode
    does not renormalize; here the result is normalized to sum 1 for
    consistency with the other baselines (a no-op under a perfect fit).
    """
    mixture = np.asarray(mixture, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n_ct = ref.shape[1]

    w = cp.Variable(n_ct)
    objective = cp.Minimize(cp.sum_squares(ref @ w - mixture))
    if constraint == "equality":
        constraints = [w >= 0, cp.sum(w) == 1]
    else:
        constraints = [w >= 0, cp.sum(w) <= 1]
    cp.Problem(objective, constraints).solve()

    return _normalize_nonneg(np.asarray(w.value, dtype=float).ravel())
