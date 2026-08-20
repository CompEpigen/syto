"""
EpiDISH reference-based deconvolution — Python port.

Ports the RPC, CBS, and CP estimators from the EpiDISH R package
(https://github.com/sjczheng/EpiDISH, GPL-2). See LICENSE.md.
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp
from sklearn.svm import NuSVR

from baselines.deconvolution.base import BaselineDeconvolver
from baselines.deconvolution.utils import rearange_deconvolution_results
from syto.data.dataset import resolve_column

_module_logger = logging.getLogger(__name__)


def _normalize_nonneg(coef: np.ndarray) -> np.ndarray:
    """Clip negatives to 0 and normalize to sum 1 (uniform if all <= 0)."""
    coef = np.asarray(coef, dtype=float).copy()
    coef[coef < 0] = 0.0
    total = coef.sum()
    if total <= 0:
        return np.full(coef.shape, 1.0 / coef.size)
    return coef / total


def _do_rpc(
    mixture: np.ndarray,
    ref: np.ndarray,
    maxit: int = 50,
    huber_k: float = 1.345,
    tol: float = 1e-4,
) -> np.ndarray:
    """Robust Partial Correlations (EpiDISH RPC) via Huber IWLS.

    Robust linear regression of the mixture beta-vector on the reference
    centroid columns (with intercept), using iteratively reweighted least
    squares with Huber weights (tuning constant ``huber_k=1.345``, MAD scale) —
    a direct numpy reimplementation of ``MASS::rlm``'s default estimator.  This
    matches R EpiDISH to <1e-3 while avoiding statsmodels' large per-fit
    overhead (~20x faster), which matters when deconvolving many pseudobulks.
    Intercept dropped, negative coefficients clipped, result normalized.
    """
    ref = np.asarray(ref, dtype=float)
    y = np.asarray(mixture, dtype=float)
    X = np.column_stack([np.ones(len(ref)), ref])  # intercept is column 0
    coef = np.linalg.lstsq(X, y, rcond=None)[0]  # OLS start

    for _ in range(maxit):
        resid = y - X @ coef
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745  # MAD
        if scale < 1e-12:
            break  # (near-)perfect fit: residuals carry no scale
        u = resid / (scale * huber_k)
        weights = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.abs(u))  # Huber psi
        Xw = X * weights[:, None]
        gram = Xw.T @ X
        rhs = Xw.T @ y
        try:
            new_coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            new_coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        denom = np.max(np.abs(coef)) + 1e-12
        converged = np.max(np.abs(new_coef - coef)) < tol * denom
        coef = new_coef
        if converged:
            break

    return _normalize_nonneg(coef[1:])  # drop intercept (R: coef[2:(N+1)])


def _do_cbs(mixture: np.ndarray, ref: np.ndarray, nu_v=(0.25, 0.5, 0.75)) -> np.ndarray:
    """CIBERSORT (EpiDISH CBS): linear nu-SVR over candidate nu values.

    Reproduces e1071's ``scale=TRUE`` by z-scoring feature columns and the
    response before fitting. For each nu, coefficients (== t(coefs) %*% SV) are
    clipped and normalized; the nu minimizing reconstruction RMSE in the
    original beta space is selected.
    """
    mixture = np.asarray(mixture, dtype=float)
    ref = np.asarray(ref, dtype=float)

    if ref.shape[0] > 3000:
        _module_logger.warning(
            "CBS (nu-SVR) scales super-linearly with feature count (%d CpGs) and "
            "is slow at atlas scale; prefer RPC or CP for large runs.",
            ref.shape[0],
        )

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


def _do_cp(
    mixture: np.ndarray, ref: np.ndarray, constraint: str = "inequality"
) -> np.ndarray:
    """Constrained Projection (EpiDISH CP, Houseman): quadratic program via OSQP.

    Minimizes ``||ref @ w - mixture||^2`` subject to ``w >= 0`` and either
    ``sum(w) <= 1`` (``inequality``) or ``sum(w) == 1`` (``equality``).  Solved
    directly with OSQP — the same solver cvxpy dispatched to internally — which
    avoids cvxpy's per-solve canonicalization overhead (~60x faster) at
    identical accuracy.  EpiDISH's inequality mode does not renormalize; here
    the result is normalized to sum 1 for consistency with the other baselines
    (a no-op under a perfect fit).
    """
    mixture = np.asarray(mixture, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n_ct = ref.shape[1]

    # 1/2 w' P w + q' w  ==  ||ref w - mixture||^2  (up to a constant)
    gram = ref.T @ ref
    P = sp.csc_matrix(2.0 * gram)
    q = -2.0 * (ref.T @ mixture)

    # Constraint rows: [ sum(w) ; w_1 ; ... ; w_T ].
    A = sp.vstack([sp.csc_matrix(np.ones((1, n_ct))), sp.eye(n_ct)]).tocsc()
    sum_lo = 1.0 if constraint == "equality" else 0.0
    lower = np.concatenate([[sum_lo], np.zeros(n_ct)])  # sum in [lo,1]; w_i >= 0
    upper = np.concatenate([[1.0], np.full(n_ct, np.inf)])

    solver = osqp.OSQP()
    solver.setup(
        P,
        q,
        A,
        lower,
        upper,
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=20000,
    )
    res = solver.solve()

    x = np.asarray(res.x, dtype=float)
    if not np.all(np.isfinite(x)):
        _module_logger.warning(
            "OSQP returned a non-finite CP solution (status=%s); using uniform.",
            res.info.status,
        )
        x = np.ones(n_ct)
    return _normalize_nonneg(x)


_METHOD_RESULT_NAME = {"RPC": "epidish", "CBS": "epidish_cbs", "CP": "epidish_cp"}


def epidish_result_name(method: str, name: Optional[str] = None) -> str:
    """Result label for an EpiDISH baseline entry.

    An explicit ``name`` always wins.  Otherwise the label is derived from the
    method so multiple EpiDISH entries (e.g. RPC + CP) don't collide under the
    same key: ``RPC -> 'epidish'``, ``CBS -> 'epidish_cbs'``, ``CP ->
    'epidish_cp'``.
    """
    if name:
        return name
    return _METHOD_RESULT_NAME.get(method, f"epidish_{method.lower()}")


class EpiDishDeconvolver(BaselineDeconvolver):
    """EpiDISH reference-based deconvolution baseline (RPC / CBS / CP)."""

    name = "epidish"
    _METHODS = {"RPC", "CBS", "CP"}
    _CONSTRAINTS = {"inequality", "equality"}

    def __init__(
        self,
        atlas,
        method: str = "RPC",
        maxit: int = 50,
        nu_v: Sequence[float] = (0.25, 0.5, 0.75),
        constraint: str = "inequality",
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}; got {method!r}.")
        if constraint not in self._CONSTRAINTS:
            raise ValueError(
                f"constraint must be one of {self._CONSTRAINTS}; got {constraint!r}."
            )
        self._atlas = atlas
        self.method = method
        self.maxit = maxit
        self.nu_v = tuple(nu_v)
        self.constraint = constraint

    @property
    def atlas(self):
        return self._atlas

    def prepare_reads(self, reads: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._atlas.prepare_reads(reads, trim=True)

    def build_input(self, reads: pd.DataFrame) -> Optional[dict]:
        meth_col = resolve_column(reads.columns, "methylation_ids")
        region_names: List[str] = []
        mixtures: List[np.ndarray] = []

        for region_name, group in reads.groupby("name", sort=False):
            if region_name not in self._atlas:
                _module_logger.warning("Region %s not in atlas; skipping.", region_name)
                continue
            cpg_lookup = self._atlas.get_cpg_lookup(region_name)
            n_cpgs = self._atlas.get_n_cpgs(region_name)

            # Scan all reads with numpy ASCII ops, collecting (abs_pos, meth)
            # across the whole region.  A single pd.Series.map then resolves all
            # positions to column indices — one map per region, not per read
            # (the per-read version was ~1000x slower on real pseudobulks).
            all_abs_pos: List[np.ndarray] = []
            all_meths: List[np.ndarray] = []
            for pattern, rs in zip(
                group[meth_col].tolist(), group["read_start"].tolist()
            ):
                arr = np.frombuffer(pattern.encode("ascii"), dtype=np.uint8)
                cpg_mask = (arr == 48) | (arr == 49)  # '0' / '1'
                offsets = np.where(cpg_mask)[0]
                if offsets.size == 0:
                    continue
                all_abs_pos.append(int(rs) + offsets)
                all_meths.append((arr[cpg_mask] == 49).astype(float))

            meth_counts = np.zeros(n_cpgs)
            tot_counts = np.zeros(n_cpgs)
            if all_abs_pos:
                positions = np.concatenate(all_abs_pos)
                meth_states = np.concatenate(all_meths)
                col_series = pd.Series(positions).map(cpg_lookup)  # one call/region
                valid = col_series.notna().values
                if valid.any():
                    cols = col_series[valid].astype(np.intp).values
                    tot_counts = np.bincount(cols, minlength=n_cpgs).astype(float)
                    meth_counts = np.bincount(
                        cols, weights=meth_states[valid], minlength=n_cpgs
                    )

            with np.errstate(invalid="ignore", divide="ignore"):
                beta = np.where(tot_counts > 0, meth_counts / tot_counts, np.nan)
            region_names.append(region_name)
            mixtures.append(beta)

        if not region_names:
            return None

        mixture = np.concatenate(mixtures)
        ref_list = self._atlas.get_beta_for_regions(region_names)  # (T, n_cpgs) each
        ref = np.concatenate([np.asarray(m, dtype=float).T for m in ref_list], axis=0)

        keep = ~np.isnan(mixture) & ~np.isnan(ref).any(axis=1)
        return {
            "mixture": mixture[keep],
            "ref": ref[keep],
            "ref_cells": list(self._atlas.ref_cells),
        }

    def merge_inputs(self, parts: List[dict]) -> Optional[dict]:
        """Stack the CpG rows of each chunk; the reference cells are shared."""
        parts = [part for part in parts if part and part["ref"].shape[0]]
        if not parts:
            return None
        ref_cells = parts[0]["ref_cells"]
        for part in parts[1:]:
            if part["ref_cells"] != ref_cells:
                raise ValueError(
                    "EpiDISH chunks disagree on the reference cell types; "
                    "they must all come from the same atlas."
                )
        return {
            "mixture": np.concatenate([part["mixture"] for part in parts]),
            "ref": np.concatenate([part["ref"] for part in parts], axis=0),
            "ref_cells": ref_cells,
        }

    def deconvolute_reads(
        self,
        reads: pd.DataFrame,
        labels_dict_reversed: Dict[str, int],
        n_labels: Optional[int] = None,
        prepare: bool = True,
    ) -> Optional[List[float]]:
        reads_sorted = self._sort_reads(reads)
        prepared = self.prepare_reads(reads_sorted) if prepare else reads_sorted
        return self.deconvolute_from_input(
            self.build_input(prepared), labels_dict_reversed, n_labels
        )

    def deconvolute_from_input(
        self,
        epi_in: Optional[dict],
        labels_dict_reversed: Dict[str, int],
        n_labels: Optional[int] = None,
    ) -> Optional[List[float]]:
        if epi_in is None or epi_in["ref"].shape[0] == 0:
            return None

        mixture, ref = epi_in["mixture"], epi_in["ref"]
        if self.method == "RPC":
            fracs = _do_rpc(mixture, ref, self.maxit)
        elif self.method == "CBS":
            fracs = _do_cbs(mixture, ref, self.nu_v)
        else:  # "CP"
            fracs = _do_cp(mixture, ref, self.constraint)

        return rearange_deconvolution_results(
            labels_dict_reversed, fracs, epi_in["ref_cells"], n_labels=n_labels
        )
