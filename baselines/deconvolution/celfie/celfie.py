"""
CelFiE: Comprehensive cell-free DNA deconvolution via EM.

Portions of the ``em`` function and its helpers are adapted from:

    Caggiano et al., Nature Communications 2021.
    https://github.com/christacaggiano/celfie
    GNU AGPL v3 — see LICENSE.md.

Key difference from CelFiE-ISH: CelFiE operates on **aggregated** per-CpG
methylation counts (not individual reads) and jointly re-estimates the
reference methylation fractions (gamma) during the EM, so the atlas counts
act as a prior rather than fixed values.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from syto.data.atlases.celfieish_atlases import CpGBetaCountsMethylationAtlas

_module_logger = logging.getLogger(__name__)
from syto.data.dataset import resolve_column
from baselines.deconvolution.base import BaselineDeconvolver
from baselines.deconvolution.utils import rearange_deconvolution_results

# ---------------------------------------------------------------------------
# EM model class (private implementation, adapted from Caggiano et al. 2021)
# ---------------------------------------------------------------------------


class _CelfieModel:
    """CelFiE EM model for a single sample.

    Encapsulates data concatenation and all EM algorithm steps.  Exposes
    ``fit`` (convergence mode with optional random restarts) and
    ``fit_with_checkpoints`` (fixed-iteration mode for reproducible snapshots).
    """

    def __init__(
        self,
        x_meth_list: List[np.ndarray],
        x_cov_list: List[np.ndarray],
        y_list: List[np.ndarray],
        y_cov_list: List[np.ndarray],
    ) -> None:
        self._x = np.hstack(x_meth_list)
        self._x_depths = np.hstack(x_cov_list)
        self._y = np.hstack(y_list)
        self._y_depths = np.hstack(y_cov_list)

    # ------------------------------------------------------------------
    # EM helpers
    # ------------------------------------------------------------------

    def _add_pseudocounts(self, value, array, meth, meth_depths):
        axis0, axis1 = np.where(array == value)
        meth[axis0, axis1] += 1
        meth_depths[axis0, axis1] += 2

    def _check_gamma(self, array):
        return (0 in array) or (1 in array)

    def _expectation(self, gamma, alpha):
        alpha = alpha.T[:, np.newaxis, :]
        gamma = gamma[..., np.newaxis]
        p0 = (1.0 - gamma) * alpha
        p1 = gamma * alpha
        p0 /= np.nansum(p0, axis=0)[np.newaxis, ...]
        p1 /= np.nansum(p1, axis=0)[np.newaxis, ...]
        return p0, p1

    def _log_likelihood(self, p0, p1, x_depths, x, y_depths, y, gamma, alpha):
        alpha = alpha.T[:, np.newaxis, :]
        gamma = gamma[..., np.newaxis]
        y = y[..., np.newaxis]
        y_depths = y_depths[..., np.newaxis]
        x = x.T[np.newaxis, ...]
        x_depths = x_depths.T[np.newaxis, ...]
        ll = 0
        ll += np.sum((y + p1 * x) * np.log(gamma))
        ll += np.sum((y_depths - y + p0 * (x_depths - x)) * np.log(1.0 - gamma))
        ll += np.sum((p1 * x + (x_depths - x) * p0) * np.log(alpha))
        return ll

    def _maximization(self, p0, p1, x, x_depths, y, y_depths):
        ones_vector = np.ones(shape=(y.shape[0]))
        new_alpha = np.zeros((x.shape[0], y.shape[0]))

        p0 = np.nan_to_num(p0)
        p1 = np.nan_to_num(p1)
        x = np.nan_to_num(x)
        x_depths = np.nan_to_num(x_depths)

        term0 = 0
        term1 = 0
        for n in range(p0.shape[2]):
            new_alpha[n, :] = np.dot(p1[:, :, n], x[n, :]) + np.matmul(
                p0[:, :, n], (x_depths[n, :] - x[n, :])
            )
            term1 += p1[:, :, n] * np.outer(ones_vector, x[n, :])
            term0 += p0[:, :, n] * np.outer(ones_vector, x_depths[n, :] - x[n, :])

        gamma = (term1 + y) / (term0 + term1 + y_depths)
        if self._check_gamma(gamma):
            self._add_pseudocounts(1, gamma, y, y_depths)
            self._add_pseudocounts(0, gamma, y, y_depths)
            gamma = (term1 + y) / (term0 + term1 + y_depths)

        return new_alpha / np.sum(new_alpha, axis=1)[:, np.newaxis], gamma

    def _init_gamma(self, y, y_depths):
        """Apply initial pseudocounts and return gamma; mutates y / y_depths."""
        with np.errstate(invalid="ignore", divide="ignore"):
            self._add_pseudocounts(1, np.nan_to_num(y / y_depths), y, y_depths)
            self._add_pseudocounts(0, np.nan_to_num(y / y_depths), y, y_depths)
            return y / y_depths

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        num_iterations: int = 50,
        convergence_criteria: float = 0.001,
        random_restarts: int = 1,
        freeze_gamma: bool = False,
    ) -> np.ndarray:
        """Run EM with optional restarts; return best-likelihood alpha ndarray(T,).

        When ``freeze_gamma`` is True the reference methylation fractions
        ``gamma`` are held at their initial atlas values (the M-step re-estimate
        is discarded), so only ``alpha`` is optimised.
        """
        best_ll = -np.inf
        best_alpha = None
        for _ in range(random_restarts):
            # Fresh copies per restart so pseudocount mutations don't accumulate.
            y, y_depths = self._y.copy(), self._y_depths.copy()
            alpha = np.random.uniform(size=(self._x.shape[0], y.shape[0]))
            alpha /= np.sum(alpha, axis=1)[:, np.newaxis]
            gamma = self._init_gamma(y, y_depths)

            i = 0
            for i in range(num_iterations):
                p0, p1 = self._expectation(gamma, alpha)
                a, g = self._maximization(p0, p1, self._x, self._x_depths, y, y_depths)
                if freeze_gamma:
                    g = gamma
                alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
                gamma_diff = np.nanmean(abs(g - gamma)) / np.nanmean(abs(gamma))
                if i and (alpha_diff + gamma_diff < convergence_criteria):
                    break
                alpha, gamma = a, g

            ll = self._log_likelihood(
                p0, p1, self._x_depths, self._x, y_depths, y, gamma, alpha
            )
            if ll > best_ll:
                best_ll = ll
                best_alpha = alpha
        return best_alpha.flatten()

    def fit_with_checkpoints(
        self, checkpoints: List[int], freeze_gamma: bool = False
    ) -> List[Tuple[int, np.ndarray]]:
        """Run EM for max(checkpoints) iterations; return snapshots.

        When ``freeze_gamma`` is True the reference methylation fractions
        ``gamma`` are held at their initial atlas values.
        """
        y, y_depths = self._y.copy(), self._y_depths.copy()
        alpha = np.random.uniform(size=(self._x.shape[0], y.shape[0]))
        alpha /= np.sum(alpha, axis=1)[:, np.newaxis]
        gamma = self._init_gamma(y, y_depths)

        checkpoint_set = set(checkpoints)
        results: List[Tuple[int, np.ndarray]] = []
        for i in range(1, max(checkpoints) + 1):
            p0, p1 = self._expectation(gamma, alpha)
            alpha, new_gamma = self._maximization(
                p0, p1, self._x, self._x_depths, y, y_depths
            )
            if not freeze_gamma:
                gamma = new_gamma
            if i in checkpoint_set:
                results.append((i, alpha.flatten().copy()))
        return results


# ---------------------------------------------------------------------------
# OOP interface
# ---------------------------------------------------------------------------


class CelFiEDeconvolver(BaselineDeconvolver):
    """CelFiE EM deconvolution baseline.

    Parameters
    ----------
    atlas : CpGBetaCountsMethylationAtlas
    num_iterations : int
    convergence_criteria : float
    random_restarts : int
    freeze_gamma : bool
        When True, hold the reference methylation fractions ``gamma`` at their
        atlas values instead of jointly re-estimating them in the EM M-step.
    sum_by_region : bool
        When True, pool per-CpG methylated/coverage counts into a single
        summary count per atlas region (CelFiE's ±250bp windowing convention)
        before running the EM, so each region contributes one feature instead
        of one feature per CpG.
    em_checkpoints : list of int, optional
        When set, ``deconvolute_reads`` returns ``[(n_steps, proportions), …]``.
    """

    name = "celfie"

    def __init__(
        self,
        atlas: "CpGBetaCountsMethylationAtlas",
        num_iterations: int = 1000,
        convergence_criteria: float = 0.0001,
        random_restarts: int = 10,
        freeze_gamma: bool = True,
        sum_by_region: bool = True,
        em_checkpoints: Optional[List[int]] = None,
    ) -> None:
        self._atlas = atlas
        self.num_iterations = num_iterations
        self.convergence_criteria = convergence_criteria
        self.random_restarts = random_restarts
        self.freeze_gamma = freeze_gamma
        self.sum_by_region = sum_by_region
        self.em_checkpoints = em_checkpoints

    @property
    def atlas(self) -> "CpGBetaCountsMethylationAtlas":
        return self._atlas

    def prepare_reads(self, reads: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._atlas.prepare_reads(reads, trim=True)

    def _build_celfie_input(
        self,
        reads: pd.DataFrame,
        atlas: "CpGBetaCountsMethylationAtlas",
    ) -> Dict:
        """Build per-region aggregated count vectors for CelFiE.

        Unlike :func:`build_celfieish_input`, this aggregates across all reads:
        for each CpG position in the atlas, count how many reads are methylated
        and how many cover that position at all.  Individual read information is
        discarded.

        Parameters
        ----------
        reads : pd.DataFrame
            Output of :func:`prepare_reads`.  Must have columns
            ``name``, ``read_start`` (trimmed, 0-based), and
            ``methylation_pattern_column``.  If a ``cpg_sig`` column is present
            (pre-extracted ``[(abs_pos, meth_state), …]`` per read), it is used
            directly instead of scanning the pattern string.
        atlas : CpGBetaCountsMethylationAtlas

        Returns
        -------
        dict with keys:
            ``"x_meth"``      — ``list[ndarray(1, n_CpGs)]``, methylated counts.
            ``"x_cov"``       — ``list[ndarray(1, n_CpGs)]``, total counts.
            ``"region_names"``— ``list[str]``, one entry per region (same order).
        """
        use_cpg_sig = "cpg_sig" in reads.columns
        region_names: List[str] = []
        x_meth_list: List[np.ndarray] = []
        x_cov_list: List[np.ndarray] = []
        methylation_pattern_column = resolve_column(reads.columns, "methylation_ids")
        for region_name, group in reads.groupby("name", sort=False):
            if region_name not in atlas:
                _module_logger.warning("Region %s not in atlas; skipping.", region_name)
                continue

            cpg_lookup = atlas.get_cpg_lookup(region_name)
            n_cpgs = atlas.get_n_cpgs(region_name)

            # ---- Batch all observations for this region, then do one bincount ----

            if use_cpg_sig:
                # Collect every (abs_position, meth_state) pair across all reads at once.
                arrays = [
                    np.asarray(sig.tolist(), dtype=np.int64)
                    for sig in group["cpg_sig"]
                    if sig is not None and len(sig) > 0
                ]
                if not arrays:
                    continue
                all_pairs = np.vstack(arrays)  # (total_obs, 2)
                positions = all_pairs[:, 0]
                meths = all_pairs[:, 1]
            else:
                # Scan pattern strings with numpy ASCII ops — avoids iterrows and
                # per-character Python iteration.
                pos_list = []
                meth_list = []
                for pattern, rs in zip(
                    group[methylation_pattern_column].tolist(),
                    group["read_start"].tolist(),
                ):
                    arr = np.frombuffer(pattern.encode("ascii"), dtype=np.uint8)
                    cpg_mask = (arr == 48) | (arr == 49)  # ord('0')=48, ord('1')=49
                    offsets = np.where(cpg_mask)[0]
                    if offsets.size == 0:
                        continue
                    pos_list.append(int(rs) + offsets)
                    meth_list.append((arr[cpg_mask] == 49).astype(np.int32))
                if not pos_list:
                    continue
                positions = np.concatenate(pos_list)
                meths = np.concatenate(meth_list)

            # Vectorised position → column-index mapping via pandas dict map,
            # then a single bincount replaces O(N) individual array increments.
            col_indices = pd.Series(positions).map(cpg_lookup)
            valid = col_indices.notna().values
            if not valid.any():
                continue
            cols = col_indices[valid].astype(np.intp).values
            m = meths[valid]

            x_meth = np.bincount(cols, weights=m.astype(np.float64), minlength=n_cpgs)
            x_cov = np.bincount(cols, minlength=n_cpgs).astype(np.float64)

            region_names.append(region_name)
            x_meth_list.append(x_meth.reshape(1, -1))
            x_cov_list.append(x_cov.reshape(1, -1))

        return {
            "x_meth": x_meth_list,
            "x_cov": x_cov_list,
            "region_names": region_names,
        }

    def build_input(self, reads: pd.DataFrame) -> dict:
        return self._build_celfie_input(reads, self._atlas)

    def deconvolute_reads(
        self,
        reads: pd.DataFrame,
        labels_dict_reversed: Dict[str, int],
        n_labels: Optional[int] = None,
        prepare: bool = True,
    ) -> Optional[Union[List[float], List[Tuple[int, List[float]]]]]:
        reads_sorted = self._sort_reads(reads)
        prepared = self.prepare_reads(reads_sorted) if prepare else reads_sorted
        if prepared.empty:
            return None
        celfie_in = self.build_input(prepared)
        if not celfie_in["x_meth"]:
            return None
        y_list, y_cov_list = self._atlas.get_meth_cov_for_regions(
            celfie_in["region_names"]
        )
        x_meth_list, x_cov_list = celfie_in["x_meth"], celfie_in["x_cov"]
        if self.sum_by_region:
            # Collapse the CpG axis so each region is one summed feature, applying
            # the identical reduction to sample (x) and atlas (y) so columns align.
            x_meth_list = [a.sum(axis=1, keepdims=True) for a in x_meth_list]
            x_cov_list = [a.sum(axis=1, keepdims=True) for a in x_cov_list]
            y_list = [a.sum(axis=1, keepdims=True) for a in y_list]
            y_cov_list = [a.sum(axis=1, keepdims=True) for a in y_cov_list]
        model = _CelfieModel(x_meth_list, x_cov_list, y_list, y_cov_list)
        ref_cells = self._atlas.ref_cells
        if self.em_checkpoints is not None:
            return [
                (
                    n_steps,
                    rearange_deconvolution_results(
                        labels_dict_reversed, alpha, ref_cells, n_labels=n_labels
                    ),
                )
                for n_steps, alpha in model.fit_with_checkpoints(
                    self.em_checkpoints, freeze_gamma=self.freeze_gamma
                )
            ]
        return rearange_deconvolution_results(
            labels_dict_reversed,
            model.fit(
                self.num_iterations,
                self.convergence_criteria,
                self.random_restarts,
                freeze_gamma=self.freeze_gamma,
            ),
            ref_cells,
            n_labels=n_labels,
        )
