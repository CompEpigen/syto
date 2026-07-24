"""
CelFiE-ISH: Read-based EM Algorithm for Deconvolution of Methylation Sequencing.

Adapted from Unterman & Berman (2022), MIT License.
Original: https://github.com/methylgrammarlab/deconvolution_models

Encoding convention (Syto):
    UNMETHYLATED = 0, METHYLATED = 1, NOVAL = 2
"""

import logging
from itertools import compress
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from baselines.deconvolution.base import BaselineDeconvolver
from baselines.deconvolution.utils import rearange_deconvolution_results
from scipy.special import logsumexp

if TYPE_CHECKING:
    from syto.data.atlases.celfieish_atlases import CpGBetaCountsMethylationAtlas

_module_logger = logging.getLogger(__name__)

UNMETHYLATED, METHYLATED, NOVAL = 0, 1, 2
pseudocount = 1e-10
from syto.data.dataset import resolve_column

# ---------------------------------------------------------------------------
# Core EM model
# ---------------------------------------------------------------------------


class _CelfieISHModel:
    """Read-based EM deconvolution of methylation sequencing data.

    Parameters
    ----------
    mixtures : list of ndarray(reads, CpGs)
        One per genomic region; values in {UNMETHYLATED, METHYLATED, NOVAL}.
    beta : list of ndarray(T, CpGs)
        Per-cell-type methylation probabilities per region; values in [0, 1].
    origins : list, optional
        Read origin labels (preserved across filtering for ``get_proba``).
    num_iterations : int
        Maximum EM iterations.
    convergence_criteria : float
        Relative change in alpha below which EM is considered converged.
    alpha : ndarray(T,), optional
        Override random initialisation with a fixed starting point.
    """

    def __init__(
        self,
        mixtures,
        beta,
        origins=None,
        num_iterations=50,
        convergence_criteria=0.001,
        alpha=None,
    ):
        self.x = self._filter_empty_rows(mixtures)
        self.beta = [self._add_pseudocounts(b) for b in beta]
        self.origins = origins

        self._filter_no_coverage()
        self.num_iterations = num_iterations
        self.convergence_criteria = convergence_criteria
        self.x_c_m = [(x == METHYLATED) for x in self.x]
        self.x_c_u = [(x == UNMETHYLATED) for x in self.x]
        self.x_c_v = [~(x == NOVAL) for x in self.x]
        self.t = self.beta[0].shape[0]
        self.alpha = alpha

        self.log_beta = [np.log(b) for b in self.beta]
        self.log_one_minus_beta = [np.log(1 - b) for b in self.beta]
        self.log_term1 = self._calc_term1()
        # Concatenate the per-region (T, n_reads) log-likelihood terms into one
        # (T, total_reads) array.  log_term1 is constant across EM iterations and
        # each read column is independent, so the E/M steps can run as a single
        # vectorised op instead of a per-region Python loop every iteration.
        # Stored in float32: the E-step softmax is the EM hot loop, and float32
        # roughly halves it.  The M-step reduction is still accumulated in
        # float64 (see two_step) to keep the proportion estimates precise.
        self.log_term1_cat = (
            np.hstack(self.log_term1)
            if self.log_term1
            else np.zeros((self.t, 0))
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calc_term1(self):
        t1 = []
        for w in range(len(self.x)):
            x_c_m = self.x_c_m[w].astype(int)
            x_c_u = self.x_c_u[w].astype(int)
            log_beta = np.nan_to_num(self.log_beta[w].T)
            log_1mb = np.nan_to_num(self.log_one_minus_beta[w].T)
            t1.append((np.matmul(x_c_m, log_beta) + np.matmul(x_c_u, log_1mb)).T)
        return t1

    def _add_pseudocounts(self, arr):
        new_arr = arr.copy()
        new_arr[arr == 0] += pseudocount
        new_arr[arr == 1] -= pseudocount
        return new_arr

    def _filter_empty_rows(self, reads):
        filtered = []
        for region in reads:
            if region.size > 0:
                filtered.append(region[~(region == NOVAL).all(axis=1), :])
            else:
                filtered.append(region)
        return filtered

    def _filter_no_coverage(self):
        ref_cov = [~np.isnan(b).all(axis=0) for b in self.beta]
        self.beta = [self.beta[i][:, ref_cov[i]] for i in range(len(self.x))]
        self.x = [self.x[i][:, ref_cov[i]] for i in range(len(self.x))]
        has_cov = np.array([(~(x == NOVAL)).any() for x in self.x])
        self.beta = list(compress(self.beta, has_cov))
        self.x = list(compress(self.x, has_cov))
        if self.origins:
            self.origins = list(compress(self.origins, has_cov))

    # ------------------------------------------------------------------
    # EM steps
    # ------------------------------------------------------------------

    def _log_expectation(self, alpha):
        z = []
        for w in range(len(self.x)):
            T, C = self.log_term1[w].shape
            a = np.tile(np.log(alpha), (C, 1)).T + self.log_term1[w]
            b = logsumexp(a, axis=0)
            z.append(np.exp(a - np.tile(b, (T, 1))))
        return z

    def _log_expectation_cat(self, alpha):
        """Vectorised E-step over all reads at once → z of shape (T, total_reads).

        Equivalent to ``np.hstack(self._log_expectation(alpha))`` — a stable
        softmax over the cell-type axis — but computed as a single array op with
        an in-place softmax rather than a per-region loop over ``scipy``'s
        ``logsumexp`` (which together dominated EM runtime at atlas scale).
        """
        la = np.log(alpha).astype(np.float32)[:, None]
        a = la + self.log_term1_cat  # (T, total_reads), float32
        a -= a.max(axis=0)  # stabilise, then softmax down the cell-type axis
        np.exp(a, out=a)
        a /= a.sum(axis=0)
        return a

    def _maximization(self, z):
        all_z = np.hstack(z)
        new_alpha = np.sum(all_z, axis=1)
        new_alpha /= np.sum(new_alpha)
        assert not np.isnan(new_alpha).any(), "alpha has NaN"
        return new_alpha

    def _test_convergence(self, new_alpha):
        return (
            np.mean(abs(new_alpha - self.alpha)) / np.mean(abs(self.alpha))
            < self.convergence_criteria
        )

    def _init_alpha(self):
        alpha = np.random.uniform(size=(self.t,))
        alpha /= np.sum(alpha)
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def two_step(self):
        """Run EM and return (alpha, n_iterations)."""
        if self.alpha is None:
            self._init_alpha()
        i = 0
        for i in range(self.num_iterations):
            z = self._log_expectation_cat(self.alpha)  # (T, total_reads)
            new_alpha = z.sum(axis=1, dtype=np.float64)  # accumulate in float64
            new_alpha /= new_alpha.sum()
            assert not np.isnan(new_alpha).any(), "alpha has NaN"
            if i and self._test_convergence(new_alpha):
                break
            self.alpha = new_alpha
        return self.alpha, i

    def run_with_checkpoints(
        self, checkpoints: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """Run EM for exactly max(checkpoints) iterations, snapshotting alpha at each checkpoint.

        Unlike :meth:`two_step`, convergence is never checked — the loop always
        runs for ``max(checkpoints)`` steps so that results at every requested
        iteration are directly comparable across pseudobulks.

        Parameters
        ----------
        checkpoints : list of int
            1-based iteration numbers at which to record alpha.

        Returns
        -------
        list of (iteration, alpha) tuples in ascending iteration order.
        """
        if self.alpha is None:
            self._init_alpha()

        checkpoint_set = set(checkpoints)
        max_iter = max(checkpoints)
        results: List[Tuple[int, np.ndarray]] = []

        for i in range(1, max_iter + 1):
            z = self._log_expectation_cat(self.alpha)
            new_alpha = z.sum(axis=1, dtype=np.float64)  # accumulate in float64
            new_alpha /= new_alpha.sum()
            assert not np.isnan(new_alpha).any(), "alpha has NaN"
            self.alpha = new_alpha
            if i in checkpoint_set:
                results.append((i, self.alpha.copy()))

        return results

    def log_likelihood(self):
        ll = 0
        for w in range(len(self.x)):
            log_t1 = self.log_term1[w]
            T, C = log_t1.shape
            ll += np.sum(
                logsumexp(np.tile(np.log(self.alpha), (C, 1)).T + log_t1, axis=0)
            )
        return ll

    def get_proba(self):
        """Return (probabilities per cell type, origins). Call after two_step."""
        z = self._log_expectation(self.alpha)
        return np.hstack(z), np.hstack(self.origins)

    def get_clipped_alpha(self, clipping=0.05):
        """Re-normalise alpha after zeroing probabilities below ``clipping``."""
        z = np.hstack(self._log_expectation(self.alpha))
        z[z < clipping] = 0
        alpha = np.sum(z, axis=1)
        assert np.sum(alpha) > 0, "all probabilities below clipping threshold"
        return alpha / np.sum(alpha)


# ---------------------------------------------------------------------------
# OOP interface
# ---------------------------------------------------------------------------


class CelFiEISHDeconvolver(BaselineDeconvolver):
    """CelFiE-ISH read-based EM deconvolution baseline.

    Parameters
    ----------
    atlas : CpGBetaCountsMethylationAtlas
    num_iterations : int
    convergence_criteria : float
    em_checkpoints : list of int, optional
        When set, ``deconvolute_reads`` runs for exactly ``max(em_checkpoints)``
        iterations and returns ``[(n_steps, proportions), …]`` instead of a
        single proportions list.
    """

    name = "celfieish"

    def __init__(
        self,
        atlas: "CpGBetaCountsMethylationAtlas",
        num_iterations: int = 50,
        convergence_criteria: float = 0.001,
        em_checkpoints: Optional[List[int]] = None,
    ) -> None:
        self._atlas = atlas
        self.num_iterations = num_iterations
        self.convergence_criteria = convergence_criteria
        self.em_checkpoints = em_checkpoints

    @property
    def atlas(self) -> "CpGBetaCountsMethylationAtlas":
        return self._atlas

    def prepare_reads(self, reads: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._atlas.prepare_reads(reads, trim=True)

    def _build_celfieish_input(
        self, reads: pd.DataFrame, atlas: "CpGBetaCountsMethylationAtlas"
    ) -> Dict:
        """Build per-region read × CpG matrices from prepared reads.

        For each atlas region that has overlapping reads, constructs an integer
        matrix of shape ``(n_reads, n_CpGs)`` using the Syto encoding
        (UNMETHYLATED=0, METHYLATED=1, NOVAL=2).  CpG columns are aligned to
        the absolute genomic positions stored in the atlas.

        Parameters
        ----------
        reads : pd.DataFrame
            Output of :func:`prepare_reads`.  Must have columns
            ``name``, ``read_start`` (trimmed, 0-based), and
            ``methylation_pattern_column``.
        atlas : CpGBetaCountsMethylationAtlas

        Returns
        -------
        dict with keys:
            ``"matrices"`` — ``list[ndarray(n_reads, n_CpGs)]``, one per region.
            ``"region_names"`` — ``list[str]``, region names in the same order.
        """
        region_names: List[str] = []
        matrices: List[np.ndarray] = []
        methylation_pattern_column = resolve_column(reads.columns, "methylation_ids")

        for region_name, group in reads.groupby("name", sort=False):
            if region_name not in atlas:
                _module_logger.warning("Region %s not in atlas; skipping.", region_name)
                continue

            cpg_lookup = atlas.get_cpg_lookup(region_name)
            n_cpgs = atlas.get_n_cpgs(region_name)
            n_reads = len(group)

            matrix = np.full((n_reads, n_cpgs), NOVAL, dtype=np.int8)

            # Scan all reads with numpy ASCII ops, collecting (row_idx, abs_pos, meth)
            # triples across the whole region.  A single pd.Series.map call then
            # resolves all positions to column indices before one fancy-index write.
            all_abs_pos: List[np.ndarray] = []
            all_row_idx: List[np.ndarray] = []
            all_meths: List[np.ndarray] = []

            for row_idx, (pattern, rs) in enumerate(
                zip(
                    group[methylation_pattern_column].tolist(),
                    group["read_start"].tolist(),
                )
            ):
                arr = np.frombuffer(pattern.encode("ascii"), dtype=np.uint8)
                cpg_mask = (arr == 48) | (arr == 49)  # ord('0')=48, ord('1')=49
                offsets = np.where(cpg_mask)[0]
                if offsets.size == 0:
                    continue

                n = offsets.size
                all_abs_pos.append(int(rs) + offsets)
                all_row_idx.append(np.full(n, row_idx, dtype=np.intp))
                all_meths.append(
                    (arr[cpg_mask] == 49).astype(np.int8)
                )  # 1=METHYLATED, 0=UNMETHYLATED

            if all_abs_pos:
                positions = np.concatenate(all_abs_pos)
                row_indices = np.concatenate(all_row_idx)
                meth_states = np.concatenate(all_meths)

                col_series = pd.Series(positions).map(cpg_lookup)  # one call per region
                valid = col_series.notna().values
                if valid.any():
                    col_indices = col_series[valid].astype(np.intp).values
                    matrix[row_indices[valid], col_indices] = meth_states[valid]

            region_names.append(region_name)
            matrices.append(matrix)

        return {"matrices": matrices, "region_names": region_names}

    def build_input(self, reads: pd.DataFrame) -> dict:
        return self._build_celfieish_input(reads, self._atlas)

    def deconvolute_reads(
        self,
        reads: pd.DataFrame,
        labels_dict_reversed: Dict[str, int],
        n_labels: Optional[int] = None,
        prepare: bool = True,
    ) -> Optional[Union[List[float], List[Tuple[int, List[float]]]]]:
        reads_sorted = self._sort_reads(reads)
        prepared = self.prepare_reads(reads_sorted) if prepare else reads_sorted
        celfieish_in = self.build_input(prepared)
        if not celfieish_in["matrices"]:
            return None
        beta_matrices = self._atlas.get_beta_for_regions(celfieish_in["region_names"])
        model = _CelfieISHModel(
            celfieish_in["matrices"],
            beta_matrices,
            num_iterations=self.num_iterations,
            convergence_criteria=self.convergence_criteria,
        )
        ref_cells = self._atlas.ref_cells
        if self.em_checkpoints is not None:
            return [
                (
                    n_steps,
                    rearange_deconvolution_results(
                        labels_dict_reversed, alpha, ref_cells, n_labels=n_labels
                    ),
                )
                for n_steps, alpha in model.run_with_checkpoints(self.em_checkpoints)
            ]
        alpha, _ = model.two_step()
        return rearange_deconvolution_results(
            labels_dict_reversed, alpha, ref_cells, n_labels=n_labels
        )
