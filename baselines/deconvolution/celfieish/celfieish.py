"""
CelFiE-ISH: Read-based EM Algorithm for Deconvolution of Methylation Sequencing.

Adapted from Unterman & Berman (2022), MIT License.
Original: https://github.com/methylgrammarlab/deconvolution_models

Encoding convention (Syto):
    UNMETHYLATED = 0, METHYLATED = 1, NOVAL = 2
"""

import logging
from itertools import compress
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp

if TYPE_CHECKING:
    from syto.data.atlases.celfieish_atlases import CelfieISHMethylationAtlas

_module_logger = logging.getLogger(__name__)

UNMETHYLATED, METHYLATED, NOVAL = 0, 1, 2
pseudocount = 1e-10


# ---------------------------------------------------------------------------
# Core EM model
# ---------------------------------------------------------------------------


class CelfieISH:
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
            z = self._log_expectation(self.alpha)
            new_alpha = self._maximization(z)
            if i and self._test_convergence(new_alpha):
                break
            self.alpha = new_alpha
        return self.alpha, i

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
# Pipeline helpers
# ---------------------------------------------------------------------------


def prepare_reads_for_celfieish(
    reads_data: pd.DataFrame,
    atlas: "CelfieISHMethylationAtlas",
    labels_dict: Optional[Dict] = None,
    cell_type_match_dict: Optional[Dict[str, str]] = None,
    trim: bool = True,
    seq_column: str = "seq",
    methylation_pattern_column: str = "pattern",
) -> pd.DataFrame:
    """Overlap reads with atlas regions and trim to region boundaries.

    Delegates to :meth:`CelfieISHMethylationAtlas.prepare_reads`.  Unlike the
    UXM pipeline, no M/U/X classification step is applied — CelFiE-ISH
    operates on the full per-read methylation matrix built by
    :func:`build_celfieish_input`.

    Parameters
    ----------
    reads_data : pd.DataFrame
        Input reads sorted by ``["chromosome", "read_start", "read_end"]``.
    atlas : CelfieISHMethylationAtlas
    labels_dict : dict, optional
        ``label_id → cell_type_name`` mapping for DMR label annotation.
    cell_type_match_dict : dict, optional
        Atlas cell-type name → project cell-type name aliases.
    trim : bool
        Clip reads to region boundaries.
    seq_column, methylation_pattern_column : str
        Column names for DNA sequence and CpG methylation pattern.

    Returns
    -------
    pd.DataFrame
        Reads annotated with ``name``, ``region_start``, ``region_end``
        and, if ``labels_dict`` provided, DMR label columns.
    """
    return atlas.prepare_reads(
        reads_data,
        trim=trim,
        seq_column=seq_column,
        methylation_pattern_column=methylation_pattern_column,
        labels_dict=labels_dict,
        cell_type_match_dict=cell_type_match_dict,
    )


def build_celfieish_input(
    reads: pd.DataFrame,
    atlas: "CelfieISHMethylationAtlas",
    methylation_pattern_column: str = "pattern",
) -> Dict:
    """Build per-region read × CpG matrices from prepared reads.

    For each atlas region that has overlapping reads, constructs an integer
    matrix of shape ``(n_reads, n_CpGs)`` using the Syto encoding
    (UNMETHYLATED=0, METHYLATED=1, NOVAL=2).  CpG columns are aligned to
    the absolute genomic positions stored in the atlas.

    Parameters
    ----------
    reads : pd.DataFrame
        Output of :func:`prepare_reads_for_celfieish`.  Must have columns
        ``name``, ``read_start`` (trimmed, 0-based), and
        ``methylation_pattern_column``.
    atlas : CelfieISHMethylationAtlas
    methylation_pattern_column : str
        Column holding the base-level methylation pattern string
        (``'0'`` = unmethylated CpG, ``'1'`` = methylated CpG).

    Returns
    -------
    dict with keys:
        ``"matrices"`` — ``list[ndarray(n_reads, n_CpGs)]``, one per region.
        ``"region_names"`` — ``list[str]``, region names in the same order.
    """
    region_names: List[str] = []
    matrices: List[np.ndarray] = []

    for region_name, group in reads.groupby("name", sort=False):
        if region_name not in atlas:
            _module_logger.warning("Region %s not in atlas; skipping.", region_name)
            continue

        cpg_lookup = atlas.get_cpg_lookup(region_name)
        n_cpgs = atlas.get_n_cpgs(region_name)
        n_reads = len(group)

        matrix = np.full((n_reads, n_cpgs), NOVAL, dtype=np.int8)

        for row_idx, (_, row) in enumerate(group.iterrows()):
            pattern = row[methylation_pattern_column]
            trimmed_start = int(row["read_start"])  # 0-based

            for char_offset, char in enumerate(pattern):
                if char == "0":
                    col = cpg_lookup.get(trimmed_start + char_offset)
                    if col is not None:
                        matrix[row_idx, col] = UNMETHYLATED
                elif char == "1":
                    col = cpg_lookup.get(trimmed_start + char_offset)
                    if col is not None:
                        matrix[row_idx, col] = METHYLATED

        region_names.append(region_name)
        matrices.append(matrix)

    return {"matrices": matrices, "region_names": region_names}


def celfieish_deconvolution(
    mixture_matrices: List[np.ndarray],
    beta_matrices: List[np.ndarray],
    num_iterations: int = 50,
    convergence_criteria: float = 0.001,
) -> np.ndarray:
    """Run CelFiE-ISH EM deconvolution and return cell-type proportions.

    Parameters
    ----------
    mixture_matrices : list of ndarray(reads, CpGs)
        Per-region read matrices from :func:`build_celfieish_input`.
    beta_matrices : list of ndarray(T, CpGs)
        Per-region atlas beta matrices from
        :meth:`CelfieISHMethylationAtlas.get_beta_for_regions`.
    num_iterations : int
    convergence_criteria : float

    Returns
    -------
    ndarray(T,)
        Cell-type proportions summing to 1.
    """
    model = CelfieISH(
        mixture_matrices,
        beta_matrices,
        num_iterations=num_iterations,
        convergence_criteria=convergence_criteria,
    )
    alpha, _ = model.two_step()
    return alpha


def rearange_celfieish_deconvolution_results(
    labels_dict_reversed: Dict[str, int],
    celfieish_proportions: np.ndarray,
    ref_cells: List[str],
    n_labels: Optional[int] = None,
) -> List[float]:
    """Reorder CelFiE-ISH proportions to match the project's label index order.

    Parameters
    ----------
    labels_dict_reversed : dict
        ``cell_type_name → label_index`` mapping.
    celfieish_proportions : ndarray(T,)
        Proportions in atlas cell-type order (``ref_cells`` ordering).
    ref_cells : list of str
        Cell-type names corresponding to axes of ``celfieish_proportions``.
    n_labels : int, optional
        Number of labels; defaults to ``len(labels_dict_reversed)``.

    Returns
    -------
    list of float
        Proportions reindexed to ``label_index`` order (0 … n_labels-1).
    """
    if n_labels is None:
        n_labels = len(labels_dict_reversed)
    ref_pos = np.array([labels_dict_reversed.get(cell, -1) for cell in ref_cells])
    return [
        celfieish_proportions[int(np.where(ref_pos == i)[0][0])]
        for i in range(n_labels)
    ]
