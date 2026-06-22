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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from syto.data.atlases.celfieish_atlases import CpGBetaCountsMethylationAtlas

_module_logger = logging.getLogger(__name__)
from syto.data.dataset import resolve_column

# ---------------------------------------------------------------------------
# Core EM (adapted from Caggiano et al. 2021, GNU AGPL v3)
# ---------------------------------------------------------------------------


def _add_pseudocounts(value, array, meth, meth_depths):
    """Add pseudocounts where gamma would cause log-likelihood to be undefined."""
    axis0, axis1 = np.where(array == value)
    meth[axis0, axis1] += 1
    meth_depths[axis0, axis1] += 2


def _check_gamma(array):
    return (0 in array) or (1 in array)


def _expectation(gamma, alpha):
    alpha = alpha.T[:, np.newaxis, :]
    gamma = gamma[..., np.newaxis]

    p0 = (1.0 - gamma) * alpha
    p1 = gamma * alpha

    p0 /= np.nansum(p0, axis=0)[np.newaxis, ...]
    p1 /= np.nansum(p1, axis=0)[np.newaxis, ...]
    return p0, p1


def _log_likelihood(p0, p1, x_depths, x, y_depths, y, gamma, alpha):
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


def _maximization(p0, p1, x, x_depths, y, y_depths):
    individuals = p0.shape[2]
    ones_vector = np.ones(shape=(y.shape[0]))
    new_alpha = np.zeros((x.shape[0], y.shape[0]))

    p0 = np.nan_to_num(p0)
    p1 = np.nan_to_num(p1)
    x = np.nan_to_num(x)
    x_depths = np.nan_to_num(x_depths)

    term0 = 0
    term1 = 0
    for n in range(individuals):
        new_alpha[n, :] = np.dot(p1[:, :, n], x[n, :]) + np.matmul(
            p0[:, :, n], (x_depths[n, :] - x[n, :])
        )
        term1 += p1[:, :, n] * np.outer(ones_vector, x[n, :])
        term0 += p0[:, :, n] * np.outer(ones_vector, x_depths[n, :] - x[n, :])

    gamma = (term1 + y) / (term0 + term1 + y_depths)

    if _check_gamma(gamma):
        _add_pseudocounts(1, gamma, y, y_depths)
        _add_pseudocounts(0, gamma, y, y_depths)
        gamma = (term1 + y) / (term0 + term1 + y_depths)

    normalized_new_alpha = new_alpha / np.sum(new_alpha, axis=1)[:, np.newaxis]
    return normalized_new_alpha, gamma


def em(
    x: np.ndarray,
    x_depths: np.ndarray,
    y: np.ndarray,
    y_depths: np.ndarray,
    num_iterations: int,
    convergence_criteria: float,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """EM deconvolution of cfDNA methylation data (Caggiano et al. 2021).

    Parameters
    ----------
    x : ndarray(n_samples, n_sites)
        Methylated read counts per CpG site per sample.
    x_depths : ndarray(n_samples, n_sites)
        Total read counts per CpG site per sample.
    y : ndarray(T, n_sites)
        Reference methylated counts per CpG per cell type.
    y_depths : ndarray(T, n_sites)
        Reference total counts per CpG per cell type.
    num_iterations : int
    convergence_criteria : float

    Returns
    -------
    alpha : ndarray(n_samples, T) — mixing proportions summing to 1 per sample
    gamma : ndarray(T, n_sites)  — estimated true methylation proportions
    ll    : float                — final log-likelihood
    i     : int                  — iterations run
    """
    alpha = np.random.uniform(size=(x.shape[0], y.shape[0]))
    alpha /= np.sum(alpha, axis=1)[:, np.newaxis]

    with np.errstate(invalid="ignore", divide="ignore"):
        _add_pseudocounts(1, np.nan_to_num(y / y_depths), y, y_depths)
        _add_pseudocounts(0, np.nan_to_num(y / y_depths), y, y_depths)
        gamma = y / y_depths

    i = 0
    for i in range(num_iterations):
        p0, p1 = _expectation(gamma, alpha)
        a, g = _maximization(p0, p1, x, x_depths, y, y_depths)

        alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
        gamma_diff = np.nanmean(abs(g - gamma)) / np.nanmean(abs(gamma))

        if i and (alpha_diff + gamma_diff < convergence_criteria):
            break

        alpha = a
        gamma = g

    ll = _log_likelihood(p0, p1, x_depths, x, y_depths, y, gamma, alpha)
    return alpha, gamma, ll, i


def em_with_checkpoints(
    x: np.ndarray,
    x_depths: np.ndarray,
    y: np.ndarray,
    y_depths: np.ndarray,
    checkpoints: List[int],
) -> List[Tuple[int, np.ndarray]]:
    """Run CelFiE EM for max(checkpoints) iterations, snapshotting alpha at each checkpoint.

    Unlike :func:`em`, convergence is never checked — the loop always runs for
    ``max(checkpoints)`` steps so that results at every iteration are directly
    comparable across pseudobulks.

    Parameters
    ----------
    x : ndarray(1, n_sites)
    x_depths : ndarray(1, n_sites)
    y : ndarray(T, n_sites)  — copied internally so the caller's array is unmodified.
    y_depths : ndarray(T, n_sites)  — same.
    checkpoints : list of int
        1-based iteration numbers at which to record alpha.

    Returns
    -------
    list of (iteration, ndarray(T,)) in ascending iteration order.
    """
    y = y.copy()
    y_depths = y_depths.copy()

    alpha = np.random.uniform(size=(x.shape[0], y.shape[0]))
    alpha /= np.sum(alpha, axis=1)[:, np.newaxis]

    with np.errstate(invalid="ignore", divide="ignore"):
        _add_pseudocounts(1, np.nan_to_num(y / y_depths), y, y_depths)
        _add_pseudocounts(0, np.nan_to_num(y / y_depths), y, y_depths)
        gamma = y / y_depths

    checkpoint_set = set(checkpoints)
    max_iter = max(checkpoints)
    results: List[Tuple[int, np.ndarray]] = []

    for i in range(1, max_iter + 1):
        p0, p1 = _expectation(gamma, alpha)
        alpha, gamma = _maximization(p0, p1, x, x_depths, y, y_depths)
        if i in checkpoint_set:
            results.append((i, alpha.flatten().copy()))

    return results


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def prepare_reads_for_celfie(
    reads_data: pd.DataFrame, atlas: "CpGBetaCountsMethylationAtlas", trim: bool = True
) -> pd.DataFrame:
    """Overlap reads with atlas regions and trim to region boundaries.

    Thin wrapper around :meth:`CpGBetaCountsMethylationAtlas.prepare_reads`.
    No M/U/X classification is applied — CelFiE operates on aggregated
    per-CpG counts built by :func:`build_celfie_input`.
    """
    return atlas.prepare_reads(reads_data, trim=trim)


def build_celfie_input(
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
        Output of :func:`prepare_reads_for_celfie`.  Must have columns
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

    return {"x_meth": x_meth_list, "x_cov": x_cov_list, "region_names": region_names}


def celfie_deconvolution(
    x_meth_list: List[np.ndarray],
    x_cov_list: List[np.ndarray],
    y_list: List[np.ndarray],
    y_cov_list: List[np.ndarray],
    num_iterations: int = 50,
    convergence_criteria: float = 0.001,
    random_restarts: int = 1,
    checkpoints: Optional[List[int]] = None,
):
    """Run CelFiE EM deconvolution.

    Parameters
    ----------
    x_meth_list, x_cov_list : list of ndarray(1, n_CpGs)
        Per-region mixture counts from :func:`build_celfie_input`.
    y_list, y_cov_list : list of ndarray(T, n_CpGs)
        Per-region reference counts from
        :meth:`CpGBetaCountsMethylationAtlas.get_meth_cov_for_regions`.
    num_iterations : int
        Max EM iterations when ``checkpoints`` is ``None``.
    convergence_criteria : float
        Early-stop threshold when ``checkpoints`` is ``None``.
    random_restarts : int
        Independent EM runs; the best log-likelihood wins.
        Ignored when ``checkpoints`` is provided (single run).
    checkpoints : list of int, optional
        When provided, runs exactly ``max(checkpoints)`` iterations without
        early stopping or random restarts and returns proportions at each step.

    Returns
    -------
    ndarray(T,)
        Cell-type proportions when ``checkpoints`` is ``None``.
    list of (iteration, ndarray(T,))
        Snapshot proportions when ``checkpoints`` is provided,
        in ascending iteration order.
    """
    x = np.hstack(x_meth_list)
    x_depths = np.hstack(x_cov_list)
    y = np.hstack(y_list)
    y_depths = np.hstack(y_cov_list)

    if checkpoints is not None:
        return em_with_checkpoints(x, x_depths, y, y_depths, checkpoints)

    best_ll = -np.inf
    best_alpha = None

    for _ in range(random_restarts):
        alpha, _, ll, _ = em(
            x, x_depths, y, y_depths, num_iterations, convergence_criteria
        )
        if ll > best_ll:
            best_ll = ll
            best_alpha = alpha

    return best_alpha.flatten()


def run_celfie_deconvolution(
    reads: pd.DataFrame,
    atlas: "CpGBetaCountsMethylationAtlas",
    labels_dict_reversed: Dict[str, int],
    n_labels: Optional[int] = None,
    prepare_reads: bool = True,
    num_iterations: int = 50,
    convergence_criteria: float = 0.001,
    random_restarts: int = 1,
    checkpoints: Optional[List[int]] = None,
):
    """Sort reads, optionally prepare with atlas, build input, deconvolve, align proportions.

    Parameters
    ----------
    reads : pd.DataFrame
        Raw processed reads (when prepare_reads=True) or atlas-annotated reads
        (when prepare_reads=False, must already have a ``name`` column).
    atlas : CpGBetaCountsMethylationAtlas
    labels_dict_reversed : dict
        ``cell_type_name → label_index`` mapping.
    n_labels : int, optional
        Defaults to ``len(labels_dict_reversed)``.
    prepare_reads : bool
        When True, calls ``atlas.prepare_reads`` to overlap and trim reads.
        Set False when reads already carry atlas-region annotations.
    num_iterations, convergence_criteria, random_restarts
        Passed to :func:`celfie_deconvolution` (convergence mode).
    checkpoints : list of int, optional
        When provided, runs exactly ``max(checkpoints)`` EM iterations.

    Returns
    -------
    list of float
        Proportions aligned to label order when checkpoints is None.
    list of (int, list of float)
        ``[(n_steps, proportions), …]`` when checkpoints is provided.
    None
        If no atlas regions overlap the reads.
    """
    reads_sorted = (
        reads.sort_values(["chromosome", "read_start", "read_end"])
        .reset_index(drop=True)
        .copy()
    )
    reads_sorted["read_start"] = reads_sorted["read_start"].astype("int64")
    reads_sorted["read_end"] = reads_sorted["read_end"].astype("int64")

    prepared = atlas.prepare_reads(reads_sorted) if prepare_reads else reads_sorted
    if prepared.empty:
        return None

    celfie_in = build_celfie_input(prepared, atlas)
    if not celfie_in["x_meth"]:
        return None

    y_list, y_cov_list = atlas.get_meth_cov_for_regions(celfie_in["region_names"])
    result = celfie_deconvolution(
        celfie_in["x_meth"],
        celfie_in["x_cov"],
        y_list,
        y_cov_list,
        num_iterations=num_iterations,
        convergence_criteria=convergence_criteria,
        random_restarts=random_restarts,
        checkpoints=checkpoints,
    )

    ref_cells = atlas.ref_cells

    if checkpoints is not None:
        return [
            (
                n_steps,
                rearange_celfie_deconvolution_results(
                    labels_dict_reversed, alpha, ref_cells, n_labels=n_labels
                ),
            )
            for n_steps, alpha in result
        ]

    return rearange_celfie_deconvolution_results(
        labels_dict_reversed, result, ref_cells, n_labels=n_labels
    )


def rearange_celfie_deconvolution_results(
    labels_dict_reversed: Dict[str, int],
    celfie_proportions: np.ndarray,
    ref_cells: List[str],
    n_labels: Optional[int] = None,
) -> List[float]:
    """Reorder CelFiE proportions to match the project's label index order.

    Parameters
    ----------
    labels_dict_reversed : dict
        ``cell_type_name → label_index``.
    celfie_proportions : ndarray(T,)
        Proportions in atlas cell-type order (``ref_cells`` ordering).
    ref_cells : list of str
        Cell-type names corresponding to axes of ``celfie_proportions``.
    n_labels : int, optional
        Defaults to ``len(labels_dict_reversed)``.

    Returns
    -------
    list of float  — proportions reindexed to label order 0 … n_labels-1.
    """
    if n_labels is None:
        n_labels = len(labels_dict_reversed)
    ref_pos = np.array([labels_dict_reversed.get(cell, -1) for cell in ref_cells])
    return [
        celfie_proportions[int(np.where(ref_pos == i)[0][0])] for i in range(n_labels)
    ]
