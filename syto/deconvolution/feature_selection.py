"""
Feature Selection for Deconvolution
====================================

Utilities for computing feature masks from purified cell-type profiles,
applying masks to full feature matrices, and generating diagnostic
visualisation plots that compare selected features across data splits.
"""

import logging
import os
import math
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────────────────────────────


def compute_feature_ratios(pure_matrix: np.ndarray) -> np.ndarray:
    """Compute max / mean ratio for each feature position.

    Parameters
    ----------
    pure_matrix : np.ndarray
        Shape ``(n_cell_types, n_gr_groups, n_pred_classes)`` feature
        matrix from :func:`extract_pure_feature_matrix`.

    Returns
    -------
    np.ndarray
        Ratio matrix of same shape as the last two dimensions of
        ``pure_matrix`` (i.e. ``(n_gr_groups, n_pred_classes)``).
    """
    max_vals = pure_matrix.max(axis=0)
    mean_vals = pure_matrix.mean(axis=0)
    # Avoid division by zero
    mean_vals = np.where(mean_vals == 0, 1e-12, mean_vals)
    return max_vals / mean_vals


def compute_feature_mask(
    ratios: np.ndarray,
    cutoff: Optional[float] = 1.1,
    guarantee_diagonal_selection: bool = False,
    guarantee_columns_selection: Optional[List[int]] = None,
    top_features: Optional[int] = None,
    return_cutoff: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Return a binary mask where ratio > cutoff, with optional guaranteed features.

    Parameters
    ----------
    ratios : np.ndarray
        Ratio matrix from :func:`compute_feature_ratios`.
    cutoff : float, optional
        Threshold above which a feature is selected. Default is 1.1.
        Ignored if `top_features` is provided.
    guarantee_diagonal_selection : bool
        If True, the diagonal elements of the mask are set to 1.
    guarantee_columns_selection : list of int, optional
        A list of column indices to unconditionally select (all rows for those columns set to 1).
    top_features : int, optional
        If provided, selects exactly this many top features based on ratios. Guaranteed features
        are prioritized. `cutoff` is dynamically calculated based on the selected non-guaranteed features.
    return_cutoff : bool
        If True, returns a tuple ``(mask, calculated_cutoff)``.

    Returns
    -------
    np.ndarray or tuple
        Binary mask with the same shape as *ratios*. If `return_cutoff` is True, returns
        ``(mask, calculated_cutoff)``.
    """
    guaranteed_mask = np.zeros_like(ratios, dtype=int)
    if guarantee_diagonal_selection:
        np.fill_diagonal(guaranteed_mask, 1)

    if guarantee_columns_selection:
        for c in guarantee_columns_selection:
            if 0 <= c < ratios.shape[1]:
                guaranteed_mask[:, c] = 1

    if top_features is not None:
        # Give priority to guaranteed features by inflating their ratios
        modified_ratios = ratios.copy()
        modified_ratios[guaranteed_mask == 1] += 1e9

        flat_ratios = modified_ratios.flatten()
        # Handle Nans
        flat_ratios[np.isnan(flat_ratios)] = -np.inf

        # Find indices of top `top_features` features
        sorted_idx = np.argsort(flat_ratios)[::-1]
        actual_top_features = min(top_features, len(flat_ratios))
        selected_idx = sorted_idx[:actual_top_features]

        mask_flat = np.zeros_like(flat_ratios, dtype=int)
        mask_flat[selected_idx] = 1
        mask = mask_flat.reshape(ratios.shape)

        # Calculate the appropriate cutoff (minimum original ratio among selected non-guaranteed features)
        selected_non_guaranteed = (mask == 1) & (guaranteed_mask == 0)
        if selected_non_guaranteed.any():
            calc_cutoff = float(ratios[selected_non_guaranteed].min())
        else:
            calc_cutoff = float(ratios[mask == 1].min()) if (mask == 1).any() else 0.0

        if return_cutoff:
            return mask, calc_cutoff
        return mask

    else:
        # Default behavior using predefined cutoff
        val_cutoff = cutoff if cutoff is not None else 1.1
        mask = (ratios > val_cutoff).astype(int)
        mask[guaranteed_mask == 1] = 1

        if return_cutoff:
            return mask, float(val_cutoff)
        return mask


def apply_feature_mask(
    matrix: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply a binary feature mask and return a flat compressed array.

    Parameters
    ----------
    matrix : np.ndarray
        Feature matrix of shape ``(n_gr_groups, n_pred_classes)`` or
        ``(n_samples, n_gr_groups, n_pred_classes)``.
    mask : np.ndarray
        Binary mask of shape ``(n_gr_groups, n_pred_classes)``.

    Returns
    -------
    np.ndarray
        If input is 2-D: 1-D array of selected features.
        If input is 3-D: 2-D array ``(n_samples, n_selected_features)``.
    """
    bool_mask = np.array(mask, dtype=bool)

    if matrix.ndim == 2:
        return np.ma.masked_array(matrix, ~bool_mask).compressed()
    elif matrix.ndim == 3:
        n_samples = matrix.shape[0]
        results = []
        for i in range(n_samples):
            compressed = np.ma.masked_array(matrix[i], ~bool_mask).compressed()
            results.append(compressed)
        return np.array(results)
    else:
        raise ValueError(f"Expected 2-D or 3-D matrix, got {matrix.ndim}-D")


# ─────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────


def generate_feature_selection_plot(
    pure_matrices_per_split: dict,
    num_input_labels: int,
    num_output_labels: int,
    cutoff: float,
    output_path: str,
    master_names: Optional[List[str]] = None,
    guarantee_diagonal_selection: bool = False,
    guarantee_columns_selection: Optional[List[int]] = None,
    splits: List = ["train", "valid", "test"],
    top_features: Optional[int] = None,
) -> str:
    """Generate a heatmap comparing feature masks across splits.

    Panels show one binary mask per split, plus a final *Differences* panel
    highlighting positions where not all splits agree.

    Parameters
    ----------
    pure_matrices_per_split : dict
        Mapping ``split_name -> pure feature matrix`` of shape
        ``(n_cell_types, n_gr_groups, n_pred_classes)`` (prediction columns
        only), as returned by
        :meth:`syto.data.pseudobulk_hdf5_utils.PseudobulkHDF5Reader.read_pure_feature_matrix`.
    num_input_labels : int
        Number of input prediction classes.
    num_output_labels : int
        Number of output cell-type labels.
    cutoff : float
        Feature selection cutoff.
    output_path : str
        File path where the figure will be saved (e.g. ``…/plot.png``).
    master_names : list[str], optional
        Cell-type names for axis labels.  Defaults to indices.
    splits : list[str]
        Split names to process.  The grid adapts automatically.
    top_features : int, optional
        Number of top features to dynamically determine cutoff.

    Returns
    -------
    str
        Path to the saved plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Build per-split matrices, ratios, and binary masks ---------------
    split_bins = {}
    split_ratios = {}
    for split_name in splits:
        pure_matrix = pure_matrices_per_split[split_name]
        ratios = compute_feature_ratios(pure_matrix)
        binary = compute_feature_mask(
            ratios,
            cutoff,
            guarantee_diagonal_selection,
            guarantee_columns_selection,
            top_features=top_features,
        )
        split_ratios[split_name] = ratios
        split_bins[split_name] = binary

    # Maximal binary mask across all splits (for reporting)
    maximal_ratios = np.array(list(split_ratios.values())).max(axis=0)
    maximal_bin = compute_feature_mask(
        maximal_ratios,
        cutoff,
        guarantee_diagonal_selection,
        guarantee_columns_selection,
        top_features=top_features,
    )

    # Difference mask: positions where not all splits agree
    all_bins = list(split_bins.values())
    agreement = np.ones_like(all_bins[0], dtype=bool)
    for b in all_bins[1:]:
        agreement &= all_bins[0] == b
    diff_mask = ~agreement

    # --- Axis labels ------------------------------------------------------
    if master_names is None:
        master_names = [str(i) for i in range(num_output_labels)]

    input_names = master_names.copy()
    if num_input_labels > num_output_labels:
        input_names += [f"Bg_{i}" for i in range(num_output_labels, num_input_labels)]

    total_features = num_output_labels * num_input_labels
    n_selected = int(maximal_bin.sum())
    logger.info(
        f"Number of features with cutoff {cutoff}: "
        f"{n_selected} from {total_features}"
    )

    # --- Layout: enough panels for each split + 1 diff panel --------------
    n_panels = len(splits) + 1  # +1 for the differences panel
    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)

    heatmap_kwargs = {
        "cmap": "viridis",
        "yticklabels": master_names,
        "xticklabels": input_names,
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    axes = np.atleast_2d(axes)  # guarantee 2-D indexing

    # Plot each split
    for idx, split_name in enumerate(splits):
        r, c = divmod(idx, ncols)
        sns.heatmap(split_bins[split_name], ax=axes[r, c], **heatmap_kwargs)
        axes[r, c].set_title(split_name.capitalize())

    # Differences panel
    r, c = divmod(len(splits), ncols)
    sns.heatmap(
        diff_mask.astype(int),
        ax=axes[r, c],
        cmap="Reds",
        yticklabels=master_names,
        xticklabels=input_names,
    )
    axes[r, c].set_title("Differences across splits")

    # Hide any leftover empty subplots
    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Feature selection (cutoff={cutoff})  —  "
        f"{n_selected}/{total_features} features selected",
        fontsize=14,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature selection plot saved to {output_path}")

    return output_path
