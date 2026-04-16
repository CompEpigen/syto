"""
Feature Selection for Deconvolution
====================================

Utilities for computing feature masks from purified cell-type profiles,
applying masks to full feature matrices, and generating diagnostic
visualisation plots that compare selected features across data splits.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────────────────────────────


def extract_pure_feature_matrix(
    pure_profiles: list,
    num_input_labels: int,
    num_output_labels: int,
    split_idx: int,
) -> np.ndarray:
    """Extract the feature matrix from purified profiles for one split.

    Parameters
    ----------
    pure_profiles : list
        Output of :func:`generate_pure_profiles` — a list of
        ``(proportions, subs, uxm_data)`` tuples, one per cell type.
    num_input_labels : int
        Number of input prediction classes.
    num_output_labels : int
        Number of output cell-type labels.
    split_idx : int
        Index into ``subs``: 0 = train, 1 = valid, 2 = test.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(num_output_labels, num_input_labels)`` where row *i*
        contains the DMR-aggregated prediction profile for cell type *i*
        within the chosen split.
    """
    target_columns = [f"prediction_{i}_wavg" for i in range(num_input_labels)]
    return np.array(
        [
            pure_profiles[i][1][split_idx][target_columns].to_numpy()
            for i in range(num_output_labels)
            if pure_profiles[i] is not None
        ]
    )


def compute_feature_ratios(pure_matrix: np.ndarray) -> np.ndarray:
    """Compute max / mean ratio for each feature position.

    Parameters
    ----------
    pure_matrix : np.ndarray
        Shape ``(n_cell_types, n_dmr_groups, n_pred_classes)`` feature
        matrix from :func:`extract_pure_feature_matrix`.

    Returns
    -------
    np.ndarray
        Ratio matrix of same shape as the last two dimensions of
        ``pure_matrix`` (i.e. ``(n_dmr_groups, n_pred_classes)``).
    """
    max_vals = pure_matrix.max(axis=0)
    mean_vals = pure_matrix.mean(axis=0)
    # Avoid division by zero
    mean_vals = np.where(mean_vals == 0, 1e-12, mean_vals)
    return max_vals / mean_vals


def compute_feature_mask(
    ratios: np.ndarray,
    cutoff: float = 1.1,
    guarantee_diagonal_selection: bool = False,
    guarantee_columns_selection: Optional[List[int]] = None,
) -> np.ndarray:
    """Return a binary mask where ratio > cutoff, with optional guaranteed features.

    Parameters
    ----------
    ratios : np.ndarray
        Ratio matrix from :func:`compute_feature_ratios`.
    cutoff : float
        Threshold above which a feature is selected.
    guarantee_diagonal_selection : bool
        If True, the diagonal elements of the mask are set to 1.
    guarantee_columns_selection : list of int, optional
        A list of column indices to unconditionally select (all rows for those columns set to 1).

    Returns
    -------
    np.ndarray
        Binary mask with the same shape as *ratios*.
    """
    mask = (ratios > cutoff).astype(int)

    if guarantee_diagonal_selection:
        np.fill_diagonal(mask, 1)

    if guarantee_columns_selection:
        for c in guarantee_columns_selection:
            if 0 <= c < mask.shape[1]:
                mask[:, c] = 1

    return mask


def apply_feature_mask(
    matrix: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply a binary feature mask and return a flat compressed array.

    Parameters
    ----------
    matrix : np.ndarray
        Feature matrix of shape ``(n_dmr_groups, n_pred_classes)`` or
        ``(n_samples, n_dmr_groups, n_pred_classes)``.
    mask : np.ndarray
        Binary mask of shape ``(n_dmr_groups, n_pred_classes)``.

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


def apply_mask_to_ios(
    ios_path: str,
    mask: np.ndarray,
    output_path: str,
    cutoff: float,
) -> dict:
    """Load full IO matrices, apply the feature mask, and save.

    Parameters
    ----------
    ios_path : str
        Path to the ``ios_full_matrices.npz`` file containing
        ``proportions``, ``features_train``, ``features_valid``,
        ``features_test``.
    mask : np.ndarray
        Binary feature mask of shape ``(n_dmr_groups, n_pred_classes)``.
    output_path : str
        Where to save the filtered ``.npz`` file.
    cutoff : float
        Cutoff value used (recorded in the filename for traceability).

    Returns
    -------
    dict
        Dictionary with ``proportions``, ``features_train``,
        ``features_valid``, ``features_test`` after masking.
    """
    data = np.load(ios_path)
    proportions = data["proportions"]
    features_train = data["features_train"]
    features_valid = data["features_valid"]
    features_test = data["features_test"]

    logger.info(
        f"Applying mask (cutoff={cutoff}) to IO matrices: "
        f"original feature shape per sample = {features_train.shape[1:]}, "
        f"selected features = {int(mask.sum())}"
    )

    features_train_masked = apply_feature_mask(features_train, mask)
    features_valid_masked = apply_feature_mask(features_valid, mask)
    features_test_masked = apply_feature_mask(features_test, mask)

    result = {
        "proportions": proportions,
        "features_train": features_train_masked,
        "features_valid": features_valid_masked,
        "features_test": features_test_masked,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **result)
    logger.info(f"Saved filtered features to {output_path}")

    return result


# ─────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────


def generate_feature_selection_plot(
    pure_profiles: list,
    num_input_labels: int,
    num_output_labels: int,
    cutoff: float,
    output_path: str,
    master_names: Optional[List[str]] = None,
    guarantee_diagonal_selection: bool = False,
    guarantee_columns_selection: Optional[List[int]] = None,
) -> str:
    """Generate a 2×2 heatmap comparing feature masks across splits.

    The four panels are:
    * *Train* — binary mask from the train split.
    * *Validation* — binary mask from the validation split.
    * *Test* — binary mask from the test split.
    * *Differences* — positions where not all three splits agree.

    Parameters
    ----------
    pure_profiles : list
        Output of :func:`generate_pure_profiles`.
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

    Returns
    -------
    str
        Path to the saved plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    pure_train = extract_pure_feature_matrix(
        pure_profiles, num_input_labels, num_output_labels, split_idx=0
    )
    pure_valid = extract_pure_feature_matrix(
        pure_profiles, num_input_labels, num_output_labels, split_idx=1
    )
    pure_test = extract_pure_feature_matrix(
        pure_profiles, num_input_labels, num_output_labels, split_idx=2
    )

    train_ratios = compute_feature_ratios(pure_train)
    valid_ratios = compute_feature_ratios(pure_valid)
    test_ratios = compute_feature_ratios(pure_test)

    train_bin = compute_feature_mask(
        train_ratios, cutoff, guarantee_diagonal_selection, guarantee_columns_selection
    )
    valid_bin = compute_feature_mask(
        valid_ratios, cutoff, guarantee_diagonal_selection, guarantee_columns_selection
    )
    test_bin = compute_feature_mask(
        test_ratios, cutoff, guarantee_diagonal_selection, guarantee_columns_selection
    )

    # Maximal binary mask across all splits (for reporting)
    maximal_ratios = np.array([train_ratios, valid_ratios, test_ratios]).max(axis=0)
    maximal_bin = compute_feature_mask(
        maximal_ratios,
        cutoff,
        guarantee_diagonal_selection,
        guarantee_columns_selection,
    )

    # Difference mask: where not all three agree
    diff_mask = ~((train_bin == valid_bin) & (valid_bin == test_bin))

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

    heatmap_kwargs = {
        "cmap": "viridis",
        "yticklabels": master_names,
        "xticklabels": input_names,
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    sns.heatmap(train_bin, ax=axes[0, 0], **heatmap_kwargs)
    axes[0, 0].set_title("Train")

    sns.heatmap(valid_bin, ax=axes[0, 1], **heatmap_kwargs)
    axes[0, 1].set_title("Validation")

    sns.heatmap(test_bin, ax=axes[1, 0], **heatmap_kwargs)
    axes[1, 0].set_title("Test")

    sns.heatmap(
        diff_mask.astype(int),
        ax=axes[1, 1],
        cmap="Reds",
        yticklabels=master_names,
        xticklabels=master_names,
    )
    axes[1, 1].set_title("Differences across splits")

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
