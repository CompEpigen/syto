"""
This module defines metrics for evaluating the calibration of read-level classifiers.
"""

from typing import Literal, Dict, Union
import logging

import numpy as np

_module_logger = logging.getLogger(__name__)


def _compute_bin_edges(
    x: np.array,
    n_bins: int,
    binning_strategy: Literal["equal_width", "equal_frequency"],
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.array:
    """
    This function returns the bin edges for the given data and binning strategy.

    Args:
        x: The input data for which to compute the bin edges.
        n_bins: The number of bins to use.
        binning_strategy: The strategy to use for binning.
            Can be either "equal_width" or "equal_frequency".
        min_value: The minimum value for equal width binning (default is 0.0).
        max_value: The maximum value for equal width binning (default is 1.0).

    Returns:
        An array of bin edges of length n_bins + 1.
    """
    # raise a warning if the data contains values outside the specified min and max range
    if np.any(x < min_value) or np.any(x > max_value):
        _module_logger.warning(
            "Data contains values outside the specified range [%s, %s]."
            "These values will be clipped to the range when computing bin edges.",
            min_value,
            max_value,
        )
    x = np.clip(x, min_value, max_value)

    if binning_strategy == "equal_width":
        return np.linspace(min_value, max_value, n_bins + 1)
    elif binning_strategy == "equal_frequency":
        # quantile-based edges. If there are ties or few unique values (common
        # for a heavily imbalanced rare class where most probs are near zero)
        # we fall back to unique quantile values to avoid zero-width bins.
        quantiles = np.linspace(min_value, max_value, n_bins + 1)
        edges = np.quantile(x, quantiles)
        edges[0], edges[-1] = min_value, max_value
        # Separate the duplicate edges by a tiny epsilon so digitize can still separate points.
        # We keep the leftmost edge at min_value, and the rightmost at max_value.
        eps = 1e-12
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = min(edges[i - 1] + eps, max_value)
        return edges
    raise ValueError(
        f"Unknown binning strategy '{binning_strategy}'."
        "Use 'equal_width' or 'equal_frequency'."
    )


def _assign_to_bins(x: np.array, bin_edges: np.array) -> np.array:
    """
    This function assigns each value in x to a bin based on the provided bin edges.

    Args:
        x: The input data to assign to bins.
        bin_edges: The edges of the bins.

    Returns:
        An array of bin indices for each value in x.
    """
    n_bins = len(bin_edges) - 1
    # np.digitize with right=True => (edges[i-1], edges[i]]
    # We want the leftmost bin to include 0.0, so we treat the first edge as
    # inclusive. Using right=True and then clipping handles this cleanly.
    idx = np.digitize(x, bin_edges[1:-1], right=True)
    return np.clip(idx, 0, n_bins - 1)


def _compute_bins_and_stats_per_bin(
    targets: np.array,
    confidences: np.array,
    n_bins: int,
    binning_strategy: Literal["equal_width", "equal_frequency"],
) -> Dict[str, Union[np.array, float]]:
    """
    This function computes the bin edges, assigns each confidence to a bin,
    and computes the average confidence and accuracy for each bin.

    Args:
        targets: The true targets (0/1 for hard labels, [0, 1] for soft labels) for each sample.
        confidences: The predicted probabilities (confidence scores) for each sample.
        n_bins: The number of bins to use.
        binning_strategy: The strategy to use for binning.
            Can be either "equal_width" or "equal_frequency".

    Returns:
        A dictionnary containing:
        - bin_edges: An array of bin edges of length n_bins + 1.
        - bin_indices: An array of bin indices for each sample.
        - counts_per_bin: An array of counts of samples in each bin.
        - avg_confidence_per_bin: An array of average confidence for each bin (NaN for empty bins).
        - accuracy_per_bin: An array of accuracy for each bin (NaN for empty bins).
        - diff_per_bin: An array of absolute difference between avg confidence and
            accuracy for each bin (NaN for empty bins).
    """
    assert (
        targets.shape == confidences.shape
    ), "Targets and confidences must have the same shape."
    assert targets.ndim == 1, "Targets and confidences must be 1D arrays."

    bin_edges = _compute_bin_edges(confidences, n_bins, binning_strategy)
    bin_indices = _assign_to_bins(confidences, bin_edges)

    counts_per_bin = np.bincount(bin_indices, minlength=n_bins)
    sum_confidence_per_bin = np.bincount(
        bin_indices, weights=confidences, minlength=n_bins
    )
    sum_targets_per_bin = np.bincount(bin_indices, weights=targets, minlength=n_bins)

    avg_confidence_per_bin = np.full(n_bins, np.nan)
    accuracy_per_bin = np.full(n_bins, np.nan)
    diff_per_bin = np.full(n_bins, np.nan)
    nonempty_bins = counts_per_bin > 0
    avg_confidence_per_bin[nonempty_bins] = (
        sum_confidence_per_bin[nonempty_bins] / counts_per_bin[nonempty_bins]
    )
    accuracy_per_bin[nonempty_bins] = (
        sum_targets_per_bin[nonempty_bins] / counts_per_bin[nonempty_bins]
    )
    diff_per_bin[nonempty_bins] = np.abs(
        avg_confidence_per_bin[nonempty_bins] - accuracy_per_bin[nonempty_bins]
    )

    return {
        "bin_edges": bin_edges,
        "bin_indices": bin_indices,
        "counts_per_bin": counts_per_bin,
        "avg_confidence_per_bin": avg_confidence_per_bin,
        "accuracy_per_bin": accuracy_per_bin,
        "diff_per_bin": diff_per_bin,
    }


def _compute_ece(
    counts_per_bin: np.array,
    diff_per_bin: np.array,
) -> float:
    """
    This function computes the Expected Calibration Error (ECE) given the counts
    and differences per bin.

    Args:
        counts_per_bin: An array of counts of samples in each bin.
        diff_per_bin: An array of absolute differences between average confidence
            and accuracy for each bin.

    Returns:
        The Expected Calibration Error (ECE) as a float.
    """
    total_samples = np.sum(counts_per_bin)
    if total_samples == 0:
        return 0.0
    ece = np.nansum((counts_per_bin / total_samples) * diff_per_bin)
    return ece


def _compute_mce(
    diff_per_bin: np.array,
    counts_per_bin: np.array,
    min_bin_count: int,
) -> float:
    """
    This function computes the Maximum Calibration Error (MCE)
    given the differences per bin.

    Args:
        diff_per_bin: An array of absolute differences between average confidence
            and accuracy for each bin.
        counts_per_bin: An array of counts of samples in each bin.
        min_bin_count: The minimum number of samples required in a bin for it to be considered

    Returns:
        The Maximum Calibration Error (MCE) as a float.
    """
    is_bin_reliable = counts_per_bin >= min_bin_count
    if is_bin_reliable.any():
        mce = float(np.nanmax(np.where(is_bin_reliable, diff_per_bin, -np.inf)))
        if mce == -np.inf:
            mce = float("nan")
    else:
        mce = float("nan")
    return mce


def compute_brier_score(
    targets: np.array,
    confidences: np.array,
) -> float:
    """
    This function computes the Brier Score given the targets and confidences.

    Args:
        targets: An array of true targets (0/1 for hard labels, [0, 1] for soft labels)
            for each sample.
        confidences: An array of predicted probabilities (confidence scores) for each sample.

    Returns:
        The Brier Score as a float.
    """
    return np.mean((targets - confidences) ** 2)


def compute_classifier_calibration_metrics(
    y_true: np.array,
    y_pred_proba: np.array,
    n_bins: int,
    min_bin_count_for_mce: int,
    binning_strategy: Literal["equal_width", "equal_frequency"] = "equal_width",
) -> Dict[str, np.array]:
    """
    This function computes:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier Score
    - Class-j-ECE for each class j
    - Classwise ECE
    for a classifier's predicted probabilities.

    Args:
        y_true: The true labels for each sample. Can be either hard labels (shape (n_samples,))
            or soft labels (shape (n_samples, n_classes)).
        y_pred_proba: The predicted probabilities for each sample and class
            (shape (n_samples, n_classes)).
        n_bins: The number of bins to use for computing ECE and MCE.
        min_bin_count_for_mce: The minimum number of samples required in a bin
            for it to be considered in MCE computation.
        binning_strategy: The strategy to use for binning.
            Can be either "equal_width" or "equal_frequency" (default is "equal_width").

    Returns:
        A dictionary containing the computed metrics:
        - "top_label_ece": The ECE for the top predicted label.
        - "top_label_mce": The MCE for the top predicted label.
        - "top_label_brier": The Brier Score for the top predicted label.
        - "class_j_ece": An array of ECE values for each class j.
        - "classwise_ece": The non weighted average class-j ECE across all classes.
    """
    # Input validation and preparation
    assert binning_strategy in [
        "equal_width",
        "equal_frequency",
    ], "Invalid binning strategy."
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    assert (
        y_pred_proba.ndim == 2
    ), "y_pred_proba must be a 2D array of shape (n_samples, n_classes)."
    is_soft_labels = y_true.ndim == 2
    if is_soft_labels:
        assert (
            y_true.shape == y_pred_proba.shape
        ), "For soft labels, y_true and y_pred_proba must have the same shape."
    else:
        assert (
            y_true.ndim == 1
        ), "For hard labels, y_true must be a 1D array of shape (n_samples,)."
        assert (
            y_true.shape[0] == y_pred_proba.shape[0]
        ), "y_true and y_pred_proba must have the same number of samples."

    ## top label calibration metrics
    # top label data preparation
    top_label_predictions = np.argmax(y_pred_proba, axis=1)
    top_label_confidence = np.max(y_pred_proba, axis=1)
    if is_soft_labels:
        top_label_targets = y_true[np.arange(y_true.shape[0]), top_label_predictions]
    else:
        top_label_targets = y_true == top_label_predictions
    top_label_targets = top_label_targets.astype(np.float64)
    # top label binning and metrics computation
    top_label_bins = _compute_bins_and_stats_per_bin(
        top_label_targets, top_label_confidence, n_bins, binning_strategy
    )
    top_label_ece = _compute_ece(
        top_label_bins["counts_per_bin"], top_label_bins["diff_per_bin"]
    )
    top_label_mce = _compute_mce(
        top_label_bins["diff_per_bin"],
        top_label_bins["counts_per_bin"],
        min_bin_count_for_mce,
    )
    top_label_brier = compute_brier_score(top_label_targets, top_label_confidence)

    ## class-j-ECE and classwise ECE
    n_classes = y_pred_proba.shape[1]
    class_j_ece = np.zeros(n_classes)
    for j in range(n_classes):
        # class-j data preparation
        if is_soft_labels:
            class_j_targets = y_true[:, j]
        else:
            class_j_targets = y_true == j
        class_j_targets = class_j_targets.astype(np.float64)
        class_j_confidence = y_pred_proba[:, j]
        class_j_bins = _compute_bins_and_stats_per_bin(
            class_j_targets, class_j_confidence, n_bins, binning_strategy
        )
        # class-j ECE computation
        class_j_ece[j] = _compute_ece(
            class_j_bins["counts_per_bin"], class_j_bins["diff_per_bin"]
        )
    classwise_ece = np.mean(class_j_ece)

    return {
        "top_label_ece": top_label_ece,
        "top_label_mce": top_label_mce,
        "top_label_brier": top_label_brier,
        "class_j_ece": class_j_ece,
        "classwise_ece": classwise_ece,
    }
