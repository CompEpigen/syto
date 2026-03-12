import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import confusion_matrix, r2_score


def plot_confusion_matrix_for_target_confidence(
    predictions,
    confidence,
    target_confidence,
    labels,
    labels_list,
    title="",
    exclude_rejected=True,
):
    confident_predictions = np.array(
        [
            int(x) if y > target_confidence else 39
            for (x, y) in zip(predictions, confidence)
        ]
    )
    # 1. Compute confusion matrix
    cf_matrix = confusion_matrix(labels, confident_predictions)
    # 2. Logic to exclude last row/col from Color Scaling
    # We slice the matrix to get everything EXCEPT the last row and last column
    if exclude_rejected:
        subset_matrix = cf_matrix[:-1, :-1]
    else:
        subset_matrix = cf_matrix
    # Find the max value in the specific cell types
    max_val_subset = np.max(subset_matrix)

    # Create the heatmap
    fig, ax = plt.subplots(1, figsize=(20, 20))

    # 3. Apply the scaling using vmax
    # 'vmax' clamps the color range. Anything higher than this (i.e., the last row/col)
    # will appear as the darkest color (saturated), but won't distort the gradient for the rest.
    cax = ax.matshow(cf_matrix, cmap="PuRd", vmin=0, vmax=max_val_subset)

    # Add colorbar
    # extend='max' indicates that values exist beyond the top of the colorbar
    plt.colorbar(cax, fraction=0.046, pad=0.04, extend="max")

    # 4. Annotate the heatmap
    # We adjust the text color threshold.
    # Since the last row/col is saturated, we force white text there for readability.
    for (i, j), val in np.ndenumerate(cf_matrix):
        # Determine text color:
        # If we are in the "Background" zone (last row or col), usually dark color -> use White text
        # Else, use standard thresholding based on the subset max
        if i == cf_matrix.shape[0] - 1 or j == cf_matrix.shape[1] - 1:
            color = (
                "white" if val > (max_val_subset * 0.3) else "black"
            )  # Adjust contrast as needed
        else:
            color = "white" if val > (max_val_subset / 2) else "black"

        # Optional: If numbers are huge, use scientific notation or hide zeros
        label_text = f"{val}" if val > 0 else ""
        ax.text(j, i, label_text, ha="center", va="center", color=color, fontsize=8)

    # Set axis labels and tick marks
    ax.set_xlabel("Prediction", fontsize=16, labelpad=20)
    ax.set_ylabel("Ground-truth", fontsize=16, labelpad=20)

    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, rotation=90, fontsize=8)  # Rotated for readability
    ax.set_yticks(range(len(labels_list)))
    ax.set_yticklabels(labels_list, fontsize=8)

    # Move x-axis ticks to bottom (matshow puts them on top by default)
    ax.xaxis.set_ticks_position("bottom")

    # Set title
    ax.set_title(f"{title} (Color scaled to {max_val_subset})", fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


def plot_predictions_by_celltype(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_dict: dict,
    figsize: tuple = (14, 10),
    alpha: float = 0.6,
    point_size: float = 20,
    show_regression_lines: bool = True,
    show_overall_r2: bool = True,
    show_per_celltype_r2: bool = True,
    cmap: str = "tab20",
    title: str = "Predicted vs Expected Proportions by Cell Type",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Scatter plot of predicted vs expected proportions, colored by cell type.

    Parameters
    ----------
    y_true : np.ndarray
        True proportions, shape (n_samples, n_cell_types).
    y_pred : np.ndarray
        Predicted proportions, shape (n_samples, n_cell_types).
    labels_dict : dict
        Mapping from position index to cell type name, e.g., {0: 'T-cell', 1: 'B-cell', ...}.
    figsize : tuple
        Figure size.
    alpha : float
        Point transparency.
    point_size : float
        Size of scatter points.
    show_regression_lines : bool
        Whether to show per-cell-type regression lines.
    show_overall_r2 : bool
        Whether to show overall R² in the title.
    show_per_celltype_r2 : bool
        Whether to show per-cell-type R² in the legend.
    cmap : str
        Colormap name.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    metrics : dict
        Dictionary containing R² scores (overall and per cell type).
    """
    n_samples, n_cell_types = y_true.shape

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i / n_cell_types) for i in range(n_cell_types)]

    # Store metrics
    metrics = {"overall_r2": None, "per_celltype_r2": {}, "per_celltype_slope": {}}

    # Compute overall R²
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    overall_r2 = r2_score(y_true_flat, y_pred_flat)
    metrics["overall_r2"] = overall_r2

    # Plot each cell type
    legend_handles = []
    legend_labels = []

    for ct_idx in range(n_cell_types):
        ct_name = labels_dict.get(ct_idx, f"Cell Type {ct_idx}")

        true_vals = y_true[:, ct_idx]
        pred_vals = y_pred[:, ct_idx]

        # Compute per-cell-type R²
        if len(np.unique(true_vals)) > 1:  # Need variance to compute R²
            ct_r2 = r2_score(true_vals, pred_vals)

            # Linear regression for the line
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept, _, _, _ = stats.linregress(pred_vals, true_vals)
        else:
            ct_r2 = np.nan
            slope, intercept = 1, 0

        metrics["per_celltype_r2"][ct_name] = ct_r2
        metrics["per_celltype_slope"][ct_name] = slope

        # Scatter plot
        scatter = ax.scatter(
            pred_vals,
            true_vals,
            c=[colors[ct_idx]],
            alpha=alpha,
            s=point_size,
            label=ct_name,
        )

        # Regression line per cell type
        if show_regression_lines and not np.isnan(ct_r2):
            x_line = np.array([0, 1])
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=colors[ct_idx],
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )

        # Legend entry
        if show_per_celltype_r2 and not np.isnan(ct_r2):
            legend_labels.append(f"{ct_name} (R²={ct_r2:.3f})")
        else:
            legend_labels.append(ct_name)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[ct_idx],
                markersize=8,
                alpha=0.8,
            )
        )

    # Perfect prediction line
    ax.plot([0, 1], [0, 1], "k-", linewidth=2, alpha=0.5, label="Perfect prediction")

    # Labels and title
    ax.set_xlabel("Predicted Proportion", fontsize=12)
    ax.set_ylabel("Expected Proportion", fontsize=12)

    if show_overall_r2:
        ax.set_title(f"{title}\nOverall R² = {overall_r2:.4f}", fontsize=14)
    else:
        ax.set_title(title, fontsize=14)

    # Set axis limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend (outside plot if many cell types)
    if n_cell_types > 10:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            ncol=1 + n_cell_types // 20,
        )
    else:
        ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=9)

    plt.tight_layout()

    return fig, ax, metrics


def plot_predictions_by_mixture_complexity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (12, 10),
    alpha: float = 0.5,
    point_size: float = 20,
    zero_threshold: float = 1e-6,
    show_regression_lines: bool = True,
    show_overall_r2: bool = True,
    show_per_complexity_r2: bool = True,
    cmap: str = "viridis",
    title: str = "Predicted vs Expected Proportions by Mixture Complexity",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Scatter plot of predicted vs expected proportions, colored by number of cell types in mixture.

    Parameters
    ----------
    y_true : np.ndarray
        True proportions, shape (n_samples, n_cell_types).
    y_pred : np.ndarray
        Predicted proportions, shape (n_samples, n_cell_types).
    figsize : tuple
        Figure size.
    alpha : float
        Point transparency.
    point_size : float
        Size of scatter points.
    zero_threshold : float
        Threshold below which a proportion is considered zero.
    show_regression_lines : bool
        Whether to show per-complexity regression lines.
    show_overall_r2 : bool
        Whether to show overall R² in the title.
    show_per_complexity_r2 : bool
        Whether to show per-complexity R² in the legend.
    cmap : str
        Colormap name.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    metrics : dict
        Dictionary containing R² scores and sample counts per complexity level.
    """
    n_samples, n_cell_types = y_true.shape

    # Compute number of cell types per sample (complexity)
    n_celltypes_per_sample = (y_true > zero_threshold).sum(axis=1)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get unique complexity levels
    unique_complexities = np.sort(np.unique(n_celltypes_per_sample))
    n_complexities = len(unique_complexities)

    # Get colormap
    cmap_obj = plt.get_cmap(cmap)
    # Map complexities to colors
    complexity_to_color = {
        c: cmap_obj(i / max(n_complexities - 1, 1))
        for i, c in enumerate(unique_complexities)
    }

    # Store metrics
    metrics = {
        "overall_r2": None,
        "per_complexity_r2": {},
        "per_complexity_n_samples": {},
        "per_complexity_slope": {},
    }

    # Compute overall R²
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    overall_r2 = r2_score(y_true_flat, y_pred_flat)
    metrics["overall_r2"] = overall_r2

    # Plot each complexity level
    legend_handles = []
    legend_labels = []

    for complexity in unique_complexities:
        # Get samples with this complexity
        sample_mask = n_celltypes_per_sample == complexity
        n_samples_complexity = sample_mask.sum()

        # Get all (pred, true) pairs for these samples
        true_vals = y_true[sample_mask].flatten()
        pred_vals = y_pred[sample_mask].flatten()

        metrics["per_complexity_n_samples"][complexity] = n_samples_complexity

        # Compute R² for this complexity level
        if len(np.unique(true_vals)) > 1:
            complexity_r2 = r2_score(true_vals, pred_vals)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept, _, _, _ = stats.linregress(pred_vals, true_vals)
        else:
            complexity_r2 = np.nan
            slope, intercept = 1, 0

        metrics["per_complexity_r2"][complexity] = complexity_r2
        metrics["per_complexity_slope"][complexity] = slope

        color = complexity_to_color[complexity]

        # Scatter plot
        ax.scatter(pred_vals, true_vals, c=[color], alpha=alpha, s=point_size)

        # Regression line per complexity
        if show_regression_lines and not np.isnan(complexity_r2):
            x_line = np.array([0, 1])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle="--", alpha=0.8, linewidth=2)

        # Legend entry
        if show_per_complexity_r2 and not np.isnan(complexity_r2):
            label = f"{complexity} cell types (n={n_samples_complexity}, R²={complexity_r2:.3f})"
        else:
            label = f"{complexity} cell types (n={n_samples_complexity})"

        legend_labels.append(label)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                alpha=0.8,
            )
        )

    # Perfect prediction line
    ax.plot([0, 1], [0, 1], "k-", linewidth=2, alpha=0.5, label="Perfect prediction")

    # Labels and title
    ax.set_xlabel("Predicted Proportion", fontsize=12)
    ax.set_ylabel("Expected Proportion", fontsize=12)

    if show_overall_r2:
        ax.set_title(f"{title}\nOverall R² = {overall_r2:.4f}", fontsize=14)
    else:
        ax.set_title(title, fontsize=14)

    # Set axis limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        fontsize=10,
        title="Mixture Complexity",
    )

    plt.tight_layout()

    return fig, ax, metrics


def plot_deconvolution_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_dict: dict,
    figsize: tuple = (20, 10),
    zero_threshold: float = 1e-6,
    save_path: Optional[str] = None,
    **kwargs,
) -> tuple[plt.Figure, dict]:
    """
    Create a combined figure with both visualization plots side by side.

    Parameters
    ----------
    y_true : np.ndarray
        True proportions, shape (n_samples, n_cell_types).
    y_pred : np.ndarray
        Predicted proportions, shape (n_samples, n_cell_types).
    labels_dict : dict
        Mapping from position index to cell type name.
    figsize : tuple
        Figure size for combined plot.
    zero_threshold : float
        Threshold for determining non-zero proportions.
    save_path : str, optional
        Path to save the figure.
    **kwargs
        Additional arguments passed to both plotting functions.

    Returns
    -------
    fig : plt.Figure
        The combined figure.
    all_metrics : dict
        Combined metrics from both plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot by cell type
    _, _, metrics_celltype = plot_predictions_by_celltype(
        y_true,
        y_pred,
        labels_dict,
        ax=axes[0],
        title="By Cell Type",
        **{k: v for k, v in kwargs.items() if k != "cmap"},
    )

    # Plot by mixture complexity
    _, _, metrics_complexity = plot_predictions_by_mixture_complexity(
        y_true,
        y_pred,
        ax=axes[1],
        zero_threshold=zero_threshold,
        title="By Mixture Complexity",
        **{k: v for k, v in kwargs.items() if k != "cmap"},
    )

    plt.suptitle("Deconvolution Model Evaluation", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    all_metrics = {
        "celltype_metrics": metrics_celltype,
        "complexity_metrics": metrics_complexity,
    }

    return fig, all_metrics
