import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import confusion_matrix, r2_score

from methyldl.deconvolution.deep_deconvolvers.diagonal_aware_deconvolver import (
    DiagonalAwareDeconvolver,
)
from methyldl.deconvolution.deep_deconvolvers.training import (
    DeconvolverOutput,
)

def print_deconvolution_metrics_summary(metrics: dict):
    """Print a formatted summary of the metrics."""
    print("=" * 60)
    print("DECONVOLUTION RESULTS SUMMARY")
    print("=" * 60)
    
    if 'celltype_metrics' in metrics:
        ct_metrics = metrics['celltype_metrics']
        print(f"\n📊 Overall R²: {ct_metrics['overall_r2']:.4f}")
        
        print("\n📋 Per Cell Type R²:")
        print("-" * 40)
        
        # Sort by R² descending
        sorted_ct = sorted(
            ct_metrics['per_celltype_r2'].items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -1,
            reverse=True
        )
        
        for ct_name, r2 in sorted_ct:
            if not np.isnan(r2):
                bar = "█" * int(r2 * 20) + "░" * (20 - int(r2 * 20))
                print(f"  {ct_name:20s} │ {bar} │ {r2:.4f}")
            else:
                print(f"  {ct_name:20s} │ {'N/A':^20s} │ N/A")
    
    if 'complexity_metrics' in metrics:
        cx_metrics = metrics['complexity_metrics']
        
        print("\n📈 Performance by Mixture Complexity:")
        print("-" * 50)
        print(f"  {'# Cell Types':^15s} │ {'N Samples':^10s} │ {'R²':^10s}")
        print("-" * 50)
        
        for complexity in sorted(cx_metrics['per_complexity_r2'].keys()):
            r2 = cx_metrics['per_complexity_r2'][complexity]
            n_samples = cx_metrics['per_complexity_n_samples'][complexity]
            
            if not np.isnan(r2):
                print(f"  {complexity:^15d} │ {n_samples:^10d} │ {r2:^10.4f}")
            else:
                print(f"  {complexity:^15d} │ {n_samples:^10d} │ {'N/A':^10s}")
    
    print("\n" + "=" * 60)

def bland_altman_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_dict: Optional[dict] = None,
    target_celltype: Optional[Union[int, str, List[int], List[str]]] = None,
    target_complexity: Optional[Union[int, List[int], tuple]] = None,
    zero_threshold: float = 1e-6,
    figsize: tuple = (12, 8),
    alpha: float = 0.5,
    point_size: float = 30,
    show_ci: bool = True,
    ci_level: float = 0.95,
    show_regression: bool = True,
    show_density: bool = False,
    color_by: Optional[str] = None,  # 'celltype', 'complexity', 'mean', None
    cmap: str = "viridis",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_stats: bool = True,
) -> tuple[plt.Figure, plt.Axes, Optional[dict]]:
    """
    Bland-Altman plot for deconvolution results with filtering options.

    Parameters
    ----------
    y_true : np.ndarray
        True proportions, shape (n_samples, n_cell_types).
    y_pred : np.ndarray
        Predicted proportions, shape (n_samples, n_cell_types).
    labels_dict : dict, optional
        Mapping from position index to cell type name.
    target_celltype : int, str, or list, optional
        Filter to specific cell type(s). Can be:
        - int: cell type index
        - str: cell type name (requires labels_dict)
        - list of int/str: multiple cell types
        - None: use all cell types
    target_complexity : int, list, or tuple, optional
        Filter to samples with specific mixture complexity. Can be:
        - int: exact number of cell types
        - list of int: specific complexity values
        - tuple (min, max): range of complexities (inclusive)
        - None: use all samples
    zero_threshold : float
        Threshold for determining non-zero proportions.
    figsize : tuple
        Figure size.
    alpha : float
        Point transparency.
    point_size : float
        Size of scatter points.
    show_ci : bool
        Whether to show confidence intervals for limits of agreement.
    ci_level : float
        Confidence level for intervals (default 0.95).
    show_regression : bool
        Whether to show regression line (checks for proportional bias).
    show_density : bool
        Whether to show density coloring instead of uniform color.
    color_by : str, optional
        Color points by: 'celltype', 'complexity', 'mean', or None.
    cmap : str
        Colormap for coloring.
    title : str, optional
        Custom title. Auto-generated if None.
    ax : plt.Axes, optional
        Existing axes to plot on.
    return_stats : bool
        Whether to return statistics dictionary.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    stats_dict : dict, optional
        Dictionary containing Bland-Altman statistics.
    """
    n_samples, n_cell_types = y_true.shape

    # Create labels_dict if not provided
    if labels_dict is None:
        labels_dict = {i: f"Cell Type {i}" for i in range(n_cell_types)}

    # Reverse mapping for name -> index
    name_to_idx = {v: k for k, v in labels_dict.items()}

    # Compute mixture complexity per sample
    complexity_per_sample = (y_true > zero_threshold).sum(axis=1)

    # --- Filter by cell type ---
    if target_celltype is not None:
        # Convert to list of indices
        if isinstance(target_celltype, (int, str)):
            target_celltype = [target_celltype]

        celltype_indices = []
        for ct in target_celltype:
            if isinstance(ct, str):
                if ct in name_to_idx:
                    celltype_indices.append(name_to_idx[ct])
                else:
                    raise ValueError(f"Cell type '{ct}' not found in labels_dict")
            else:
                celltype_indices.append(ct)
    else:
        celltype_indices = list(range(n_cell_types))

    # --- Filter by complexity ---
    if target_complexity is not None:
        if isinstance(target_complexity, int):
            valid_samples = complexity_per_sample == target_complexity
        elif isinstance(target_complexity, (list, np.ndarray)):
            valid_samples = np.isin(complexity_per_sample, target_complexity)
        elif isinstance(target_complexity, tuple):
            min_c, max_c = target_complexity
            valid_samples = (complexity_per_sample >= min_c) & (
                complexity_per_sample <= max_c
            )
        else:
            raise ValueError("target_complexity must be int, list, or tuple")
    else:
        valid_samples = np.ones(n_samples, dtype=bool)

    # --- Extract filtered data ---
    # Get true and predicted values for selected cell types and samples
    y_true_filtered = y_true[valid_samples][:, celltype_indices]
    y_pred_filtered = y_pred[valid_samples][:, celltype_indices]
    complexity_filtered = complexity_per_sample[valid_samples]

    # Create arrays for plotting
    # Each point is (sample, celltype) pair
    n_filtered_samples = valid_samples.sum()
    n_filtered_celltypes = len(celltype_indices)

    true_vals = y_true_filtered.flatten()
    pred_vals = y_pred_filtered.flatten()

    # For coloring
    sample_indices = np.repeat(np.arange(n_filtered_samples), n_filtered_celltypes)
    celltype_labels = np.tile(celltype_indices, n_filtered_samples)
    complexity_labels = np.repeat(complexity_filtered, n_filtered_celltypes)

    # --- Bland-Altman calculations ---
    means = (true_vals + pred_vals) / 2
    diffs = pred_vals - true_vals  # predicted - true

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    # Limits of agreement
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Confidence intervals for mean and LoA
    n = len(diffs)
    se_mean = std_diff / np.sqrt(n)
    se_loa = np.sqrt(3 * std_diff**2 / n)

    t_crit = stats.t.ppf((1 + ci_level) / 2, n - 1)

    ci_mean = (mean_diff - t_crit * se_mean, mean_diff + t_crit * se_mean)
    ci_upper = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)
    ci_lower = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)

    # Regression for proportional bias
    if len(means) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(means, diffs)
        has_proportional_bias = p_value < 0.05
    else:
        slope, intercept, r_value, p_value = 0, mean_diff, 0, 1
        has_proportional_bias = False

    # --- Create figure ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- Determine colors ---
    if color_by == "celltype":
        cmap_obj = plt.get_cmap("tab20" if n_filtered_celltypes <= 20 else cmap)
        unique_ct = np.unique(celltype_labels)
        color_map = {
            ct: cmap_obj(i / max(len(unique_ct) - 1, 1))
            for i, ct in enumerate(unique_ct)
        }
        colors = [color_map[ct] for ct in celltype_labels]
    elif color_by == "complexity":
        cmap_obj = plt.get_cmap(cmap)
        unique_cx = np.unique(complexity_labels)
        min_cx, max_cx = unique_cx.min(), unique_cx.max()
        colors = [
            cmap_obj((cx - min_cx) / max(max_cx - min_cx, 1))
            for cx in complexity_labels
        ]
    elif color_by == "mean":
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj((means - means.min()) / max(means.max() - means.min(), 1e-10))
    elif show_density:
        # Density-based coloring
        from scipy.stats import gaussian_kde

        xy = np.vstack([means, diffs])
        try:
            density = gaussian_kde(xy)(xy)
            colors = density
            cmap_obj = plt.get_cmap(cmap)
        except:
            colors = "steelblue"
    else:
        colors = "steelblue"

    # --- Plot points ---
    scatter = ax.scatter(
        means,
        diffs,
        c=colors,
        alpha=alpha,
        s=point_size,
        cmap=cmap if isinstance(colors, np.ndarray) else None,
    )

    # --- Plot reference lines ---
    x_range = np.array([means.min() - 0.02, means.max() + 0.02])

    # Zero line
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Mean difference line
    ax.axhline(
        mean_diff,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean diff: {mean_diff:.4f}",
    )

    # Limits of agreement
    ax.axhline(
        loa_upper,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"+1.96 SD: {loa_upper:.4f}",
    )
    ax.axhline(
        loa_lower,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"-1.96 SD: {loa_lower:.4f}",
    )

    # Confidence intervals
    if show_ci:
        ax.fill_between(x_range, ci_mean[0], ci_mean[1], color="blue", alpha=0.1)
        ax.fill_between(x_range, ci_upper[0], ci_upper[1], color="red", alpha=0.1)
        ax.fill_between(x_range, ci_lower[0], ci_lower[1], color="red", alpha=0.1)

    # Regression line (proportional bias)
    if show_regression:
        y_reg = slope * x_range + intercept
        linestyle = "-" if has_proportional_bias else ":"
        ax.plot(
            x_range,
            y_reg,
            color="green",
            linestyle=linestyle,
            linewidth=1.5,
            label=f"Regression (p={p_value:.3f})",
        )

    # --- Labels and title ---
    ax.set_xlabel("Mean of Predicted and True Proportion", fontsize=12)
    ax.set_ylabel("Difference (Predicted - True)", fontsize=12)

    # Generate title
    if title is None:
        title_parts = ["Bland-Altman Plot"]

        if target_celltype is not None:
            if len(celltype_indices) == 1:
                ct_name = labels_dict.get(
                    celltype_indices[0], f"Cell Type {celltype_indices[0]}"
                )
                title_parts.append(f"Cell Type: {ct_name}")
            else:
                title_parts.append(f"{len(celltype_indices)} Cell Types")

        if target_complexity is not None:
            if isinstance(target_complexity, int):
                title_parts.append(f"Complexity: {target_complexity}")
            elif isinstance(target_complexity, tuple):
                title_parts.append(
                    f"Complexity: {target_complexity[0]}-{target_complexity[1]}"
                )
            else:
                title_parts.append(f"Complexity: {target_complexity}")

        title = " | ".join(title_parts)

    ax.set_title(
        f"{title}\n(n={len(diffs)} points from {n_filtered_samples} samples)",
        fontsize=12,
    )

    # --- Legend ---
    ax.legend(loc="upper right", fontsize=9)

    # --- Color legend if needed ---
    if color_by == "celltype" and n_filtered_celltypes <= 20:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[ct],
                markersize=8,
                label=labels_dict.get(ct, f"CT{ct}"),
            )
            for ct in unique_ct
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=8,
            title="Cell Type",
            bbox_to_anchor=(1.02, 1),
        )
    elif color_by == "complexity":
        sm = plt.cm.ScalarMappable(
            cmap=cmap_obj, norm=plt.Normalize(vmin=min_cx, vmax=max_cx)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Mixture Complexity")
    elif color_by == "mean" or (show_density and isinstance(colors, np.ndarray)):
        plt.colorbar(
            scatter, ax=ax, label="Mean Proportion" if color_by == "mean" else "Density"
        )

    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)

    plt.tight_layout()

    # --- Compile statistics ---
    stats_dict = None
    if return_stats:
        stats_dict = {
            "n_points": len(diffs),
            "n_samples": n_filtered_samples,
            "n_celltypes": n_filtered_celltypes,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "loa_upper": loa_upper,
            "loa_lower": loa_lower,
            "ci_mean": ci_mean,
            "ci_loa_upper": ci_upper,
            "ci_loa_lower": ci_lower,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "regression_r": r_value,
            "regression_p": p_value,
            "has_proportional_bias": has_proportional_bias,
            "percent_within_loa": np.mean((diffs >= loa_lower) & (diffs <= loa_upper))
            * 100,
        }

    return fig, ax, stats_dict


def bland_altman_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_dict: dict,
    by: str = "celltype",  # 'celltype' or 'complexity'
    celltypes_to_show: Optional[List[Union[int, str]]] = None,
    complexities_to_show: Optional[List[int]] = None,
    filter_mode: Literal["all", "nonzero", "either_nonzero"] = "nonzero",
    zero_threshold: float = 1e-4,
    ncols: int = 4,
    subplot_size: tuple = (4, 3),
    shared_ylim: bool = True,
    ylim_padding: float = 0.1,
    show_ci: bool = False,
    show_regression: bool = True,
    alpha: float = 0.4,
    point_size: float = 15,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray, dict]:
    """
    Create a grid of Bland-Altman plots with shared y-axis limits.

    Parameters
    ----------
    y_true : np.ndarray
        True proportions, shape (n_samples, n_cell_types).
    y_pred : np.ndarray
        Predicted proportions, shape (n_samples, n_cell_types).
    labels_dict : dict
        Mapping from position index to cell type name.
    by : str
        'celltype' to create one plot per cell type, or
        'complexity' to create one plot per mixture complexity level.
    celltypes_to_show : list, optional
        Specific cell types to show (if by='celltype'). Shows all if None.
    complexities_to_show : list, optional
        Specific complexity levels to show (if by='complexity'). Shows all if None.
    filter_mode : str
        'all', 'nonzero' (recommended), or 'either_nonzero'.
    zero_threshold : float
        Threshold for determining non-zero proportions.
    ncols : int
        Number of columns in the grid.
    subplot_size : tuple
        Size of each subplot.
    shared_ylim : bool
        If True, all subplots share the same y-axis limits (RECOMMENDED).
    ylim_padding : float
        Fractional padding to add to y-axis limits (0.1 = 10% padding).
    show_ci : bool
        Whether to show confidence intervals.
    show_regression : bool
        Whether to show regression lines.
    alpha : float
        Point transparency.
    point_size : float
        Size of scatter points.
    cmap : str
        Colormap for coloring.
    save_path : str, optional
        Path to save the figure.
    **kwargs
        Additional arguments passed to individual plots.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.
    all_stats : dict
        Dictionary mapping cell type/complexity to statistics.
    """
    n_samples, n_cell_types = y_true.shape
    complexity_per_sample = (y_true > zero_threshold).sum(axis=1)

    # Initialize filter
    prop_filter = ProportionFilter(zero_threshold=zero_threshold)

    # Determine items to plot
    if by == "celltype":
        if celltypes_to_show is None:
            items = list(range(n_cell_types))
        else:
            name_to_idx = {v: k for k, v in labels_dict.items()}
            items = []
            for ct in celltypes_to_show:
                if isinstance(ct, str):
                    items.append(name_to_idx[ct])
                else:
                    items.append(ct)

        item_labels = [labels_dict.get(i, f"CT{i}") for i in items]

    elif by == "complexity":
        unique_complexities = np.unique(complexity_per_sample)
        if complexities_to_show is None:
            items = list(unique_complexities)
        else:
            items = [c for c in complexities_to_show if c in unique_complexities]

        item_labels = [f"{c} cell types" for c in items]
    else:
        raise ValueError("by must be 'celltype' or 'complexity'")

    n_items = len(items)
    nrows = int(np.ceil(n_items / ncols))

    # --- FIRST PASS: Compute statistics for all items to find global y-limits ---
    all_stats = {}
    all_diffs = []  # Collect all differences for y-limit calculation
    all_means = []  # Collect all means for x-limit calculation

    for item, label in zip(items, item_labels):
        try:
            # Get data for this item
            if by == "celltype":
                true_vals = y_true[:, item].flatten()
                pred_vals = y_pred[:, item].flatten()
            else:  # complexity
                sample_mask = complexity_per_sample == item
                true_vals = y_true[sample_mask].flatten()
                pred_vals = y_pred[sample_mask].flatten()

            # Apply filtering
            if filter_mode != "all":
                _, _, mask = prop_filter.filter_data(
                    true_vals, pred_vals, mode=filter_mode
                )
                true_vals = true_vals[mask]
                pred_vals = pred_vals[mask]

            if len(true_vals) < 2:
                all_stats[label] = None
                continue

            # Compute Bland-Altman values
            means = (true_vals + pred_vals) / 2
            diffs = pred_vals - true_vals

            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)

            loa_upper = mean_diff + 1.96 * std_diff
            loa_lower = mean_diff - 1.96 * std_diff

            # Regression
            if len(means) > 2:
                slope, intercept, r_value, p_value, _ = stats.linregress(means, diffs)
            else:
                slope, intercept, r_value, p_value = 0, mean_diff, 0, 1

            stats_dict = {
                "means": means,
                "diffs": diffs,
                "true_vals": true_vals,
                "pred_vals": pred_vals,
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "loa_upper": loa_upper,
                "loa_lower": loa_lower,
                "n_points": len(diffs),
                "regression_slope": slope,
                "regression_intercept": intercept,
                "regression_p": p_value,
                "has_proportional_bias": p_value < 0.05,
                "percent_within_loa": np.mean(
                    (diffs >= loa_lower) & (diffs <= loa_upper)
                )
                * 100,
            }

            all_stats[label] = stats_dict
            all_diffs.extend(diffs)
            all_means.extend(means)

        except Exception as e:
            all_stats[label] = {"error": str(e)}

    # --- Compute global axis limits ---
    if shared_ylim and len(all_diffs) > 0:
        all_diffs = np.array(all_diffs)
        all_means = np.array(all_means)

        # Y-limits based on all differences
        y_min = all_diffs.min()
        y_max = all_diffs.max()

        # Also consider LoA lines
        for label, stat in all_stats.items():
            if stat is not None and "error" not in stat:
                y_min = min(y_min, stat["loa_lower"])
                y_max = max(y_max, stat["loa_upper"])

        # Add padding
        y_range = y_max - y_min
        global_ylim = (y_min - ylim_padding * y_range, y_max + ylim_padding * y_range)

        # X-limits based on all means
        x_min = max(0, all_means.min() - 0.02)
        x_max = min(1, all_means.max() + 0.02)
        global_xlim = (x_min, x_max)
    else:
        global_ylim = None
        global_xlim = None

    # --- SECOND PASS: Create plots ---
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
        squeeze=False,
    )

    for idx, (item, label) in enumerate(zip(items, item_labels)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        stat = all_stats.get(label)

        if stat is None or "error" in stat:
            error_msg = (
                stat.get("error", "Insufficient data") if stat else "Insufficient data"
            )
            ax.text(
                0.5,
                0.5,
                f"Error:\n{error_msg[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_title(label, fontsize=10)
            if global_ylim:
                ax.set_ylim(global_ylim)
            if global_xlim:
                ax.set_xlim(global_xlim)
            continue

        means = stat["means"]
        diffs = stat["diffs"]

        # Scatter plot
        ax.scatter(means, diffs, alpha=alpha, s=point_size, c="steelblue")

        # Reference lines
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(stat["mean_difference"], color="blue", linestyle="-", linewidth=1.5)
        ax.axhline(stat["loa_upper"], color="red", linestyle="--", linewidth=1.2)
        ax.axhline(stat["loa_lower"], color="red", linestyle="--", linewidth=1.2)

        # Regression line
        if show_regression and len(means) > 2:
            x_line = np.array([means.min(), means.max()])
            y_line = stat["regression_slope"] * x_line + stat["regression_intercept"]
            linestyle = "-" if stat["has_proportional_bias"] else ":"
            ax.plot(x_line, y_line, color="green", linestyle=linestyle, linewidth=1.2)

        # Confidence intervals
        if show_ci:
            n = stat["n_points"]
            se_mean = stat["std_difference"] / np.sqrt(n)
            se_loa = np.sqrt(3 * stat["std_difference"] ** 2 / n)
            t_crit = stats.t.ppf(0.975, n - 1)

            x_range = np.array([means.min(), means.max()])

            ci_mean = (
                stat["mean_difference"] - t_crit * se_mean,
                stat["mean_difference"] + t_crit * se_mean,
            )
            ci_upper = (
                stat["loa_upper"] - t_crit * se_loa,
                stat["loa_upper"] + t_crit * se_loa,
            )
            ci_lower = (
                stat["loa_lower"] - t_crit * se_loa,
                stat["loa_lower"] + t_crit * se_loa,
            )

            ax.fill_between(x_range, ci_mean[0], ci_mean[1], color="blue", alpha=0.1)
            ax.fill_between(x_range, ci_upper[0], ci_upper[1], color="red", alpha=0.1)
            ax.fill_between(x_range, ci_lower[0], ci_lower[1], color="red", alpha=0.1)

        # Set axis limits
        if global_ylim:
            ax.set_ylim(global_ylim)
        if global_xlim:
            ax.set_xlim(global_xlim)

        # Title with stats
        bias_marker = "⚠" if stat["has_proportional_bias"] else ""
        ax.set_title(
            f"{label}\n"
            f'Bias: {stat["mean_difference"]:.3f}, '
            f'LoA: [{stat["loa_lower"]:.3f}, {stat["loa_upper"]:.3f}] '
            f"{bias_marker}\n"
            f'n={stat["n_points"]:,}',
            fontsize=9,
        )

        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_items, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Common labels
    fig.supxlabel("Mean of Predicted and True Proportion", fontsize=12)
    fig.supylabel("Difference (Predicted - True)", fontsize=12)

    filter_desc = {
        "all": "All points",
        "nonzero": "True > 0 only",
        "either_nonzero": "True > 0 OR Pred > 0",
    }

    fig.suptitle(
        f"Bland-Altman Analysis by {by.capitalize()}\n"
        f"Filter: {filter_desc[filter_mode]} | "
        f'{"Shared" if shared_ylim else "Independent"} Y-axis',
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Clean up stats for return (remove raw data to save memory)
    return_stats = {}
    for label, stat in all_stats.items():
        if stat is not None and "error" not in stat:
            return_stats[label] = {
                k: v
                for k, v in stat.items()
                if k not in ["means", "diffs", "true_vals", "pred_vals"]
            }
        else:
            return_stats[label] = stat

    return fig, axes, return_stats


def bland_altman_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_dict: dict,
    compare_by: str = "complexity",  # 'complexity' or 'celltype_group'
    groups: Optional[dict] = None,
    zero_threshold: float = 1e-6,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, dict]:
    """
    Compare Bland-Altman statistics across different groups.

    Parameters
    ----------
    y_true : np.ndarray
        True proportions, shape (n_samples, n_cell_types).
    y_pred : np.ndarray
        Predicted proportions, shape (n_samples, n_cell_types).
    labels_dict : dict
        Mapping from position index to cell type name.
    compare_by : str
        'complexity' to compare across mixture complexities, or
        'celltype_group' to compare across cell type groups (requires groups dict).
    groups : dict, optional
        For celltype_group comparison: {'Group Name': [list of cell type indices]}.
    zero_threshold : float
        Threshold for determining non-zero proportions.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    comparison_stats : dict
        Dictionary with comparison statistics.
    """
    n_samples, n_cell_types = y_true.shape
    complexity_per_sample = (y_true > zero_threshold).sum(axis=1)

    comparison_stats = {}

    if compare_by == "complexity":
        unique_complexities = np.sort(np.unique(complexity_per_sample))

        for cx in unique_complexities:
            _, _, stats = bland_altman_plot(
                y_true,
                y_pred,
                labels_dict,
                target_complexity=cx,
                zero_threshold=zero_threshold,
                ax=plt.figure().add_subplot(111),  # Dummy figure
            )
            plt.close()
            comparison_stats[f"{cx} cell types"] = stats

        group_names = [f"{cx}" for cx in unique_complexities]
        x_label = "Mixture Complexity (# Cell Types)"

    elif compare_by == "celltype_group":
        if groups is None:
            raise ValueError("groups dict required for celltype_group comparison")

        for group_name, celltype_indices in groups.items():
            _, _, stats = bland_altman_plot(
                y_true,
                y_pred,
                labels_dict,
                target_celltype=celltype_indices,
                zero_threshold=zero_threshold,
                ax=plt.figure().add_subplot(111),
            )
            plt.close()
            comparison_stats[group_name] = stats

        group_names = list(groups.keys())
        x_label = "Cell Type Group"
    else:
        raise ValueError("compare_by must be 'complexity' or 'celltype_group'")

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    names = list(comparison_stats.keys())
    x = np.arange(len(names))

    # Plot 1: Mean Difference (Bias)
    ax = axes[0]
    biases = [comparison_stats[n]["mean_difference"] for n in names]
    ci_lower = [comparison_stats[n]["ci_mean"][0] for n in names]
    ci_upper = [comparison_stats[n]["ci_mean"][1] for n in names]

    ax.bar(x, biases, color="steelblue", alpha=0.7)
    ax.errorbar(
        x,
        biases,
        yerr=[
            np.array(biases) - np.array(ci_lower),
            np.array(ci_upper) - np.array(biases),
        ],
        fmt="none",
        color="black",
        capsize=5,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        (
            group_names
            if compare_by == "celltype_group"
            else [str(c) for c in unique_complexities]
        ),
        rotation=45,
        ha="right",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean Difference (Bias)")
    ax.set_title("Systematic Bias")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Limits of Agreement Width
    ax = axes[1]
    loa_widths = [
        comparison_stats[n]["loa_upper"] - comparison_stats[n]["loa_lower"]
        for n in names
    ]

    ax.bar(x, loa_widths, color="coral", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        (
            group_names
            if compare_by == "celltype_group"
            else [str(c) for c in unique_complexities]
        ),
        rotation=45,
        ha="right",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("LoA Width (Upper - Lower)")
    ax.set_title("Agreement Range")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Percent within LoA
    ax = axes[2]
    pct_within = [comparison_stats[n]["percent_within_loa"] for n in names]

    colors = [
        "green" if p >= 95 else "orange" if p >= 90 else "red" for p in pct_within
    ]
    ax.bar(x, pct_within, color=colors, alpha=0.7)
    ax.axhline(95, color="green", linestyle="--", linewidth=1, label="95% expected")
    ax.set_xticks(x)
    ax.set_xticklabels(
        (
            group_names
            if compare_by == "celltype_group"
            else [str(c) for c in unique_complexities]
        ),
        rotation=45,
        ha="right",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("% Within LoA")
    ax.set_title("Points Within Limits")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f'Bland-Altman Comparison by {compare_by.replace("_", " ").title()}',
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, comparison_stats


def print_bland_altman_stats(stats: dict, name: str = ""):
    """Print formatted Bland-Altman statistics."""
    print("=" * 60)
    print(f"BLAND-ALTMAN STATISTICS{': ' + name if name else ''}")
    print("=" * 60)

    print(f"\n📊 Sample Size:")
    print(
        f"   Points: {stats['n_points']}, Samples: {stats['n_samples']}, Cell Types: {stats['n_celltypes']}"
    )

    print(f"\n📈 Agreement Metrics:")
    print(f"   Mean Difference (Bias): {stats['mean_difference']:.4f}")
    print(f"   SD of Differences:      {stats['std_difference']:.4f}")
    print(
        f"   95% CI of Mean:         [{stats['ci_mean'][0]:.4f}, {stats['ci_mean'][1]:.4f}]"
    )

    print(f"\n📏 Limits of Agreement (95%):")
    print(
        f"   Upper LoA: {stats['loa_upper']:.4f}  (95% CI: [{stats['ci_loa_upper'][0]:.4f}, {stats['ci_loa_upper'][1]:.4f}])"
    )
    print(
        f"   Lower LoA: {stats['loa_lower']:.4f}  (95% CI: [{stats['ci_loa_lower'][0]:.4f}, {stats['ci_loa_lower'][1]:.4f}])"
    )
    print(f"   LoA Width: {stats['loa_upper'] - stats['loa_lower']:.4f}")

    print(f"\n📉 Proportional Bias Check:")
    print(f"   Regression Slope:     {stats['regression_slope']:.4f}")
    print(f"   Regression Intercept: {stats['regression_intercept']:.4f}")
    print(f"   p-value:              {stats['regression_p']:.4f}")
    bias_status = "⚠️  YES" if stats["has_proportional_bias"] else "✅ NO"
    print(f"   Proportional Bias:    {bias_status}")

    print(f"\n✅ Coverage:")
    print(f"   % Within LoA: {stats['percent_within_loa']:.1f}% (expected ~95%)")

    print("\n" + "=" * 60)


class DeconvolverVisualizer:
    """
    Comprehensive visualization tools for interpretability analysis.
    """

    def __init__(
        self,
        model: DiagonalAwareDeconvolver,
        cell_type_names: Optional[List[str]] = None,
        dmr_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.cell_type_names = cell_type_names or [
            f"CT_{i}" for i in range(model.n_cell_types)
        ]
        self.dmr_names = dmr_names or [f"DMR_{i}" for i in range(model.n_dmr)]

    def plot_ablation_comparison(
        self,
        ablation_results: Dict[str, torch.Tensor],
        y_true: Optional[torch.Tensor] = None,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Visualize predictions under different ablation conditions.

        Shows how zeroing out different feature pathways affects predictions,
        revealing the contribution of each pathway.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

        conditions = list(ablation_results.keys())

        for idx, condition in enumerate(conditions):
            ax = fig.add_subplot(gs[idx // 4, idx % 4])

            pred = ablation_results[condition][sample_idx].detach().cpu().numpy()

            colors = plt.cm.viridis(pred / pred.max() if pred.max() > 0 else pred)
            bars = ax.bar(range(len(pred)), pred, color=colors)

            if y_true is not None:
                true = y_true[sample_idx].detach().cpu().numpy()
                ax.scatter(
                    range(len(true)),
                    true,
                    color="red",
                    marker="x",
                    s=50,
                    zorder=5,
                    label="True",
                )
                ax.legend(loc="upper right", fontsize=8)

            ax.set_title(condition.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Cell Type")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1)

            # Show top 3 predicted cell types
            top3 = np.argsort(pred)[-3:][::-1]
            ax.set_xticks(top3)
            ax.set_xticklabels(
                [self.cell_type_names[i] for i in top3], rotation=45, fontsize=8
            )

        fig.suptitle("Ablation Study: Feature Pathway Contributions", fontsize=14)
        return fig

    def plot_feature_space(
        self,
        outputs: List[DeconvolverOutput],
        labels: Optional[torch.Tensor] = None,
        feature_type: str = "combined",
        method: str = "pca",
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """
        Visualize learned feature representations using dimensionality reduction.

        Useful for understanding:
        - How well features separate different cell type compositions
        - Clustering structure in the learned representations

        Args:
            outputs: List of DeconvolverOutput from multiple samples
            labels: Optional labels for coloring (e.g., dominant cell type)
            feature_type: 'diag', 'confusion', 'reject', or 'combined'
            method: 'pca', 'tsne', or 'umap'
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # Stack features
        feature_map = {
            "diag": "diag_features",
            "confusion": "confusion_features",
            "reject": "reject_features",
            "combined": "combined_features",
        }

        features = (
            torch.cat([getattr(o, feature_map[feature_type]) for o in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(features)
            var_explained = reducer.explained_variance_ratio_
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, len(features) - 1))
            embedded = reducer.fit_transform(features)
            var_explained = None
        else:
            raise ValueError(f"Unknown method: {method}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Colored by labels if provided
        ax1 = axes[0]
        if labels is not None:
            labels_np = (
                labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
            )
            scatter = ax1.scatter(
                embedded[:, 0],
                embedded[:, 1],
                c=labels_np,
                cmap="tab20",
                alpha=0.7,
                s=50,
            )
            plt.colorbar(scatter, ax=ax1, label="Label")
        else:
            ax1.scatter(embedded[:, 0], embedded[:, 1], alpha=0.7, s=50)

        ax1.set_xlabel(f"{method.upper()} 1")
        ax1.set_ylabel(f"{method.upper()} 2")
        ax1.set_title(f"{feature_type.title()} Features - {method.upper()}")

        if var_explained is not None:
            ax1.set_xlabel(f"{method.upper()} 1 ({var_explained[0]:.1%} var)")
            ax1.set_ylabel(f"{method.upper()} 2 ({var_explained[1]:.1%} var)")

        # Plot 2: Feature statistics
        ax2 = axes[1]
        feature_norms = np.linalg.norm(features, axis=1)
        ax2.hist(feature_norms, bins=30, edgecolor="black", alpha=0.7)
        ax2.axvline(
            feature_norms.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {feature_norms.mean():.2f}",
        )
        ax2.set_xlabel("Feature Norm")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{feature_type.title()} Feature Magnitude Distribution")
        ax2.legend()

        plt.tight_layout()
        return fig

    def plot_diagonal_analysis(
        self,
        x: torch.Tensor,
        output: DeconvolverOutput,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Detailed analysis of diagonal pathway contribution.

        Visualizes:
        - Raw diagonal values from input
        - Correlation between diagonal and predictions
        - Per-cell-type diagonal importance
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Extract data
        diag_input = (
            torch.diagonal(x[sample_idx, :, : self.model.n_dmr], dim1=0, dim2=1)
            .detach()
            .cpu()
            .numpy()
        )
        pred = output.proportions[sample_idx].detach().cpu().numpy()
        diag_feat = output.diag_features[sample_idx].detach().cpu().numpy()

        # Plot 1: Diagonal input values
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(
            range(len(diag_input)),
            diag_input,
            color=plt.cm.Blues(diag_input / diag_input.max()),
        )
        ax1.set_xlabel("DMR Group / Cell Type Index")
        ax1.set_ylabel("Diagonal Value (P(CT_i | DMR_i))")
        ax1.set_title("Raw Diagonal Input Values")

        # Highlight top values
        top5 = np.argsort(diag_input)[-5:]
        for i in top5:
            ax1.annotate(
                self.cell_type_names[i], (i, diag_input[i]), fontsize=8, rotation=45
            )

        # Plot 2: Diagonal vs Prediction scatter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(diag_input, pred, alpha=0.7, s=60)

        # Add correlation line
        z = np.polyfit(diag_input, pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(diag_input.min(), diag_input.max(), 100)
        ax2.plot(
            x_line,
            p(x_line),
            "r--",
            alpha=0.8,
            label=f"r={np.corrcoef(diag_input, pred)[0,1]:.3f}",
        )

        ax2.set_xlabel("Diagonal Input Value")
        ax2.set_ylabel("Predicted Proportion")
        ax2.set_title("Diagonal-Prediction Correlation")
        ax2.legend()

        # Plot 3: Encoded diagonal features heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        diag_feat_2d = (
            diag_feat.reshape(-1, 16)
            if len(diag_feat) >= 16
            else diag_feat.reshape(1, -1)
        )
        sns.heatmap(
            diag_feat_2d,
            cmap="RdBu_r",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Activation"},
        )
        ax3.set_title("Encoded Diagonal Features")
        ax3.set_xlabel("Feature Dimension")
        ax3.set_ylabel("Feature Block")

        # Plot 4: Per-cell-type contribution analysis
        ax4 = fig.add_subplot(gs[1, 1])
        contribution = diag_input * pred  # Simple interaction
        colors = ["green" if c > 0.01 else "gray" for c in contribution]
        ax4.barh(range(len(contribution)), contribution, color=colors)
        ax4.set_xlabel("Diagonal × Prediction")
        ax4.set_ylabel("Cell Type Index")
        ax4.set_title("Diagonal Contribution Score")
        ax4.set_yticks(range(0, len(contribution), 5))

        fig.suptitle(f"Diagonal Pathway Analysis (Sample {sample_idx})", fontsize=14)
        return fig

    def plot_confusion_patterns(
        self,
        x: torch.Tensor,
        output: DeconvolverOutput,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """
        Visualize off-diagonal confusion patterns.

        Shows:
        - Full prediction matrix heatmap
        - Off-diagonal structure that might indicate cell type similarities
        - Confusion feature activation patterns
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Get the cell type portion of input matrix
        matrix = x[sample_idx, :, : self.model.n_dmr].detach().cpu().numpy()

        # Plot 1: Full matrix heatmap
        ax1 = axes[0]
        sns.heatmap(
            matrix,
            cmap="YlOrRd",
            ax=ax1,
            xticklabels=5,
            yticklabels=5,
            cbar_kws={"label": "Prediction Probability"},
        )
        ax1.set_xlabel("Predicted Cell Type")
        ax1.set_ylabel("DMR Group")
        ax1.set_title("Full Prediction Matrix")

        # Highlight diagonal
        for i in range(min(matrix.shape)):
            ax1.add_patch(
                plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="blue", linewidth=2)
            )

        # Plot 2: Off-diagonal only (zeroed diagonal)
        ax2 = axes[1]
        off_diag = matrix.copy()
        np.fill_diagonal(off_diag, 0)
        sns.heatmap(
            off_diag,
            cmap="YlOrRd",
            ax=ax2,
            xticklabels=5,
            yticklabels=5,
            cbar_kws={"label": "Off-Diagonal Value"},
        )
        ax2.set_xlabel("Predicted Cell Type")
        ax2.set_ylabel("DMR Group")
        ax2.set_title("Off-Diagonal Confusion Patterns")

        # Plot 3: Confusion features
        ax3 = axes[2]
        conf_feat = output.confusion_features[sample_idx].detach().cpu().numpy()
        conf_feat_2d = (
            conf_feat.reshape(-1, 16)
            if len(conf_feat) >= 16
            else conf_feat.reshape(1, -1)
        )
        sns.heatmap(
            conf_feat_2d,
            cmap="RdBu_r",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Activation"},
        )
        ax3.set_title("Encoded Confusion Features")
        ax3.set_xlabel("Feature Dimension")
        ax3.set_ylabel("Feature Block")

        plt.suptitle(f"Confusion Pattern Analysis (Sample {sample_idx})", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_rejection_analysis(
        self,
        x: torch.Tensor,
        output: DeconvolverOutput,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Analyze rejection column patterns.

        Shows:
        - Rejection probability per DMR group
        - Relationship between rejection and prediction confidence
        - Rejection feature activations
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        reject_col = x[sample_idx, :, -1].detach().cpu().numpy()
        pred = output.proportions[sample_idx].detach().cpu().numpy()
        reject_feat = output.reject_features[sample_idx].detach().cpu().numpy()

        # Plot 1: Rejection values per DMR
        ax1 = axes[0]
        colors = plt.cm.Reds(reject_col / max(reject_col.max(), 0.01))
        ax1.bar(range(len(reject_col)), reject_col, color=colors)
        ax1.set_xlabel("DMR Group")
        ax1.set_ylabel("Rejection Probability")
        ax1.set_title("Per-DMR Rejection Values")
        ax1.axhline(
            reject_col.mean(),
            color="blue",
            linestyle="--",
            label=f"Mean: {reject_col.mean():.3f}",
        )
        ax1.legend()

        # Plot 2: Rejection vs Prediction Entropy
        ax2 = axes[1]
        pred_entropy = -np.sum(pred * np.log(pred + 1e-10))
        total_rejection = reject_col.sum()

        # Create a summary visualization
        metrics = {
            "Total Rejection": total_rejection,
            "Mean Rejection": reject_col.mean(),
            "Max Rejection": reject_col.max(),
            "Pred Entropy": pred_entropy,
            "Max Pred": pred.max(),
        }

        bars = ax2.bar(metrics.keys(), metrics.values(), color="steelblue")
        ax2.set_ylabel("Value")
        ax2.set_title("Rejection & Confidence Metrics")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Annotate bars
        for bar, val in zip(bars, metrics.values()):
            ax2.annotate(
                f"{val:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 3: Rejection features
        ax3 = axes[2]
        ax3.bar(
            range(len(reject_feat)),
            reject_feat,
            color=plt.cm.RdBu_r(
                (reject_feat - reject_feat.min())
                / (reject_feat.max() - reject_feat.min() + 1e-10)
            ),
        )
        ax3.set_xlabel("Feature Dimension")
        ax3.set_ylabel("Activation")
        ax3.set_title("Encoded Rejection Features")
        ax3.axhline(0, color="black", linewidth=0.5)

        plt.suptitle(f"Rejection Pattern Analysis (Sample {sample_idx})", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_pathway_contributions(
        self, x: torch.Tensor, sample_idx: int = 0, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Compare predictions with each pathway in isolation and combined.

        This is a key interpretability visualization showing how each
        pathway independently predicts and how they combine.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        with torch.no_grad():
            # Get predictions with different ablations
            full_pred = (
                self.model(x[sample_idx : sample_idx + 1]).squeeze().cpu().numpy()
            )

            # Only diagonal
            diag_only = (
                self.model(
                    x[sample_idx : sample_idx + 1],
                    zero_confusion=True,
                    zero_reject=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Only confusion
            conf_only = (
                self.model(
                    x[sample_idx : sample_idx + 1], zero_diagonal=True, zero_reject=True
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Only rejection
            rej_only = (
                self.model(
                    x[sample_idx : sample_idx + 1],
                    zero_diagonal=True,
                    zero_confusion=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        predictions = {
            "Full Model": full_pred,
            "Diagonal Only": diag_only,
            "Confusion Only": conf_only,
            "Rejection Only": rej_only,
        }

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

        for ax, (name, pred), color in zip(axes.flat, predictions.items(), colors):
            bars = ax.bar(range(len(pred)), pred, color=color, alpha=0.7)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("Cell Type")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1)

            # Annotate top 3
            top3 = np.argsort(pred)[-3:][::-1]
            for i in top3:
                if pred[i] > 0.01:
                    ax.annotate(
                        f"{self.cell_type_names[i]}\n{pred[i]:.2f}",
                        (i, pred[i]),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        fig.suptitle(
            f"Pathway Contribution Analysis (Sample {sample_idx})", fontsize=14
        )
        plt.tight_layout()
        return fig

    def plot_feature_correlation_matrix(
        self, outputs: List[DeconvolverOutput], figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize correlations between different feature types.

        Useful for understanding redundancy and independence of pathways.
        """
        # Stack all features
        diag = (
            torch.cat([o.diag_features for o in outputs], dim=0).detach().cpu().numpy()
        )
        conf = (
            torch.cat([o.confusion_features for o in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        rej = (
            torch.cat([o.reject_features for o in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        # Compute mean activations per sample
        diag_mean = diag.mean(axis=1)
        conf_mean = conf.mean(axis=1)
        rej_mean = rej.mean(axis=1)

        # Stack and compute correlation
        all_means = np.stack([diag_mean, conf_mean, rej_mean], axis=1)

        # Also add feature norms
        diag_norm = np.linalg.norm(diag, axis=1)
        conf_norm = np.linalg.norm(conf, axis=1)
        rej_norm = np.linalg.norm(rej, axis=1)

        all_features = np.stack(
            [diag_mean, conf_mean, rej_mean, diag_norm, conf_norm, rej_norm], axis=1
        )

        feature_names = [
            "Diag Mean",
            "Conf Mean",
            "Rej Mean",
            "Diag Norm",
            "Conf Norm",
            "Rej Norm",
        ]

        fig, ax = plt.subplots(figsize=figsize)
        corr_matrix = np.corrcoef(all_features.T)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            xticklabels=feature_names,
            yticklabels=feature_names,
            vmin=-1,
            vmax=1,
        )

        ax.set_title("Feature Pathway Correlation Matrix")
        plt.tight_layout()
        return fig

    def create_interpretability_report(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        sample_indices: List[int] = None,
        save_dir: str = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive interpretability report.

        Creates all visualizations for selected samples.

        Args:
            x: Input batch
            y_true: Optional ground truth
            sample_indices: Which samples to analyze (default: first 3)
            save_dir: Optional directory to save figures

        Returns:
            Dictionary of figure names to figures
        """
        if sample_indices is None:
            sample_indices = list(range(min(3, len(x))))

        figures = {}

        # Get outputs for all samples
        with torch.no_grad():
            outputs = [
                self.model(x[i : i + 1], return_features=True) for i in sample_indices
            ]

        # Ablation study
        ablation_results = self.model.ablation_study(x)
        figures["ablation"] = self.plot_ablation_comparison(
            ablation_results, y_true, sample_indices[0]
        )

        # Per-sample analyses
        for idx, sample_idx in enumerate(sample_indices):
            output = outputs[idx]

            figures[f"diagonal_s{sample_idx}"] = self.plot_diagonal_analysis(
                x, output, sample_idx
            )
            figures[f"confusion_s{sample_idx}"] = self.plot_confusion_patterns(
                x, output, sample_idx
            )
            figures[f"rejection_s{sample_idx}"] = self.plot_rejection_analysis(
                x, output, sample_idx
            )
            figures[f"pathways_s{sample_idx}"] = self.plot_pathway_contributions(
                x, sample_idx
            )

        # Cross-sample analyses
        all_outputs = [
            self.model(x[i : i + 1], return_features=True) for i in range(len(x))
        ]
        figures["feature_correlation"] = self.plot_feature_correlation_matrix(
            all_outputs
        )

        # Feature space visualization
        if len(x) > 10:
            labels = x[:, :, : self.model.n_dmr].diagonal(dim1=1, dim2=2).argmax(dim=1)
            figures["feature_space"] = self.plot_feature_space(
                all_outputs, labels, feature_type="combined", method="pca"
            )

        # Save if directory provided
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(
                    os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight"
                )

        return figures


def cohen_d(group1, group2):
    """
    Computes Cohen's d to measure effect size between two distributions.
    Implemented based on ConceptsOfBiometrics.pdf, slide 77.

    Parameters:
    - group1, group2: Arrays of numerical values.

    Returns:
    - Cohen's d value (effect size).
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)  # Unbiased std
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return abs((mean1 - mean2)) / pooled_std


def plot_filled_kde(
    values, groups, alpha=0.4, ax=None, bw_adjust=0.4, name="Default", figsize=(10, 6)
):
    """
    Plots overlapping KDE (Kernel Density Estimation) curves with filled areas for two groups (0 and 1).
    Also computes Cohen's d to measure effect size.

    Parameters:
    - values: A list or numpy array of numerical values.
    - groups: A list or numpy array of group labels (0 or 1).
    - alpha: Transparency level for the filled area under KDE curves (default is 0.4).
    - ax: Matplotlib axis to plot on (optional).
    """
    # Convert to numpy arrays
    values = np.array(values)
    groups = np.array(groups)

    # Ensure groups contain only 0s and 1s
    unique_groups = np.unique(groups)
    if len(unique_groups) > 2 or set(unique_groups) - {0, 1}:
        raise ValueError("Groups should contain only 0 and 1.")

    # Separate values by group
    values_0 = values[groups == 0]  # Impostor scores
    values_1 = values[groups == 1]  # Genuine scores

    # Compute Uniquness and permanence
    uniqueness = np.abs(np.mean(values_0) - np.mean(values_1))

    n1, n2 = len(values_0), len(values_1)
    std1, std2 = np.std(values_0, ddof=1), np.std(values_1, ddof=1)  # Unbiased std
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    d_value = cohen_d(values_0, values_1)
    # permanence = pooled_std

    # If no axis is provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # KDE plots with filled area
    sns.kdeplot(
        values_0,
        color="blue",
        fill=True,
        alpha=alpha,
        label="Normal",
        linewidth=2,
        ax=ax,
        bw_adjust=bw_adjust,
    )
    sns.kdeplot(
        values_1,
        color="red",
        fill=True,
        alpha=alpha,
        label="Tumour",
        linewidth=2,
        ax=ax,
        bw_adjust=bw_adjust,
    )

    # Labels and legend
    ax.set_xlabel("Methylation rate", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    # ax.set_title(f"{name} Cohen's d: {d_value:.2f}")
    print(
        f"{name} Cohen's d: {d_value:.2f}, Means Difference: {uniqueness:.3f}, \n Pooled StDev: {pooled_std:.3f})"
    )
    ax.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # If ax is None, show the plot
    if ax is None:
        plt.show()


def create_methylation_boxplots(df, min_methyl_labels_count=0):
    """
    Create four box plots of methylation levels based on prediction outcomes.

    Parameters:
    df: DataFrame with 'label', 'prediction', and 'methylation_level' columns
    """
    df = df[df["total_methylated_labels"] > min_methyl_labels_count]
    # Define the four groups
    correctly_predicted = df[(df["label"] == df["prediction"]) & (df["label"] != 39)]
    incorrectly_rejected = df[(df["label"] != 39) & (df["prediction"] == 39)]
    incorrectly_predicted = df[(df["label"] == 39) & (df["prediction"] != 39)]
    correctly_rejected = df[(df["label"] == 39) & (df["prediction"] == 39)]

    # Print sample sizes
    print("Sample sizes:")
    print(
        f"1. Correctly predicted (label == prediction, label != 39): {len(correctly_predicted)}"
    )
    print(
        f"2. Incorrectly rejected (label != 39, prediction == 39): {len(incorrectly_rejected)}"
    )
    print(
        f"3. Incorrectly predicted (label == 39, prediction != 39): {len(incorrectly_predicted)}"
    )
    print(
        f"4. Correctly rejected (label == prediction == 39): {len(correctly_rejected)}"
    )
    print()

    # Prepare data for plotting
    plot_data = []
    groups = []

    for data, name in [
        (correctly_predicted, "Correctly\nPredicted"),
        (incorrectly_rejected, "Incorrectly\nRejected"),
        (incorrectly_predicted, "Incorrectly\nPredicted"),
        (correctly_rejected, "Correctly\nRejected"),
    ]:
        if len(data) > 0:
            plot_data.append(data["methylation_level"].values)
            groups.append(name)
        else:
            plot_data.append([])
            groups.append(name)

    # Create the box plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create box plot
    bp = ax.boxplot(
        plot_data, labels=groups, patch_artist=True, showmeans=True, meanline=True
    )

    # Customize colors
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize plot
    ax.set_ylabel("Methylation Level", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediction Groups", fontsize=12, fontweight="bold")
    ax.set_title(
        "Methylation Level Distribution by Prediction Outcome\n(label 39 = rejection class)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add sample size annotations
    for i, (data, group) in enumerate(zip(plot_data, groups), 1):
        ax.text(
            i,
            ax.get_ylim()[0],
            f"n={len(data)}",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
        )

    plt.tight_layout()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    for data, name in zip(plot_data, groups):
        if len(data) > 0:
            print(f"\n{name.replace(chr(10), ' ')}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Median: {np.median(data):.4f}")
            print(f"  Std: {np.std(data):.4f}")
            print(f"  Min: {np.min(data):.4f}")
            print(f"  Max: {np.max(data):.4f}")
        else:
            print(f"\n{name.replace(chr(10), ' ')}: No data")
    ax.tick_params(axis="x", pad=20)
    plt.show()

    return fig, ax


def create_methylation_violinplots(
    df,
    title="Methylation Level Distribution by Prediction Outcome\n(label 39 = rejection class)",
):
    """
    Create four violin plots of methylation levels based on prediction outcomes.

    Parameters:
    df: DataFrame with 'label', 'prediction', and 'methylation_level' columns
    """

    # Define the four groups
    correctly_predicted = df[(df["label"] == df["prediction"]) & (df["label"] != 39)]
    incorrectly_rejected = df[(df["label"] != 39) & (df["prediction"] == 39)]
    incorrectly_predicted = df[
        (df["label"] == 39)
        & (df["prediction"] != 39)
        & (df["prediction"] != df["original_label"])
    ]
    correctly_rejected = df[(df["label"] == 39) & (df["prediction"] == 39)]

    # Print sample sizes
    print("Sample sizes:")
    print(
        f"1. Correctly predicted (label == prediction, label != 39): {len(correctly_predicted)}"
    )
    print(
        f"2. Incorrectly rejected (label != 39, prediction == 39): {len(incorrectly_rejected)}"
    )
    print(
        f"3. Incorrectly predicted (label == 39, prediction != 39): {len(incorrectly_predicted)}"
    )
    print(
        f"4. Correctly rejected (label == prediction == 39): {len(correctly_rejected)}"
    )
    print()

    # Prepare data for plotting
    plot_data = []

    for data, group_num, name in [
        (correctly_predicted, 1, "Correctly\nPredicted"),
        (incorrectly_rejected, 2, "Incorrectly\nRejected"),
        (incorrectly_predicted, 3, "Incorrectly\nPredicted"),
        (correctly_rejected, 4, "Correctly\nRejected"),
    ]:
        if len(data) > 0:
            temp_df = pd.DataFrame(
                {
                    "methylation_level": data["methylation_level"].values,
                    "group": name,
                    "group_num": group_num,
                }
            )
            plot_data.append(temp_df)

    # Combine all data
    plot_df = pd.concat(plot_data, ignore_index=True)

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]

    # Create violin plot
    parts = ax.violinplot(
        [
            plot_df[plot_df["group_num"] == i]["methylation_level"].values
            for i in range(1, 5)
        ],
        positions=range(1, 5),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    # Color the violins
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Customize other elements
    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.5)

    # Set x-tick labels with group names
    group_labels = [
        "Correctly\nPredicted",
        "Incorrectly\nRejected",
        "Incorrectly\nPredicted",
        "Correctly\nRejected",
    ]
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(group_labels, fontsize=11, fontweight="bold")

    # Add sample sizes below group labels
    sample_sizes = [
        len(correctly_predicted),
        len(incorrectly_rejected),
        len(incorrectly_predicted),
        len(correctly_rejected),
    ]

    # Increase padding between tick labels and axis
    ax.tick_params(axis="x", pad=15, labelsize=11)
    ax.tick_params(axis="y", pad=8, labelsize=10)

    # Add sample size annotations with extra spacing
    y_min = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    annotation_y = y_min - (0.08 * y_range)  # Position below x-axis

    for i, n in enumerate(sample_sizes, 1):
        ax.text(
            i,
            annotation_y,
            f"(n={n})",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            fontweight="bold",
        )

    # Customize plot
    ax.set_ylabel("Methylation Level", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_xlabel("Prediction Groups", fontsize=13, fontweight="bold", labelpad=25)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)

    # Adjust y-axis to make room for sample size labels
    y_lim = ax.get_ylim()
    ax.set_ylim([y_lim[0] - (0.12 * y_range), y_lim[1]])

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=colors[0],
            alpha=0.7,
            edgecolor="black",
            label="Correctly Predicted",
        ),
        Patch(
            facecolor=colors[1],
            alpha=0.7,
            edgecolor="black",
            label="Incorrectly Rejected",
        ),
        Patch(
            facecolor=colors[2],
            alpha=0.7,
            edgecolor="black",
            label="Incorrectly Predicted",
        ),
        Patch(
            facecolor=colors[3],
            alpha=0.7,
            edgecolor="black",
            label="Correctly Rejected",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    for data, name in [
        (correctly_predicted, "Correctly Predicted"),
        (incorrectly_rejected, "Incorrectly Rejected"),
        (incorrectly_predicted, "Incorrectly Predicted"),
        (correctly_rejected, "Correctly Rejected"),
    ]:
        if len(data) > 0:
            print(f"\n{name}:")
            print(f"  Count: {len(data)}")
            print(f"  Mean: {data['methylation_level'].mean():.4f}")
            print(f"  Median: {data['methylation_level'].median():.4f}")
            print(f"  Std: {data['methylation_level'].std():.4f}")
            print(f"  Min: {data['methylation_level'].min():.4f}")
            print(f"  Max: {data['methylation_level'].max():.4f}")
            print(f"  Q1 (25%): {data['methylation_level'].quantile(0.25):.4f}")
            print(f"  Q3 (75%): {data['methylation_level'].quantile(0.75):.4f}")
        else:
            print(f"\n{name}: No data")

    plt.show()

    return fig, ax


def plot_methylation_with_predictions(df, figsize=(14, 6), colors=None):
    """
    Create a box plot of methylation levels with secondary y-axis showing
    average prediction scores for the matching tissue type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns:
        - methylation_level: methylation values
        - dmr_ctype_matched: tissue type labels (x-axis)
        - dmr_ctype_label: numeric label corresponding to prediction column
        - prediction_0, prediction_1, ..., prediction_N: prediction scores
    figsize : tuple
        Figure size (default: (14, 6))
    colors : dict, optional
        Custom colors for tissue types

    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axes
    """

    # Prepare data
    # Group by tissue type and get the corresponding label
    tissue_types = df["dmr_ctype_matched"].unique()
    tissue_order = sorted(tissue_types)

    # For each tissue type, get the corresponding label and average prediction
    tissue_predictions = {}
    tissue_labels = {}

    for tissue in tissue_order:
        tissue_data = df[df["dmr_ctype_matched"] == tissue]

        # Get the label (should be consistent within tissue type)
        label = tissue_data["dmr_ctype_label"].mode()[0]  # Most common label
        tissue_labels[tissue] = int(label)

        # Get the corresponding prediction column
        pred_col = f"prediction_{int(label)}"
        if pred_col in tissue_data.columns:
            avg_pred = tissue_data[pred_col].mean()
            tissue_predictions[tissue] = avg_pred
        else:
            tissue_predictions[tissue] = np.nan

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # === PRIMARY AXIS: Box plot of methylation levels ===
    bp_data = [
        df[df["dmr_ctype_matched"] == tissue]["methylation_level"].values
        for tissue in tissue_order
    ]

    # Create box plot
    bp = ax1.boxplot(
        bp_data,
        positions=range(len(tissue_order)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(
            facecolor="lightblue", alpha=0.7, edgecolor="black", linewidth=1.5
        ),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
    )

    ax1.set_ylabel("Methylation Level", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Tissue Type (DMR)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, axis="y")

    # === SECONDARY AXIS: Average predictions ===
    x_positions = range(len(tissue_order))
    predictions = [tissue_predictions[tissue] for tissue in tissue_order]

    # Plot as line with markers
    line = ax2.plot(
        x_positions,
        predictions,
        color="darkgreen",
        marker="D",
        markersize=10,
        linewidth=2.5,
        label="Avg Prediction Score",
        zorder=10,
    )

    # Add value labels on the markers
    for x, y, tissue in zip(x_positions, predictions, tissue_order):
        if not np.isnan(y):
            ax2.text(
                x,
                y + 0.03,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax2.set_ylabel(
        "Average Prediction Score\n(for matching tissue type)",
        fontsize=12,
        fontweight="bold",
        color="darkgreen",
    )
    ax2.tick_params(axis="y", labelcolor="darkgreen")
    ax2.set_ylim(-0.05, 1.05)

    # Set x-axis labels
    ax1.set_xticks(range(len(tissue_order)))
    ax1.set_xticklabels(
        [f"{tissue}\n(label={tissue_labels[tissue]})" for tissue in tissue_order],
        rotation=45,
        ha="right",
    )

    # Add legend
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Add custom legend entries
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="lightblue",
            edgecolor="black",
            label="Methylation Level (boxplot)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="darkgreen",
            marker="D",
            markersize=10,
            linewidth=2.5,
            label="Avg Prediction Score",
        ),
    ]

    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10)

    # Title
    plt.title(
        "Methylation Levels vs Model Predictions by Tissue Type\n"
        + "Boxplot: Distribution of methylation | Line: Average prediction for matching label",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_methylation_with_predictions_grouped(df, figsize=(16, 6)):
    """
    Alternative version with grouped bar plot for predictions alongside boxplot.
    """

    # Prepare data
    tissue_types = df["dmr_ctype_matched"].unique()
    tissue_order = sorted(tissue_types)

    # For each tissue type, get statistics
    stats_data = []
    for tissue in tissue_order:
        tissue_data = df[df["dmr_ctype_matched"] == tissue]
        label = int(tissue_data["dmr_ctype_label"].mode()[0])
        pred_col = f"prediction_{label}"

        stats_data.append(
            {
                "tissue": tissue,
                "label": label,
                "n_samples": len(tissue_data),
                "mean_methylation": tissue_data["methylation_level"].mean(),
                "median_methylation": tissue_data["methylation_level"].median(),
                "mean_prediction": (
                    tissue_data[pred_col].mean()
                    if pred_col in tissue_data.columns
                    else np.nan
                ),
                "std_prediction": (
                    tissue_data[pred_col].std()
                    if pred_col in tissue_data.columns
                    else np.nan
                ),
            }
        )

    stats_df = pd.DataFrame(stats_data)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # === LEFT PLOT: Box plot ===
    bp_data = [
        df[df["dmr_ctype_matched"] == tissue]["methylation_level"].values
        for tissue in tissue_order
    ]

    bp = ax1.boxplot(
        bp_data,
        positions=range(len(tissue_order)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(
            facecolor="lightblue", alpha=0.7, edgecolor="black", linewidth=1.5
        ),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )

    ax1.set_ylabel("Methylation Level", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Tissue Type", fontsize=12, fontweight="bold")
    ax1.set_title("Distribution of Methylation Levels", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xticks(range(len(tissue_order)))
    ax1.set_xticklabels(tissue_order, rotation=45, ha="right")

    # === RIGHT PLOT: Bar plot of predictions ===
    x_pos = np.arange(len(tissue_order))
    bars = ax2.bar(
        x_pos,
        stats_df["mean_prediction"],
        yerr=stats_df["std_prediction"],
        width=0.6,
        color="darkgreen",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
    )

    # Add value labels on bars
    for i, (tissue, val) in enumerate(zip(tissue_order, stats_df["mean_prediction"])):
        if not np.isnan(val):
            ax2.text(
                i,
                val + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax2.set_ylabel("Average Prediction Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tissue Type", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Model Predictions for Matching Labels", fontsize=13, fontweight="bold"
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [
            f'{tissue}\n(label={stats_df.iloc[i]["label"]})'
            for i, tissue in enumerate(tissue_order)
        ],
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    return fig, (ax1, ax2), stats_df


def plot_methylation_vs_prediction_scatter(df, figsize=(12, 8)):
    """
    Create scatter plot of methylation level vs prediction score for each tissue.
    Each tissue type gets its own subplot.
    """

    tissue_types = df["dmr_ctype_matched"].unique()
    n_tissues = len(tissue_types)

    # Calculate grid dimensions
    n_cols = min(3, n_tissues)
    n_rows = (n_tissues + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_tissues > 1 else [axes]

    for idx, tissue in enumerate(sorted(tissue_types)):
        ax = axes[idx]
        tissue_data = df[df["dmr_ctype_matched"] == tissue]

        label = int(tissue_data["dmr_ctype_label"].mode()[0])
        pred_col = f"prediction_{label}"

        if pred_col in tissue_data.columns:
            x = tissue_data["methylation_level"]
            y = tissue_data[pred_col]

            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=30, edgecolor="black", linewidth=0.5)

            # Add correlation
            corr = x.corr(y)

            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_line,
                p(x_line),
                "r--",
                linewidth=2,
                alpha=0.8,
                label=f"Corr: {corr:.3f}",
            )

            ax.set_xlabel("Methylation Level", fontsize=10)
            ax.set_ylabel(f"Prediction Score\n(label {label})", fontsize=10)
            ax.set_title(
                f"{tissue}\n(n={len(tissue_data)})", fontsize=11, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(n_tissues, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Methylation Level vs Model Prediction by Tissue Type",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    return fig, axes


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

import ast
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_soft_label_distribution(df, target_dmr_ctype, target_label, labels_dict, num_classes=40):
    """
    Plots the distribution of soft label probabilities across all classes,
    using a dictionary to map class indices to biological names.
    """
    # 1. Filter the dataset
    subset = df[(df['dmr_ctype'] == target_dmr_ctype) & (df['original_label'] == target_label)].copy()
    
    if len(subset) == 0:
        print(f"No reads found for dmr_ctype='{target_dmr_ctype}' and label={target_label}")
        return
        
    print(f"Found {len(subset):,} reads for this subset. Generating plot...")

    # 2. Extract and stack the soft labels into a 2D numpy array
    try:
        soft_labels_matrix = np.vstack(subset['soft_label'].values)
    except ValueError:
        # Safely evaluate string representations of lists if necessary
        subset['soft_label'] = subset['soft_label'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        soft_labels_matrix = np.vstack(subset['soft_label'].values)

    # 3. Set up the plot
    plt.figure(figsize=(18, 7)) # Made slightly wider to accommodate text labels
    
    ax = sns.boxplot(
        data=soft_labels_matrix, 
        color="lightgray", 
        fliersize=1,     
        linewidth=1.2,
        showfliers=True  
    )

    # 4. Highlight the target label box
    if target_label < len(ax.patches):
        box = ax.patches[target_label]
        box.set_facecolor('dodgerblue')
        box.set_edgecolor('darkblue')
        box.set_linewidth(2)

    # 5. Formatting with the dictionary mappings
    target_name = labels_dict.get(target_label, f"Class {target_label}")
    
    plt.title(
        f"Soft Label Distributions for {target_dmr_ctype} Regions\n"
        f"(Ground Truth: {target_name} [{target_label}] | Reads: {len(subset):,})", 
        fontsize=14, fontweight='bold'
    )
    plt.xlabel("Cell Type", fontsize=12)
    plt.ylabel("Probability Score", fontsize=12)
    
    # Generate the string labels for the x-axis using the dictionary
    # Fall back to the integer string if an index is missing from the dict
    x_labels = [labels_dict.get(i, str(i)) for i in range(num_classes)]
    
    # Apply the string labels and force a 90-degree rotation so they don't overlap
    plt.xticks(ticks=range(num_classes), labels=x_labels, rotation=90)
    plt.ylim(-0.05, 1.05)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Update legend to show the mapped name
    legend_elements = [
        Patch(facecolor='dodgerblue', edgecolor='darkblue', label=f'Target: {target_name}'),
        Patch(facecolor='lightgray', edgecolor='#333333', label='Other Cell Types')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()
