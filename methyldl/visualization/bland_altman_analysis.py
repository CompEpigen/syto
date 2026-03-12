from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats


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
