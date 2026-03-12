import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
