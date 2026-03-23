import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import List, Literal, Optional, Callable, Union, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from scipy import stats
import warnings
import pysam
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import numba as nb
import re
from matplotlib.gridspec import GridSpec
import os
from sklearn.metrics import confusion_matrix
from methyldl.deconvolution.uxm import (
    uxm_deconvolution,
    rearange_uxm_deconvolution_results,
)

cell_type_match_dict = {
    "Adipocytes": "Adipocytes",
    "Bladder-Epithelium": "Bladder-Ep",
    "Bladder-Epithelial": "Bladder-Ep",
    "Blood-B": "Blood-B",
    "Blood-B-Mem": "Blood-B",
    "Blood-Granulocytes": "Blood-Granul",
    "Blood-Monocytes": "Blood-Mono+Macro",
    "Colon-Macrophages": "Blood-Mono+Macro",
    "Lung-Alveolar-Macrophages": "Blood-Mono+Macro",
    "Lung-Interstitial-Macrophages": "Blood-Mono+Macro",
    "Liver-Macrophages": "Blood-Mono+Macro",
    "Blood-NK": "Blood-NK",
    "Blood-T-CD3": "Blood-T",
    "Blood-T-CD4": "Blood-T",
    "Blood-T-CD8": "Blood-T",
    "Blood-T-CenMem": "Blood-T",
    "Blood-T-Eff-CD8": "Blood-T",
    "Blood-T-EffMem-CD4": "Blood-T",
    "Blood-T-EffMem-CD8": "Blood-T",
    "Blood-T-Naive-CD4": "Blood-T",
    "Blood-T-CenMem-CD4": "Blood-T",
    "Blood-T-Naive-CD8": "Blood-T",
    "Bone-Osteoblasts": "Bone-Osteob",
    "Breast-Basal-Epithelium": "Breast-Basal-Ep",
    "Breast-Basal-Epithelial": "Breast-Basal-Ep",
    "Breast-Luminal-Epithelium": "Breast-Luminal-Ep",
    "Breast-Luminal-Epithelial": "Breast-Luminal-Ep",
    "Colon-Right-Epithelium": "Colon-Ep",
    "Colon-Right-Epithelial": "Colon-Ep",
    "Colon-Left-Epithelium": "Colon-Ep",
    "Colon-Left-Epithelial": "Colon-Ep",
    "Colon-Fibroblasts": "Colon-Fibro",
    "Dermal-Fibroblasts": "Dermal-Fibro",
    "Aorta-Endothel": "Endothel",
    "Saphenous-Vein-Endothel": "Endothel",
    "Kidney-Glomerular-Endothelium": "Endothel",
    "Kidney-Tubular-Endothelium": "Endothel",
    "Kidney-Tubular-Endothel": "Endothel",
    "Kidney-Glomerular-Endothel": "Endothel",
    "Liver-Endothelium": "Endothel",
    "Lung-Alveolar-Endothelium": "Endothel",
    "Lung-Alveolar-Endothel": "Endothel",
    "Pancreas-Endothelium": "Endothel",
    "Pancreas-Endothel": "Endothel",
    "Pancreas-Islet-Endothel": "Endothel",
    "Pancreas-Islet-Endothelium": "Endothel",
    "Epidermal-Keratinocytes": "Epid-Kerat",
    "Bone_marrow-Erythrocyte_progenitors": "Eryth-prog",
    "Fallopian-Epithelium": "Fallopian-Ep",
    "Fallopian-Epithelial": "Fallopian-Ep",
    "Gallbladder-Epithelium": "Gallbladder",
    "Gallbladder-Epithelial": "Gallbladder",
    "Gastric-antrum-Epithelium": "Gastric-Ep",
    "Gastric-antrum-Epithelial": "Gastric-Ep",
    "Gastric-fundus-Epithelium": "Gastric-Ep",
    "Gastric-fundus-Epithelial": "Gastric-Ep",
    "Gastric-body-Epithelium": "Gastric-Ep",
    "Gastric-body-Epithelial": "Gastric-Ep",
    "Tonsil-Palatine-Epithelium": "Head-Neck-Ep",
    "Tonsil-Palatine-Epithelial": "Head-Neck-Ep",
    "Tongue-Epithelium": "Head-Neck-Ep",
    "Tongue-Epithelial": "Head-Neck-Ep",
    "Tonsil-Pharyngeal-Epithelium": "Head-Neck-Ep",
    "Tonsil-Pharyngeal-Epithelial": "Head-Neck-Ep",
    "Tongue_base-Epithelium": "Head-Neck-Ep",
    "Tongue_base-Epithelial": "Head-Neck-Ep",
    "Larynx-Epithelium": "Head-Neck-Ep",
    "Larynx-Epithelial": "Head-Neck-Ep",
    "Esophagus-Epithelium": "Head-Neck-Ep",
    "Esophagus-Epithelial": "Head-Neck-Ep",
    "Pharynx-Epithelium": "Head-Neck-Ep",
    "Pharynx-Epithelial": "Head-Neck-Ep",
    "Heart-Cardiomyocyte": "Heart-Cardio",
    "Heart-Fibroblasts": "Heart-Fibro",
    "Kidney-Glomerular-Epithelium": "Kidney-Ep",
    "Kidney-Glomerular-Epithelial": "Kidney-Ep",
    "Kidney-Tubular-Epithelium": "Kidney-Ep",
    "Kidney-Tubular-Epithelial": "Kidney-Ep",
    "Kidney-Glomerular-Podocytes": "Kidney-Ep",
    "Liver-Hepatocytes": "Liver-Hep",
    "Lung-Alveolar-Epithelium": "Lung-Ep-Alveo",
    "Lung-Alveolar-Epithelial": "Lung-Ep-Alveo",
    "Lung-Bronchus-Epithelium": "Lung-Ep-Bron",
    "Lung-Bronchus-Epithelial": "Lung-Ep-Bron",
    "Cortex-Neuron": "Neuron",
    "Neuron": "Neuron",
    "Cerebellum-Neuron": "Neuron",
    "Oligodendrocytes": "Oligodend",
    "Ovary-Epithelium": "Ovary-Ep",
    "Ovary-Epithelial": "Ovary-Ep",
    "Pancreas-Acinar": "Pancreas-Acinar",
    "Pancreas-Alpha": "Pancreas-Alpha",
    "Pancreas-Beta": "Pancreas-Beta",
    "Pancreas-Delta": "Pancreas-Delta",
    "Pancreas-Duct": "Pancreas-Duct",
    "Prostate-Epithelium": "Prostate-Ep",
    "Prostate-Epithelial": "Prostate-Ep",
    "Skeletal-Muscle": "Skeletal-Musc",
    "Small-int-Epithelium": "Small-Int-Ep",
    "Small-int-Epithelial": "Small-Int-Ep",
    "Aorta-Smooth-Muscle": "Smooth-Musc",
    "Coronary-Artery-Smooth-Muscle": "Smooth-Musc",
    "Bladder-Smooth-Muscle": "Smooth-Musc",
    "Prostate-Smooth-Muscle": "Smooth-Musc",
    "Lung-Bronchus-Smooth-Muscle": "Smooth-Musc",
    "Thyroid-Epithelium": "Thyroid-Ep",
    "Thyroid-Epithelial": "Thyroid-Ep",
}


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


def aggregate_predictions_by_dmr(
    df: pd.DataFrame,
    group_cols: List[str] = ["dmr_label", "file", "original_label"],
    prediction_cols: Optional[List[str]] = None,
    weight_col: str = "total_marked_cpgs",
    create_weight_from_cpgs: bool = True,
    fill_in_missing_labels: bool = False,
    labels_dict: dict = None,
) -> pd.DataFrame:
    """
    Aggregate predictions for each class across all reads at the DMR level.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with read-level predictions.
    group_cols : List[str]
        Columns to group by for aggregation. Default: ['dmr_label', 'file', 'original_label']
    prediction_cols : Optional[List[str]]
        List of prediction column names. If None, auto-detects columns starting with 'prediction_'.
    weight_col : str
        Column name to use for weighted aggregation. Default: 'total_marked_cpgs'
    create_weight_from_cpgs : bool
        If True and weight_col doesn't exist, creates it from methylated_CpGs + unmethylated_CpGs.
    fill_in_missing_labels: bool
        If True, fills in additional rows from labels_dict labels not presented in the data. All predictions in
        these rows will be zero
    labels_dict: dict
        Must be provided if fill_in_missing_labels is set to True

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with both simple average and weighted average predictions.
    """

    df = df.copy()
    if fill_in_missing_labels:
        if labels_dict is None:
            raise ValueError(
                "labels_dict must be provided when fill_in_missing_labels is set to True"
            )

    # Auto-detect prediction columns if not provided
    if prediction_cols is None:
        prediction_cols = [
            col
            for col in df.columns
            if col.startswith("prediction_") and col != "prediction"
        ]
        prediction_cols = sorted(prediction_cols, key=lambda x: int(x.split("_")[1]))
        prediction_cols.append("methylation_level")

    # Create weight column if needed
    if weight_col not in df.columns and create_weight_from_cpgs:
        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df[weight_col] = df["methylated_CpGs"] + df["unmethylated_CpGs"]
        else:
            raise ValueError(
                f"Weight column '{weight_col}' not found and cannot create from CpG columns."
            )

    # Handle case where weights might be 0 (avoid division by zero)
    df["_weight"] = df[weight_col].clip(lower=1e-10)

    # Simple average aggregation
    simple_avg = df.groupby(group_cols)[prediction_cols].mean()
    simple_avg.columns = [f"{col}_avg" for col in simple_avg.columns]

    # Weighted average aggregation
    def weighted_average(group):
        weights = group["_weight"].values
        total_weight = weights.sum()

        result = {}
        for col in prediction_cols:
            values = group[col].values
            result[f"{col}_wavg"] = np.average(values, weights=weights)

        result["total_weight"] = total_weight
        result["n_reads"] = len(group)
        return pd.Series(result)

    weighted_avg = df.groupby(group_cols).apply(weighted_average, include_groups=False)

    # Combine results
    result = simple_avg.join(weighted_avg)

    # Add additional metadata columns (take first value per group)
    metadata_cols = [
        col
        for col in df.columns
        if col not in prediction_cols + group_cols + ["_weight", weight_col]
        and col in ["ctype", "dmr_ctype", "label", "chromosome"]
    ]

    if metadata_cols:
        metadata = df.groupby(group_cols)[metadata_cols].first()
        result = result.join(metadata)
    result.reset_index(inplace=True)

    if fill_in_missing_labels:
        labels_dict_pd = pd.DataFrame(labels_dict, index=["dmr_ctype"]).T.reset_index()
        labels_dict_pd.columns = ["dmr_ctype_label", "dmr_ctype"]
        if set(labels_dict_pd["dmr_ctype_label"]).difference(
            set(result["dmr_ctype_label"])
        ):
            result = pd.merge(
                result, labels_dict_pd, on=["dmr_ctype_label", "dmr_ctype"], how="outer"
            )
            result[result.isna()] = 0
            result["label"] = -1
            result["total_weight"] = result["total_weight"].apply(lambda x: max(x, 1))

    return result.reset_index()


def get_final_prediction(
    aggregated_df: pd.DataFrame,
    methods: list = ["avg", "wavg"],
    prediction_prefix: str = "prediction_",
) -> pd.DataFrame:
    """
    Get final class prediction from aggregated probabilities.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Output from aggregate_predictions_by_dmr()
    method : str
        'avg' for simple average or 'wavg' for weighted average
    prediction_prefix : str
        Prefix for prediction columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added final_prediction and final_confidence columns.
    """

    df = aggregated_df.copy()
    for method in methods:
        suffix = f"_{method}"

        # Find relevant prediction columns
        pred_cols = [
            col
            for col in df.columns
            if col.startswith(prediction_prefix) and col.endswith(suffix)
        ]
        pred_cols = sorted(
            pred_cols,
            key=lambda x: int(x.replace(prediction_prefix, "").replace(suffix, "")),
        )

        if not pred_cols:
            raise ValueError(f"No prediction columns found with suffix '{suffix}'")

        # Get predictions as array
        pred_array = df[pred_cols].values

        # Final prediction is argmax, confidence is max probability
        df[f"final_prediction_{method}"] = pred_array.argmax(axis=1)
        df[f"final_confidence_{method}"] = pred_array.max(axis=1)

    return df


def random_select_with_weights(elements, n):
    """
    Randomly selects up to n elements from a list and generates
    corresponding weights that sum to 1.

    Args:
        elements: List of elements to select from
        n: Maximum number of elements to select

    Returns:
        tuple: (selected_elements, weights) where weights sum to 1
    """
    # Determine how many elements to select (up to n, but not more than available)
    k = random.randint(1, n)

    # Randomly select k elements without replacement
    selected = random.sample(elements, k)

    # Generate random weights and normalize to sum to 1
    weights = [random.random() for _ in range(k)]
    total = sum(weights)
    weights = [w / total for w in weights]

    return selected, weights


# TODO: make sure that reference cellss are matching labels --> probably need to be reordered
def generate_pseudo_bulk(
    total_samples,
    labels,
    proportions,
    train_data,
    valid_data,
    test_data,
    atlas,
    ref_cells,
    labels_dict_reversed,
    return_reads=False,
    n_labels=39,
):
    ref_pos = np.array(
        [
            labels_dict_reversed[cell] if cell in labels_dict_reversed.keys() else -1
            for cell in ref_cells
        ]
    )
    assert np.round(np.sum(proportions), 4) == 1, "Proportions must sum up to one"
    n_samples_list = [int(total_samples * x) for x in proportions]
    target_columns = ["dmr_ctype_label", "dmr_ctype"]
    target_columns.extend([f"prediction_{i}_wavg" for i in range(n_labels)])
    target_columns.extend(
        ["methylation_level_wavg", "total_weight", "n_reads", "chromosome", "label"]
    )
    subs = []
    uxm_data = []
    reads = []
    sample_name = "pseudo_balk_sample"
    for df in [train_data, valid_data, test_data]:
        df["total_marked_cpgs"] = df["NCPGS"]
        df["methylation_level"] = df["M_rate"]
        df.rename(columns={"chr": "chromosome"}, inplace=True)
        df["direction"] = "U"
        grouped = df.groupby(["original_label", "dmr_ctype_label"], sort=False)
        sub = []
        for n, label in zip(n_samples_list, labels):
            for dmr_ctype_label in range(39):
                group = grouped.get_group((label, dmr_ctype_label))
                sub.append(group.sample(int(n / 39), replace=True))
        sub = pd.concat(sub)
        sub_aggregated = aggregate_predictions_by_dmr(
            sub, group_cols=["dmr_ctype_label", "dmr_ctype"]
        )
        sub_aggregated = sub_aggregated[target_columns]
        subs.append(sub_aggregated)
        results = sub
        results_agg = (
            results.groupby(["name", "direction"])
            .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
            .reset_index()
        )
        results_agg["count"] = (
            results_agg["record_M"] + results_agg["record_U"] + results_agg["record_X"]
        )
        results_agg["sf"] = results_agg["record_U"] / results_agg["count"]
        sf = deepcopy(results_agg[["name", "direction"]])
        sf[sample_name] = results_agg["sf"]
        counts = results_agg[["name", "direction", "count"]]
        counts.columns = ["name", "direction", sample_name]
        uxm_proportions = uxm_deconvolution(
            atlas, ref_cells, sf, counts, sample_names=[sample_name]
        )[0]
        uxm_deconv_results = {
            x: np.round(y, 4) for (x, y) in zip(ref_cells, uxm_proportions)
        }
        uxm_deconv_results_alligned = rearange_uxm_deconvolution_results(
            labels_dict_reversed, uxm_proportions, ref_cells
        )
        uxm_data.append((sf, counts, uxm_deconv_results, uxm_deconv_results_alligned))
        if return_reads:
            reads.append(sub)

    proportions_dict = {x: y for x, y in zip(labels, proportions)}
    proportions_full = [proportions_dict.get(x, 0) for x in range(39)]
    if return_reads:
        return labels, proportions_full, subs, uxm_data, reads
    return labels, proportions_full, subs, uxm_data


# Global variables for worker processes (initialized once per worker)
_worker_data = {}


def init_worker(
    train_data,
    valid_data,
    test_data,
    allowed_labels,
    n_cells_max,
    n_read_per_split,
    n_labels,
):
    """Initialize worker process with shared data and pre-computed groups."""

    global _worker_data

    # Set unique random seed per process

    seed = mp.current_process().pid

    random.seed(seed)

    np.random.seed(seed)

    for df in [train_data, valid_data, test_data]:
        df["total_marked_cpgs"] = df["NCPGS"]
        df["methylation_level"] = df["M_rate"]
        df.rename(columns={"chr": "chromosome"}, inplace=True)
        df["direction"] = "U"
    # Pre-compute grouped dataframes (expensive operation done once per worker)

    _worker_data["grouped_train"] = train_data.groupby(
        ["original_label", "dmr_ctype_label"], sort=False
    )

    _worker_data["grouped_valid"] = valid_data.groupby(
        ["original_label", "dmr_ctype_label"], sort=False
    )

    _worker_data["grouped_test"] = test_data.groupby(
        ["original_label", "dmr_ctype_label"], sort=False
    )

    _worker_data["allowed_labels"] = allowed_labels

    _worker_data["n_cells_max"] = n_cells_max

    _worker_data["n_read_per_split"] = n_read_per_split

    _worker_data["target_columns"] = ["dmr_ctype_label", "dmr_ctype"]
    _worker_data["target_columns"].extend(
        [f"prediction_{i}_wavg" for i in range(n_labels)]
    )
    _worker_data["target_columns"].extend(["n_reads", "chromosome", "label"])
    _worker_data["n_labels"] = n_labels


def worker_task(batch_indices):
    """Process a batch of examples."""

    global _worker_data
    results = []
    exceptions = []

    for idx in batch_indices:
        # try:
        labels, proportions = random_select_with_weights(
            _worker_data["allowed_labels"], _worker_data["n_cells_max"]
        )
        _, proportions, subs, uxm_data = generate_pseudo_bulk_optimized(
            _worker_data["n_read_per_split"],
            labels,
            proportions,
            _worker_data["grouped_train"],
            _worker_data["grouped_valid"],
            _worker_data["grouped_test"],
            _worker_data["target_columns"],
            _worker_data["n_labels"],
        )

        results.append((proportions, subs, uxm_data))

    # except Exception as e:

    #     exceptions.append((labels, proportions, str(e)))

    return results, exceptions


def generate_pseudo_bulk_optimized(
    total_samples,
    labels,
    proportions,
    grouped_train,
    grouped_valid,
    grouped_test,
    target_columns,
    n_labels,
):
    """Optimized version using pre-computed groups with UXM deconvolution."""
    assert np.round(np.sum(proportions), 4) == 1, "Proportions must sum up to one"

    n_samples_list = [int(total_samples * x) for x in proportions]
    sample_name = "pseudo_balk_sample"

    subs = []
    uxm_data = []

    for grouped in [grouped_train, grouped_valid, grouped_test]:
        sub_parts = []
        samples_per_dmr = {
            label: int(n / n_labels) for label, n in zip(labels, n_samples_list)
        }

        for label in labels:
            n_per_dmr = samples_per_dmr[label]
            for dmr_ctype_label in range(n_labels):
                try:
                    group = grouped.get_group((label, dmr_ctype_label))
                    sub_parts.append(group.sample(n_per_dmr, replace=True))
                except KeyError:
                    continue

        if sub_parts:
            sub = pd.concat(sub_parts, ignore_index=True)

            # Compute UXM deconvolution inputs before aggregation
            results = sub
            results_agg = (
                results.groupby(["name", "direction"])
                .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
                .reset_index()
            )
            results_agg["count"] = (
                results_agg["record_M"]
                + results_agg["record_U"]
                + results_agg["record_X"]
            )
            results_agg["sf"] = results_agg["record_U"] / results_agg["count"]

            sf = deepcopy(results_agg[["name", "direction"]])
            sf[sample_name] = results_agg["sf"]

            counts = results_agg[["name", "direction", "count"]].copy()
            counts.columns = ["name", "direction", sample_name]

            # uxm_proportions = uxm_deconvolution(atlas, ref_cells, sf, counts, sample_names=[sample_name])[0]
            # uxm_deconv_results = {x: np.round(y, 4) for (x, y) in zip(ref_cells, uxm_proportions)}
            # uxm_deconv_results_alligned = rearange_uxm_deconvolution_results(
            #     labels_dict_reversed, uxm_proportions, ref_cells
            # )
            # uxm_data.append((sf, counts, uxm_deconv_results, uxm_deconv_results_alligned))
            uxm_data.append((sf, counts))
            # Aggregate predictions
            sub = aggregate_predictions_by_dmr_optimized(
                sub, group_cols=["dmr_ctype_label", "dmr_ctype"]
            )
            sub = sub[target_columns]
            subs.append(sub)

    proportions_dict = dict(zip(labels, proportions))
    proportions_full = [proportions_dict.get(x, 0) for x in range(n_labels)]

    return labels, proportions_full, subs, uxm_data


def aggregate_predictions_by_dmr_optimized(df, group_cols):
    """Optimized aggregation using vectorized operations."""

    prediction_cols = [
        col
        for col in df.columns
        if col.startswith("prediction_") and col != "prediction"
    ]
    prediction_cols = sorted(prediction_cols, key=lambda x: int(x.split("_")[1]))
    prediction_cols.append("methylation_level")

    weight_col = "total_marked_cpgs"
    df = df.copy()
    if weight_col not in df.columns:
        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df[weight_col] = df["methylated_CpGs"] + df["unmethylated_CpGs"]
        else:
            raise ValueError(f"Weight column '{weight_col}' not found")

    # Clip weights to avoid division by zero
    df["_weight"] = df[weight_col].clip(lower=1e-10)
    # Pre-compute weighted values for each prediction column
    for col in prediction_cols:
        df[f"_weighted_{col}"] = df[col] * df["_weight"]
    # Build aggregation dictionary

    agg_dict = {}
    # Sum of weighted values and weights
    for col in prediction_cols:
        agg_dict[f"_weighted_{col}_sum"] = (f"_weighted_{col}", "sum")
    agg_dict["_weight_sum"] = ("_weight", "sum")
    agg_dict["n_reads"] = ("_weight", "count")
    agg_dict["total_weight"] = (weight_col, "sum")

    # Metadata columns - take first value
    metadata_cols = [col for col in ["label", "chromosome"] if col in df.columns]
    for col in metadata_cols:
        agg_dict[col] = (col, "first")

    # Perform aggregation
    grouped = df.groupby(group_cols, sort=True)
    result = grouped.agg(**agg_dict).reset_index()

    # Compute weighted averages from sums
    for col in prediction_cols:
        result[f"{col}_wavg"] = result[f"_weighted_{col}_sum"] / result["_weight_sum"]
        result.drop(columns=[f"_weighted_{col}_sum"], inplace=True)

    result.drop(columns=["_weight_sum"], inplace=True)
    return result


def random_select_with_weights(elements, n):
    """Same as original."""

    k = random.randint(1, n)

    selected = random.sample(elements, k)

    weights = [random.random() for _ in range(k)]

    total = sum(weights)

    weights = [w / total for w in weights]

    return selected, weights


def run_ios_generation_parallel(
    train_data,
    valid_data,
    test_data,
    file_name,
    n_io_examples=30000,
    n_workers=None,
    batch_size=100,
    checkpoint_interval=1000,
    start_checkpoint_idx=0,
    n_labels=39,
):
    """Main function to run parallel processing."""

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    allowed_labels = list(range(n_labels))
    n_cells_max = 10
    n_read_per_split = int(4.75 * 1e5)

    # Create batches of indices
    all_indices = list(range(n_io_examples))
    batches = [
        all_indices[i : i + batch_size] for i in range(0, len(all_indices), batch_size)
    ]

    all_ios = []
    all_exceptions = []

    checkpoint_idx = start_checkpoint_idx
    # Use ProcessPoolExecutor with initializer

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(
            train_data,
            valid_data,
            test_data,
            allowed_labels,
            n_cells_max,
            n_read_per_split,
            n_labels,
        ),
    ) as executor:

        # Submit all batches
        futures = {
            executor.submit(worker_task, batch): i for i, batch in enumerate(batches)
        }

        # Process results as they complete
        with tqdm(total=n_io_examples, desc="Generating examples") as pbar:
            completed = 0
            for future in as_completed(futures):
                results, exceptions = future.result()
                all_ios.extend(results)
                all_exceptions.extend(exceptions)
                completed += len(results) + len(exceptions)
                pbar.update(len(results) + len(exceptions))

                # Checkpoint
                if len(all_ios) % checkpoint_interval < batch_size:
                    checkpoint_idx += checkpoint_interval
                    with open(
                        file_name.replace(".pkl", f"_{checkpoint_idx}.pkl"), "wb"
                    ) as f:
                        pickle.dump(all_ios, f)
                    all_ios = (
                        []
                    )  # Initialize from the beggining so the object is not growing in memory

    return all_ios, all_exceptions


class FlattenMLPDeconvolver(nn.Module):
    """
    Flatten 39×40 → 1560 features, then MLP.
    Simple but ignores spatial structure.
    """

    def __init__(
        self,
        n_dmr_groups=39,
        n_classes=40,
        n_cell_types=39,
        hidden_dims=[512, 256, 128],
    ):
        super().__init__()

        input_dim = n_dmr_groups * n_classes

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_cell_types))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 39, 40)
        x = x.flatten(start_dim=1)  # (batch, 1560)
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)


class DMRAttentionDeconvolver(nn.Module):
    """
    Process each DMR row independently, then use attention
    to weight their contributions to final prediction.
    """

    def __init__(
        self,
        n_dmr_groups=39,
        n_pred_classes=40,
        n_cell_types=39,
        embed_dim=64,
        num_heads=4,
    ):
        super().__init__()

        self.n_dmr_groups = n_dmr_groups
        self.n_cell_types = n_cell_types

        # Encode each DMR's prediction distribution
        self.row_encoder = nn.Sequential(
            nn.Linear(n_pred_classes, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable DMR embeddings (captures which DMR we're looking at)
        self.dmr_embeddings = nn.Parameter(torch.randn(n_dmr_groups, embed_dim) * 0.02)

        # Self-attention across DMRs
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Final prediction head
        self.proportion_head = nn.Sequential(
            nn.Linear(embed_dim * n_dmr_groups, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cell_types),
        )

    def forward(self, x):
        # x: (batch, 39, 40) — each row is a distribution over predictions
        batch_size = x.shape[0]

        # Encode each DMR row
        row_features = self.row_encoder(x)  # (batch, 39, embed_dim)

        # Add DMR positional embeddings
        row_features = row_features + self.dmr_embeddings.unsqueeze(0)

        # Self-attention: DMRs attend to each other
        attn_out, _ = self.attention(row_features, row_features, row_features)
        row_features = self.attn_norm(row_features + attn_out)

        # Flatten and predict
        pooled = row_features.flatten(start_dim=1)  # (batch, 39 * embed_dim)
        logits = self.proportion_head(pooled)

        return torch.softmax(logits, dim=-1)


@dataclass
class DeconvolverOutput:
    """Structured output from the deconvolver containing predictions and intermediate features."""

    proportions: torch.Tensor  # (batch, n_cell_types) - final predicted proportions
    diag_features: torch.Tensor  # (batch, hidden_dim) - diagonal pathway features
    confusion_features: torch.Tensor  # (batch, hidden_dim) - confusion pathway features
    reject_features: torch.Tensor  # (batch, hidden_dim//2) - rejection pathway features
    combined_features: (
        torch.Tensor
    )  # (batch, combined_dim) - pre-head combined features
    logits: torch.Tensor  # (batch, n_cell_types) - pre-softmax logits


class DiagonalAwareDeconvolver(nn.Module):
    """
    Explicitly model the diagonal (DMR_i → CellType_i) relationship
    plus off-diagonal confusion patterns.

    Enhanced with:
    - Configurable encoder pathways (diagonal, confusion, reject)
    - Feature extraction for all intermediate representations
    - Ablation controls to zero-out feature pathways at inference
    - Interpretability-focused forward passes

    Args:
        n_dmr_groups: Number of DMR groups (default: 39)
        n_pred_classes: Number of prediction classes including rejection (default: 40)
        n_cell_types: Number of output cell types (default: 39)
        hidden_dim: Hidden dimension for encoders (default: 128)
        use_diagonal: Whether to include diagonal encoder pathway (default: True)
        use_confusion: Whether to include confusion/off-diagonal encoder pathway (default: True)
        use_reject: Whether to include rejection column encoder pathway (default: True)

    Examples:
        # Full model (all pathways)
        model = DiagonalAwareDeconvolver()

        # Diagonal only (like XGBoost baseline but neural)
        model = DiagonalAwareDeconvolver(use_confusion=False, use_reject=False)

        # Diagonal + rejection (matches XGBoost feature set)
        model = DiagonalAwareDeconvolver(use_confusion=False)

        # Confusion patterns only
        model = DiagonalAwareDeconvolver(use_diagonal=False, use_reject=False)
    """

    def __init__(
        self,
        n_dmr_groups: int = 39,
        n_pred_classes: int = 40,
        n_cell_types: int = 39,
        hidden_dim: int = 128,
        use_diagonal: bool = True,
        use_confusion: bool = True,
        use_reject: bool = True,
    ):
        super().__init__()

        # Validate at least one pathway is enabled
        if not any([use_diagonal, use_confusion, use_reject]):
            raise ValueError("At least one encoder pathway must be enabled")

        # Store configuration
        self.n_dmr = n_dmr_groups
        self.n_pred_classes = n_pred_classes
        self.n_cell_types = n_cell_types
        self.hidden_dim = hidden_dim

        # Pathway flags
        self.use_diagonal = use_diagonal
        self.use_confusion = use_confusion
        self.use_reject = use_reject

        # Feature dimensions (0 if pathway disabled)
        self.diag_feature_dim = hidden_dim if use_diagonal else 0
        self.confusion_feature_dim = hidden_dim if use_confusion else 0
        self.reject_feature_dim = (hidden_dim // 2) if use_reject else 0
        self.combined_dim = (
            self.diag_feature_dim + self.confusion_feature_dim + self.reject_feature_dim
        )

        # Build only the enabled encoders
        if use_diagonal:
            self.diagonal_encoder = nn.Sequential(
                nn.Linear(n_dmr_groups, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.diagonal_encoder = None

        if use_confusion:
            self.confusion_encoder = nn.Sequential(
                nn.Linear(n_dmr_groups * n_pred_classes, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:
            self.confusion_encoder = None

        if use_reject:
            self.reject_encoder = nn.Sequential(
                nn.Linear(n_dmr_groups, hidden_dim // 2), nn.GELU()
            )
        else:
            self.reject_encoder = None

        # Combine all features and predict proportions
        self.proportion_head = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_cell_types),
        )

    @property
    def config(self) -> Dict:
        """Return model configuration."""
        return {
            "n_dmr_groups": self.n_dmr,
            "n_pred_classes": self.n_pred_classes,
            "n_cell_types": self.n_cell_types,
            "hidden_dim": self.hidden_dim,
            "use_diagonal": self.use_diagonal,
            "use_confusion": self.use_confusion,
            "use_reject": self.use_reject,
            "combined_dim": self.combined_dim,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self) -> str:
        pathways = []
        if self.use_diagonal:
            pathways.append("diagonal")
        if self.use_confusion:
            pathways.append("confusion")
        if self.use_reject:
            pathways.append("reject")

        return (
            f"{self.__class__.__name__}(\n"
            f"  pathways={pathways},\n"
            f"  combined_dim={self.combined_dim},\n"
            f"  n_parameters={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )

    def extract_input_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract raw input features before encoding.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)

        Returns:
            Dictionary containing available features based on enabled pathways
        """
        features = {"full_matrix": x}

        if self.use_diagonal:
            features["diagonal"] = torch.diagonal(x[:, :, : self.n_dmr], dim1=1, dim2=2)

        if self.use_reject:
            features["reject_col"] = x[:, :, -1]

        if self.use_confusion:
            cell_type_matrix = x[:, :, : self.n_dmr].clone()
            mask = torch.eye(self.n_dmr, device=x.device, dtype=torch.bool)
            cell_type_matrix[:, mask] = 0
            features["off_diagonal"] = cell_type_matrix

        return features

    def encode_features(
        self,
        x: torch.Tensor,
        zero_diagonal: bool = False,
        zero_confusion: bool = False,
        zero_reject: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode input through enabled pathways with optional ablation.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)
            zero_diagonal: If True, zero out diagonal features (only if pathway enabled)
            zero_confusion: If True, zero out confusion features (only if pathway enabled)
            zero_reject: If True, zero out rejection features (only if pathway enabled)

        Returns:
            Tuple of (diag_features, confusion_features, reject_features)
            None for disabled pathways
        """
        diag_features = None
        confusion_features = None
        reject_features = None

        # Diagonal pathway
        if self.use_diagonal:
            diagonal = torch.diagonal(x[:, :, : self.n_dmr], dim1=1, dim2=2)
            diag_features = self.diagonal_encoder(diagonal)
            if zero_diagonal:
                diag_features = torch.zeros_like(diag_features)

        # Confusion pathway
        if self.use_confusion:
            full_flat = x.flatten(start_dim=1)
            confusion_features = self.confusion_encoder(full_flat)
            if zero_confusion:
                confusion_features = torch.zeros_like(confusion_features)

        # Rejection pathway
        if self.use_reject:
            reject_col = x[:, :, -1]
            reject_features = self.reject_encoder(reject_col)
            if zero_reject:
                reject_features = torch.zeros_like(reject_features)

        return diag_features, confusion_features, reject_features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        zero_diagonal: bool = False,
        zero_confusion: bool = False,
        zero_reject: bool = False,
    ) -> torch.Tensor | DeconvolverOutput:
        """
        Forward pass with optional feature extraction and ablation.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)
            return_features: If True, return DeconvolverOutput with all features
            zero_diagonal: If True, zero out diagonal pathway (if enabled)
            zero_confusion: If True, zero out confusion pathway (if enabled)
            zero_reject: If True, zero out rejection pathway (if enabled)

        Returns:
            If return_features=False: Tensor of proportions (batch, n_cell_types)
            If return_features=True: DeconvolverOutput dataclass
        """
        # Encode with optional ablation
        diag_features, confusion_features, reject_features = self.encode_features(
            x, zero_diagonal, zero_confusion, zero_reject
        )

        # Combine only enabled features
        features_to_cat = []
        if diag_features is not None:
            features_to_cat.append(diag_features)
        if confusion_features is not None:
            features_to_cat.append(confusion_features)
        if reject_features is not None:
            features_to_cat.append(reject_features)

        combined = torch.cat(features_to_cat, dim=-1)

        # Get logits and proportions
        logits = self.proportion_head(combined)
        proportions = torch.softmax(logits, dim=-1)

        if return_features:
            return DeconvolverOutput(
                proportions=proportions,
                diag_features=diag_features,
                confusion_features=confusion_features,
                reject_features=reject_features,
                combined_features=combined,
                logits=logits,
            )
        return proportions

    def ablation_study(
        self, x: torch.Tensor, y_true: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run systematic ablation study on enabled feature pathways.

        Only tests ablations that are valid for the current configuration.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)
            y_true: Optional ground truth for computing metrics

        Returns:
            Dictionary with predictions under each valid ablation condition
        """
        results = {}

        with torch.no_grad():
            # Always include full model (no ablation)
            results["full_model"] = self.forward(x)

            # Single pathway ablations (only for enabled pathways)
            if self.use_diagonal:
                results["no_diagonal"] = self.forward(x, zero_diagonal=True)
            if self.use_confusion:
                results["no_confusion"] = self.forward(x, zero_confusion=True)
            if self.use_reject:
                results["no_reject"] = self.forward(x, zero_reject=True)

            # "Only X" conditions (requires at least 2 pathways enabled)
            n_enabled = sum([self.use_diagonal, self.use_confusion, self.use_reject])

            if n_enabled >= 2:
                if self.use_diagonal:
                    results["only_diagonal"] = self.forward(
                        x,
                        zero_confusion=self.use_confusion,
                        zero_reject=self.use_reject,
                    )
                if self.use_confusion:
                    results["only_confusion"] = self.forward(
                        x, zero_diagonal=self.use_diagonal, zero_reject=self.use_reject
                    )
                if self.use_reject:
                    results["only_reject"] = self.forward(
                        x,
                        zero_diagonal=self.use_diagonal,
                        zero_confusion=self.use_confusion,
                    )

        return results

    def get_feature_importance(
        self, x: torch.Tensor, target_cell_type: int, method: str = "gradient"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance for a target cell type.

        Args:
            x: Input tensor (batch, n_dmr_groups, n_pred_classes)
            target_cell_type: Index of cell type to analyze
            method: 'gradient' or 'integrated_gradient'

        Returns:
            Dictionary with importance scores for enabled input features
        """
        x = x.clone().requires_grad_(True)

        output = self.forward(x, return_features=True)
        target_prob = output.proportions[:, target_cell_type].sum()
        target_prob.backward()

        importance = {"input_gradient": x.grad.detach()}

        if self.use_diagonal:
            importance["diagonal_importance"] = (
                x.grad[:, :, : self.n_dmr].diagonal(dim1=1, dim2=2).detach()
            )

        if self.use_reject:
            importance["reject_importance"] = x.grad[:, :, -1].detach()

        return importance


class CNNDeconvolver(nn.Module):
    """
    Treat 39×40 matrix as a single-channel image.
    """

    def __init__(self, n_dmr_groups=39, n_pred_classes=40, n_cell_types=39):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # (1, 39, 40) → (32, 39, 40)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # (32, 39, 40) → (64, 19, 20)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # (64, 19, 20) → (128, 9, 10)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (128, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cell_types),
        )

    def forward(self, x):
        # x: (batch, 39, 40)
        x = x.unsqueeze(1)  # (batch, 1, 39, 40)
        x = self.conv_layers(x)
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)


@dataclass
class TrainingHistory:
    """Stores training metrics over epochs."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_mae: list = field(default_factory=list)
    val_mse: list = field(default_factory=list)
    val_kl: list = field(default_factory=list)
    val_max_error: list = field(default_factory=list)
    val_cosine_sim: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    def to_dict(self):
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_mae": self.val_mae,
            "val_mse": self.val_mse,
            "val_kl": self.val_kl,
            "val_max_error": self.val_max_error,
            "val_cosine_sim": self.val_cosine_sim,
            "learning_rates": self.learning_rates,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
        }


class EarlyStopping:
    """
    Early stopping handler.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        'min' for metrics where lower is better (loss, MAE),
        'max' for metrics where higher is better (cosine_sim).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        mode: Literal["min", "max"] = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def compute_deconvolution_metrics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> dict:
    """Compute all evaluation metrics."""
    with torch.no_grad():
        # MAE
        mae = (pred - target).abs().mean().item()

        # MSE
        mse = ((pred - target) ** 2).mean().item()

        # KL Divergence
        kl = (
            (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
            .sum(dim=-1)
            .mean()
            .item()
        )

        # Max error (worst case)
        max_error = (pred - target).abs().max().item()

        # Cosine similarity (average across batch)
        cosine_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()

    return {
        "mae": mae,
        "mse": mse,
        "kl": kl,
        "max_error": max_error,
        "cosine_sim": cosine_sim,
    }


def train_matrix_deconvolver(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    early_stopping_metric: Literal[
        "val_loss", "val_mae", "val_mse", "val_kl", "val_max_error", "val_cosine_sim"
    ] = "val_mae",
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-5,
    scheduler_type: Literal["cosine", "plateau", "none"] = "plateau",
    loss_weights: dict = None,
    verbose: int = 1,
    save_path: Optional[str] = None,
) -> tuple[nn.Module, TrainingHistory]:
    """
    Train a deconvolution model with early stopping support.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    X_train, y_train : np.ndarray
        Training data and labels.
    X_val, y_val : np.ndarray
        Validation data and labels.
    n_epochs : int
        Maximum number of epochs.
    batch_size : int
        Batch size for training.
    lr : float
        Initial learning rate.
    weight_decay : float
        L2 regularization weight.
    device : str
        Device to train on ('cuda' or 'cpu').
    early_stopping_metric : str
        Metric to monitor for early stopping.
        Options: 'val_loss', 'val_mae', 'val_mse', 'val_kl', 'val_max_error', 'val_cosine_sim'
    early_stopping_patience : int
        Number of epochs to wait for improvement.
    early_stopping_min_delta : float
        Minimum change to qualify as improvement.
    scheduler_type : str
        Learning rate scheduler type: 'cosine', 'plateau', or 'none'.
    loss_weights : dict
        Weights for loss components: {'mse': float, 'kl': float}.
        Default: {'mse': 1.0, 'kl': 0.5}
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed).
    save_path : str, optional
        Path to save the best model.

    Returns
    -------
    model : nn.Module
        The trained model (loaded with best weights).
    history : TrainingHistory
        Training history with all metrics.
    """

    # Default loss weights
    if loss_weights is None:
        loss_weights = {"mse": 1.0, "kl": 0.5}

    # Determine early stopping mode
    maximize_metrics = {"val_cosine_sim"}
    es_mode = "max" if early_stopping_metric in maximize_metrics else "min"

    # Setup data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup model, optimizer, scheduler
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=es_mode, patience=5, factor=0.5, min_lr=1e-6
        )
    else:
        scheduler = None

    # Loss function
    def loss_fn(pred, target, eps=1e-8):
        mse = nn.functional.mse_loss(pred, target)
        kl = (
            (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
            .sum(dim=-1)
            .mean()
        )
        return loss_weights["mse"] * mse + loss_weights["kl"] * kl

    # Initialize tracking
    history = TrainingHistory()
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode=es_mode,
    )
    best_model_state = None

    # Training loop
    for epoch in range(n_epochs):
        # === Training phase ===
        model.train()
        train_loss = 0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # === Validation phase ===
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                val_loss += loss_fn(pred, y).item()
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        val_loss /= len(val_loader)

        # Compute all metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_deconvolution_metrics(all_preds, all_targets)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_mae.append(metrics["mae"])
        history.val_mse.append(metrics["mse"])
        history.val_kl.append(metrics["kl"])
        history.val_max_error.append(metrics["max_error"])
        history.val_cosine_sim.append(metrics["cosine_sim"])
        history.learning_rates.append(current_lr)

        # Get the metric value for early stopping
        metric_map = {
            "val_loss": val_loss,
            "val_mae": metrics["mae"],
            "val_mse": metrics["mse"],
            "val_kl": metrics["kl"],
            "val_max_error": metrics["max_error"],
            "val_cosine_sim": metrics["cosine_sim"],
        }
        current_metric = metric_map[early_stopping_metric]

        # Check if this is the best model
        is_best = False
        if early_stopping.best_score is None:
            is_best = True
        elif (
            es_mode == "min"
            and current_metric < early_stopping.best_score - early_stopping_min_delta
        ):
            is_best = True
        elif (
            es_mode == "max"
            and current_metric > early_stopping.best_score + early_stopping_min_delta
        ):
            is_best = True

        if is_best:
            best_model_state = deepcopy(model.state_dict())
            history.best_epoch = epoch
            if save_path:
                torch.save(model.state_dict(), save_path)

        # Update scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(current_metric)
            else:
                scheduler.step()

        # Logging
        if verbose >= 1 and (epoch + 1) % max(1, n_epochs // 100) == 0:
            print(
                f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {metrics['mae']:.4f} | "
                f"Val Max Error: {metrics['max_error']:.4f} | "
                f"LR: {current_lr:.2e}" + (" *" if is_best else "")
            )

        if verbose >= 2:
            print(
                f"         Val MSE: {metrics['mse']:.4f} | "
                f"Val KL: {metrics['kl']:.4f} | "
                f"Val MaxErr: {metrics['max_error']:.4f}"
            )

        # Early stopping check
        if early_stopping(current_metric, epoch):
            history.stopped_early = True
            if verbose >= 1:
                print(
                    f"\nEarly stopping triggered at epoch {epoch+1}. "
                    f"Best epoch: {early_stopping.best_epoch+1} "
                    f"({early_stopping_metric}={early_stopping.best_score:.4f})"
                )
            break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose >= 1:
        print(f"\nTraining complete. Best epoch: {history.best_epoch+1}")
        print(f"Best {early_stopping_metric}: {early_stopping.best_score:.4f}")

    return model, history


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


def print_deconvolution_metrics_summary(metrics: dict):
    """Print a formatted summary of the metrics."""
    print("=" * 60)
    print("DECONVOLUTION RESULTS SUMMARY")
    print("=" * 60)

    if "celltype_metrics" in metrics:
        ct_metrics = metrics["celltype_metrics"]
        print(f"\n📊 Overall R²: {ct_metrics['overall_r2']:.4f}")

        print("\n📋 Per Cell Type R²:")
        print("-" * 40)

        # Sort by R² descending
        sorted_ct = sorted(
            ct_metrics["per_celltype_r2"].items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -1,
            reverse=True,
        )

        for ct_name, r2 in sorted_ct:
            if not np.isnan(r2):
                bar = "█" * int(r2 * 20) + "░" * (20 - int(r2 * 20))
                print(f"  {ct_name:20s} │ {bar} │ {r2:.4f}")
            else:
                print(f"  {ct_name:20s} │ {'N/A':^20s} │ N/A")

    if "complexity_metrics" in metrics:
        cx_metrics = metrics["complexity_metrics"]

        print("\n📈 Performance by Mixture Complexity:")
        print("-" * 50)
        print(f"  {'# Cell Types':^15s} │ {'N Samples':^10s} │ {'R²':^10s}")
        print("-" * 50)

        for complexity in sorted(cx_metrics["per_complexity_r2"].keys()):
            r2 = cx_metrics["per_complexity_r2"][complexity]
            n_samples = cx_metrics["per_complexity_n_samples"][complexity]

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


# Constants for numba
CG_ASCII = np.array([ord("C"), ord("G")], dtype=np.uint8)


@nb.njit(fastmath=True, cache=True)
def cpg_scan(seq_bytes, ml_values, tr=122):
    """Return CpG offsets (0-based) and methylation states for one read (ONT)."""
    n = len(seq_bytes) - 1
    pos_buf = []
    state_buf = []
    cg_idx = 0
    for i in range(n):
        if seq_bytes[i] == CG_ASCII[0] and seq_bytes[i + 1] == CG_ASCII[1]:
            pos_buf.append(i)
            if cg_idx < len(ml_values):
                state_buf.append(1 if ml_values[cg_idx] > tr else 0)  # m/unm
            else:
                state_buf.append(2)  # missing
            cg_idx += 1
    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def cpg_scan_wgbs_forward(ref_bytes, read_bytes):
    """
    Scan CpGs in WGBS data (forward strand).
    Compare reference vs read to infer methylation.
    """
    n = len(ref_bytes) - 1
    pos_buf = []
    state_buf = []

    for i in range(n):
        if ref_bytes[i] == ord("C") and ref_bytes[i + 1] == ord("G"):
            pos_buf.append(i)

            if i < len(read_bytes):
                if read_bytes[i] == ord("C"):
                    state_buf.append(1)  # Methylated
                elif read_bytes[i] == ord("T"):
                    state_buf.append(0)  # Unmethylated
                else:
                    state_buf.append(2)  # Ambiguous
            else:
                state_buf.append(2)

    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def cpg_scan_wgbs_reverse(ref_bytes, read_bytes):
    """
    Scan CpGs in WGBS data (reverse strand).
    """
    n = len(ref_bytes) - 1
    pos_buf = []
    state_buf = []

    for i in range(n):
        if ref_bytes[i] == ord("C") and ref_bytes[i + 1] == ord("G"):
            pos_buf.append(i)

            if i + 1 < len(read_bytes):
                if read_bytes[i + 1] == ord("G"):
                    state_buf.append(1)  # Methylated
                elif read_bytes[i + 1] == ord("A"):
                    state_buf.append(0)  # Unmethylated
                else:
                    state_buf.append(2)  # Ambiguous
            else:
                state_buf.append(2)

    return pos_buf, state_buf


def parse_mm_tag(mm_tag):
    """
    Parse MM tag to extract modification types and their positions.

    MM tag format: "A+a,10,5,3;C+m,2,1,4;G+g,5,2;"
    Returns dict with modification info and cumulative counts for ML indexing.

    Parameters:
    -----------
    mm_tag : str
        MM tag string from BAM file

    Returns:
    --------
    dict with keys:
        'modifications': list of tuples (base, modification_code, skip_positions)
        'cpg_mod_index': index of C+m modification (-1 if not found)
        'cpg_ml_offset': offset in ML array where C+m probabilities start
        'cpg_count': number of C+m positions
    """
    if not mm_tag:
        return None

    # Split by semicolon to get each modification type
    mod_groups = mm_tag.split(";")

    modifications = []
    cumulative_count = 0
    cpg_mod_index = -1
    cpg_ml_offset = 0
    cpg_count = 0

    for idx, group in enumerate(mod_groups):
        if not group.strip():
            continue

        # Parse format: "C+m,2,1,4" or "C+m?,2,1,4"
        parts = group.split(",")
        if len(parts) < 1:
            continue

        # Extract base and modification code
        mod_info = parts[0]
        # Match patterns like "C+m" or "C+m?" or "A+a"
        match = re.match(r"([ACGT])\+([a-z\?]+)", mod_info)
        if not match:
            continue

        base = match.group(1)
        mod_code = match.group(2)

        # Extract skip positions
        skip_positions = [int(x) for x in parts[1:] if x.strip()]
        mod_count = len(skip_positions)

        modifications.append((base, mod_code, skip_positions))

        # Check if this is C+m (CpG methylation)
        # Can be "C+m" or "C+m?" where ? indicates uncertainty
        if base == "C" and mod_code.startswith("m"):
            cpg_mod_index = idx
            cpg_ml_offset = cumulative_count
            cpg_count = mod_count

        cumulative_count += mod_count

    return {
        "modifications": modifications,
        "cpg_mod_index": cpg_mod_index,
        "cpg_ml_offset": cpg_ml_offset,
        "cpg_count": cpg_count,
        "total_mods": cumulative_count,
    }


def extract_cpg_ml_values(ml_array, mm_info):
    """
    Extract only the ML probability values corresponding to C+m modifications.

    Parameters:
    -----------
    ml_array : array-like
        Full ML probability array
    mm_info : dict
        Output from parse_mm_tag()

    Returns:
    --------
    np.array
        ML values for C+m positions only
    """
    if mm_info is None or mm_info["cpg_mod_index"] == -1:
        return np.array([])

    offset = mm_info["cpg_ml_offset"]
    count = mm_info["cpg_count"]

    # Extract the slice of ML values for C+m
    if offset + count <= len(ml_array):
        return np.array(ml_array[offset : offset + count])
    else:
        # Handle edge case where ML array is shorter than expected
        return np.array(ml_array[offset:])


import numpy as np
from collections import defaultdict


def analyze_read_dmr_overlap(read_start, read_end, chrom, dmr_trees, strict=False):
    """
    Analyze overlap between a read segment and DMRs with calculated overlap metrics.

    Parameters:
    - strict: If True, only returns overlaps where read is FULLY contained in DMR.
    """
    if chrom not in dmr_trees:
        return _empty_overlap_result()

    # Query the tree
    overlapping = dmr_trees[chrom][read_start:read_end]

    # Optional Strict Filter (Read must be fully inside DMR)
    if strict and overlapping:
        overlapping = {
            iv for iv in overlapping if iv.begin <= read_start and iv.end >= read_end
        }

    if not overlapping:
        return _empty_overlap_result()

    # Initialize lists to store data for all overlapping DMRs
    dmr_labels = []
    dmr_types = []
    overlap_bps = []
    overlap_pcts = []

    read_len = read_end - read_start

    for interval in overlapping:
        dmr_data = interval.data
        dmr_start, dmr_end = interval.begin, interval.end
        dmr_len = dmr_end - dmr_start

        # 1. Calculate Overlap in BP
        # Intersection = max of starts to min of ends
        intersect_start = max(read_start, dmr_start)
        intersect_end = min(read_end, dmr_end)
        overlap_len = max(0, intersect_end - intersect_start)

        # 2. Calculate Percentage (Maximal Percentage Intersection)
        # Denominator is the length of the shorter entity (Read vs DMR)
        denominator = min(read_len, dmr_len)

        if denominator > 0:
            pct = (overlap_len / denominator) * 100.0
        else:
            pct = 0.0

        # Append to lists
        dmr_labels.append(str(dmr_data["name"]))
        dmr_types.append(str(dmr_data["type"]))
        overlap_bps.append(str(overlap_len))
        overlap_pcts.append(f"{pct:.1f}")  # Format as "75.0"

    return {
        "overlaps_dmr": True,
        "num_dmrs_overlapped": len(overlapping),
        "dmr_labels": ",".join(dmr_labels),
        "dmr_types": ",".join(dmr_types),
        "overlap_bps": ",".join(overlap_bps),  # e.g., "30,15"
        "overlap_pcts": ",".join(overlap_pcts),  # e.g., "75.0,20.0"
        # 'dmr_mean_meth_target': ...
    }


def _empty_overlap_result():
    return {
        "overlaps_dmr": False,
        "num_dmrs_overlapped": 0,
        "dmr_labels": "",
        "dmr_types": "",
        "overlap_bps": "",
        "overlap_pcts": "",
        "dmr_mean_meth_target": None,
        "dmr_mean_meth_background": None,
        "dmr_mean_diff": None,
    }


def chunk_read_data(
    seq,
    methylation_encoding,
    cpg_positions,
    meth_states,
    read_start,
    read_end,
    chrom,
    chunk_size,
    dmr_trees,
    strict,
):
    """
    Split a read into chunks of specified size and filter for DMR overlaps.
    Creates ONE ENTRY PER OVERLAPPING DMR with sequences clipped to DMR boundaries.

    This matches wgbstools behavior where fragments are clipped to marker boundaries
    before U/X/M classification.

    Returns:
    --------
    list of dict
        List of chunk dictionaries, one per overlapping DMR with:
        - seq, methylation_encoding: Original chunk data
        - seq_clipped, methylation_clipped: Clipped to DMR boundaries
        - clipped_methylated, clipped_unmethylated: CpG counts in clipped region
        - dmr_label, dmr_type: Single DMR info (not comma-separated)
    """
    chunks = []
    read_length = len(seq)

    for chunk_start_offset in range(0, read_length, chunk_size):
        chunk_end_offset = min(chunk_start_offset + chunk_size, read_length)

        chunk_gen_start = read_start + chunk_start_offset
        chunk_gen_end = read_start + chunk_end_offset

        chunk_seq = seq[chunk_start_offset:chunk_end_offset]
        chunk_meth_enc = methylation_encoding[chunk_start_offset:chunk_end_offset]

        # Collect CpGs for this chunk
        chunk_cpgs = []
        chunk_cpg_states = []
        for cpg_pos, cpg_state in zip(cpg_positions, meth_states):
            if chunk_gen_start <= cpg_pos < chunk_gen_end:
                chunk_cpgs.append(cpg_pos)
                chunk_cpg_states.append(cpg_state)

        # Original chunk CpG stats
        total_cpgs = len(chunk_cpgs)
        methylated_cpgs = sum(1 for s in chunk_cpg_states if s == 1)
        unmethylated_cpgs = sum(1 for s in chunk_cpg_states if s == 0)
        methylation_rate = (methylated_cpgs / total_cpgs) if total_cpgs > 0 else 0.0

        # Get list of overlapping DMRs (not aggregated)
        overlapping_dmrs = _get_overlapping_dmrs(
            chunk_gen_start, chunk_gen_end, chrom, dmr_trees, strict
        )

        # Create ONE ENTRY PER DMR with clipped sequences
        for dmr_info in overlapping_dmrs:
            dmr_start = dmr_info["dmr_start"]
            dmr_end = dmr_info["dmr_end"]

            # Calculate clipped region (intersection of chunk and DMR)
            clip_gen_start = max(chunk_gen_start, dmr_start)
            clip_gen_end = min(chunk_gen_end, dmr_end)

            # Calculate offsets within chunk for clipping
            clip_offset_start = clip_gen_start - chunk_gen_start
            clip_offset_end = clip_gen_end - chunk_gen_start

            # Clip sequence and methylation encoding to DMR boundaries
            seq_clipped = chunk_seq[clip_offset_start:clip_offset_end]
            meth_clipped = chunk_meth_enc[clip_offset_start:clip_offset_end]

            # Count CpGs in clipped region only (for UXM classification)
            clipped_methylated = 0
            clipped_unmethylated = 0
            clipped_total = 0
            for cpg_pos, cpg_state in zip(chunk_cpgs, chunk_cpg_states):
                if clip_gen_start <= cpg_pos < clip_gen_end:
                    clipped_total += 1
                    if cpg_state == 1:
                        clipped_methylated += 1
                    elif cpg_state == 0:
                        clipped_unmethylated += 1

            clipped_meth_rate = (
                (clipped_methylated / clipped_total) if clipped_total > 0 else 0.0
            )

            chunk_data = {
                # Original chunk data
                "seq": chunk_seq,
                "methylation_encoding": chunk_meth_enc,
                "chromosome": chrom,
                "chunk_start": chunk_gen_start,
                "chunk_end": chunk_gen_end,
                "chunk_length": chunk_end_offset - chunk_start_offset,
                "chunk_offset_in_read": chunk_start_offset,
                "total_cpgs": total_cpgs,
                "methylated_cpgs": methylated_cpgs,
                "unmethylated_cpgs": unmethylated_cpgs,
                "methylation_rate": methylation_rate,
                # Clipped data (for UXM classification)
                "seq_clipped": seq_clipped,
                "methylation_clipped": meth_clipped,
                "clip_start": clip_gen_start,
                "clip_end": clip_gen_end,
                "clip_length": len(seq_clipped),
                "clipped_methylated": clipped_methylated,
                "clipped_unmethylated": clipped_unmethylated,
                "clipped_total": clipped_total,
                "clipped_meth_rate": clipped_meth_rate,
                # Single DMR info (not comma-separated)
                "overlaps_dmr": True,
                "dmr_label": dmr_info["name"],
                "dmr_type": dmr_info["type"],
                "dmr_start": dmr_start,
                "dmr_end": dmr_end,
                "overlap_bp": dmr_info["overlap_bp"],
                "overlap_pct": dmr_info["overlap_pct"],
            }
            chunks.append(chunk_data)

    return chunks


def _get_overlapping_dmrs(read_start, read_end, chrom, dmr_trees, strict=False):
    """
    Get list of overlapping DMRs with individual info (not aggregated).
    Returns one dict per overlapping DMR for iteration.
    """
    if chrom not in dmr_trees:
        return []

    overlapping = dmr_trees[chrom][read_start:read_end]

    if strict and overlapping:
        overlapping = {
            iv for iv in overlapping if iv.begin <= read_start and iv.end >= read_end
        }

    if not overlapping:
        return []

    read_len = read_end - read_start
    results = []

    for interval in overlapping:
        dmr_data = interval.data
        dmr_start, dmr_end = interval.begin, interval.end
        dmr_len = dmr_end - dmr_start

        # Calculate overlap
        intersect_start = max(read_start, dmr_start)
        intersect_end = min(read_end, dmr_end)
        overlap_len = max(0, intersect_end - intersect_start)

        denominator = min(read_len, dmr_len)
        pct = (overlap_len / denominator) * 100.0 if denominator > 0 else 0.0

        results.append(
            {
                "name": str(dmr_data["name"]),
                "type": str(dmr_data["type"]),
                "dmr_start": dmr_start,
                "dmr_end": dmr_end,
                "overlap_bp": overlap_len,
                "overlap_pct": pct,
            }
        )

    return results


def process_single_read(
    read,
    data_type,
    methyl_tr=122,
    ref_fasta=None,
    interesting_chromosomes=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
    min_cpgs=1,
):
    """
    Process a single read and return its methylation data as-is (no chunking or DMR filtering).
    This function can be used independently for testing.

    Parameters:
    -----------
    read : pysam.AlignedSegment
        BAM read object
    data_type : str
        'ont' or 'wgbs'
    methyl_tr : int
        Methylation threshold for ONT (default: 122)
    ref_fasta : pysam.FastaFile or None
        Reference genome (required for WGBS)
    interesting_chromosomes : list or None
        List of chromosomes to process (None = process all)
    min_mapq : int
        Minimum mapping quality threshold (default: 10, matching SAMtools -q 10)
    require_flags : int
        SAM flags that must ALL be set (default: 3 = paired + properly paired,
        matching SAMtools -f 3). Set to 0 to disable required flag filtering.
        Flag breakdown: 1 (paired) + 2 (properly paired)
    exclude_flags : int
        SAM flags to exclude if ANY are set (default: 1796 = unmapped + secondary + failed QC + duplicate,
        matching SAMtools -F 1796). Set to 0 to disable flag filtering.
        Flag breakdown: 4 (unmapped) + 256 (secondary) + 512 (failed QC) + 1024 (duplicate)
    min_cpgs : int
        Minimum number of CpG sites required in the read (default: 1).
        Reads with fewer CpGs are skipped.

    Returns:
    --------
    list of dict
        List with a single dictionary containing read-level methylation data,
        or empty list if read is filtered out.
    """
    # Filter by chromosome if specified
    if interesting_chromosomes and read.reference_name not in interesting_chromosomes:
        return []

    # Quality filters matching SAMtools -f 3 -F 1796 -q 10
    # Check if ALL require_flags are set (SAMtools -f)
    if require_flags is not None:
        if require_flags and (read.flag & require_flags) != require_flags:
            return []

    # Check if ANY of the exclude_flags are set (SAMtools -F)
    if exclude_flags is not None:
        if exclude_flags and (read.flag & exclude_flags):
            return []

    # Skip reads below minimum mapping quality (SAMtools -q)
    if read.mapping_quality < min_mapq:
        return []

    chrom = read.reference_name
    r_beg, r_end = read.reference_start, read.reference_end

    # ========== Process based on data type ==========
    if data_type == "ont":
        # ONT processing with MM/ML tag parsing
        try:
            ml_values_full = read.get_tag("ML")
            mm_tag = read.get_tag("MM")
        except KeyError:
            return []

        # Parse MM tag to find C+m modifications
        mm_info = parse_mm_tag(mm_tag)
        if mm_info is None or mm_info["cpg_mod_index"] == -1:
            # No C+m modification found in this read
            return []

        # Extract only the ML values for C+m
        ml_values = extract_cpg_ml_values(ml_values_full, mm_info)

        if len(ml_values) == 0:
            return []

        seq = read.get_forward_sequence()
        read_length = len(seq)

        # Scan for CpGs using extracted C+m probabilities
        pos_in_read, meth_states = cpg_scan(seq.encode(), ml_values, methyl_tr)
        total_cpgs = len(pos_in_read)

        # Filter by minimum CpG count
        if total_cpgs < min_cpgs:
            return []

        cpg_positions = [r_beg + off for off in pos_in_read]

        # Build methylation encoding
        meth_enc = ["2" for _ in range(read_length)]
        for x, y in zip(pos_in_read, meth_states):
            meth_enc[x] = str(y)
        methylation_encoding = "".join(meth_enc)

    else:  # WGBS
        if ref_fasta is None:
            raise ValueError("Reference genome (ref_fasta) required for WGBS data")

        # Get reference sequence for this region
        try:
            # We are parsing +1 one charachter from ref_seq in case it ends with CG.
            ref_seq = ref_fasta.fetch(chrom, r_beg, r_end + 1).upper()
            # We are checking if string ends with CG and only keeping last character if it does
            if ref_seq[-2:] != "CG":
                ref_seq = ref_seq[:-1]
        except:
            return []

        read_seq = clean_cigar_sequence(read)
        if read_seq is None:
            return []

        read_length = len(ref_seq)

        # Scan for CpGs and infer methylation
        if read.is_reverse:
            pos_in_read, meth_states = cpg_scan_wgbs_reverse(
                ref_seq.encode(), read_seq.encode()
            )
        else:
            pos_in_read, meth_states = cpg_scan_wgbs_forward(
                ref_seq.encode(), read_seq.encode()
            )

        total_cpgs = len(pos_in_read)

        # Filter by minimum CpG count
        if total_cpgs < min_cpgs:
            return []

        cpg_positions = [r_beg + off for off in pos_in_read]

        seq = ref_seq

        # Build methylation encoding
        meth_enc = ["2" for _ in range(read_length)]
        for x, y in zip(pos_in_read, meth_states):
            if x < len(meth_enc):
                meth_enc[x] = str(y)
        methylation_encoding = "".join(meth_enc)

    # ========== Calculate CpG statistics ==========
    methylated_cpgs = sum(1 for s in meth_states if s == 1)
    unmethylated_cpgs = sum(1 for s in meth_states if s == 0)
    methylation_rate = methylated_cpgs / total_cpgs if total_cpgs > 0 else 0.0

    # ========== Return read data as a single entry ==========
    read_data = {
        "read_name": read.query_name,
        "chromosome": chrom,
        "read_start": r_beg,
        "read_end": r_end,
        "read_length": r_end - r_beg,
        "seq": seq,
        "methylation_encoding": methylation_encoding,
        "cpg_positions": cpg_positions,
        "meth_states": list(meth_states),
        "total_cpgs": total_cpgs,
        "methylated_cpgs": methylated_cpgs,
        "unmethylated_cpgs": unmethylated_cpgs,
        "methylation_rate": methylation_rate,
        "mapping_quality": read.mapping_quality,
        "is_reverse": read.is_reverse,
        "data_type": data_type,
    }

    return [read_data]


def merge_paired_reads(df, verbose=True):
    """
    Merge paired-end reads into single fragments, similar to wgbs_tools' bam2pat.

    For paired-end sequencing, both mates represent the same DNA fragment and should
    be merged. This function:
    1. Groups reads by read_name AND dmr_label (both mates share the same name, merge per DMR)
    2. Merges methylation information, handling overlapping CpGs
    3. Creates a single fragment entry spanning both mates

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from process_bam_with_chunking with columns:
        - read_name: Read identifier (shared by both mates)
        - dmr_label: Single DMR associated with this entry
        - read_start, read_end: Genomic coordinates
        - chromosome: Chromosome
        - methylation_encoding: String encoding methylation states
        - seq: Sequence
        - clipped_methylated, clipped_unmethylated, clipped_total: CpG counts in clipped region
        - and other metadata columns
    verbose : bool
        Print progress information

    Returns:
    --------
    pd.DataFrame
        DataFrame with paired mates merged into single fragment entries.
        Each entry is for a single DMR with clipped CpG counts for UXM classification.
    """
    if len(df) == 0:
        return df

    # Determine grouping columns - use dmr_label if available
    if "dmr_label" in df.columns:
        group_cols = ["read_name", "dmr_label"]
    else:
        group_cols = ["read_name"]

    # Count reads per (name, dmr_label) to identify pairs vs singletons
    read_counts = df.groupby(group_cols).size()
    singletons = read_counts[read_counts == 1].index
    pairs = read_counts[read_counts == 2].index
    multiplets = read_counts[read_counts > 2].index

    if verbose:
        print(f"Read distribution (grouped by {group_cols}):")
        print(f"  Singletons: {len(singletons)}")
        print(f"  Paired (2 mates): {len(pairs)}")
        print(
            f"  Multiplets (>2): {len(multiplets)} (will be treated as separate entries)"
        )

    merged_data = []

    # Process singletons - keep as is
    if len(group_cols) == 2:
        singleton_mask = df.set_index(group_cols).index.isin(singletons)
        singleton_df = df[singleton_mask]
    else:
        singleton_df = df[df["read_name"].isin(singletons)]
    for _, row in singleton_df.iterrows():
        merged_data.append(row.to_dict())

    # Process pairs - merge mates
    if len(group_cols) == 2:
        pairs_mask = df.set_index(group_cols).index.isin(pairs)
        pairs_df = df[pairs_mask]
    else:
        pairs_df = df[df["read_name"].isin(pairs)]

    for key, group in pairs_df.groupby(group_cols):
        if len(group) != 2:
            continue

        mate1, mate2 = group.iloc[0], group.iloc[1]

        # Ensure mate1 is the one with smaller start position
        if mate2["read_start"] < mate1["read_start"]:
            mate1, mate2 = mate2, mate1

        # Merge the pair
        merged_fragment = _merge_mate_pair(mate1, mate2)
        merged_data.append(merged_fragment)

    # Process multiplets - keep each entry separately (unusual case)
    if len(group_cols) == 2:
        multiplet_mask = df.set_index(group_cols).index.isin(multiplets)
        multiplet_df = df[multiplet_mask]
    else:
        multiplet_df = df[df["read_name"].isin(multiplets)]
    for _, row in multiplet_df.iterrows():
        merged_data.append(row.to_dict())

    result_df = pd.DataFrame(merged_data)

    if verbose:
        print(
            f"Merged result: {len(result_df)} fragments (from {len(df)} read entries)"
        )

    return result_df


def _merge_mate_pair(mate1, mate2):
    """
    Merge two mates of a paired-end read into a single fragment.

    Parameters:
    -----------
    mate1 : pd.Series
        First mate (should have smaller read_start)
    mate2 : pd.Series
        Second mate

    Returns:
    --------
    dict
        Merged fragment with combined methylation information
    """
    # Calculate fragment coordinates
    frag_start = min(mate1["read_start"], mate2["read_start"])
    frag_end = max(mate1["read_end"], mate2["read_end"])
    frag_length = frag_end - frag_start

    # Merge methylation encodings
    # Create a position-aware mapping for CpGs
    merged_encoding, merged_cpg_stats = _merge_methylation_encodings(
        mate1["seq"],
        mate1["methylation_encoding"],
        mate1["read_start"],
        mate2["seq"],
        mate2["methylation_encoding"],
        mate2["read_start"],
        frag_start,
        frag_end,
    )

    # Handle DMR info - use singular dmr_label if available, fall back to dmr_labels
    dmr_label = mate1.get("dmr_label", "") or mate1.get("dmr_labels", "")
    dmr_type = mate1.get("dmr_type", "") or mate1.get("dmr_types", "")
    dmr_start = mate1.get("dmr_start", 0)
    dmr_end = mate1.get("dmr_end", 0)

    # Merge clipped sequences if available
    # For merged fragments, we need to recalculate clipped data within DMR boundaries
    if "seq_clipped" in mate1.index and dmr_start and dmr_end:
        # Merge clipped sequences within DMR boundaries
        clipped_encoding, clipped_stats = _merge_methylation_encodings(
            mate1.get("seq_clipped", ""),
            mate1.get("methylation_clipped", ""),
            mate1.get("clip_start", mate1["read_start"]),
            mate2.get("seq_clipped", ""),
            mate2.get("methylation_clipped", ""),
            mate2.get("clip_start", mate2["read_start"]),
            dmr_start,
            dmr_end,
        )
        clipped_methylated = clipped_stats["methylated"]
        clipped_unmethylated = clipped_stats["unmethylated"]
        clipped_total = clipped_stats["total"]
    else:
        # Fallback if no clipped data
        clipped_methylated = mate1.get("clipped_methylated", 0) + mate2.get(
            "clipped_methylated", 0
        )
        clipped_unmethylated = mate1.get("clipped_unmethylated", 0) + mate2.get(
            "clipped_unmethylated", 0
        )
        clipped_total = clipped_methylated + clipped_unmethylated
        clipped_encoding = {"seq": "", "methylation_encoding": ""}

    clipped_meth_rate = clipped_methylated / clipped_total if clipped_total > 0 else 0.0

    # Build merged fragment entry
    merged = {
        "read_name": mate1["read_name"],
        "chromosome": mate1["chromosome"],
        "read_start": frag_start,  # Fragment start (earliest)
        "read_end": frag_end,  # Fragment end (latest)
        "read_length": frag_length,
        "seq": merged_encoding["seq"],
        "methylation_encoding": merged_encoding["methylation_encoding"],
        "total_cpgs": merged_cpg_stats["total"],
        "methylated_cpgs": merged_cpg_stats["methylated"],
        "unmethylated_cpgs": merged_cpg_stats["unmethylated"],
        "methylation_rate": merged_cpg_stats["methylation_rate"],
        "read_total_cpgs": merged_cpg_stats["total"],  # For compatibility
        "mapping_quality": min(mate1["mapping_quality"], mate2["mapping_quality"]),
        "is_reverse": mate1["is_reverse"],  # Keep first mate's orientation
        "data_type": mate1["data_type"],
        "is_merged_pair": True,
        "mate1_start": mate1["read_start"],
        "mate1_end": mate1["read_end"],
        "mate2_start": mate2["read_start"],
        "mate2_end": mate2["read_end"],
        # DMR info (singular)
        "overlaps_dmr": mate1.get("overlaps_dmr", False)
        or mate2.get("overlaps_dmr", False),
        "dmr_label": dmr_label,
        "dmr_type": dmr_type,
        "dmr_start": dmr_start,
        "dmr_end": dmr_end,
        # Clipped data for UXM classification
        "seq_clipped": clipped_encoding.get("seq", ""),
        "methylation_clipped": clipped_encoding.get("methylation_encoding", ""),
        "clipped_methylated": clipped_methylated,
        "clipped_unmethylated": clipped_unmethylated,
        "clipped_total": clipped_total,
        "clipped_meth_rate": clipped_meth_rate,
        # Chunk info (use fragment coordinates)
        "chunk_start": frag_start,
        "chunk_end": frag_end,
        "chunk_length": frag_length,
    }

    return merged


def _merge_methylation_encodings(
    seq1, enc1, start1, seq2, enc2, start2, frag_start, frag_end
):
    """
    Merge methylation encodings from two mates, matching wgbstools' merge_PE behavior.

    Consensus logic:
    - If both mates report same state, use that state
    - If one is unknown ('2'), use the other's value
    - If they DISAGREE (both known but different), use '2' (unknown)
    """
    frag_length = frag_end - frag_start

    # Initialize merged arrays with 'unknown' state
    merged_enc = ["2"] * frag_length
    merged_seq = ["N"] * frag_length

    # Fill in mate1 data
    for i in range(len(seq1)):
        if i >= len(enc1):
            break
        pos = (start1 - frag_start) + i
        if 0 <= pos < frag_length:
            merged_seq[pos] = seq1[i]
            merged_enc[pos] = enc1[i]

    # Fill in mate2 data with consensus logic matching wgbstools
    for i in range(len(seq2)):
        if i >= len(enc2):
            break
        pos = (start2 - frag_start) + i
        if 0 <= pos < frag_length:
            existing = merged_enc[pos]
            merged_seq[pos] = seq2[i]

            meth = enc2[i]

            if existing == "2":
                # No data from mate1 (or unknown), use mate2
                merged_enc[pos] = meth
            elif meth == "2":
                # No data from mate2, keep mate1
                pass
            elif existing == meth:
                # Both agree, keep the value
                pass
            else:
                # CONFLICT: both are known ('0' or '1') but different
                # Mark as unknown, matching wgbstools behavior
                merged_enc[pos] = "2"

    # Calculate CpG statistics from merged encoding
    # Only count confident calls ('1' and '0'), not unknowns ('2')
    methylated = sum(1 for x in merged_enc if x == "1")
    unmethylated = sum(1 for x in merged_enc if x == "0")
    total = methylated + unmethylated
    meth_rate = methylated / total if total > 0 else 0.0

    return {"seq": "".join(merged_seq), "methylation_encoding": "".join(merged_enc)}, {
        "total": total,
        "methylated": methylated,
        "unmethylated": unmethylated,
        "methylation_rate": meth_rate,
    }


def _merge_comma_separated(str1, str2):
    """Merge two comma-separated strings, keeping unique values."""
    if not str1 and not str2:
        return ""
    items1 = set(str1.split(",")) if str1 else set()
    items2 = set(str2.split(",")) if str2 else set()
    merged = items1 | items2
    merged.discard("")  # Remove empty strings
    return ",".join(sorted(merged))


def count_reference_cpgs(ref_fasta, chrom, start, end):
    """
    Count CpG dinucleotides in the reference genome for a given region.

    This is used to count the total number of CpGs a fragment spans,
    including those in the insert region between paired-end mates.

    Parameters:
    -----------
    ref_fasta : pysam.FastaFile
        Open reference genome file
    chrom : str
        Chromosome name
    start : int
        Start position (0-based)
    end : int
        End position (exclusive)

    Returns:
    --------
    int
        Number of CpG dinucleotides in the region
    """
    try:
        seq = ref_fasta.fetch(chrom, start, end + 1).upper()
        return seq.count("CG")
    except:
        return 0


def add_reference_cpg_counts(df, reference_path, verbose=True):
    """
    Add reference-based CpG counts to the DataFrame.

    This counts ALL CpGs in each fragment's genomic span using the reference genome,
    matching wgbs_tools' behavior. The difference between reference CpGs and
    called CpGs (M + U) gives the number of "unknown" CpGs.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with read/fragment data containing:
        - chromosome, read_start, read_end: Fragment coordinates
        - methylated_cpgs, unmethylated_cpgs: Called CpG counts
    reference_path : str
        Path to reference genome FASTA
    verbose : bool
        Print progress

    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns:
        - ref_cpg_count: Total CpGs in fragment span (from reference)
        - unknown_cpgs: CpGs in span but not called (ref_cpg_count - M - U)
    """
    if len(df) == 0:
        return df

    if verbose:
        print(f"Counting reference CpGs for {len(df)} fragments...")

    ref_fasta = pysam.FastaFile(reference_path)

    ref_counts = []
    for _, row in df.iterrows():
        count = count_reference_cpgs(
            ref_fasta, row["chromosome"], row["read_start"], row["read_end"]
        )
        ref_counts.append(count)

    ref_fasta.close()

    df = df.copy()
    df["ref_cpg_count"] = ref_counts
    df["unknown_cpgs"] = (
        df["ref_cpg_count"] - df["methylated_cpgs"] - df["unmethylated_cpgs"]
    )
    # Ensure non-negative (edge cases)
    df["unknown_cpgs"] = df["unknown_cpgs"].clip(lower=0)

    if verbose:
        total_ref = df["ref_cpg_count"].sum()
        total_m = df["methylated_cpgs"].sum()
        total_u = df["unmethylated_cpgs"].sum()
        total_x = df["unknown_cpgs"].sum()
        print(
            f"Reference CpG counts - Total: {total_ref}, M: {total_m}, U: {total_u}, X: {total_x}"
        )

    return df


def detect_data_type(bam_path, sample_size=1000):
    """
    Detect if BAM file is ONT (has ML tags) or WGBS (no ML tags).
    """
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        reads_checked = 0
        has_ml_tag = 0

        for read in bam:
            if read.is_unmapped:
                continue

            try:
                read.get_tag("ML")
                has_ml_tag += 1
            except KeyError:
                pass

            reads_checked += 1
            if reads_checked >= sample_size:
                break

        if reads_checked == 0:
            raise ValueError("No mapped reads found in BAM file")

        if has_ml_tag / reads_checked > 0.5:
            return "ont"
        else:
            return "wgbs"


def process_tabular_chunk(args):
    """
    Process chunk with support for both ONT and WGBS data.
    Now uses the refactored process_single_read function.

    Quality filtering: By default, filters reads matching SAMtools -f 3 -F 1796 -q 10
    (requires paired + properly paired; excludes unmapped, secondary, failed QC, duplicates; MAPQ >= 10).
    Also filters reads with fewer than min_cpgs CpG sites.
    """
    (
        chrom,
        chunk_start,
        chunk_end,
        bam_path,
        interesting_chromosomes,
        estimate_cov,
        methyl_tr,
        data_type,
        reference_path,
        min_mapq,
        require_flags,
        exclude_flags,
        min_cpgs,
    ) = args

    tabular_data = []

    # Open reference genome if WGBS
    ref_fasta = None
    if data_type == "wgbs":
        if reference_path is None:
            raise ValueError("Reference genome path required for WGBS data")
        ref_fasta = pysam.FastaFile(reference_path)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, chunk_start, chunk_end):
            # Skip reads that don't START within this chunk to avoid duplicate counting
            # bam.fetch() returns all reads that OVERLAP the region, but we only want
            # to process each read once (in the chunk where it starts)
            if read.reference_start < chunk_start or read.reference_start >= chunk_end:
                continue

            # Process single read with quality filters
            read_data = process_single_read(
                read=read,
                data_type=data_type,
                methyl_tr=methyl_tr,
                ref_fasta=ref_fasta,
                interesting_chromosomes=interesting_chromosomes,
                min_mapq=min_mapq,
                require_flags=require_flags,
                exclude_flags=exclude_flags,
                min_cpgs=min_cpgs,
            )

            # Add all read entries (each read returns a list with one dict)
            tabular_data.extend(read_data)

    if ref_fasta is not None:
        ref_fasta.close()

    return tabular_data


def clean_cigar_sequence(read):
    """
    Process CIGAR string to align read sequence to reference coordinates.

    Similar to wgbstools' clean_CIGAR function:
    - Removes inserted bases (I) from the read sequence
    - Adds 'N' placeholders for deleted bases (D) in reference
    - Removes soft-clipped bases (S)

    Returns:
    --------
    str: Processed read sequence that aligns 1:1 with reference positions
    """
    if read.cigartuples is None:
        return read.query_alignment_sequence

    seq = read.query_sequence  # Full query sequence including soft clips
    result = []
    seq_pos = 0  # Position in query sequence

    for op, length in read.cigartuples:
        if op == 0:  # M - Match/mismatch: consume both query and reference
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length
        elif op == 1:  # I - Insertion: consume query only, skip these bases
            seq_pos += length  # Skip inserted bases
        elif op == 2:  # D - Deletion: consume reference only, add placeholder
            result.append("N" * length)  # Add N's for deleted reference positions
        elif op == 3:  # N - Reference skip (intron): add placeholder
            result.append("N" * length)
        elif op == 4:  # S - Soft clip: consume query only, skip these bases
            seq_pos += length
        elif op == 5:  # H - Hard clip: doesn't consume query
            pass
        elif op == 7:  # = - Sequence match: same as M
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length
        elif op == 8:  # X - Sequence mismatch: same as M
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length

    return "".join(result)


def process_bam_with_chunking(
    bam_path,
    chromosomes,
    n_jobs=4,
    chunk_size_genomic=1_000_000,
    methyl_tr=122,
    reference_path=None,
    data_type=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
    min_cpgs=1,
    merge_pairs=True,
):
    """
    Process BAM file and extract read-level methylation data.
    Supports both ONT and WGBS data.

    For ONT: Properly handles MM tags with multiple modifications (A+a, C+m, etc.)
    For WGBS: Uses reference genome comparison

    Parameters:
    -----------
    bam_path : str
        Path to BAM file
    chromosomes : list
        List of chromosomes to process
    n_jobs : int
        Number of parallel jobs (default: 4)
    chunk_size_genomic : int
        Genomic chunk size for parallel processing (default: 1M)
    methyl_tr : int
        Methylation threshold for ONT (default: 122)
    reference_path : str
        Path to reference genome FASTA (required for WGBS)
    data_type : str
        'ont' or 'wgbs'. If None, will auto-detect.
    min_mapq : int
        Minimum mapping quality threshold (default: 10, matching SAMtools -q 10).
        Set to 0 to disable MAPQ filtering.
    require_flags : int
        SAM flags that must ALL be set (default: 3 = paired + properly paired,
        matching SAMtools -f 3). Set to 0 to disable required flag filtering.
        Flag breakdown: 1 (paired) + 2 (properly paired)
    exclude_flags : int
        SAM flags to exclude if ANY are set (default: 1796, matching SAMtools -F 1796).
        Flag breakdown: 4 (unmapped) + 256 (secondary) + 512 (failed QC) + 1024 (duplicate)
        Set to 0 to disable flag filtering.
    min_cpgs : int
        Minimum number of CpG sites required in the read (default: 1).
        Reads with fewer CpGs are skipped.
    merge_pairs : bool
        If True, merge paired-end reads (mates) into single fragments, similar to
        wgbs_tools' bam2pat. Both mates are combined with their methylation patterns
        merged, spanning the full fragment. Also adds reference-based CpG counts
        including 'unknown_cpgs' for CpGs in the insert region. (default: True)

    Returns:
    --------
    pd.DataFrame
        DataFrame with read-level methylation data.
        If merge_pairs=True, returns merged fragment entries with additional columns:
        - ref_cpg_count: Total CpGs in fragment span (from reference genome)
        - unknown_cpgs: CpGs in span but not called (in insert region between mates)
        These match wgbs_tools' behavior for counting CpGs.
    """
    from multiprocessing import Pool

    # Auto-detect data type if not specified
    if data_type is None:
        print("Auto-detecting data type...")
        data_type = detect_data_type(bam_path)
        print(f"Detected data type: {data_type.upper()}")

    # Validate inputs
    if data_type == "wgbs" and reference_path is None:
        raise ValueError("Reference genome path is required for WGBS data")

    # Get chromosome lengths from BAM
    print("Reading BAM file...")
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        chr_lengths = {ref: length for ref, length in zip(bam.references, bam.lengths)}

    # Generate genomic chunks for parallel processing
    tasks = []
    for chrom in chromosomes:
        if chrom not in chr_lengths:
            print(f"Warning: {chrom} not found in BAM file")
            continue

        chr_len = chr_lengths[chrom]
        for start in range(0, chr_len, chunk_size_genomic):
            end = min(start + chunk_size_genomic, chr_len)
            tasks.append(
                (
                    chrom,
                    start,
                    end,
                    bam_path,
                    chromosomes,
                    False,
                    methyl_tr,
                    data_type,
                    reference_path,
                    min_mapq,
                    require_flags,
                    exclude_flags,
                    min_cpgs,
                )
            )

    print(f"Processing {len(tasks)} genomic chunks with {n_jobs} workers...")

    # Process in parallel
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(process_tabular_chunk, tasks)
    else:
        results = [process_tabular_chunk(task) for task in tasks]

    # Flatten results
    all_data = []
    for result in results:
        all_data.extend(result)

    print(f"Collected {len(all_data)} reads")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Optionally merge paired-end reads into fragments
    if merge_pairs and len(df) > 0:
        print("Merging paired-end reads into fragments...")
        df = merge_paired_reads(df, verbose=True)

        # # Add reference CpG counts to get accurate unknown counts
        # # This counts all CpGs in the fragment span, including insert region
        # if reference_path is not None:
        #     df = add_reference_cpg_counts(df, reference_path, verbose=True)

    return df


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


def chunk_tokens(tokens, window_size, stride):
    """
    Splits a list of tokens into overlapping chunks of fixed size.
    Ensures no chunk is shorter than window_size (unless the total read is shorter).
    """
    total_len = len(tokens)

    # Case 1: Read is shorter than the window. Return as is.
    if total_len <= window_size:
        yield tokens
        return

    # Case 2: Sliding window
    # We iterate until the window would go out of bounds
    for i in range(0, total_len - window_size + 1, stride):
        yield tokens[i : i + window_size]

    # Case 3: Handle the "Tail"
    # If the last sliding window didn't exactly align with the end,
    # we yield one final chunk containing the *last* window_size elements.
    # This creates a variable overlap for the last segment, but ensures full context.
    if total_len % stride != 0:
        yield tokens[-window_size:]


def generate_valid_tokens(read_data, k=3):
    """
    Yields valid k-mers and their ORIGINAL indices.
    Skips any k-mer containing 'N'.
    """
    seq = read_data["input_ids"]
    pattern = read_data["methylation_ids"]

    # We iterate up to len(seq) - k + 1
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]

        # 1. Check for 'N' in the window
        if "N" in kmer:
            continue

        center_idx = i + 1
        methylation_code = pattern[center_idx]

        # Yield the clean k-mer and its specific methylation label
        yield [kmer, methylation_code]


def aggregate_chuncked_predictions_weighted(
    pred_df, weight_col="ncpgs_marked", group_col="read_name"
):
    """
    Calculates the weighted average of prediction columns grouped by read_name.
    """
    # 1. Identify prediction columns (prediction_0 ... prediction_39)
    pred_cols = [c for c in pred_df.columns if c.startswith("prediction_")]

    # 2. Create a working copy to avoid SettingWithCopy warnings
    df = pred_df.copy()

    # 3. Vectorized Weighting: Multiply all prediction columns by the weight column
    # This is much faster than doing it inside the groupby
    df[pred_cols] = df[pred_cols].multiply(df[weight_col], axis=0)

    # 4. Group by read_name and sum both the weighted predictions and the weights
    grouped = df.groupby(group_col)

    # Sum the weighted scores
    summed_preds = grouped[pred_cols].sum()

    # Sum the weights (the total number of CpGs seen across all chunks for this read)
    summed_weights = grouped[weight_col].sum()

    # 5. Divide to get the Weighted Average
    # We use .div with axis=0 to align by the index (read_name)
    final_averaged_df = summed_preds.div(summed_weights, axis=0)

    # 6. Cleanup
    # Handle cases where total weight might be 0 (avoid division by zero errors)
    # Though your filtering likely prevents this, it's good practice.
    final_averaged_df = final_averaged_df.fillna(0)

    # Reset index so 'read_name' becomes a column again
    return final_averaged_df.reset_index()


def prepare_methylbert_list_inference(
    results_df, dmr_label_column, seq_length=150, stride=75
):
    """
    Prepares inference data with sliding window chunking.
    params:
        stride: How far to move the window (75 = 50% overlap for 150bp window)
    """
    data_list = [
        [
            "dna_seq",
            "methyl_seq",
            "dmr_ctype",
            "dmr_label",
            "ctype",
            "original_label",
            "read_name",
            "ncpgs_marked",
        ]
    ]

    for i, row in results_df.iterrows():
        # 1. Get the CLEAN stream of tokens (Ns removed)
        processed_read_full = list(generate_valid_tokens(row))
        read_name = row["read_name"]
        # If read was entirely Ns or empty, skip
        if not processed_read_full:
            continue

        # 2. Chunk the valid tokens
        # We process the read in chunks of 'seq_length'
        for chunk in chunk_tokens(
            processed_read_full, window_size=seq_length, stride=stride
        ):

            dna = " ".join([x[0] for x in chunk])
            methyl = "".join([x[1] for x in chunk])
            ncpgs_marked = methyl.count("0") + methyl.count("1")
            # Metadata propagation
            # Note: You might want to track which chunk this is (e.g., read_id_0, read_id_1)
            # but for bulk inference, this format works.
            label = 0
            o_label = 0
            dmr_label = row[dmr_label_column]
            dmr_ctype = row["dmr_ctype_label"]

            data_list.append(
                [
                    dna,
                    methyl,
                    dmr_ctype,
                    dmr_label,
                    label,
                    o_label,
                    read_name,
                    ncpgs_marked,
                ]
            )

    return data_list
