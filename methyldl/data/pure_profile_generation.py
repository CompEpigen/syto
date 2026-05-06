"""
Pure Profile Generation
=======================

Generate one purified prediction profile per cell type by aggregating
reads that belong exclusively to a single cell type.  These profiles
serve as reference vectors for least-squares deconvolution methods
(NNLS, PSLS) and as the basis for the feature-selection cutoff step.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from methyldl.modelling.prediction_aggregation import _fill_in_missing_labels

from methyldl.data.pseudo_bulk_generation import (
    _build_target_columns,
    generate_pseudo_bulk_optimized,
)

logger = logging.getLogger(__name__)


def generate_pure_profiles(
    splits: Dict[str, pd.DataFrame],
    num_output_labels: int = 39,
    num_input_labels: int = 39,
    n_read_per_split: int = 475_000,
    labels_dict: dict = None,
) -> List[Tuple[np.ndarray, Dict[str, pd.DataFrame], list]]:
    """Generate one purified profile per cell type.

    For each cell type *i* in ``range(num_labels)`` the function creates
    a pseudo-bulk mixture where *100 %* of the reads come from cell type
    *i*.  Reads are sampled **with replacement** from each split so that
    every cell type profile is based on the same total read budget
    regardless of how many reads of that cell type exist.

    Parameters
    ----------
    splits : Dict[str, pd.DataFrame]
        Read-level DataFrames enriched with prediction columns
        (``prediction_0 … prediction_{num_input_labels-1}``) where keys are split names (e.g. 'train', 'valid').
    num_output_labels : int
        Number of output cell-type classes.
    num_input_labels : int
        Number of input prediction classes (can be larger than output if there's a reject class).
    n_read_per_split : int
        Number of reads to sample per split per profile.

    Returns
    -------
    list[tuple]
        A list of length ``num_output_labels``.  Each element is a tuple
        ``(proportions_full, subs, uxm_data)`` returned by
        :func:`generate_pseudo_bulk_optimized`.

        * ``proportions_full`` — length-``num_output_labels`` one-hot array.
        * ``subs`` — dictionary of aggregated DataFrames per split.
        * ``uxm_data`` — corresponding UXM data (or ``None``).
    """
    # Pre-compute grouped DataFrames (done once, reused per cell type)
    grouped_splits = {}
    for name, df in splits.items():
        if "total_marked_cpgs" not in df.columns:
            df["total_marked_cpgs"] = df["NCPGS"]
        if "methylation_level" not in df.columns:
            df["methylation_level"] = df["M_rate"]
        if "chromosome" not in df.columns and "chr" in df.columns:
            df.rename(columns={"chr": "chromosome"}, inplace=True)
        if "direction" not in df.columns:
            df["direction"] = "U"

        grouped_splits[name] = df.groupby(
            ["original_label", "dmr_ctype_label"], sort=False
        )

    target_columns = _build_target_columns(num_input_labels)

    pure_profiles: List[Tuple[np.ndarray, List[pd.DataFrame], list]] = []

    for i in tqdm(range(num_output_labels), desc="Generating pure profiles"):
        proportions = [0.0] * num_output_labels
        proportions[i] = 1.0
        labels = [i]
        cell_proportions = [1.0]

        try:
            _, proportions_full, subs, uxm_data = generate_pseudo_bulk_optimized(
                total_samples=n_read_per_split,
                labels=labels,
                proportions=cell_proportions,
                grouped_splits=grouped_splits,
                target_columns=target_columns,
                num_labels=num_output_labels,
                generate_uxm_inputs=False,
            )
            if labels_dict is not None:
                for key in subs.keys():
                    subs[key] = _fill_in_missing_labels(
                        subs[key],
                        group_cols=["dmr_label", "file", "original_label"],
                        labels_dict=labels_dict,
                        substitution_strategy="uniform_number",
                    )
            pure_profiles.append((proportions_full, subs, uxm_data))
        except Exception as e:
            logger.warning(f"Failed to generate pure profile for cell type {i}: {e}")
            pure_profiles.append(None)

    return pure_profiles


def compute_uniform_prior_matrix(
    pure_profiles: List[Tuple[np.ndarray, Dict[str, pd.DataFrame], list]],
    split_key: str = "train",
    num_input_labels: int = 39,
) -> pd.DataFrame:
    """Compute a uniform-mixture prior by averaging DMR predictions across pure profiles.

    For each DMR row, this function averages the weighted-average prediction
    columns (``prediction_*_wavg``) across all non-``None`` pure profiles,
    producing a matrix that represents how the aggregated DMR table would look
    for a mixture composed uniformly of all cell types.

    Parameters
    ----------
    pure_profiles : list[tuple]
        Output of :func:`generate_pure_profiles`.  Each element is a tuple
        ``(proportions, subs_dict, uxm_data)`` or ``None`` for failed profiles.
    split_key : str
        Key into ``subs_dict`` (e.g. ``"train"``, ``"valid"``, ``"test"``).
    num_input_labels : int
        Number of prediction classes to average over.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``dmr_ctype_label`` with columns
        ``prediction_0_wavg … prediction_{N-1}_wavg`` and
        ``methylation_level_wavg``.  Each row represents the average
        prediction profile for that DMR group across all cell types.
    """
    pred_cols = [f"prediction_{i}_wavg" for i in range(num_input_labels)]
    all_cols = pred_cols + ["methylation_level_wavg"]

    valid_profiles = [p for p in pure_profiles if p is not None]
    if not valid_profiles:
        raise ValueError("No valid pure profiles provided; all entries are None.")

    # Collect per-profile DataFrames for the requested split
    dfs: List[pd.DataFrame] = []
    for proportions, subs, uxm_data in valid_profiles:
        if split_key not in subs:
            raise KeyError(
                f"Split key '{split_key}' not found in pure profile subs. "
                f"Available keys: {list(subs.keys())}"
            )
        df = subs[split_key].copy()
        # Ensure consistent index
        df = df.sort_values("dmr_ctype_label").reset_index(drop=True)
        dfs.append(df[["dmr_ctype_label", "dmr_ctype"] + all_cols])

    # Stack and average across all profiles for each DMR row
    combined = pd.concat(dfs, ignore_index=True)
    prior = combined.groupby(["dmr_ctype_label", "dmr_ctype"])[all_cols].mean()
    prior = prior.reset_index().sort_values("dmr_ctype_label").reset_index(drop=True)

    logger.info(
        f"Computed uniform prior matrix with {len(prior)} DMR rows "
        f"from {len(valid_profiles)} pure profiles (split='{split_key}')"
    )
    return prior


def save_uniform_prior(prior: pd.DataFrame, path: str) -> None:
    """Save a uniform prior matrix to an ``.npz`` file.

    Parameters
    ----------
    prior : pd.DataFrame
        Output of :func:`compute_uniform_prior_matrix`.
    path : str
        Destination ``.npz`` file path.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    pred_cols = [c for c in prior.columns if c.startswith("prediction_")]
    all_value_cols = pred_cols
    if "methylation_level_wavg" in prior.columns:
        all_value_cols = all_value_cols + ["methylation_level_wavg"]

    np.savez_compressed(
        path,
        values=prior[all_value_cols].to_numpy(),
        dmr_ctype_labels=prior["dmr_ctype_label"].to_numpy(),
        dmr_ctypes=prior["dmr_ctype"].to_numpy().astype(str),
        column_names=np.array(all_value_cols, dtype=str),
    )
    logger.info(f"Saved uniform prior matrix to {path}")


def load_uniform_prior(path: str) -> pd.DataFrame:
    """Load a uniform prior matrix from an ``.npz`` file.

    Parameters
    ----------
    path : str
        Path to the ``.npz`` file created by :func:`save_uniform_prior`.

    Returns
    -------
    pd.DataFrame
        Reconstructed prior DataFrame with ``dmr_ctype_label``,
        ``dmr_ctype``, and prediction value columns.
    """
    data = np.load(path, allow_pickle=True)
    column_names = list(data["column_names"])
    df = pd.DataFrame(data["values"], columns=column_names)
    df["dmr_ctype_label"] = data["dmr_ctype_labels"]
    df["dmr_ctype"] = data["dmr_ctypes"]

    # Re-order columns for consistency
    meta_cols = ["dmr_ctype_label", "dmr_ctype"]
    df = df[meta_cols + column_names]
    df = df.sort_values("dmr_ctype_label").reset_index(drop=True)

    logger.info(f"Loaded uniform prior matrix ({len(df)} rows) from {path}")
    return df
