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
            pure_profiles.append((proportions_full, subs, uxm_data))
        except Exception as e:
            logger.warning(f"Failed to generate pure profile for cell type {i}: {e}")
            pure_profiles.append(None)

    return pure_profiles
