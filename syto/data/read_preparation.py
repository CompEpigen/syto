"""
Read Preparation Utilities

Adds derived columns required by the pseudo-bulk generator and optionally
trims reads to the genomic span of overlapping atlas regions.
"""

from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from syto.data.atlases.abstract_atlas import AbstractAtlas


def prepare_splits_for_pseudobulk(
    splits: Dict[str, pd.DataFrame],
    target_columns: Optional[List[str]] = None,
    num_labels: int = 39,
    trim: bool = True,
    atlas: Optional["AbstractAtlas"] = None
) -> Dict[str, pd.DataFrame]:
    """Add derived columns needed by the pseudobulk generator and optionally
    overlap reads with atlas regions.

    The following columns are added to every split DataFrame:

    * ``is_cell_informative_region`` — ``original_label == dmr_ctype_label``
    * ``prediction`` — argmax of the prediction columns
    * ``confidence`` — max of the prediction columns
    * ``total_marked_cpgs`` — ``methylated_CpGs + unmethylated_CpGs``
    * ``direction`` — set to ``"U"`` (placeholder)
    * ``read_start`` — alias for ``trimmed_start``
    * ``read_end`` — alias for ``trimmed_end``
    * ``chromosome`` — rename of ``chr`` if present

    Parameters
    ----------
    splits : Dict[str, pd.DataFrame]
        Input split DataFrames.
    target_columns : list of str, optional
        Prediction column names.  Default: ``prediction_0 … prediction_{num_labels-1}``.
    num_labels : int
        Number of labels.  Default: 39.
    trim : bool
        If ``True`` (default), overlap reads with atlas regions and clip each
        read to its overlapping region's boundaries.  Requires ``atlas``.
    atlas : AbstractAtlas, optional
        Pre-loaded atlas object.  Required when ``trim=True``.
    seq_column, methylation_pattern_column : str
        Column names forwarded to :meth:`AbstractAtlas.trim_reads`.
    """
    if target_columns is None:
        target_columns = [f"prediction_{i}" for i in range(num_labels)]

    splits_copy = {name: df.copy() for name, df in splits.items()}

    for split_name, df in splits_copy.items():
        if "original_label" in df.columns and "dmr_ctype_label" in df.columns:
            df["is_cell_informative_region"] = (
                df["original_label"] == df["dmr_ctype_label"]
            )

        available_pred_cols = [c for c in target_columns if c in df.columns]
        if available_pred_cols:
            pred_array = df[available_pred_cols].to_numpy()
            df["prediction"] = np.argmax(pred_array, axis=1)
            df["confidence"] = np.max(pred_array, axis=1)

        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df["total_marked_cpgs"] = df["methylated_CpGs"] + df["unmethylated_CpGs"]

        df["direction"] = "U"

        if "trimmed_start" in df.columns:
            df["read_start"] = df["trimmed_start"]
        if "trimmed_end" in df.columns:
            df["read_end"] = df["trimmed_end"]

        if "chr" in df.columns and "chromosome" not in df.columns:
            df.rename(columns={"chr": "chromosome"}, inplace=True)

    if trim:
        if atlas is None:
            raise ValueError("atlas is required when trim=True")

        for split_name, df in splits_copy.items():
            df_sorted = df.sort_values(
                by=["chromosome", "read_start", "read_end"], ascending=True
            ).reset_index(drop=True)
            overlapped = atlas.overlap_reads(
                df_sorted
            )
            splits_copy[split_name] = atlas.trim_reads(
                overlapped
            )

    return splits_copy
