"""
Read Preparation Utilities

Adds derived columns required by the pseudo-bulk generator and
optionally runs UXM read-preparation on each split.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def prepare_splits_for_pseudobulk(
    splits: Dict[str, pd.DataFrame],
    labels_dict: Dict[int, str],
    target_columns: Optional[List[str]] = None,
    num_labels: int = 39,
    generate_uxm_inputs: bool = True,
    atlas_path: Optional[str] = None,
    cell_type_match_dict: Optional[dict] = None,
    seq_column: str = "seq",
    methylation_pattern_column: str = "pattern",
    read_name_column: str = "read_name",
) -> Dict[str, pd.DataFrame]:
    """
    Add derived columns needed by ``generate_pseudo_bulk*`` and
    optionally run ``prepare_reads_for_uxm``.

    The following columns are added to every split DataFrame:

    * ``is_cell_informative_region`` — whether ``original_label == dmr_ctype_label``
    * ``prediction`` — argmax of the prediction columns
    * ``confidence`` — max of the prediction columns
    * ``total_marked_cpgs`` — ``methylated_CpGs + unmethylated_CpGs``
    * ``direction`` — set to ``"U"`` (placeholder)
    * ``read_start`` — alias for ``trimmed_start``
    * ``read_end`` — alias for ``trimmed_end``
    * ``chromosome`` — rename of ``chr`` if present

    Parameters
    ----------
    train_split, valid_split, test_split : pd.DataFrame
        Input split DataFrames. (Deprecated parameters removed)
    splits : Dict[str, pd.DataFrame]
        Input split DataFrames.
    labels_dict : dict
        ``{int_label: cell_type_name}`` mapping.
    target_columns : list of str, optional
        Prediction column names.  Default: ``[prediction_0 .. prediction_{num_labels-1}]``.
    num_labels : int
        Number of labels.  Default: 39.
    generate_uxm_inputs : bool
        If ``True``, run ``prepare_reads_for_uxm`` on each split.
        Requires ``atlas_path`` and ``cell_type_match_dict``.
    atlas_path : str, optional
        Path to the UXM atlas TSV.  Required when ``generate_uxm_inputs=True``.
    cell_type_match_dict : dict, optional
        Cell-type name mapping.  Required when ``generate_uxm_inputs=True``.
    seq_column, methylation_pattern_column, read_name_column : str
        Column names forwarded to ``prepare_reads_for_uxm``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        ``{"train": df_train, "valid": df_valid, ...}`` — enriched DataFrames.
    """
    if target_columns is None:
        target_columns = [f"prediction_{i}" for i in range(num_labels)]

    splits_copy = {name: df.copy() for name, df in splits.items()}

    for split_name, df in splits_copy.items():
        # Derived columns from predictions
        if "original_label" in df.columns and "dmr_ctype_label" in df.columns:
            df["is_cell_informative_region"] = (
                df["original_label"] == df["dmr_ctype_label"]
            )

        available_pred_cols = [c for c in target_columns if c in df.columns]
        if available_pred_cols:
            pred_array = df[available_pred_cols].to_numpy()
            df["prediction"] = np.argmax(pred_array, axis=1)
            df["confidence"] = np.max(pred_array, axis=1)

        # CpG count
        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df["total_marked_cpgs"] = df["methylated_CpGs"] + df["unmethylated_CpGs"]

        # Direction placeholder
        df["direction"] = "U"

        # Positional aliases
        if "trimmed_start" in df.columns:
            df["read_start"] = df["trimmed_start"]
        if "trimmed_end" in df.columns:
            df["read_end"] = df["trimmed_end"]

        # Chromosome column rename
        if "chr" in df.columns and "chromosome" not in df.columns:
            df.rename(columns={"chr": "chromosome"}, inplace=True)

    if generate_uxm_inputs:
        if atlas_path is None:
            raise ValueError("atlas_path is required when generate_uxm_inputs=True")
        if cell_type_match_dict is None:
            raise ValueError(
                "cell_type_match_dict is required when generate_uxm_inputs=True"
            )

        from syto.deconvolution.uxm import load_atlas, prepare_reads_for_uxm

        atlas, _ref_cells = load_atlas(atlas_path)
        atlas = atlas.sort_values(
            by=["chr", "start", "end"], ascending=True
        ).reset_index(drop=True)

        prepared = {}
        for split_name, df in splits_copy.items():
            df_sorted = df.sort_values(
                by=["chromosome", "read_start", "read_end"], ascending=True
            ).reset_index(drop=True)

            uxm_df = prepare_reads_for_uxm(
                df_sorted,
                atlas,
                labels_dict,
                cell_type_match_dict,
                seq_column=seq_column,
                debug=False,
                methylation_pattern_column=methylation_pattern_column,
                read_name_column=read_name_column,
                progress_prefix=f"[{split_name.capitalize()}] ",
            )
            prepared[split_name] = uxm_df
        return prepared

    return splits_copy
