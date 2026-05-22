"""
Split Rebalancing Utilities

Ensures that every train/valid/test split has all possible combinations
of (dmr_ctype_label, original_label) by redistributing files that appear
in multiple splits.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def rebalance_splits(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_col: str = "original_label",
    dmr_label_col: str = "dmr_ctype_label",
    file_col: str = "file",
    manual_common_labels: Optional[List[int]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Re-distribute files across splits so every split contains all
    ``(dmr_ctype_label, original_label)`` combinations.

    Files that are exclusive to one split stay there.  Files that
    appear in more than one split, or that belong to manually flagged
    labels, are pooled and re-split with stratification by file.

    Parameters
    ----------
    train_data, valid_data, test_data : pd.DataFrame
        The original split DataFrames.  Must contain ``label_col``,
        ``dmr_label_col``, and ``file_col`` columns.
    label_col : str
        Column with the ground-truth cell-type label.
    dmr_label_col : str
        Column with the DMR cell-type label.
    file_col : str
        Column identifying the source file/sample.
    manual_common_labels : list of int, optional
        Label ids whose files should always be treated as common
        (redistributed).  Default: ``[28, 35]``.
    random_state : int
        Seed for reproducible stratified splitting.

    Returns
    -------
    tuple of pd.DataFrame
        ``(train_split, valid_split, test_split)`` with indices reset.
    """
    if manual_common_labels is None:
        manual_common_labels = [28, 35]

    def _informative_files(df: pd.DataFrame) -> set:
        """Files where at least one read has matching label and DMR label."""
        mask = df[dmr_label_col] == df[label_col]
        return set(df.loc[mask, file_col].unique())

    train_file_set = _informative_files(train_data)
    valid_file_set = _informative_files(valid_data)
    test_file_set = _informative_files(test_data)

    # Combine all data for redistribution
    all_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)

    # Files belonging to manually flagged labels
    manual_common_files = set(
        all_data.loc[all_data[label_col].isin(manual_common_labels), file_col].unique()
    )

    # Files exclusive to each split
    train_files = (
        train_file_set.difference(valid_file_set)
        .difference(test_file_set)
        .difference(manual_common_files)
    )
    valid_files = (
        valid_file_set.difference(train_file_set)
        .difference(test_file_set)
        .difference(manual_common_files)
    )
    test_files = (
        test_file_set.difference(train_file_set)
        .difference(valid_file_set)
        .difference(manual_common_files)
    )

    # Files that appear in more than one split, plus manual ones
    common_files = (
        train_file_set.intersection(valid_file_set).intersection(test_file_set)
    ).union(manual_common_files)

    # Build exclusive-split subsets
    train_split = all_data[all_data[file_col].isin(train_files)]
    valid_split = all_data[all_data[file_col].isin(valid_files)]
    test_split = all_data[all_data[file_col].isin(test_files)]

    # Stratified re-split of the common pool
    common_subset = all_data[all_data[file_col].isin(common_files)]

    if len(common_subset) > 0:
        common_train, common_test = train_test_split(
            common_subset,
            test_size=0.333,
            stratify=np.array(common_subset[file_col]),
            random_state=random_state,
        )
        common_train, common_valid = train_test_split(
            common_train,
            test_size=0.5,
            stratify=np.array(common_train[file_col]),
            random_state=random_state,
        )

        train_split = pd.concat([train_split, common_train], ignore_index=True)
        valid_split = pd.concat([valid_split, common_valid], ignore_index=True)
        test_split = pd.concat([test_split, common_test], ignore_index=True)

    return train_split, valid_split, test_split
