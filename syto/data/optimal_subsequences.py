"""
Module for selecting optimal subsequences from DNA reads based on methylation patterns.

"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm


def select_optimal_subsequence(
    input_ids: str,
    methylation_ids: str,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    stride: int = 50,
    entropy_weight_factor: float = 0.1,
) -> Tuple[str, str, int, float]:
    """
    Select optimal subsequence from DNA read based on methylation patterns.

    Parameters:
    -----------
    input_ids : str
        DNA sequence string
    methylation_ids : str
        Methylation status string (0=unmethylated, 1=methylated, 2=irrelevant/unknown)
    target_read_length : int
        Desired length of the subsequence
    selection_criteria : str
        Either "counts" (maximize labeled CpGs) or "entropy" (minimize entropy)
    min_labeled_cpgs : int
        Minimum number of labeled CpGs required for entropy calculation
    stride:int
        Stride of a sliding window for approximation (bigger stride --> more efficient, less accurated)
    entropy_weight_factor : float
        Factor to balance entropy vs count of labeled CpGs (for entropy mode)

    Returns:
    --------
    Tuple[str, str, int, float]
        (selected_dna_sequence, selected_methylation_sequence, labeled_cpg_count, entropy_score)
    """

    # Validate inputs
    if len(input_ids) != len(methylation_ids):
        raise ValueError("DNA sequence and methylation sequence must have equal length")

    # if target_read_length > len(input_ids):
    #     raise ValueError("Target read length cannot exceed sequence length")

    if selection_criteria not in ["counts", "entropy"]:
        raise ValueError("Selection criteria must be either 'counts' or 'entropy'")

    read_length = len(input_ids)
    best_score = float("-inf") if selection_criteria == "counts" else float("inf")
    best_start_idx = 0
    best_stats = (0, float("inf"))  # (labeled_count, entropy)
    if read_length > target_read_length:
        # Slide window across the sequence
        for start_idx in range(0, read_length - target_read_length + 1, stride):
            end_idx = start_idx + target_read_length

            # Extract subsequence
            subseq_methylation = methylation_ids[start_idx:end_idx]

            # Count labeled methylation sites (excluding '2')
            labeled_positions = [pos for pos in subseq_methylation if pos != "2"]
            labeled_count = len(labeled_positions)

            # Skip if no labeled positions
            if labeled_count == 0:
                continue

            # Calculate entropy for labeled positions
            if labeled_count >= min_labeled_cpgs:
                # Convert to numeric for entropy calculation
                numeric_labels = [int(pos) for pos in labeled_positions]
                unique, counts = np.unique(numeric_labels, return_counts=True)
                probabilities = counts / len(numeric_labels)
                entropy_score = entropy(probabilities, base=2)  # Use base 2 for bits
            else:
                entropy_score = float(
                    "inf"
                )  # Penalize sequences with too few labeled CpGs

            # Determine score based on selection criteria
            if selection_criteria == "counts":
                score = labeled_count
                is_better = score > best_score
            else:  # entropy
                if labeled_count < min_labeled_cpgs:
                    # Heavily penalize sequences with insufficient labeled CpGs
                    score = float("inf")
                else:
                    # Lower entropy is better, but also consider count of labeled CpGs
                    # Add small penalty for fewer labeled CpGs to break ties
                    score = entropy_score + entropy_weight_factor * (
                        1.0 / labeled_count
                    )
                is_better = score < best_score

            # Update best subsequence if this one is better
            if is_better:
                best_score = score
                best_start_idx = start_idx
                best_stats = (labeled_count, entropy_score)

        # Extract the best subsequence
        best_end_idx = best_start_idx + target_read_length
        selected_dna = input_ids[best_start_idx:best_end_idx]
        selected_methylation = methylation_ids[best_start_idx:best_end_idx]
    else:  # delegates to vectorized version if sequence is shorter than target
        return select_optimal_subsequence_vectorized(
            input_ids,
            methylation_ids,
            target_read_length,
            selection_criteria,
            min_labeled_cpgs,
            stride,
            entropy_weight_factor,
        )

    return selected_dna, selected_methylation, best_stats[0], best_stats[1]


def select_optimal_subsequence_rolling_window(
    input_ids: str,
    methylation_ids: str,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    stride: int = 50,
    entropy_weight_factor: float = 0.1,
) -> Tuple[str, str, int, float]:
    """
    Ultra-optimized version using rolling window statistics for maximum efficiency.
    Best for very long sequences with small strides.
    """

    if len(input_ids) != len(methylation_ids):
        raise ValueError("DNA sequence and methylation sequence must have equal length")

    if selection_criteria not in ["counts", "entropy"]:
        raise ValueError("Selection criteria must be either 'counts' or 'entropy'")

    read_length = len(input_ids)

    if read_length <= target_read_length:
        return select_optimal_subsequence_vectorized(
            input_ids,
            methylation_ids,
            target_read_length,
            selection_criteria,
            min_labeled_cpgs,
            stride,
            entropy_weight_factor,
        )

    # Convert to numpy arrays
    meth_array = np.array(list(methylation_ids))

    # Create binary masks for efficient counting
    is_labeled = meth_array != "2"
    is_methylated = meth_array == "1"
    is_unmethylated = meth_array == "0"

    best_score = float("-inf") if selection_criteria == "counts" else float("inf")
    best_start_idx = 0
    best_stats = (0, float("inf"))

    # Use stride-based sampling for efficiency
    for start_idx in range(0, read_length - target_read_length + 1, stride):
        end_idx = start_idx + target_read_length

        # Fast counting using pre-computed masks
        window_labeled = is_labeled[start_idx:end_idx]
        labeled_count = np.sum(window_labeled)

        if labeled_count == 0:
            continue

        # Calculate entropy efficiently
        if labeled_count >= min_labeled_cpgs:
            window_methylated = is_methylated[start_idx:end_idx]
            window_unmethylated = is_unmethylated[start_idx:end_idx]

            meth_count = np.sum(window_methylated)
            unmeth_count = np.sum(window_unmethylated)

            # Fast entropy calculation for binary case
            if meth_count > 0 and unmeth_count > 0:
                p_meth = meth_count / labeled_count
                p_unmeth = unmeth_count / labeled_count
                entropy_score = -(
                    p_meth * np.log2(p_meth) + p_unmeth * np.log2(p_unmeth)
                )
            else:
                entropy_score = 0.0  # All same class = no entropy
        else:
            entropy_score = float("inf")

        # Score calculation
        if selection_criteria == "counts":
            score = labeled_count
            is_better = score > best_score
        else:
            if labeled_count < min_labeled_cpgs:
                score = float("inf")
            else:
                score = entropy_score + entropy_weight_factor * (1.0 / labeled_count)
            is_better = score < best_score

        if is_better:
            best_score = score
            best_start_idx = start_idx
            best_stats = (labeled_count, entropy_score)

    # Extract best subsequence
    best_end_idx = best_start_idx + target_read_length
    selected_dna = input_ids[best_start_idx:best_end_idx]
    selected_methylation = methylation_ids[best_start_idx:best_end_idx]

    return selected_dna, selected_methylation, best_stats[0], best_stats[1]


def extract_optimal_subsequence_dataframe_chunked(
    df: pd.DataFrame,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    entropy_weight_factor: float = 0.1,
    stride: int = 50,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """
    Memory-efficient version that processes data in chunks.
    Recommended for very large datasets.
    """

    results = []

    for chunk_start in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end]

        chunk_results = []
        for idx, row in chunk_df.iterrows():
            try:
                selected_dna, selected_meth, labeled_count, entropy_score = (
                    select_optimal_subsequence_rolling_window(
                        row["input_ids"],
                        row["methylation_ids"],
                        target_read_length,
                        selection_criteria,
                        min_labeled_cpgs,
                        stride,
                        entropy_weight_factor,
                    )
                )

                result = {
                    "selected_input_ids": selected_dna,
                    "selected_methylation_ids": selected_meth,
                    "labeled_cpg_count": labeled_count,
                    "entropy_score": entropy_score,
                    "selection_criteria": selection_criteria,
                }

                for col in chunk_df.columns:
                    if col not in ["input_ids", "methylation_ids"]:
                        result[col] = row[col]

                chunk_results.append(result)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        results.extend(chunk_results)

        # Optional: Clear memory periodically
        if chunk_start % (chunk_size * 10) == 0:
            import gc

            gc.collect()

    return pd.DataFrame(results)


def select_optimal_subsequence_vectorized(
    input_ids: str,
    methylation_ids: str,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    stride: int = 50,
    entropy_weight_factor: float = 0.1,
) -> Tuple[str, str, int, float]:
    """
    Highly optimized version using numpy vectorization and sliding window optimizations.
    """

    # Validate inputs
    if len(input_ids) != len(methylation_ids):
        raise ValueError("DNA sequence and methylation sequence must have equal length")

    if selection_criteria not in ["counts", "entropy"]:
        raise ValueError("Selection criteria must be either 'counts' or 'entropy'")

    read_length = len(input_ids)

    # Handle case where sequence is shorter than target
    if read_length <= target_read_length:
        labeled_positions = [pos for pos in methylation_ids if pos != "2"]
        labeled_count = len(labeled_positions)

        if labeled_count >= min_labeled_cpgs:
            numeric_labels = np.array([int(pos) for pos in labeled_positions])
            _, counts = np.unique(numeric_labels, return_counts=True)
            probabilities = counts / len(numeric_labels)
            entropy_score = entropy(probabilities, base=2)
        else:
            entropy_score = float("inf")

        return input_ids, methylation_ids, labeled_count, entropy_score

    # Convert methylation string to numpy array for faster operations
    meth_array = np.array(list(methylation_ids), dtype="U1")

    # Pre-compute all window positions
    window_starts = np.arange(0, read_length - target_read_length + 1, stride)

    if len(window_starts) == 0:
        # Fallback if stride is too large
        window_starts = np.array([0])

    best_score = float("-inf") if selection_criteria == "counts" else float("inf")
    best_start_idx = 0
    best_stats = (0, float("inf"))

    # Vectorized window processing
    for start_idx in window_starts:
        end_idx = start_idx + target_read_length

        # Extract window using numpy slicing
        window_meth = meth_array[start_idx:end_idx]

        # Count labeled positions using vectorized operations
        labeled_mask = window_meth != "2"
        labeled_count = np.sum(labeled_mask)

        if labeled_count == 0:
            continue

        # Calculate entropy using vectorized operations
        if labeled_count >= min_labeled_cpgs:
            labeled_values = window_meth[labeled_mask].astype(int)
            unique, counts = np.unique(labeled_values, return_counts=True)
            probabilities = counts / labeled_count
            entropy_score = entropy(probabilities, base=2)
        else:
            entropy_score = float("inf")

        # Determine score based on selection criteria
        if selection_criteria == "counts":
            score = labeled_count
            is_better = score > best_score
        else:  # entropy
            if labeled_count < min_labeled_cpgs:
                score = float("inf")
            else:
                score = entropy_score + entropy_weight_factor * (1.0 / labeled_count)
            is_better = score < best_score

        # Update best subsequence if this one is better
        if is_better:
            best_score = score
            best_start_idx = start_idx
            best_stats = (labeled_count, entropy_score)

    # Extract the best subsequence
    best_end_idx = best_start_idx + target_read_length
    selected_dna = input_ids[best_start_idx:best_end_idx]
    selected_methylation = methylation_ids[best_start_idx:best_end_idx]

    return selected_dna, selected_methylation, best_stats[0], best_stats[1]


def extract_optimal_subsequence_dataframe(
    df: pd.DataFrame,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    entropy_weight_factor: float = 0.1,
    stride: int = 50,
) -> pd.DataFrame:
    """
    Process entire dataframe to select optimal subsequences for all reads.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'input_ids' and 'methylation_ids' columns
    target_read_length : int
        Desired length of subsequences
    selection_criteria : str
        Either "counts" or "entropy"
    min_labeled_cpgs : int
        Minimum labeled CpGs for entropy calculation
    entropy_weight_factor : float
        Weight factor for entropy calculation

    Returns:
    --------
    pd.DataFrame
        DataFrame with original columns plus selected subsequences and statistics
    """

    results = []

    for idx, row in tqdm(df.iterrows()):
        try:
            selected_dna, selected_meth, labeled_count, entropy_score = (
                select_optimal_subsequence(
                    row["input_ids"],
                    row["methylation_ids"],
                    target_read_length,
                    selection_criteria,
                    min_labeled_cpgs,
                    stride,
                    entropy_weight_factor,
                )
            )

            result = {
                "original_input_ids": row["input_ids"],
                "original_methylation_ids": row["methylation_ids"],
                "selected_input_ids": selected_dna,
                "selected_methylation_ids": selected_meth,
                "labeled_cpg_count": labeled_count,
                "entropy_score": entropy_score,
                "selection_criteria": selection_criteria,
            }

            # Add other columns from original dataframe
            for col in df.columns:
                if col not in ["input_ids", "methylation_ids"]:
                    result[col] = row[col]

            results.append(result)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    return pd.DataFrame(results)
