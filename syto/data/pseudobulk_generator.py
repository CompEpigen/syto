""" """

from typing import Literal, Dict, Tuple, Any

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
import numpy as np

from syto.modelling.prediction_aggregation import (
    aggregate_predictions_by_grg_optimized,
)


def _sample_reads_per_grg_uniform_multinomial(
    n_samples_per_class: np.ndarray, n_grg: int
) -> np.ndarray:
    """
    For each class i, allocate n_samples_per_class[i] reads across the n_grg GRG groups
    using a multinomial distribution with equal probabilities.

    Simulates natural sequencing noise by randomly distributing each
    cell type's total reads across the available GRGs. This causes
    total depth to fluctuate per locus, but guarantees the local
    proportions correctly center around the global mixture proportions.

    Args:
        n_samples_per_class: An array of shape (n_classes,)
            containing the number of reads to sample for each class.
        n_grg: The number of GR groups to distribute reads across.
    """
    n_classes = len(n_samples_per_class)
    result = np.zeros((n_classes, n_grg), dtype=int)

    # Assumption: A read has an equal baseline probability of landing in any GRG
    pvals = [1.0 / n_grg] * n_grg

    for i in range(n_classes):
        n_samples = n_samples_per_class[i]
        # Sample the number of reads for each GRG using a multinomial distribution
        result[i] = np.random.multinomial(n_samples, pvals)

    return result.astype(np.int64)


def _sample_read_ids_from_grouped_dataframe(
    n_samples_per_class_per_grg: np.ndarray,
    indices_per_class_and_grg: Dict[Tuple[Any, Any], np.ndarray],
    seed: int,
) -> np.ndarray:
    """
    Reproducibly sample read IDs from a grouped dataframe based on the specified
    number of samples per class and GR groups.

    Args:
        n_samples_per_class_per_grg: A 2D array of shape (n_classes, n_gr_groups)
            specifying the number of reads to sample for each (class, GRG) combination.
        indices_per_class_and_grg: A dictionary mapping (class_label, grg_label) tuples
            to arrays of read IDs corresponding to that group in the original dataframe.
        seed: An integer seed for the random number generator to ensure reproducibility.
    """
    n_reads_to_sample = n_samples_per_class_per_grg.sum()
    read_ids = np.empty(n_reads_to_sample, dtype=np.int64)
    current_index = 0
    # Create a fixed random generator for reproducibility
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    # sort index for reproducibility
    for (class_label, grg_label), indices in sorted(indices_per_class_and_grg.items()):
        n_reads_to_sample = n_samples_per_class_per_grg[class_label, grg_label]
        if len(indices) == 0:
            if n_reads_to_sample > 0:
                raise ValueError(
                    f"Cannot sample {n_reads_to_sample} from the group"
                    f"({class_label}, {grg_label}) with size 0."
                )
            continue
        read_ids[current_index : current_index + n_reads_to_sample] = rng.choice(
            indices, size=n_reads_to_sample, replace=True
        )
        current_index += n_reads_to_sample
    return read_ids


class PseudobulkGenerator:
    """
    Class for generating pseudobulk from dataframe of predictions per read.
    """

    @classmethod
    def generate_single_pseudobulk(
        cls,
        n_reads_to_sample: int,
        n_gr_groups: int,
        indices_per_class_and_grg: Dict[Tuple[Any, Any], np.ndarray],
        read_df: pd.DataFrame,
        target_proportions: np.ndarray,
        grg_grouping_columns: list[str],
        columns_to_keep: list[str] = None,
        grg_sampling_type: Literal["uniform_multinomial"] = "uniform_multinomial",
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, pd.DataFrame]:
        """

        Args:
            n_reads_to_sample: The total number of reads to sample for the pseudobulk.
            n_gr_groups: The number of GR groups to distribute reads across.
            indices_per_class_and_grg: A dictionary mapping (class_label, grg_label) tuples
                to arrays of read IDs corresponding to that group in the original dataframe.
            read_df: The original dataframe containing read-level predictions
            target_proportions: An array of of proportions summing to 1, shape (n_classes,).
            grg_sampling_type: The method to use for sampling reads across GR groups.
                 Currently only supports "uniform_multinomial"
        """
        assert np.isclose(
            target_proportions.sum(), 1.0
        ), "target_proportions must sum to 1"
        assert grg_sampling_type in [
            "uniform_multinomial"
        ], f"Unsupported grg_sampling_type: {grg_sampling_type}"
        target_proportions = np.asarray(target_proportions, dtype=np.float64)
        n_classes = len(target_proportions)

        # Compute the number of reads to sample from each class
        n_samples_per_class = np.array(
            [int(n_reads_to_sample * p) for p in target_proportions]
        )
        actual_n_reads_sampled = n_samples_per_class.sum()
        actual_proportions = n_samples_per_class / actual_n_reads_sampled

        # For each class, compute the number of reads to sample from each GR group
        if grg_sampling_type == "uniform_multinomial":
            n_samples_per_class_per_grg = _sample_reads_per_grg_uniform_multinomial(
                n_samples_per_class, n_gr_groups
            )  # shape (n_classes, n_gr_groups)
        else:
            raise ValueError(f"Unsupported grg_sampling_type: {grg_sampling_type}")

        # Sample the read IDS for each class and GR group, and concatenate them into a single array
        seed = np.random.randint(0, 1_000_000)
        read_ids = _sample_read_ids_from_grouped_dataframe(
            n_samples_per_class_per_grg,
            indices_per_class_and_grg,
            seed=seed,
        )  # shape (actual_n_reads_sampled,)

        # Aggregate the reads by GR group into a feature matrix
        aggregated_features = aggregate_predictions_by_grg_optimized(
            read_df.iloc[read_ids], grg_grouping_columns,
            weight_col="NCPGS",
        )
        if columns_to_keep is not None:
            aggregated_features = aggregated_features[columns_to_keep]

        return {
            "actual_proportions": actual_proportions,
            "actual_n_reads_sampled": actual_n_reads_sampled,
            "n_samples_per_class_per_grg": n_samples_per_class_per_grg,
            "read_ids": read_ids,
            "aggregated_features": aggregated_features,
        }
