"""Pseudobulk generation module.

This module provides functionality for generating pseudobulk samples from
read-level prediction dataframes. It supports:
- Crash-resilient batch processing with checkpointing
- Parallel generation using Dask
- HDF5 output format with full reproducibility metadata
"""

from typing import Literal, Dict, Tuple, Any, Union, List
from pathlib import Path
import logging

import dask
from dask import delayed
import pandas as pd
import numpy as np

from syto.modelling.prediction_aggregation import (
    aggregate_by_grg_from_np_arrays,
    fill_in_missing_gr_groups,
)
from syto.data.hdf5_utils import (
    PseudobulkResult,
    PureProfileResult,
    GenerationMetadata,
    GenerationParameters,
    CheckpointManager,
    HDF5BatchWriter,
    HDF5ConsolidationWriter,
)

_module_logger = logging.getLogger(__name__)


def build_target_columns(num_prediction_classes: int = 39) -> List[str]:
    """Build the default list of target columns for GR-aggregated output.

    Args:
        num_prediction_classes: Number of classifier output classes.

    Returns:
        List of column names to keep in aggregated output.
    """
    cols = ["dmr_ctype_label", "dmr_ctype"]
    cols += [f"prediction_{i}_wavg" for i in range(num_prediction_classes)]
    cols += ["methylation_level_wavg", "total_weight", "n_reads", "chromosome", "label"]
    return cols


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

    Returns:
        Array of shape (n_classes, n_grg) with allocated read counts.
    """
    n_classes = len(n_samples_per_class)
    result = np.zeros((n_classes, n_grg), dtype=np.int64)

    # Use numpy array for pvals (faster than list)
    pvals = np.full(n_grg, 1.0 / n_grg)

    for i in range(n_classes):
        result[i] = np.random.multinomial(n_samples_per_class[i], pvals)

    return result


def _sample_read_ids_from_grouped_dataframe(
    n_samples_per_class_per_grg: np.ndarray,
    indices_per_class_and_grg: Dict[Tuple[int, int], np.ndarray],
    seed: int,
) -> np.ndarray:
    """
    Reproducibly sample read IDs from a grouped dataframe based on the specified
    number of samples per class and GR groups.

    Args:
        n_samples_per_class_per_grg: A 2D array of shape (n_classes, n_gr_groups)
            specifying the number of reads to sample for each (class_index, grg_index)
            combination.
        indices_per_class_and_grg: A dictionary mapping (class_index, grg_index) tuples
            to arrays of row indices corresponding to that group in the original dataframe.
            Keys must be integer indices (0-based), not raw label values.
        seed: An integer seed for the random number generator to ensure reproducibility.
    """
    n_reads_to_sample = n_samples_per_class_per_grg.sum()
    read_ids = np.full(n_reads_to_sample, -1, dtype=np.int64)
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
    """Class for generating pseudobulk samples from read-level predictions.

    This class orchestrates the generation of pseudobulk samples across multiple
    data splits (train/valid/test), with crash-resilient checkpointing and
    parallel execution using Dask.

    Attributes:
        splits_df: Dictionary mapping split names to DataFrames with read-level predictions.
        output_directory: Directory for intermediate files and final HDF5 output.
        target_proportions_per_split: Dictionary mapping split names to target proportion arrays.
        batch_size: Number of pseudobulks to generate per batch.
        n_workers: Number of parallel workers for Dask.
        metadata: Generation metadata (classifier, labeling scheme, etc.).
        parameters: Generation parameters (cell types, substitution method, etc.).
    """

    def __init__(
        self,
        splits_df: Dict[str, pd.DataFrame],
        output_directory: Union[str, Path],
        target_proportions_per_split: Dict[str, np.ndarray],
        batch_size: int,
        n_workers: int,
        metadata: GenerationMetadata,
        parameters: GenerationParameters,
        n_reads_to_sample: int = 475_000,
        class_label_column: str = "original_label",
        grg_label_column: str = "dmr_ctype_label",
        grg_grouping_columns: List[str] = None,
        columns_to_keep: List[str] = None,
        logger: logging.Logger = _module_logger,
    ):
        """Initialize the PseudobulkGenerator.

        Args:
            splits_df: Dictionary mapping split names to DataFrames containing
                read-level predictions. Expected columns include predictions,
                NCPGS, and grouping columns.
            output_directory: Directory for output files (batches, checkpoints, final HDF5).
            target_proportions_per_split: Dictionary mapping split names to 2D arrays
                of shape (n_pseudobulks, n_classes) where each row sums to 1.
            batch_size: Number of pseudobulks per batch file.
            n_workers: Number of Dask workers for parallel generation.
            metadata: Metadata about the generation (classifier, labeling scheme, etc.).
            parameters: Generation parameters (cell types mapping, substitution method, etc.).
            n_reads_to_sample: Total reads to sample per pseudobulk.
            class_label_column: Column name containing the class labels for each read.
            grg_label_column: Column name containing the GRG labels for grouping reads.
                Must be present in grg_grouping_columns.
            grg_grouping_columns: Columns to group by for GRG aggregation.
                Defaults to ["dmr_ctype_label", "dmr_ctype"].
            columns_to_keep: Columns to keep in aggregated output. If None, uses
                the default columns from build_target_columns()
            logger: Logger instance.
        """
        # Validate target proportions
        for split_name, proportions in target_proportions_per_split.items():
            assert (
                proportions.ndim == 2
            ), f"target_proportions for {split_name} must be a 2D array"
            assert np.allclose(
                proportions.sum(axis=1), 1.0
            ), f"Each row of target_proportions for {split_name} must sum to 1"

        self.splits_df = splits_df
        self.output_directory = Path(output_directory)
        self.target_proportions_per_split = target_proportions_per_split
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.metadata = metadata
        self.parameters = parameters
        self.n_reads_to_sample = n_reads_to_sample
        self.class_label_column = class_label_column
        self.grg_label_column = grg_label_column
        self.grg_grouping_columns = grg_grouping_columns or [
            "dmr_ctype_label",
            "dmr_ctype",
        ]
        self.n_classes = len(parameters.cell_types_mapping)
        self.n_gr_groups = len(parameters.gr_groups_mapping)
        # Default to legacy target columns if not specified
        self.columns_to_keep = columns_to_keep or build_target_columns(self.n_classes)
        self.logger = logger

        # Validate that grg_label_column is in grg_grouping_columns
        assert self.grg_label_column in self.grg_grouping_columns, (
            f"grg_label_column '{self.grg_label_column}' must be in "
            f"grg_grouping_columns {self.grg_grouping_columns}"
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_directory, self.logger)

        # Pre-compute indices and numpy arrays for each split
        self._precompute_group_indices()
        self._preextract_numpy_arrays()

        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _precompute_group_indices(self) -> None:
        """Pre-compute group indices for each split for efficient sampling.

        Converts (class_label_value, grg_label_value) from the data to
        (class_index, grg_index) using the cell_types_mapping and gr_groups_mapping.
        This ensures the indices dict keys match the array indices used in
        n_samples_per_class_per_grg.
        """
        self._indices_per_split: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}

        # Build reverse mappings: label_value -> index
        # cell_types_mapping is {name: index}, we need to map from class_label_column values
        # For Loyfer data, class labels are typically integers matching the index
        # gr_groups_mapping is {grg_label: index}
        grg_to_index = self.parameters.gr_groups_mapping

        for split_name, df in self.splits_df.items():
            # Group by (class_label, grg_label) and get indices directly
            grouped = df.groupby(
                [self.class_label_column, self.grg_label_column], sort=False
            )
            # .indices returns a dict mapping group keys to numpy arrays of row indices
            raw_indices_dict = grouped.indices

            # Convert (class_label_value, grg_label_value) -> (class_index, grg_index)
            indices_dict: Dict[Tuple[int, int], np.ndarray] = {}
            for (class_label, grg_label), row_indices in raw_indices_dict.items():
                # class_label is already an integer index (original_label column)
                class_index = int(class_label)
                # grg_label needs to be converted using the mapping
                # Handle both string and numeric grg labels
                grg_key = (
                    str(grg_label) if str(grg_label) in grg_to_index else grg_label
                )
                if grg_key in grg_to_index:
                    grg_index = grg_to_index[grg_key]
                else:
                    # If grg_label is already an integer index, use it directly
                    grg_index = int(grg_label)
                indices_dict[(class_index, grg_index)] = row_indices

            self._indices_per_split[split_name] = indices_dict

            self.logger.debug(
                "Split %s: %d groups of (%s, %s)",
                split_name,
                len(indices_dict),
                self.class_label_column,
                self.grg_label_column,
            )

    def _preextract_numpy_arrays(self) -> None:
        """Pre-extract numpy arrays from DataFrames for faster aggregation.

        Extracts GR group indices, weights, and prediction columns as numpy arrays
        for each split. This avoids DataFrame operations during pseudobulk generation.
        """
        self._numpy_arrays_per_split: Dict[str, Dict[str, np.ndarray]] = {}

        # Build list of prediction columns
        pred_cols = [f"prediction_{i}" for i in range(self.n_classes)]

        for split_name, df in self.splits_df.items():
            arrays = {
                "gr_group_idx_array": df[self.grg_label_column].values.astype(np.int64),
                "weight_array": df["NCPGS"].values.astype(np.float64),
                # Store as Fortran-order for faster column access
                "pred_matrix": np.asfortranarray(
                    df[pred_cols].values.astype(np.float64)
                ),
            }
            self._numpy_arrays_per_split[split_name] = arrays

            self.logger.debug(
                "Split %s: pre-extracted arrays (gr_group_idx: %s, weight: %s, pred: %s)",
                split_name,
                arrays["gr_group_idx_array"].shape,
                arrays["weight_array"].shape,
                arrays["pred_matrix"].shape,
            )

    def run(self) -> Path:
        """Generate pseudobulk samples for all splits and consolidate into HDF5.

        This method orchestrates the full generation workflow:
        1. Check for existing checkpoint and resume if possible
        2. For each split (train → valid → test):
           a. Generate pseudobulk batches in parallel using Dask
           b. Generate pure profiles
           c. Mark split as completed
        3. Consolidate all batches into final HDF5 file
        4. Cleanup intermediate files

        Returns:
            Path to the final consolidated HDF5 file.
        """
        # Check for existing config or create new one
        config = self.checkpoint_manager.load_config()
        if config is not None:
            # Validate parameters match
            current_hash = self.parameters.to_hash()
            if config.parameters_hash != current_hash:
                raise ValueError(
                    f"Parameter mismatch on resume. "
                    f"Expected hash {config.parameters_hash}, got {current_hash}. "
                    f"Delete checkpoint files to start fresh."
                )
            self.logger.info(
                "Resuming generation from checkpoint. Completed splits: %s",
                config.splits_completed,
            )
        else:
            # Create new config
            n_pseudobulks_per_split = {
                split_name: len(proportions)
                for split_name, proportions in self.target_proportions_per_split.items()
            }
            config = self.checkpoint_manager.create_config(
                parameters_hash=self.parameters.to_hash(),
                splits_order=list(self.splits_df.keys()),
                n_pseudobulks_per_split=n_pseudobulks_per_split,
            )
            self.logger.info(
                "Starting new generation with %s pseudobulks", n_pseudobulks_per_split
            )

        # Dictionary to hold pure profiles for consolidation
        pure_profiles: Dict[str, PureProfileResult] = {}

        # Generate pseudobulk for each split
        for split_name in config.splits_order:
            if split_name in config.splits_completed:
                self.logger.info("Skipping completed split: %s", split_name)
                # Load pure profiles if they exist
                # TODO: Load from batch files during consolidation
                continue

            self.logger.info("Generating pseudobulk for split: %s", split_name)

            # Generate pseudobulks for this split
            self._generate_single_split(split_name)

            # Generate pure profiles for this split
            pure_profile = self._generate_pure_profiles(split_name)
            pure_profiles[split_name] = pure_profile

            # Mark split as completed
            self.checkpoint_manager.mark_split_completed(split_name)
            self.logger.info("Completed split: %s", split_name)

        # Consolidate all batches into final HDF5
        self.logger.info("Consolidating batches into final HDF5...")
        output_path = self.output_directory / "pseudobulk.h5"
        consolidator = HDF5ConsolidationWriter(output_path, self.logger)
        final_path = consolidator.consolidate(
            checkpoint_manager=self.checkpoint_manager,
            metadata=self.metadata,
            parameters=self.parameters,
            input_dfs=self.splits_df,
            pure_profiles=pure_profiles,
        )

        # Cleanup
        self.logger.info("Cleaning up intermediate files...")
        self.checkpoint_manager.cleanup_all()

        self.logger.info("Generation complete: %s", final_path)
        return final_path

    def _generate_single_split(self, split_name: str) -> None:
        """Generate all pseudobulk batches for a single split using Dask.

        Args:
            split_name: Name of the split to generate (e.g., 'train', 'valid', 'test').
        """
        split_df = self.splits_df[split_name]
        target_proportions = self.target_proportions_per_split[split_name]
        n_pseudobulks = len(target_proportions)

        # Check for existing checkpoint or create new one
        checkpoint = self.checkpoint_manager.load_split_checkpoint(split_name)
        if checkpoint is None:
            checkpoint = self.checkpoint_manager.create_split_checkpoint(
                split_name, n_pseudobulks, self.batch_size
            )

        # Get missing batches
        missing_batches = self.checkpoint_manager.get_missing_batches(split_name)
        if not missing_batches:
            self.logger.info("All batches complete for split %s", split_name)
            return

        self.logger.info(
            "Generating %d batches for split %s (%d pseudobulks total)",
            len(missing_batches),
            split_name,
            n_pseudobulks,
        )

        # Initialize batch writer
        batches_dir = self.checkpoint_manager.get_split_batches_dir(split_name)
        batch_writer = HDF5BatchWriter(batches_dir, self.logger)

        # Get pre-computed indices for this split
        indices_dict = self._indices_per_split[split_name]

        # Process batches using Dask
        for batch_idx in missing_batches:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_pseudobulks)

            self.logger.debug(
                "Generating batch %d: pseudobulks %d-%d",
                batch_idx,
                start_idx,
                end_idx - 1,
            )

            # Create delayed tasks for each pseudobulk in the batch
            delayed_results = []
            numpy_arrays = self._numpy_arrays_per_split[split_name]
            for idx in range(start_idx, end_idx):
                result = delayed(self.generate_single_pseudobulk)(
                    n_reads_to_sample=self.n_reads_to_sample,
                    n_gr_groups=self.n_gr_groups,
                    n_classes=self.n_classes,
                    indices_per_class_and_grg=indices_dict,
                    numpy_arrays=numpy_arrays,
                    target_proportions=target_proportions[idx],
                    grg_grouping_column=self.grg_label_column,
                    columns_to_keep=self.columns_to_keep,
                    grg_sampling_type=self.parameters.gr_sampling_method,
                    index=idx,
                )
                delayed_results.append(result)

            # Compute all results in parallel
            with dask.config.set(scheduler="threads", num_workers=self.n_workers):
                results: List[PseudobulkResult] = dask.compute(*delayed_results)

            # Write batch atomically
            batch_writer.write_batch(batch_idx, list(results))

            # Mark batch as completed
            self.checkpoint_manager.mark_batch_completed(split_name, batch_idx)

            self.logger.info(
                "Completed batch %d/%d for split %s",
                batch_idx,
                checkpoint.total_batches - 1,
                split_name,
            )

    def _generate_pure_profiles(self, split_name: str) -> PureProfileResult:
        """Generate pure profiles (100% single cell type) for a split.

        This follows the same algorithm as `syto.data.pure_profile_generation`:
        for each cell type, create a pseudobulk where 100% of reads come from
        that cell type, then aggregate predictions by GRG.

        Args:
            split_name: Name of the split.

        Returns:
            PureProfileResult containing feature matrices and uniform prior.
        """
        self.logger.info("Generating pure profiles for split %s", split_name)

        split_df = self.splits_df[split_name]
        indices_dict = self._indices_per_split[split_name]
        n_classes = len(self.parameters.cell_types_mapping)

        # Generate one pure profile per class
        pure_results: List[PseudobulkResult] = []

        # Get expected GR group IDs from parameters (values of the dict)
        expected_gr_ids = sorted(self.parameters.gr_groups_mapping.values())

        for class_idx in range(n_classes):
            # Create one-hot proportions (100% of reads from this cell type)
            proportions = np.zeros(n_classes)
            proportions[class_idx] = 1.0

            result = self.generate_single_pseudobulk(
                n_reads_to_sample=self.n_reads_to_sample,
                n_gr_groups=self.n_gr_groups,
                n_classes=self.n_classes,
                indices_per_class_and_grg=indices_dict,
                numpy_arrays=self._numpy_arrays_per_split[split_name],
                target_proportions=proportions,
                grg_grouping_column=self.grg_label_column,
                columns_to_keep=self.columns_to_keep,
                grg_sampling_type=self.parameters.gr_sampling_method,
                index=class_idx,
            )

            # Fill in missing GR groups using the substitution method
            filled_features = fill_in_missing_gr_groups(
                df=result.aggregated_features,
                expected_gr_ids=expected_gr_ids,
                gr_label_column=self.grg_label_column,
                n_classes=n_classes,
                substitution_strategy=self.parameters.substitution_method,
                uniform_prior=None,  # Will be computed after first pass if needed
            )
            result.aggregated_features = filled_features
            pure_results.append(result)

        # Extract feature matrices (shape: n_classes x n_gr_groups x n_features)
        # Filter to columns_to_keep (defaults to build_target_columns)
        # Separate numeric and string columns for numpy compatibility
        target_cols = self.columns_to_keep
        numeric_cols = [
            col
            for col in target_cols
            if col in pure_results[0].aggregated_features.columns
            and pd.api.types.is_numeric_dtype(
                pure_results[0].aggregated_features[col].dtype
            )
        ]
        string_cols = [
            col
            for col in target_cols
            if col in pure_results[0].aggregated_features.columns
            and not pd.api.types.is_numeric_dtype(
                pure_results[0].aggregated_features[col].dtype
            )
        ]

        # Stack numeric features into feature_matrices
        numeric_features_list = [
            r.aggregated_features[numeric_cols] for r in pure_results
        ]
        feature_matrices = np.stack([df.values for df in numeric_features_list], axis=0)

        # Stack string features into string_matrices (if any)
        string_matrices = None
        if string_cols:
            string_features_list = [
                r.aggregated_features[string_cols].astype(str) for r in pure_results
            ]
            string_matrices = np.stack(
                [df.values for df in string_features_list], axis=0
            )

        # Compute uniform prior (average across all classes)
        uniform_prior = feature_matrices.mean(axis=0)

        # Mark pure profiles as done
        self.checkpoint_manager.mark_pure_profiles_done(split_name)

        return PureProfileResult(
            split_name=split_name,
            feature_matrices=feature_matrices,
            uniform_prior=uniform_prior,
            numeric_columns=numeric_cols,
            string_matrices=string_matrices,
            string_columns=string_cols if string_cols else None,
        )

    @classmethod
    def generate_single_pseudobulk(
        cls,
        n_reads_to_sample: int,
        n_gr_groups: int,
        n_classes: int,
        indices_per_class_and_grg: Dict[Tuple[int, int], np.ndarray],
        numpy_arrays: Dict[str, np.ndarray],
        target_proportions: np.ndarray,
        grg_grouping_column: str,
        columns_to_keep: list[str] = None,
        grg_sampling_type: Literal["uniform_multinomial"] = "uniform_multinomial",
        index: int = 0,
    ) -> PseudobulkResult:
        """Generate a single pseudobulk sample by sampling and aggregating reads.

        This method samples reads from each cell type according to target proportions,
        distributes them across GR groups using the specified sampling method, and
        aggregates the predictions into feature matrices.

        Args:
            n_reads_to_sample: The total number of reads to sample for the pseudobulk.
            n_gr_groups: The number of GR groups to distribute reads across.
            n_classes: Number of prediction classes.
            indices_per_class_and_grg: A dictionary mapping (class_index, grg_index) tuples
                to arrays of row indices corresponding to that group in the original dataframe.
                Keys must be integer indices (0-based), not raw label values.
            numpy_arrays: Pre-extracted numpy arrays dict with keys:
                - 'gr_group_idx_array': GR group indices (n_rows,)
                - 'weight_array': weights (n_rows,)
                - 'pred_matrix': prediction matrix (n_rows, n_classes), Fortran-order
            target_proportions: An array of proportions summing to 1, shape (n_classes,).
            grg_grouping_column: Column name for the GR group column (used for output DataFrame).
            columns_to_keep: Columns to keep in aggregated output. If None, keeps all.
            grg_sampling_type: The method to use for sampling reads across GR groups.
                Currently only supports "uniform_multinomial".
            index: Index to assign to this pseudobulk result.

        Returns:
            PseudobulkResult containing the sampled pseudobulk data.
        """
        assert np.isclose(
            target_proportions.sum(), 1.0
        ), "target_proportions must sum to 1"
        assert grg_sampling_type in [
            "uniform_multinomial"
        ], f"Unsupported grg_sampling_type: {grg_sampling_type}"
        target_proportions = np.asarray(target_proportions, dtype=np.float64)

        # Compute the number of reads to sample from each class
        n_samples_per_class = np.array(
            [int(n_reads_to_sample * p) for p in target_proportions]
        )

        # For each class, compute the number of reads to sample from each GR group
        if grg_sampling_type == "uniform_multinomial":
            n_samples_per_class_per_grg = _sample_reads_per_grg_uniform_multinomial(
                n_samples_per_class, n_gr_groups
            )  # shape (n_classes, n_gr_groups)
        else:
            raise ValueError(f"Unsupported grg_sampling_type: {grg_sampling_type}")

        # post processing step: if there are some (class, grg) groups with 0 reads available in
        # the dataset, we cannot sample from them.
        # We redistribute the reads that were supposed to come from those groups
        # uniformly across the other groups of the same class
        mask_has_reads = np.zeros_like(n_samples_per_class_per_grg, dtype=bool)
        for (class_idx, grg_idx), indices in indices_per_class_and_grg.items():
            if len(indices) > 0:
                mask_has_reads[class_idx, grg_idx] = True
        for class_idx in range(n_samples_per_class_per_grg.shape[0]):
            # Identify groups with and without reads for this class
            has_reads = mask_has_reads[class_idx]
            quota_to_distribute = n_samples_per_class_per_grg[
                class_idx, ~has_reads
            ].sum()
            if quota_to_distribute > 0:
                n_samples_per_class_per_grg[class_idx, has_reads] += int(
                    quota_to_distribute / has_reads.sum()
                )

        # compute actual proportions after adjustment
        n_reads_really_sampled = n_samples_per_class_per_grg.sum()
        actual_proportions = (
            n_samples_per_class_per_grg.sum(axis=1) / n_reads_really_sampled
        )  # shape (n_classes,)
        seed = np.random.randint(0, 2**31 - 1)

        # Sample the read IDS for each class and GR group, and concatenate them into a single array
        read_ids = _sample_read_ids_from_grouped_dataframe(
            n_samples_per_class_per_grg,
            indices_per_class_and_grg,
            seed=seed,
        )  # shape (n_reads_really_sampled,)

        # Aggregate directly from pre-extracted numpy arrays (no DataFrame slicing)
        weighted_avgs, counts, total_weights = aggregate_by_grg_from_np_arrays(
            read_ids=read_ids,
            gr_group_idx_array=numpy_arrays["gr_group_idx_array"],
            weight_array=numpy_arrays["weight_array"],
            pred_matrix=numpy_arrays["pred_matrix"],
            n_gr_groups=n_gr_groups,
            n_pred_cols=n_classes,
        )

        # Build result DataFrame from numpy arrays
        result_data = {grg_grouping_column: np.arange(n_gr_groups, dtype=np.int64)}
        for i in range(n_classes):
            result_data[f"prediction_{i}_wavg"] = weighted_avgs[:, i]
        result_data["n_reads"] = counts
        result_data["total_weight"] = total_weights
        aggregated_features = pd.DataFrame(result_data)

        # Filter to only groups that have data
        if not counts.all():
            aggregated_features = aggregated_features[counts > 0].reset_index(drop=True)

        if columns_to_keep is not None:
            columns_to_keep = [
                col for col in columns_to_keep if col in aggregated_features.columns
            ]
            aggregated_features = aggregated_features[columns_to_keep]

        return PseudobulkResult(
            index=index,
            target_proportions=target_proportions,
            actual_proportions=actual_proportions,
            n_reads_really_sampled=n_reads_really_sampled,
            n_samples_per_class_per_grg=n_samples_per_class_per_grg,
            seed=seed,
            aggregated_features=aggregated_features,
        )
