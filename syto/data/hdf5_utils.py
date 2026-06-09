"""HDF5 utilities for pseudobulk generation.

This module provides:
- HDF5 schema constants defining the structure of pseudobulk output files
- HDF5Writer class for atomic batch file writing
- Checkpoint management for crash-resilient generation
"""

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import hashlib
import json
import logging
import os
import tempfile

import h5py
import numpy as np
import pandas as pd

_module_logger = logging.getLogger(__name__)


# =============================================================================
# HDF5 Schema Constants
# =============================================================================


class HDF5Schema:
    """Constants defining the HDF5 file structure for pseudobulk data."""

    # Top-level groups
    INPUTS = "inputs"
    PARAMETERS = "parameters"
    OUTPUTS = "outputs"

    # Inputs subgroups
    INPUTS_METADATA = f"{INPUTS}/metadata"
    INPUTS_TRAIN = f"{INPUTS}/train"
    INPUTS_VALID = f"{INPUTS}/valid"
    INPUTS_TEST = f"{INPUTS}/test"

    # Metadata attributes
    ATTR_GR_ID_COLUMN = "grg_id_column"
    ATTR_LABELING_SCHEME = "labeling_scheme"
    ATTR_CLASSIFIER = "classifier"
    ATTR_DATA_WATERMARK = "data_watermark"
    DATASET_DATA_STATS = "data_stats"

    # Parameters datasets/attributes
    PARAMS_CELL_TYPES_MAPPING = f"{PARAMETERS}/cell_types_mapping"
    PARAMS_GR_GROUPS_MAPPING = f"{PARAMETERS}/gr_groups_mapping"
    ATTR_SUBSTITUTION_METHOD = "substitution_method"
    ATTR_GR_SAMPLING_METHOD = "grg_sampling_method"

    # Outputs structure (per split)
    @staticmethod
    def outputs_split(split_name: str) -> str:
        """Get the HDF5 path for a split's outputs group."""
        return f"outputs/{split_name}"

    @staticmethod
    def pseudobulks_group(split_name: str) -> str:
        """Get the HDF5 path for a split's pseudobulks group."""
        return f"outputs/{split_name}/pseudobulks"

    @staticmethod
    def pseudobulk_index(split_name: str, index: int) -> str:
        """Get the HDF5 path for a specific pseudobulk by index."""
        return f"outputs/{split_name}/pseudobulks/i_{index}"

    @staticmethod
    def pure_profiles_group(split_name: str) -> str:
        """Get the HDF5 path for a split's pure profiles group."""
        return f"outputs/{split_name}/pure_profiles"

    # Pseudobulk datasets
    DATASET_TARGET_PROPORTIONS = "target_proportions"
    DATASET_ACTUAL_PROPORTIONS = "actual_proportions"
    DATASET_N_READS_PER_GR = "n_reads_per_gr"
    DATASET_AGGREGATED_FEATURES = "aggregated_features"
    GROUP_SAMPLED_INDEXES = "sampled_indexes"
    ATTR_SEED = "seed"
    ATTR_SAMPLING_FUNCTION = "sampling_function"

    # Pure profiles datasets
    DATASET_FEATURE_MATRICES = "feature_matrices"
    DATASET_UNIFORM_PRIOR = "uniform_prior"

    # Compression settings
    COMPRESSION = "gzip"
    COMPRESSION_LEVEL = 4


# =============================================================================
# Dataclasses for Results
# =============================================================================


@dataclass
class PseudobulkResult:
    """Result of generating a single pseudobulk sample."""

    index: int
    target_proportions: np.ndarray
    actual_proportions: np.ndarray
    n_reads_really_sampled: int
    n_samples_per_class_per_grg: np.ndarray
    seed: int
    aggregated_features: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "target_proportions": self.target_proportions,
            "actual_proportions": self.actual_proportions,
            "n_reads_really_sampled": self.n_reads_really_sampled,
            "n_samples_per_class_per_grg": self.n_samples_per_class_per_grg,
            "seed": self.seed,
            "aggregated_features": self.aggregated_features,
        }


@dataclass
class PureProfileResult:
    """Result of generating pure profiles for a split."""

    split_name: str
    feature_matrices: np.ndarray  # shape (n_classes, n_gr_groups, n_numeric_features)
    uniform_prior: np.ndarray  # shape (n_gr_groups, n_numeric_features)
    numeric_columns: List[str]  # column names for numeric features
    string_matrices: Optional[np.ndarray] = (
        None  # shape (n_classes, n_gr_groups, n_string_cols)
    )
    string_columns: Optional[List[str]] = None  # column names for string features


@dataclass
class GenerationMetadata:
    """Metadata about the generation process."""

    grg_id_column: str
    labeling_scheme: str
    classifier: str
    data_watermark: str
    data_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary for serialization."""
        return {
            "grg_id_column": self.grg_id_column,
            "labeling_scheme": self.labeling_scheme,
            "classifier": self.classifier,
            "data_watermark": self.data_watermark,
            "data_stats": self.data_stats,
        }


@dataclass
class GenerationParameters:
    """Parameters for pseudobulk generation."""

    cell_types_mapping: Dict[str, int]  # Mapping from cell type name to class index
    gr_groups_mapping: Dict[str, int]  # Mapping from GR group label to GR group index
    substitution_method: Literal["uniform_number", "prior_blending", "prior_imputation"]
    grg_sampling_method: Literal["uniform_multinomial"]

    def to_hash(self) -> str:
        """Compute a hash of the parameters for change detection."""
        param_str = json.dumps(
            {
                "cell_types_mapping": self.cell_types_mapping,
                "gr_groups_mapping": self.gr_groups_mapping,
                "substitution_method": self.substitution_method,
                "grg_sampling_method": self.grg_sampling_method,
            },
            sort_keys=True,
        )
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]


# =============================================================================
# Checkpoint Management
# =============================================================================


@dataclass
class SplitCheckpoint:
    """Checkpoint state for a single split."""

    split: str
    status: Literal["not_started", "in_progress", "completed"]
    total_batches: int
    batch_size: int
    completed_batches: List[int] = field(default_factory=list)
    pure_profiles_done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint state to a dictionary for JSON serialization."""
        return {
            "split": self.split,
            "status": self.status,
            "total_batches": self.total_batches,
            "batch_size": self.batch_size,
            "completed_batches": self.completed_batches,
            "pure_profiles_done": self.pure_profiles_done,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitCheckpoint":
        """Create a SplitCheckpoint from a dictionary."""
        return cls(
            split=data["split"],
            status=data["status"],
            total_batches=data["total_batches"],
            batch_size=data["batch_size"],
            completed_batches=data.get("completed_batches", []),
            pure_profiles_done=data.get("pure_profiles_done", False),
        )


@dataclass
class GenerationConfig:
    """Overall generation configuration and state."""

    generation_started_at: str
    parameters_hash: str
    splits_order: List[str]
    splits_completed: List[str]
    n_pseudobulks_per_split: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary for JSON serialization."""
        return {
            "generation_started_at": self.generation_started_at,
            "parameters_hash": self.parameters_hash,
            "splits_order": self.splits_order,
            "splits_completed": self.splits_completed,
            "n_pseudobulks_per_split": self.n_pseudobulks_per_split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create a GenerationConfig from a dictionary."""
        return cls(
            generation_started_at=data["generation_started_at"],
            parameters_hash=data["parameters_hash"],
            splits_order=data["splits_order"],
            splits_completed=data.get("splits_completed", []),
            n_pseudobulks_per_split=data["n_pseudobulks_per_split"],
        )


class CheckpointManager:
    """Manages checkpoint files for crash-resilient generation."""

    CONFIG_FILENAME = "generation_config.json"
    CHECKPOINT_FILENAME = "checkpoint.json"
    BATCHES_DIR = "batches"

    def __init__(self, output_dir: Path, logger: logging.Logger = _module_logger):
        """Initialize the checkpoint manager.

        Args:
            output_dir: Directory where checkpoint files will be stored.
            logger: Logger instance for logging messages.
        """
        self.output_dir = Path(output_dir)
        self.logger = logger

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file atomically using temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix=path.stem, dir=path.parent
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file if it exists."""
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # --- Generation Config ---

    def get_config_path(self) -> Path:
        """Get the path to the generation config file."""
        return self.output_dir / self.CONFIG_FILENAME

    def load_config(self) -> Optional[GenerationConfig]:
        """Load generation config if it exists."""
        data = self._read_json(self.get_config_path())
        if data is None:
            return None
        return GenerationConfig.from_dict(data)

    def save_config(self, config: GenerationConfig) -> None:
        """Save generation config atomically."""
        self._atomic_write_json(self.get_config_path(), config.to_dict())

    def create_config(
        self,
        parameters_hash: str,
        splits_order: List[str],
        n_pseudobulks_per_split: Dict[str, int],
    ) -> GenerationConfig:
        """Create a new generation config."""
        config = GenerationConfig(
            generation_started_at=datetime.now().isoformat(),
            parameters_hash=parameters_hash,
            splits_order=splits_order,
            splits_completed=[],
            n_pseudobulks_per_split=n_pseudobulks_per_split,
        )
        self.save_config(config)
        return config

    # --- Split Checkpoint ---

    def get_split_dir(self, split_name: str) -> Path:
        """Get the directory path for a specific split."""
        return self.output_dir / split_name

    def get_split_checkpoint_path(self, split_name: str) -> Path:
        """Get the path to a split's checkpoint file."""
        return self.get_split_dir(split_name) / self.CHECKPOINT_FILENAME

    def get_split_batches_dir(self, split_name: str) -> Path:
        """Get the directory path for a split's batch files."""
        return self.get_split_dir(split_name) / self.BATCHES_DIR

    def load_split_checkpoint(self, split_name: str) -> Optional[SplitCheckpoint]:
        """Load checkpoint for a specific split."""
        data = self._read_json(self.get_split_checkpoint_path(split_name))
        if data is None:
            return None
        return SplitCheckpoint.from_dict(data)

    def save_split_checkpoint(self, checkpoint: SplitCheckpoint) -> None:
        """Save split checkpoint atomically."""
        self._atomic_write_json(
            self.get_split_checkpoint_path(checkpoint.split), checkpoint.to_dict()
        )

    def create_split_checkpoint(
        self, split_name: str, total_pseudobulks: int, batch_size: int
    ) -> SplitCheckpoint:
        """Create a new split checkpoint."""
        total_batches = (total_pseudobulks + batch_size - 1) // batch_size
        checkpoint = SplitCheckpoint(
            split=split_name,
            status="in_progress",
            total_batches=total_batches,
            batch_size=batch_size,
            completed_batches=[],
            pure_profiles_done=False,
        )
        self.get_split_batches_dir(split_name).mkdir(parents=True, exist_ok=True)
        self.save_split_checkpoint(checkpoint)
        return checkpoint

    def mark_batch_completed(self, split_name: str, batch_index: int) -> None:
        """Mark a batch as completed."""
        checkpoint = self.load_split_checkpoint(split_name)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for split {split_name}")
        if batch_index not in checkpoint.completed_batches:
            checkpoint.completed_batches.append(batch_index)
            checkpoint.completed_batches.sort()
        self.save_split_checkpoint(checkpoint)

    def mark_pure_profiles_done(self, split_name: str) -> None:
        """Mark pure profiles as completed for a split."""
        checkpoint = self.load_split_checkpoint(split_name)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for split {split_name}")
        checkpoint.pure_profiles_done = True
        self.save_split_checkpoint(checkpoint)

    def mark_split_completed(self, split_name: str) -> None:
        """Mark a split as fully completed."""
        checkpoint = self.load_split_checkpoint(split_name)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for split {split_name}")
        checkpoint.status = "completed"
        self.save_split_checkpoint(checkpoint)

        # Update global config
        config = self.load_config()
        if config and split_name not in config.splits_completed:
            config.splits_completed.append(split_name)
            self.save_config(config)

    def get_missing_batches(self, split_name: str) -> List[int]:
        """Get list of batch indices that need to be (re)generated."""
        checkpoint = self.load_split_checkpoint(split_name)
        if checkpoint is None:
            return []
        all_batches = set(range(checkpoint.total_batches))
        completed = set(checkpoint.completed_batches)
        return sorted(all_batches - completed)

    # --- Cleanup ---

    def cleanup_split(self, split_name: str) -> None:
        """Remove batch files and checkpoint for a split after consolidation."""

        batches_dir = self.get_split_batches_dir(split_name)
        if batches_dir.exists():
            shutil.rmtree(batches_dir)
            self.logger.info("Removed batches directory: %s", batches_dir)

        checkpoint_path = self.get_split_checkpoint_path(split_name)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info("Removed checkpoint: %s", checkpoint_path)

        # Remove the split directory itself if empty
        split_dir = self.get_split_dir(split_name)
        if split_dir.exists() and not any(split_dir.iterdir()):
            split_dir.rmdir()
            self.logger.info("Removed split directory: %s", split_dir)

    def cleanup_all(self) -> None:
        """Remove all checkpoints and batch files after successful consolidation."""
        config = self.load_config()
        if config:
            for split_name in config.splits_order:
                self.cleanup_split(split_name)

        config_path = self.get_config_path()
        if config_path.exists():
            config_path.unlink()
            self.logger.info("Removed generation config: %s", config_path)


# =============================================================================
# HDF5 Batch Writer
# =============================================================================


class HDF5BatchWriter:
    """Writes pseudobulk batches to HDF5 files atomically."""

    def __init__(
        self,
        batches_dir: Path,
        logger: logging.Logger = _module_logger,
    ):
        """Initialize the HDF5 batch writer.

        Args:
            batches_dir: Directory where batch HDF5 files will be written.
            logger: Logger instance for logging messages.
        """
        self.batches_dir = Path(batches_dir)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def get_batch_path(self, batch_index: int) -> Path:
        """Get the path for a batch file by index."""
        return self.batches_dir / f"batch_{batch_index:06d}.h5"

    def get_temp_batch_path(self, batch_index: int) -> Path:
        """Get the temporary path for a batch file during atomic write."""
        return self.batches_dir / f"batch_{batch_index:06d}.h5.tmp"

    def write_batch(self, batch_index: int, results: List[PseudobulkResult]) -> Path:
        """Write a batch of pseudobulk results to HDF5 atomically."""
        final_path = self.get_batch_path(batch_index)
        temp_path = self.get_temp_batch_path(batch_index)

        try:
            with h5py.File(temp_path, "w") as f:
                f.attrs["batch_index"] = batch_index
                f.attrs["n_pseudobulks"] = len(results)

                for result in results:
                    grp = f.create_group(f"pseudobulk_{result.index}")
                    grp.attrs["index"] = result.index
                    grp.attrs["seed"] = result.seed
                    grp.attrs["n_reads_really_sampled"] = result.n_reads_really_sampled
                    grp.attrs["sampling_function"] = (
                        "_sample_read_ids_from_grouped_dataframe"
                    )

                    grp.create_dataset(
                        "target_proportions",
                        data=result.target_proportions,
                        compression=HDF5Schema.COMPRESSION,
                        compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                    )
                    grp.create_dataset(
                        "actual_proportions",
                        data=result.actual_proportions,
                        compression=HDF5Schema.COMPRESSION,
                        compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                    )
                    grp.create_dataset(
                        "n_reads_per_gr",
                        data=result.n_samples_per_class_per_grg,
                        compression=HDF5Schema.COMPRESSION,
                        compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                    )
                    # Store aggregated features - handle numeric and string columns
                    numeric_cols = result.aggregated_features.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    string_cols = result.aggregated_features.select_dtypes(
                        include=[object]
                    ).columns.tolist()

                    # Store numeric features
                    if numeric_cols:
                        grp.create_dataset(
                            "aggregated_features",
                            data=result.aggregated_features[numeric_cols].values,
                            compression=HDF5Schema.COMPRESSION,
                            compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                        )
                        grp.attrs["feature_columns"] = numeric_cols

                    # Store string features as variable-length strings
                    if string_cols:
                        str_dtype = h5py.special_dtype(vlen=str)
                        str_data = (
                            result.aggregated_features[string_cols].astype(str).values
                        )
                        grp.create_dataset(
                            "aggregated_features_strings",
                            data=str_data,
                            dtype=str_dtype,
                        )
                        grp.attrs["string_columns"] = string_cols

            # Atomic rename
            os.rename(temp_path, final_path)
            self.logger.debug("Wrote batch %d to %s", batch_index, final_path)
            return final_path

        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def read_batch(self, batch_index: int) -> List[PseudobulkResult]:
        """Read a batch of pseudobulk results from HDF5."""
        path = self.get_batch_path(batch_index)
        results = []

        with h5py.File(path, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("pseudobulk_"):
                    continue
                grp = f[key]

                # Reconstruct aggregated_features DataFrame
                df_parts = []
                columns_order = []

                # Read numeric features
                if "aggregated_features" in grp:
                    numeric_cols = list(grp.attrs.get("feature_columns", []))
                    numeric_df = pd.DataFrame(
                        grp["aggregated_features"][...],
                        columns=numeric_cols,
                    )
                    df_parts.append(numeric_df)
                    columns_order.extend(numeric_cols)

                # Read string features
                if "aggregated_features_strings" in grp:
                    string_cols = list(grp.attrs.get("string_columns", []))
                    string_df = pd.DataFrame(
                        grp["aggregated_features_strings"][...],
                        columns=string_cols,
                    )
                    df_parts.append(string_df)
                    columns_order.extend(string_cols)

                # Combine DataFrames
                if df_parts:
                    aggregated_features = pd.concat(df_parts, axis=1)
                else:
                    aggregated_features = pd.DataFrame()

                result = PseudobulkResult(
                    index=grp.attrs["index"],
                    target_proportions=grp["target_proportions"][...],
                    actual_proportions=grp["actual_proportions"][...],
                    n_reads_really_sampled=grp.attrs["n_reads_really_sampled"],
                    n_samples_per_class_per_grg=grp["n_reads_per_gr"][...],
                    seed=grp.attrs["seed"],
                    aggregated_features=aggregated_features,
                )
                results.append(result)

        return results

    def batch_exists(self, batch_index: int) -> bool:
        """Check if a batch file exists (and is not a temp file)."""
        return self.get_batch_path(batch_index).exists()


# =============================================================================
# HDF5 Consolidation Writer
# =============================================================================


class HDF5ConsolidationWriter:
    """Consolidates batch files into the final HDF5 output."""

    def __init__(
        self,
        output_path: Path,
        logger: logging.Logger = _module_logger,
    ):
        """Initialize the HDF5 consolidation writer.

        Args:
            output_path: Path where the final consolidated HDF5 file will be written.
            logger: Logger instance for logging messages.
        """
        self.output_path = Path(output_path)
        self.logger = logger

    def consolidate(
        self,
        checkpoint_manager: CheckpointManager,
        metadata: GenerationMetadata,
        parameters: GenerationParameters,
        input_dfs: Dict[str, pd.DataFrame],
        pure_profiles: Dict[str, PureProfileResult],
    ) -> Path:
        """Consolidate all batch files into final HDF5."""
        temp_path = self.output_path.with_suffix(".h5.tmp")

        try:
            with h5py.File(temp_path, "w") as f:
                # Write inputs group with metadata
                self._write_inputs(f, metadata, input_dfs)

                # Write parameters
                self._write_parameters(f, parameters)

                # Write outputs for each split
                config = checkpoint_manager.load_config()
                for split_name in config.splits_order:
                    self._write_split_outputs(
                        f,
                        split_name,
                        checkpoint_manager,
                        pure_profiles.get(split_name),
                    )

            # Atomic rename
            os.rename(temp_path, self.output_path)
            self.logger.info("Consolidated HDF5 written to %s", self.output_path)
            return self.output_path

        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _write_inputs(
        self,
        f: h5py.File,
        metadata: GenerationMetadata,
        input_dfs: Dict[str, pd.DataFrame],
    ) -> None:
        """Write inputs group with metadata and input DataFrames."""
        inputs_grp = f.create_group(HDF5Schema.INPUTS)

        # Metadata subgroup
        meta_grp = inputs_grp.create_group("metadata")
        meta_grp.attrs[HDF5Schema.ATTR_GR_ID_COLUMN] = metadata.grg_id_column
        meta_grp.attrs[HDF5Schema.ATTR_LABELING_SCHEME] = metadata.labeling_scheme
        meta_grp.attrs[HDF5Schema.ATTR_CLASSIFIER] = metadata.classifier
        meta_grp.attrs[HDF5Schema.ATTR_DATA_WATERMARK] = metadata.data_watermark
        if metadata.data_stats:
            meta_grp.create_dataset(
                HDF5Schema.DATASET_DATA_STATS,
                data=json.dumps(metadata.data_stats),
            )

        # Input DataFrames per split
        for split_name, df in input_dfs.items():
            split_grp = inputs_grp.create_group(split_name)

            # Separate numeric and string columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            string_cols = df.select_dtypes(include=[object]).columns.tolist()

            # Store numeric columns
            if numeric_cols:
                split_grp.create_dataset(
                    "numeric_data",
                    data=df[numeric_cols].values,
                    compression=HDF5Schema.COMPRESSION,
                    compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                )
                split_grp.attrs["numeric_columns"] = numeric_cols

            # Store string columns as variable-length strings
            if string_cols:
                str_dtype = h5py.special_dtype(vlen=str)
                str_data = df[string_cols].astype(str).values
                split_grp.create_dataset(
                    "string_data",
                    data=str_data,
                    dtype=str_dtype,
                )
                split_grp.attrs["string_columns"] = string_cols

            split_grp.attrs["n_rows"] = len(df)

    def _write_parameters(self, f: h5py.File, parameters: GenerationParameters) -> None:
        """Write parameters group."""
        params_grp = f.create_group(HDF5Schema.PARAMETERS)

        # Cell types mapping as structured dataset
        cell_types = list(parameters.cell_types_mapping.items())
        dt = np.dtype([("name", "S100"), ("id", "i4")])
        cell_types_arr = np.array(
            [(name.encode(), id_) for name, id_ in cell_types], dtype=dt
        )
        params_grp.create_dataset("cell_types_mapping", data=cell_types_arr)

        # GR groups mapping as structured dataset
        gr_groups = list(parameters.gr_groups_mapping.items())
        gr_groups_arr = np.array(
            [(name.encode(), id_) for name, id_ in gr_groups], dtype=dt
        )
        params_grp.create_dataset("gr_groups_mapping", data=gr_groups_arr)

        # Attributes
        params_grp.attrs[HDF5Schema.ATTR_SUBSTITUTION_METHOD] = (
            parameters.substitution_method
        )
        params_grp.attrs[HDF5Schema.ATTR_GR_SAMPLING_METHOD] = (
            parameters.grg_sampling_method
        )

    def _write_split_outputs(
        self,
        f: h5py.File,
        split_name: str,
        checkpoint_manager: CheckpointManager,
        pure_profile: Optional[PureProfileResult],
    ) -> None:
        """Write outputs for a single split."""
        split_grp = f.create_group(HDF5Schema.outputs_split(split_name))
        pseudobulks_grp = split_grp.create_group("pseudobulks")

        # Read and write all batches
        batches_dir = checkpoint_manager.get_split_batches_dir(split_name)
        batch_writer = HDF5BatchWriter(batches_dir)

        checkpoint = checkpoint_manager.load_split_checkpoint(split_name)
        for batch_idx in sorted(checkpoint.completed_batches):
            results = batch_writer.read_batch(batch_idx)
            for result in results:
                pb_grp = pseudobulks_grp.create_group(f"i_{result.index}")
                pb_grp.attrs[HDF5Schema.ATTR_SEED] = result.seed
                pb_grp.attrs[HDF5Schema.ATTR_SAMPLING_FUNCTION] = (
                    "_sample_read_ids_from_grouped_dataframe"
                )
                pb_grp.attrs["n_reads_really_sampled"] = result.n_reads_really_sampled

                pb_grp.create_dataset(
                    HDF5Schema.DATASET_TARGET_PROPORTIONS,
                    data=result.target_proportions,
                    compression=HDF5Schema.COMPRESSION,
                    compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                )
                pb_grp.create_dataset(
                    HDF5Schema.DATASET_ACTUAL_PROPORTIONS,
                    data=result.actual_proportions,
                    compression=HDF5Schema.COMPRESSION,
                    compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                )
                pb_grp.create_dataset(
                    HDF5Schema.DATASET_N_READS_PER_GR,
                    data=result.n_samples_per_class_per_grg,
                    compression=HDF5Schema.COMPRESSION,
                    compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                )
                # Store aggregated features - handle numeric and string columns
                numeric_cols = result.aggregated_features.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                string_cols = result.aggregated_features.select_dtypes(
                    include=[object]
                ).columns.tolist()

                # Store numeric features
                if numeric_cols:
                    pb_grp.create_dataset(
                        HDF5Schema.DATASET_AGGREGATED_FEATURES,
                        data=result.aggregated_features[numeric_cols].values,
                        compression=HDF5Schema.COMPRESSION,
                        compression_opts=HDF5Schema.COMPRESSION_LEVEL,
                    )
                    pb_grp.attrs["feature_columns"] = numeric_cols

                # Store string features as variable-length strings
                if string_cols:
                    str_dtype = h5py.special_dtype(vlen=str)
                    str_data = (
                        result.aggregated_features[string_cols].astype(str).values
                    )
                    pb_grp.create_dataset(
                        "aggregated_features_strings",
                        data=str_data,
                        dtype=str_dtype,
                    )
                    pb_grp.attrs["string_columns"] = string_cols

        # Write pure profiles if available
        if pure_profile:
            pp_grp = split_grp.create_group("pure_profiles")
            pp_grp.create_dataset(
                HDF5Schema.DATASET_FEATURE_MATRICES,
                data=pure_profile.feature_matrices,
                compression=HDF5Schema.COMPRESSION,
                compression_opts=HDF5Schema.COMPRESSION_LEVEL,
            )
            pp_grp.create_dataset(
                HDF5Schema.DATASET_UNIFORM_PRIOR,
                data=pure_profile.uniform_prior,
                compression=HDF5Schema.COMPRESSION,
                compression_opts=HDF5Schema.COMPRESSION_LEVEL,
            )
            pp_grp.attrs["numeric_columns"] = pure_profile.numeric_columns

            # Write string columns if available
            if pure_profile.string_matrices is not None and pure_profile.string_columns:
                str_dtype = h5py.special_dtype(vlen=str)
                pp_grp.create_dataset(
                    "string_matrices",
                    data=pure_profile.string_matrices,
                    dtype=str_dtype,
                )
                pp_grp.attrs["string_columns"] = pure_profile.string_columns
