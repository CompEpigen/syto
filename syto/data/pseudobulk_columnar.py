"""Columnar backend for pseudobulk stores.

The consolidated HDF5 layout writes one HDF5 group per pseudobulk.  At the
scale the project actually runs at -- 8.3 million pseudobulks across 33 runs --
that per-group bookkeeping costs more space than the data it wraps.  This
module stores the same information as a directory of Parquet and JSON, which
removes the overhead and is readable without ``h5py``::

    <store>/
    ├── manifest.json              format_version, splits, counts, columns, dtypes
    ├── metadata.json              grg_id_column, labeling_scheme, classifier, watermark
    ├── parameters.json            cell-type / GR-group mappings, sampling methods
    ├── inputs/<split>.parquet     the per-split reads, in generation row order
    └── outputs/<split>/
        ├── features.parquet       n_pseudobulks*n_grg rows, one row per GR group
        ├── proportions.parquet    one row per pseudobulk (seed, target, actual)
        ├── n_reads_per_gr.parquet n_pseudobulks*n_classes rows
        └── pure_profiles.npz      pure profiles and the uniform prior

Two invariants matter more than anything else here:

* **Row order in ``inputs/<split>.parquet`` is load-bearing.**  Read-level
  reconstruction re-derives row positions into that frame from a stored seed,
  so a frame in any other order silently yields the wrong reads while every
  shape and count still checks out.  Parquet preserves row order across row
  groups; :meth:`PseudobulkColumnarWriter.write_inputs` asserts on the count
  and the reader never sorts.
* **``pb_index`` is stored explicitly** in the flattened tables rather than
  being implied by position, so a reshape can be checked rather than trusted.

Note on dtypes: the HDF5 layout stacks every numeric input column into one
``float64`` array, so integer columns come back as floats.  This backend
preserves whatever dtypes it is given.  Converting an existing HDF5 store
therefore round-trips exactly (the values read out are already float64), while
a store written directly by the generator keeps the original integer columns.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from syto.data.pseudobulk_store import COLUMNAR_MANIFEST, BasePseudobulkReader

_module_logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Written alongside the payload columns so a reshape can be verified.
COL_PB_INDEX = "pb_index"
COL_GRG_ROW = "grg_row"
COL_CLASS_ROW = "class_row"

FEATURES_FILE = "features.parquet"
PROPORTIONS_FILE = "proportions.parquet"
N_READS_PER_GR_FILE = "n_reads_per_gr.parquet"
PURE_PROFILES_FILE = "pure_profiles.npz"

DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 9

# Lossless by default: narrowing the features is a deliberate, measured choice
# made by the caller, not a silent property of the format.
DEFAULT_FEATURE_DTYPE = "float64"


def _target_column(i: int) -> str:
    return f"target_{i}"


def _actual_column(i: int) -> str:
    return f"actual_{i}"


def _grg_column(j: int) -> str:
    return f"grg_{j}"


@dataclass
class SplitManifest:
    """Per-split shape information recorded in ``manifest.json``."""

    n_pseudobulks: int
    n_grg: int
    n_classes: int
    feature_columns: List[str]
    has_pure_profiles: bool
    n_input_rows: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "n_pseudobulks": self.n_pseudobulks,
            "n_grg": self.n_grg,
            "n_classes": self.n_classes,
            "feature_columns": self.feature_columns,
            "has_pure_profiles": self.has_pure_profiles,
            "n_input_rows": self.n_input_rows,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitManifest":
        """Rebuild from the dictionary written to ``manifest.json``."""
        return cls(
            n_pseudobulks=int(data["n_pseudobulks"]),
            n_grg=int(data["n_grg"]),
            n_classes=int(data["n_classes"]),
            feature_columns=list(data["feature_columns"]),
            has_pure_profiles=bool(data["has_pure_profiles"]),
            n_input_rows=int(data["n_input_rows"]),
        )


# =============================================================================
# Writer
# =============================================================================


class PseudobulkColumnarWriter:
    """Write a columnar pseudobulk store.

    Exposes both a streaming API (``write_inputs`` / ``open_split`` /
    ``write_pure_profiles`` / ``finalize``) and a :meth:`consolidate` entry
    point whose signature matches
    :meth:`~syto.data.pseudobulk_hdf5_utils.PseudobulkHDF5ConsolidationWriter.consolidate`,
    so the generator can target either backend.
    """

    def __init__(
        self,
        output_path: "Path | str",
        feature_dtype: str = DEFAULT_FEATURE_DTYPE,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        logger: logging.Logger = _module_logger,
    ):
        """Initialise the writer.

        Args:
            output_path: Directory the store is written to. Created if absent.
            feature_dtype: Storage dtype for the aggregated features. The
                default is lossless; ``"float32"`` roughly halves the store.
            compression: Parquet compression codec.
            compression_level: Codec level.
            logger: Logger instance.
        """
        self.output_path = Path(output_path)
        self.feature_dtype = np.dtype(feature_dtype)
        self.compression = compression
        self.compression_level = compression_level
        self.logger = logger

        self._splits: Dict[str, SplitManifest] = {}
        self._metadata: Optional[Dict[str, Any]] = None
        self._parameters: Optional[Dict[str, Any]] = None
        self._input_rows: Dict[str, int] = {}
        self._pure_profiles: Dict[str, bool] = {}

    # -- Paths ---------------------------------------------------------------

    def _split_dir(self, split_name: str) -> Path:
        return self.output_path / "outputs" / split_name

    def _inputs_path(self, split_name: str) -> Path:
        return self.output_path / "inputs" / f"{split_name}.parquet"

    # -- Metadata ------------------------------------------------------------

    def write_metadata(self, metadata: Any, parameters: Any) -> None:
        """Record generation metadata and parameters.

        Args:
            metadata: A ``GenerationMetadata`` (or any object exposing
                ``to_dict``, or a plain dictionary).
            parameters: A ``GenerationParameters``, similarly duck-typed.
        """
        self._metadata = metadata.to_dict() if hasattr(metadata, "to_dict") else dict(metadata)
        if hasattr(parameters, "cell_types_mapping"):
            self._parameters = {
                "cell_types_mapping": dict(parameters.cell_types_mapping),
                "gr_groups_mapping": dict(parameters.gr_groups_mapping),
                "substitution_method": parameters.substitution_method,
                "grg_sampling_method": parameters.grg_sampling_method,
            }
        else:
            self._parameters = dict(parameters)

    def write_inputs(self, split_name: str, df: pd.DataFrame) -> Path:
        """Write the per-split input reads, preserving row order exactly.

        Args:
            split_name: Name of the split.
            df: The reads passed to the generator for this split.

        Returns:
            Path to the written Parquet file.
        """
        dest = self._inputs_path(split_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            dest,
            compression=self.compression,
            compression_level=self.compression_level,
            version="2.6",
        )
        written = pq.ParquetFile(dest).metadata.num_rows
        if written != len(df):
            raise ValueError(
                f"inputs/{split_name}: wrote {written} rows, expected {len(df)}."
            )
        self._input_rows[split_name] = written
        return dest

    def write_pure_profiles(self, split_name: str, pure_profile: Any) -> Optional[Path]:
        """Write the pure profiles and uniform prior for a split.

        Args:
            split_name: Name of the split.
            pure_profile: A ``PureProfileResult``, or ``None`` to skip.

        Returns:
            Path to the written ``.npz``, or ``None`` if there was nothing.
        """
        if pure_profile is None:
            self._pure_profiles[split_name] = False
            return None
        if getattr(pure_profile, "string_matrices", None) is not None:
            raise NotImplementedError(
                "String columns in pure profiles are not supported by the "
                "columnar format."
            )
        dest = self._split_dir(split_name) / PURE_PROFILES_FILE
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            feature_matrices=np.asarray(pure_profile.feature_matrices),
            uniform_prior=np.asarray(pure_profile.uniform_prior),
            numeric_columns=np.asarray(pure_profile.numeric_columns, dtype=object),
        )
        self._pure_profiles[split_name] = True
        return dest

    # -- Pseudobulk outputs ---------------------------------------------------

    def open_split(self, split_name: str, feature_columns: List[str]) -> "SplitWriter":
        """Open a streaming writer for one split's pseudobulk outputs.

        Args:
            split_name: Name of the split.
            feature_columns: Ordered aggregated-feature column names.

        Returns:
            A :class:`SplitWriter` usable as a context manager.
        """
        return SplitWriter(self, split_name, feature_columns)

    def _record_split(self, split_name: str, manifest: SplitManifest) -> None:
        manifest.has_pure_profiles = self._pure_profiles.get(split_name, False)
        manifest.n_input_rows = self._input_rows.get(split_name, 0)
        self._splits[split_name] = manifest

    # -- Finalisation ---------------------------------------------------------

    def finalize(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Write ``manifest.json``, ``metadata.json`` and ``parameters.json``.

        Args:
            extra: Additional key/value pairs recorded in the manifest, for
                example the checksum of a converted source file.

        Returns:
            Path to the store directory.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        # Re-stamp the derived fields, in case pure profiles or inputs were
        # written after the split was closed.
        for split_name, manifest in self._splits.items():
            manifest.has_pure_profiles = self._pure_profiles.get(split_name, False)
            manifest.n_input_rows = self._input_rows.get(split_name, 0)

        manifest = {
            "format_version": FORMAT_VERSION,
            "feature_dtype": str(self.feature_dtype),
            "compression": self.compression,
            "splits": {name: m.to_dict() for name, m in self._splits.items()},
        }
        if extra:
            manifest.update(extra)

        (self.output_path / COLUMNAR_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        (self.output_path / "metadata.json").write_text(
            json.dumps(self._metadata or {}, indent=2, sort_keys=True, default=str)
        )
        (self.output_path / "parameters.json").write_text(
            json.dumps(self._parameters or {}, indent=2, sort_keys=True, default=str)
        )
        self.logger.info("Columnar pseudobulk store written to %s", self.output_path)
        return self.output_path

    # -- Generator-compatible entry point -------------------------------------

    def consolidate(
        self,
        checkpoint_manager: Any,
        metadata: Any,
        parameters: Any,
        input_dfs: Dict[str, pd.DataFrame],
        pure_profiles: Dict[str, Any],
        batch_chunk: int = 500,
    ) -> Path:
        """Consolidate generation batches into a columnar store.

        Mirrors
        :meth:`~syto.data.pseudobulk_hdf5_utils.PseudobulkHDF5ConsolidationWriter.consolidate`
        so the generator can choose a backend without further changes.

        Args:
            checkpoint_manager: The run's ``CheckpointManager``.
            metadata: ``GenerationMetadata`` for the run.
            parameters: ``GenerationParameters`` for the run.
            input_dfs: Per-split input read frames.
            pure_profiles: Per-split ``PureProfileResult``.
            batch_chunk: Pseudobulks buffered per Parquet write.

        Returns:
            Path to the store directory.
        """
        from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5BatchWriter

        temp_path = self.output_path.with_name(self.output_path.name + ".tmp")
        final_path = self.output_path
        if temp_path.exists():
            shutil.rmtree(temp_path)
        self.output_path = temp_path

        try:
            self.write_metadata(metadata, parameters)
            for split_name, df in input_dfs.items():
                self.write_inputs(split_name, df)

            config = checkpoint_manager.load_config()
            for split_name in config.splits_order:
                self.write_pure_profiles(split_name, pure_profiles.get(split_name))

                batches_dir = checkpoint_manager.get_split_batches_dir(split_name)
                batch_reader = PseudobulkHDF5BatchWriter(batches_dir)
                checkpoint = checkpoint_manager.load_split_checkpoint(split_name)

                writer: Optional[SplitWriter] = None
                buffer: List[Any] = []
                for batch_idx in sorted(checkpoint.completed_batches):
                    for result in batch_reader.read_batch(batch_idx):
                        if writer is None:
                            columns = list(result.aggregated_features.columns)
                            writer = self.open_split(split_name, columns)
                            writer.open()
                        buffer.append(result)
                        if len(buffer) >= batch_chunk:
                            writer.write_results(buffer)
                            buffer = []
                if writer is not None:
                    if buffer:
                        writer.write_results(buffer)
                    writer.close()

            self.finalize()
            if final_path.exists():
                shutil.rmtree(final_path)
            temp_path.rename(final_path)
            self.output_path = final_path
            return final_path
        except Exception:
            self.output_path = final_path
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
            raise


class SplitWriter:
    """Streaming writer for one split's pseudobulk outputs."""

    def __init__(
        self,
        parent: PseudobulkColumnarWriter,
        split_name: str,
        feature_columns: List[str],
    ):
        """Initialise the split writer.

        Args:
            parent: The owning :class:`PseudobulkColumnarWriter`.
            split_name: Name of the split.
            feature_columns: Ordered aggregated-feature column names.
        """
        self.parent = parent
        self.split_name = split_name
        self.feature_columns = list(feature_columns)
        self.n_written = 0
        self.n_grg: Optional[int] = None
        self.n_classes: Optional[int] = None

        self._features_writer: Optional[pq.ParquetWriter] = None
        self._proportions_writer: Optional[pq.ParquetWriter] = None
        self._counts_writer: Optional[pq.ParquetWriter] = None

    def __enter__(self) -> "SplitWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        """Create the split's output directory."""
        self.parent._split_dir(self.split_name).mkdir(parents=True, exist_ok=True)

    # -- Schemas --------------------------------------------------------------

    def _features_schema(self) -> pa.Schema:
        dtype = pa.from_numpy_dtype(self.parent.feature_dtype)
        return pa.schema(
            [(COL_PB_INDEX, pa.int32()), (COL_GRG_ROW, pa.int16())]
            + [(c, dtype) for c in self.feature_columns]
        )

    def _proportions_schema(self) -> pa.Schema:
        return pa.schema(
            [
                (COL_PB_INDEX, pa.int32()),
                ("seed", pa.int64()),
                ("n_reads_sampled", pa.int64()),
            ]
            + [(_target_column(i), pa.float64()) for i in range(self.n_classes)]
            + [(_actual_column(i), pa.float64()) for i in range(self.n_classes)]
        )

    def _counts_schema(self) -> pa.Schema:
        return pa.schema(
            [(COL_PB_INDEX, pa.int32()), (COL_CLASS_ROW, pa.int16())]
            + [(_grg_column(j), pa.int32()) for j in range(self.n_grg)]
        )

    def _writer(self, path: Path, schema: pa.Schema) -> pq.ParquetWriter:
        return pq.ParquetWriter(
            path,
            schema,
            compression=self.parent.compression,
            compression_level=self.parent.compression_level,
            version="2.6",
        )

    # -- Writing --------------------------------------------------------------

    def write_results(self, results: List[Any]) -> None:
        """Append a chunk of ``PseudobulkResult`` objects.

        Args:
            results: Pseudobulk results, in ascending index order.
        """
        if not results:
            return

        first = results[0]
        if self.n_grg is None:
            self.n_grg = int(first.n_samples_per_class_per_grg.shape[1])
            self.n_classes = int(first.n_samples_per_class_per_grg.shape[0])
            split_dir = self.parent._split_dir(self.split_name)
            self._features_writer = self._writer(
                split_dir / FEATURES_FILE, self._features_schema()
            )
            self._proportions_writer = self._writer(
                split_dir / PROPORTIONS_FILE, self._proportions_schema()
            )
            self._counts_writer = self._writer(
                split_dir / N_READS_PER_GR_FILE, self._counts_schema()
            )

        self._features_writer.write_table(self._features_table(results))
        self._proportions_writer.write_table(self._proportions_table(results))
        self._counts_writer.write_table(self._counts_table(results))
        self.n_written += len(results)

    def _features_table(self, results: List[Any]) -> pa.Table:
        dtype = self.parent.feature_dtype
        blocks, pb_idx, grg_rows = [], [], []
        for r in results:
            frame = r.aggregated_features
            extra = set(frame.columns) - set(self.feature_columns)
            if extra:
                raise ValueError(
                    f"Pseudobulk {r.index} has unexpected feature columns: "
                    f"{sorted(extra)}."
                )
            values = frame[self.feature_columns].to_numpy(dtype=dtype, copy=False)
            if values.shape[0] != self.n_grg:
                raise ValueError(
                    f"Pseudobulk {r.index} has {values.shape[0]} feature rows, "
                    f"expected {self.n_grg}. Ragged stores are not supported."
                )
            blocks.append(values)
            pb_idx.append(np.full(values.shape[0], r.index, dtype=np.int32))
            grg_rows.append(np.arange(values.shape[0], dtype=np.int16))

        stacked = np.concatenate(blocks, axis=0)
        arrays = [
            pa.array(np.concatenate(pb_idx)),
            pa.array(np.concatenate(grg_rows)),
        ] + [pa.array(stacked[:, i]) for i in range(stacked.shape[1])]
        return pa.Table.from_arrays(arrays, schema=self._features_schema())

    def _proportions_table(self, results: List[Any]) -> pa.Table:
        targets = np.stack([np.asarray(r.target_proportions, dtype=np.float64) for r in results])
        actuals = np.stack([np.asarray(r.actual_proportions, dtype=np.float64) for r in results])
        arrays = [
            pa.array(np.array([r.index for r in results], dtype=np.int32)),
            pa.array(np.array([r.seed for r in results], dtype=np.int64)),
            pa.array(np.array([r.n_reads_sampled for r in results], dtype=np.int64)),
        ]
        arrays += [pa.array(targets[:, i]) for i in range(self.n_classes)]
        arrays += [pa.array(actuals[:, i]) for i in range(self.n_classes)]
        return pa.Table.from_arrays(arrays, schema=self._proportions_schema())

    def _counts_table(self, results: List[Any]) -> pa.Table:
        blocks, pb_idx, class_rows = [], [], []
        for r in results:
            counts = np.asarray(r.n_samples_per_class_per_grg)
            if counts.shape != (self.n_classes, self.n_grg):
                raise ValueError(
                    f"Pseudobulk {r.index} has n_reads_per_gr shape "
                    f"{counts.shape}, expected {(self.n_classes, self.n_grg)}."
                )
            blocks.append(counts.astype(np.int32, copy=False))
            pb_idx.append(np.full(self.n_classes, r.index, dtype=np.int32))
            class_rows.append(np.arange(self.n_classes, dtype=np.int16))

        stacked = np.concatenate(blocks, axis=0)
        arrays = [
            pa.array(np.concatenate(pb_idx)),
            pa.array(np.concatenate(class_rows)),
        ] + [pa.array(stacked[:, j]) for j in range(self.n_grg)]
        return pa.Table.from_arrays(arrays, schema=self._counts_schema())

    def close(self) -> None:
        """Close the Parquet writers and record the split in the manifest."""
        for writer in (
            self._features_writer,
            self._proportions_writer,
            self._counts_writer,
        ):
            if writer is not None:
                writer.close()
        self._features_writer = None
        self._proportions_writer = None
        self._counts_writer = None

        self.parent._record_split(
            self.split_name,
            SplitManifest(
                n_pseudobulks=self.n_written,
                n_grg=int(self.n_grg or 0),
                n_classes=int(self.n_classes or 0),
                feature_columns=self.feature_columns,
                has_pure_profiles=False,
                n_input_rows=0,
            ),
        )


# =============================================================================
# Reader
# =============================================================================


class PseudobulkColumnarReader(BasePseudobulkReader):
    """Read a columnar pseudobulk store.

    Satisfies the same contract as
    :class:`~syto.data.pseudobulk_hdf5_utils.PseudobulkHDF5Reader`; prefer
    :func:`~syto.data.pseudobulk_store.open_pseudobulk_store` over constructing
    it directly.
    """

    def __init__(
        self,
        path: "Path | str",
        logger: logging.Logger = _module_logger,
    ):
        """Initialise the reader.

        Args:
            path: Path to the store directory.
            logger: Logger instance.

        Raises:
            FileNotFoundError: If the directory holds no ``manifest.json``.
        """
        self.path = Path(path)
        self.logger = logger
        manifest_path = self.path / COLUMNAR_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No {COLUMNAR_MANIFEST} in {self.path}.")
        self._manifest = json.loads(manifest_path.read_text())
        self._splits = {
            name: SplitManifest.from_dict(data)
            for name, data in self._manifest.get("splits", {}).items()
        }
        self._metadata: Optional[Dict[str, Any]] = None
        self._parameters: Optional[Dict[str, Any]] = None

    # -- Paths and small JSON -------------------------------------------------

    def _split_dir(self, split_name: str) -> Path:
        return self.path / "outputs" / split_name

    def _split_manifest(self, split_name: str) -> SplitManifest:
        if split_name not in self._splits:
            raise KeyError(
                f"No pseudobulks found for split '{split_name}' in {self.path} "
                f"(available: {sorted(self._splits)})."
            )
        return self._splits[split_name]

    def _load_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            path = self.path / "metadata.json"
            self._metadata = json.loads(path.read_text()) if path.is_file() else {}
        return self._metadata

    def _load_parameters(self) -> Dict[str, Any]:
        if self._parameters is None:
            path = self.path / "parameters.json"
            self._parameters = json.loads(path.read_text()) if path.is_file() else {}
        return self._parameters

    # -- Contract: enumeration ------------------------------------------------

    def list_splits(self) -> List[str]:
        """List the split names present in the store."""
        return list(self._splits)

    def count_pseudobulks(self, split_name: str) -> int:
        """Return how many pseudobulks are stored for a split.

        Raises:
            KeyError: If no pseudobulks are found for *split_name*.
        """
        return self._split_manifest(split_name).n_pseudobulks

    # -- Contract: matrices ---------------------------------------------------

    def read_pseudobulk_matrices(
        self, split_name: str, num_pred_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read pseudobulk feature matrices and target proportions for a split.

        Args:
            split_name: Name of the split (e.g. ``"train"``).
            num_pred_classes: Number of prediction classes to keep.

        Returns:
            Tuple ``(features, target_proportions)`` where ``features`` has
            shape ``(n_pseudobulks, n_gr_groups, num_pred_classes)`` and
            ``target_proportions`` has shape ``(n_pseudobulks, n_classes)``.
        """
        manifest = self._split_manifest(split_name)
        pred_idx = self._prediction_indices(manifest.feature_columns, num_pred_classes)
        wanted = [manifest.feature_columns[i] for i in pred_idx]

        table = pq.read_table(
            self._split_dir(split_name) / FEATURES_FILE, columns=wanted
        )
        if table.num_rows == 0:
            raise ValueError(
                f"No pseudobulk samples found for split '{split_name}' in {self.path}."
            )
        features = np.column_stack(
            [table.column(name).to_numpy(zero_copy_only=False) for name in wanted]
        ).reshape(manifest.n_pseudobulks, manifest.n_grg, len(wanted))

        target_columns = [_target_column(i) for i in range(manifest.n_classes)]
        proportions = pq.read_table(
            self._split_dir(split_name) / PROPORTIONS_FILE, columns=target_columns
        )
        targets = np.column_stack(
            [proportions.column(c).to_numpy(zero_copy_only=False) for c in target_columns]
        )
        return features, targets

    def read_pure_feature_matrix(
        self, split_name: str, num_pred_classes: int
    ) -> np.ndarray:
        """Read pure cell-type profiles for a split.

        Returns:
            Array of shape ``(n_cell_types, n_gr_groups, num_pred_classes)``.
        """
        path = self._split_dir(split_name) / PURE_PROFILES_FILE
        if not path.is_file():
            raise KeyError(
                f"No pure profiles found for split '{split_name}' in {self.path}."
            )
        with np.load(path, allow_pickle=True) as data:
            matrices = data["feature_matrices"]
            columns = self._decode_columns(data["numeric_columns"])
        pred_idx = self._prediction_indices(columns, num_pred_classes)
        return matrices[:, :, pred_idx]

    def read_uniform_prior(self, split_name: str) -> pd.DataFrame:
        """Read the uniform prior matrix for a split.

        Returns:
            DataFrame with the pure-profile numeric columns.
        """
        path = self._split_dir(split_name) / PURE_PROFILES_FILE
        if not path.is_file():
            raise KeyError(
                f"No pure profiles found for split '{split_name}' in {self.path}."
            )
        with np.load(path, allow_pickle=True) as data:
            prior = data["uniform_prior"]
            columns = self._decode_columns(data["numeric_columns"])
        return pd.DataFrame(prior, columns=columns)

    # -- Contract: reconstruction primitives ----------------------------------

    def iter_pseudobulk_params(
        self, split_name: str
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(seed, n_samples_per_class_per_grg, target_proportions)``.

        Yields in pseudobulk index order.
        """
        manifest = self._split_manifest(split_name)
        split_dir = self._split_dir(split_name)

        target_columns = [_target_column(i) for i in range(manifest.n_classes)]
        proportions = pq.read_table(
            split_dir / PROPORTIONS_FILE, columns=["seed"] + target_columns
        )
        seeds = proportions.column("seed").to_numpy(zero_copy_only=False)
        targets = np.column_stack(
            [proportions.column(c).to_numpy(zero_copy_only=False) for c in target_columns]
        )

        grg_columns = [_grg_column(j) for j in range(manifest.n_grg)]
        counts_table = pq.read_table(split_dir / N_READS_PER_GR_FILE, columns=grg_columns)
        counts = np.column_stack(
            [counts_table.column(c).to_numpy(zero_copy_only=False) for c in grg_columns]
        ).reshape(manifest.n_pseudobulks, manifest.n_classes, manifest.n_grg)

        for i in range(manifest.n_pseudobulks):
            yield int(seeds[i]), counts[i], targets[i]

    def _read_input_dataframe(self, split_name: str) -> pd.DataFrame:
        """Read back the per-split input reads, in generation row order."""
        path = self.path / "inputs" / f"{split_name}.parquet"
        if not path.is_file():
            raise KeyError(
                f"No input data found for split '{split_name}' in {self.path}."
            )
        return pq.read_table(path).to_pandas()

    def _read_grg_label_column(self) -> str:
        """Read the GR-group label column name used during generation."""
        metadata = self._load_metadata()
        if "grg_id_column" not in metadata:
            raise KeyError(f"'grg_id_column' missing from metadata.json in {self.path}.")
        return str(metadata["grg_id_column"])

    def _read_gr_groups_mapping(self) -> Dict[str, int]:
        """Read the GR-group label -> index mapping."""
        parameters = self._load_parameters()
        mapping = parameters.get("gr_groups_mapping", {})
        return {str(name): int(idx) for name, idx in mapping.items()}
