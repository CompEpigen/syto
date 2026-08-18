"""Format-independent access to generated pseudobulks.

Pseudobulks are produced once and then read by several pipelines
(deconvolution fitting, calibration, baseline deconvolution, inference and the
wizard).  Those consumers only ever need a handful of operations, listed here
as :class:`PseudobulkStore`.  Keeping that contract explicit lets the same
pipelines run against more than one on-disk layout: the consolidated
``pseudobulk.h5`` written by
:class:`~syto.data.pseudobulk_hdf5_utils.PseudobulkHDF5ConsolidationWriter`, and
the columnar store used for publication.

Use :func:`open_pseudobulk_store` rather than constructing a reader directly;
it inspects the path and returns the right backend.

Backends share more than they differ.  Everything that is genuinely
format-specific reduces to six primitives, and :class:`BasePseudobulkReader`
derives the rest -- notably the read-level reconstruction, which is subtle
enough that having one implementation matters more than the small amount of
code it saves.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd

_module_logger = logging.getLogger(__name__)

# Name of the file that identifies a directory as a columnar store.
COLUMNAR_MANIFEST = "manifest.json"

# Conventional file name of a consolidated HDF5 store inside a run directory.
HDF5_DEFAULT_NAME = "pseudobulk.h5"

HDF5_SUFFIXES = {".h5", ".hdf5"}


@runtime_checkable
class PseudobulkStore(Protocol):
    """The operations pipelines perform on a set of generated pseudobulks."""

    def list_splits(self) -> List[str]:
        """Return the split names present in the store."""
        ...

    def count_pseudobulks(self, split_name: str) -> int:
        """Return how many pseudobulks a split holds."""
        ...

    def read_pseudobulk_matrices(
        self, split_name: str, num_pred_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(features, target_proportions)`` for a split.

        ``features`` has shape ``(n_pseudobulks, n_gr_groups, num_pred_classes)``
        and ``target_proportions`` has shape ``(n_pseudobulks, n_classes)``.
        """
        ...

    def read_pure_feature_matrix(
        self, split_name: str, num_pred_classes: int
    ) -> np.ndarray:
        """Return pure cell-type profiles ``(n_cell_types, n_gr_groups, num_pred_classes)``."""
        ...

    def read_uniform_prior(self, split_name: str) -> pd.DataFrame:
        """Return the uniform prior matrix for a split."""
        ...

    def build_reconstruction_state(
        self, split_name: str, class_label_column: str = "original_label"
    ) -> Tuple[pd.DataFrame, Dict[tuple, np.ndarray]]:
        """Return the shared state needed to reconstruct sampled read subsets."""
        ...

    def iter_pseudobulk_params(
        self, split_name: str
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(seed, n_samples_per_class_per_grg, target_proportions)`` per pseudobulk."""
        ...

    def iter_pseudobulk_read_subsets(
        self, split_name: str, class_label_column: str = "original_label"
    ) -> Iterator[Tuple[pd.DataFrame, np.ndarray]]:
        """Yield ``(reads, target_proportions)`` per pseudobulk."""
        ...


class BasePseudobulkReader(ABC):
    """Shared implementation behind every :class:`PseudobulkStore` backend.

    Subclasses implement the six abstract primitives below; the reconstruction
    logic, column selection and label decoding are provided here so that both
    backends reproduce sampling identically.
    """

    path: Path
    logger: logging.Logger

    # -- Format-specific primitives ------------------------------------------

    @abstractmethod
    def list_splits(self) -> List[str]:
        """Return the split names present in the store."""

    @abstractmethod
    def count_pseudobulks(self, split_name: str) -> int:
        """Return how many pseudobulks a split holds."""

    @abstractmethod
    def read_pseudobulk_matrices(
        self, split_name: str, num_pred_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(features, target_proportions)`` for a split."""

    @abstractmethod
    def read_pure_feature_matrix(
        self, split_name: str, num_pred_classes: int
    ) -> np.ndarray:
        """Return the pure cell-type profile matrices for a split."""

    @abstractmethod
    def read_uniform_prior(self, split_name: str) -> pd.DataFrame:
        """Return the uniform prior matrix for a split."""

    @abstractmethod
    def iter_pseudobulk_params(
        self, split_name: str
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield the per-pseudobulk sampling parameters in index order."""

    @abstractmethod
    def _read_input_dataframe(self, split_name: str) -> pd.DataFrame:
        """Return the per-split input reads, in generation row order.

        The row order is load-bearing: :meth:`iter_pseudobulk_read_subsets`
        re-derives positions into this frame from the stored seed, so a frame
        in any other order silently yields the wrong reads.
        """

    @abstractmethod
    def _read_grg_label_column(self) -> str:
        """Return the GR-group label column used during generation."""

    @abstractmethod
    def _read_gr_groups_mapping(self) -> Dict[str, int]:
        """Return the GR-group label -> index mapping."""

    # -- Shared helpers -------------------------------------------------------

    @staticmethod
    def _decode_columns(raw: Any) -> List[str]:
        """Decode a stored column-name attribute into a list of ``str``."""
        return [
            c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw
        ]

    @staticmethod
    def _prediction_indices(columns: List[str], num_pred_classes: int) -> List[int]:
        """Return the column indices of the ``prediction_{i}_wavg`` features.

        Args:
            columns: Ordered list of feature-column names.
            num_pred_classes: Number of prediction classes to extract.

        Returns:
            List of indices (length ``num_pred_classes``) into ``columns``.
        """
        indices: List[int] = []
        for i in range(num_pred_classes):
            name = f"prediction_{i}_wavg"
            if name not in columns:
                raise KeyError(
                    f"Expected feature column '{name}' not found in the store "
                    f"(available columns: {columns})."
                )
            indices.append(columns.index(name))
        return indices

    @staticmethod
    def _build_indices_per_class_and_grg(
        df: pd.DataFrame,
        class_label_column: str,
        grg_label_column: str,
        gr_groups_mapping: Dict[str, int],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Group row positions of ``df`` by ``(class_index, grg_index)``.

        Mirrors
        :meth:`~syto.data.pseudobulk_generator.PseudobulkGenerator._precompute_group_indices`
        so that the resulting dictionary matches the one used at generation
        time (required for the sampling RNG to reproduce the same draws).

        Args:
            df: Input DataFrame for the split.
            class_label_column: Column holding the (integer) class index.
            grg_label_column: Column holding the GR-group label or index.
            gr_groups_mapping: Mapping from GR-group label to GR-group index.

        Returns:
            Dictionary mapping ``(class_index, grg_index)`` to arrays of row
            positions (suitable for ``DataFrame.iloc``).
        """
        for col in (class_label_column, grg_label_column):
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in the reconstructed input "
                    f"DataFrame (available columns: {list(df.columns)})."
                )

        grouped = df.groupby([class_label_column, grg_label_column], sort=False)
        indices_dict: Dict[Tuple[int, int], np.ndarray] = {}
        for (class_label, grg_label), row_indices in grouped.indices.items():
            class_index = int(class_label)
            grg_key = (
                str(grg_label) if str(grg_label) in gr_groups_mapping else grg_label
            )
            if grg_key in gr_groups_mapping:
                grg_index = gr_groups_mapping[grg_key]
            else:
                grg_index = int(grg_label)
            indices_dict[(class_index, grg_index)] = row_indices

        return indices_dict

    # -- Derived operations ---------------------------------------------------

    def build_reconstruction_state(
        self,
        split_name: str,
        class_label_column: str = "original_label",
    ) -> Tuple[pd.DataFrame, Dict[tuple, np.ndarray]]:
        """Load the shared state required to reconstruct pseudobulk read subsets.

        Call this once per split before iterating with
        :meth:`iter_pseudobulk_params`.  The returned objects are read-only and
        safe to share across threads.

        Args:
            split_name: Name of the split (e.g. ``"train"``).
            class_label_column: Column in the input DataFrame holding the
                (integer) class index of each read.

        Returns:
            ``(input_df, indices_per_class_and_grg)`` where ``input_df`` is the
            full per-split read DataFrame and ``indices_per_class_and_grg`` maps
            ``(class_index, grg_index)`` to arrays of row positions for
            ``DataFrame.iloc``.
        """
        input_df = self._read_input_dataframe(split_name)
        grg_label_column = self._read_grg_label_column()
        gr_groups_mapping = self._read_gr_groups_mapping()
        indices_per_class_and_grg = self._build_indices_per_class_and_grg(
            input_df, class_label_column, grg_label_column, gr_groups_mapping
        )
        return input_df, indices_per_class_and_grg

    def iter_pseudobulk_read_subsets(
        self,
        split_name: str,
        class_label_column: str = "original_label",
    ) -> Iterator[Tuple[pd.DataFrame, np.ndarray]]:
        """Reconstruct the read-level subset sampled for each pseudobulk.

        For every pseudobulk in the split, re-derives the exact rows of the
        input DataFrame that were sampled (with replacement) to build it, using
        the stored ``seed`` and ``n_reads_per_gr`` together with
        :func:`syto.data.pseudobulk_generator._sample_read_ids_from_grouped_dataframe`.
        This is the inverse of
        :meth:`~syto.data.pseudobulk_generator.PseudobulkGenerator.generate_single_pseudobulk`.

        Args:
            split_name: Name of the split (e.g. ``"train"``).
            class_label_column: Column in the input DataFrame holding the
                (integer) class index of each read. Must match the
                ``class_label_column`` used during generation
                (default: ``"original_label"``).

        Yields:
            Tuples ``(reads, target_proportions)`` in order of pseudobulk
            index, where ``reads`` is the DataFrame subset of sampled reads
            (rows may repeat, since sampling is performed with replacement)
            and ``target_proportions`` has shape ``(n_classes,)``.
        """
        from syto.data.pseudobulk_generator import (
            _sample_read_ids_from_grouped_dataframe,
        )

        input_df, indices_per_class_and_grg = self.build_reconstruction_state(
            split_name, class_label_column
        )

        for seed, n_samples_per_class_per_grg, target_proportions in (
            self.iter_pseudobulk_params(split_name)
        ):
            read_ids = _sample_read_ids_from_grouped_dataframe(
                n_samples_per_class_per_grg,
                indices_per_class_and_grg,
                seed=seed,
            )
            reads = input_df.iloc[read_ids].reset_index(drop=True)
            yield reads, target_proportions


def detect_store_format(path: "Path | str") -> str:
    """Classify a pseudobulk store path.

    Args:
        path: Path to a consolidated ``.h5`` file, or to a directory holding
            either a columnar store or a ``pseudobulk.h5``.

    Returns:
        ``"columnar"`` or ``"hdf5"``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the path exists but is not a recognisable store.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No pseudobulk store at {path}.")

    if path.is_dir():
        if (path / COLUMNAR_MANIFEST).is_file():
            return "columnar"
        if (path / HDF5_DEFAULT_NAME).is_file():
            return "hdf5"
        raise ValueError(
            f"{path} is a directory but contains neither '{COLUMNAR_MANIFEST}' "
            f"nor '{HDF5_DEFAULT_NAME}'."
        )

    if path.suffix.lower() in HDF5_SUFFIXES:
        return "hdf5"
    raise ValueError(
        f"{path} is not a recognised pseudobulk store (expected a "
        f"{'/'.join(sorted(HDF5_SUFFIXES))} file or a store directory)."
    )


def open_pseudobulk_store(
    path: "Path | str",
    logger: logging.Logger = _module_logger,
) -> PseudobulkStore:
    """Open a pseudobulk store, choosing the backend from the path.

    Args:
        path: Path to a consolidated ``.h5`` file, or to a directory holding
            either a columnar store or a ``pseudobulk.h5``.
        logger: Logger passed through to the backend.

    Returns:
        A reader satisfying :class:`PseudobulkStore`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the path is not a recognisable store.
    """
    path = Path(path)
    kind = detect_store_format(path)

    if kind == "columnar":
        try:
            from syto.data.pseudobulk_columnar import PseudobulkColumnarReader
        except ImportError as exc:  # pragma: no cover - until the backend lands
            raise NotImplementedError(
                f"{path} is a columnar pseudobulk store, but the columnar "
                "reader is not available in this build."
            ) from exc
        return PseudobulkColumnarReader(path, logger=logger)

    # Imported here so that pseudobulk_hdf5_utils can depend on this module.
    from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader

    if path.is_dir():
        path = path / HDF5_DEFAULT_NAME
    return PseudobulkHDF5Reader(path, logger=logger)
