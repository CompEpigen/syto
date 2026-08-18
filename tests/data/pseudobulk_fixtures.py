"""Miniature but structurally faithful pseudobulk stores for tests.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from syto.data.pseudobulk_hdf5_utils import (
    CheckpointManager,
    GenerationMetadata,
    GenerationParameters,
    PseudobulkHDF5BatchWriter,
    PseudobulkHDF5ConsolidationWriter,
    PseudobulkResult,
    PureProfileResult,
)

GRG_LABEL_COLUMN = "dmr_ctype_label"
CLASS_LABEL_COLUMN = "original_label"


def feature_columns(n_predictions: int) -> List[str]:
    """Return the aggregated-feature column names for a store."""
    return (
        [GRG_LABEL_COLUMN]
        + [f"prediction_{i}_wavg" for i in range(n_predictions)]
        + ["total_weight", "n_reads"]
    )


def make_reads(n_classes: int, n_grg: int, rows_per_group: int, seed: int) -> pd.DataFrame:
    """Build an input read frame covering every (class, GR group) pair."""
    rng = np.random.default_rng(seed)
    classes, grgs = [], []
    for c in range(n_classes):
        for g in range(n_grg):
            classes.extend([c] * rows_per_group)
            grgs.extend([g] * rows_per_group)
    n = len(classes)
    return pd.DataFrame(
        {
            CLASS_LABEL_COLUMN: np.array(classes, dtype=np.int64),
            GRG_LABEL_COLUMN: np.array(grgs, dtype=np.int64),
            "NCPGS": rng.integers(1, 9, n).astype(np.int64),
            "name": [f"read_{i}" for i in range(n)],
            "dmr_ctype": [f"ct_{c}" for c in classes],
        }
    )


class PseudobulkFixture:
    """A generated set of batches that can be consolidated by any backend."""

    def __init__(
        self,
        root: Path,
        n_classes: int = 3,
        n_grg: int = 2,
        n_predictions: Optional[int] = None,
        n_pseudobulks: int = 6,
        splits: Sequence[str] = ("train", "valid"),
        batch_size: int = 4,
        rows_per_group: int = 3,
        seed: int = 0,
    ):
        """Generate batches for every split.

        Args:
            root: Directory to build in.
            n_classes: Number of cell-type classes.
            n_grg: Number of GR groups.
            n_predictions: Prediction columns; defaults to *n_classes*. Set it
                higher to model the background-class runs.
            n_pseudobulks: Pseudobulks per split.
            splits: Split names.
            batch_size: Pseudobulks per generation batch.
            rows_per_group: Input reads per (class, GR group) pair.
            seed: RNG seed.
        """
        self.root = Path(root)
        self.n_classes = n_classes
        self.n_grg = n_grg
        self.n_predictions = n_predictions if n_predictions is not None else n_classes
        self.n_pseudobulks = n_pseudobulks
        self.splits = list(splits)
        self.columns = feature_columns(self.n_predictions)

        rng = np.random.default_rng(seed)
        self.input_dfs: Dict[str, pd.DataFrame] = {
            s: make_reads(n_classes, n_grg, rows_per_group, seed + i)
            for i, s in enumerate(self.splits)
        }

        self.manager = CheckpointManager(self.root / "ckpt")
        self.manager.create_config(
            parameters_hash="fixture",
            splits_order=self.splits,
            n_pseudobulks_per_split={s: n_pseudobulks for s in self.splits},
        )

        self.pure_profiles: Dict[str, PureProfileResult] = {}
        for split in self.splits:
            self.manager.create_split_checkpoint(
                split, total_pseudobulks=n_pseudobulks, batch_size=batch_size
            )
            batch_writer = PseudobulkHDF5BatchWriter(
                self.manager.get_split_batches_dir(split)
            )
            index = 0
            batch_idx = 0
            while index < n_pseudobulks:
                results = []
                for _ in range(min(batch_size, n_pseudobulks - index)):
                    target = rng.dirichlet(np.ones(n_classes))
                    results.append(
                        PseudobulkResult(
                            index=index,
                            target_proportions=target,
                            actual_proportions=rng.dirichlet(np.ones(n_classes)),
                            n_reads_sampled=int(rng.integers(5, 20)),
                            n_samples_per_class_per_grg=rng.integers(
                                0, rows_per_group + 1, (n_classes, n_grg)
                            ).astype(np.int64),
                            seed=1000 + index,
                            aggregated_features=pd.DataFrame(
                                rng.random((n_grg, len(self.columns))),
                                columns=self.columns,
                            ),
                        )
                    )
                    index += 1
                batch_writer.write_batch(batch_idx, results)
                self.manager.mark_batch_completed(split, batch_idx)
                batch_idx += 1

            self.pure_profiles[split] = PureProfileResult(
                split_name=split,
                feature_matrices=rng.random((n_classes, n_grg, len(self.columns))),
                uniform_prior=rng.random((n_grg, len(self.columns))),
                numeric_columns=self.columns,
                string_matrices=None,
                string_columns=None,
            )

        self.metadata = GenerationMetadata(
            grg_id_column=GRG_LABEL_COLUMN,
            labeling_scheme="soft_labels",
            classifier="fixture",
            data_watermark="fixture:1",
            data_stats={},
        )
        self.parameters = GenerationParameters(
            cell_types_mapping={f"ct_{i}": i for i in range(n_classes)},
            gr_groups_mapping={str(i): i for i in range(n_grg)},
            substitution_method="uniform_number",
            grg_sampling_method="uniform_multinomial",
        )

    def _consolidate_kwargs(self) -> dict:
        return {
            "checkpoint_manager": self.manager,
            "metadata": self.metadata,
            "parameters": self.parameters,
            "input_dfs": self.input_dfs,
            "pure_profiles": self.pure_profiles,
        }

    def build_hdf5(self, path: Optional[Path] = None) -> Path:
        """Consolidate the batches into a ``pseudobulk.h5``."""
        path = Path(path) if path else self.root / "pseudobulk.h5"
        PseudobulkHDF5ConsolidationWriter(path).consolidate(**self._consolidate_kwargs())
        return path

    def build_columnar(
        self, path: Optional[Path] = None, feature_dtype: str = "float64"
    ) -> Path:
        """Consolidate the batches into a columnar store."""
        from syto.data.pseudobulk_columnar import PseudobulkColumnarWriter

        path = Path(path) if path else self.root / "columnar"
        PseudobulkColumnarWriter(path, feature_dtype=feature_dtype).consolidate(
            **self._consolidate_kwargs()
        )
        return path

    def build_both(self) -> Tuple[Path, Path]:
        """Build both backends from the same batches."""
        return self.build_hdf5(), self.build_columnar()
