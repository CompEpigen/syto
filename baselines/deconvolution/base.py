from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class BaselineDeconvolver(ABC):
    """Abstract base class for read-based baseline deconvolution methods.

    Subclasses implement three modular steps — prepare_reads, build_input,
    deconvolute_reads — and expose the atlas they rely on as a property.
    """

    @property
    @abstractmethod
    def atlas(self):
        """Atlas object used for read preparation and reference cell types."""

    @abstractmethod
    def prepare_reads(self, reads: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Overlap reads with atlas regions and apply method-specific preprocessing."""

    @abstractmethod
    def build_input(self, reads: pd.DataFrame, **kwargs) -> dict:
        """Convert prepared reads to the model-specific input representation."""

    @abstractmethod
    def deconvolute_reads(
        self,
        reads: pd.DataFrame,
        labels_dict_reversed: Dict[str, int],
        n_labels: Optional[int] = None,
        prepare: bool = True,
    ):
        """Run the full deconvolution pipeline.

        Parameters
        ----------
        reads : pd.DataFrame
            Raw reads (prepare=True) or atlas-annotated reads (prepare=False).
        labels_dict_reversed : dict
            ``cell_type_name → label_index`` mapping.
        n_labels : int, optional
            Output size; defaults to ``len(labels_dict_reversed)``.
        prepare : bool
            When True, calls ``prepare_reads`` before building input.

        Returns
        -------
        list of float
            Proportions aligned to label order, or None if no regions overlap.
        """

    @staticmethod
    def _sort_reads(reads: pd.DataFrame) -> pd.DataFrame:
        """Sort reads by genomic position and cast coordinate columns to int64."""
        return (
            reads.sort_values(["chromosome", "read_start", "read_end"])
            .reset_index(drop=True)
            .copy()
            .assign(
                read_start=lambda df: df["read_start"].astype("int64"),
                read_end=lambda df: df["read_end"].astype("int64"),
            )
        )
