""" """

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from syto.data.dataset import resolve_column


class AbstractAtlas(ABC):
    """
    Abstract base class for atlases
    """

    REQUIRED_COLUMNS = {"chr", "start", "end", "target", "name"}
    VALID_REFERENCE_GENOMES = {"hg38", "hg19"}
    VALID_CHROMOSOMES = {f"chr{i}" for i in range(1, 23)}.union({"chrX", "chrY"})

    @property
    @abstractmethod
    def reference_genome(self) -> str:
        """Return the reference genome associated with the atlas."""

    @property
    @abstractmethod
    def atlas(self) -> pd.DataFrame:
        """Return the atlas as a pandas DataFrame.

        The atlas DataFrame should have at least the following columns:
        - 'chr': Chromosome name
        - 'start': Start position of the genomic region (1-based)
        - 'end': End position of the genomic region (1-based, inclusive)
        - 'name': Identifier of the genomic region (e.g. "chr1:1000-2000")
        - 'target': The characteristic group for the region (e.g. cell type)
        """

    # ------------------------------------------------------------------
    # Reads preparation methods
    # ------------------------------------------------------------------

    def overlap_reads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate each read with the atlas region(s) it overlaps.

        Reads not overlapping any atlas region are dropped.  A read
        overlapping multiple regions is duplicated once per region.  All
        original columns are preserved; three columns are added:

        * ``name`` — atlas region identifier
        * ``region_start``, ``region_end`` — region boundaries (1-based)

        Coordinates and sequences are **not** modified here; call
        :meth:`trim_reads` afterwards if clipping is required.

        Parameters
        ----------
        df : pd.DataFrame
            Input reads sorted by ``["chromosome", "read_start", "read_end"]``.
        seq_column, methylation_pattern_column : str
            Not used directly but forwarded by :meth:`prepare_reads`
            conventions.

        Returns
        -------
        pd.DataFrame
            Annotated reads with ``name``, ``region_start``, ``region_end``.
        """
        atlas = self.atlas.sort_values(["chr", "start", "end"]).reset_index(drop=True)
        df = df.sort_values(["chromosome", "read_start", "read_end"]).reset_index(
            drop=True
        )
        n_records = len(df)

        if n_records == 0 or len(atlas) == 0:
            empty = df.iloc[0:0].copy()
            empty["name"] = pd.Series(dtype=str)
            empty["region_start"] = pd.Series(dtype=int)
            empty["region_end"] = pd.Series(dtype=int)
            return empty

        # Build per-chromosome start pointer into the sorted reads DataFrame
        chrom_start_idx: Dict[str, int] = {}
        current_chrom = df.iloc[0]["chromosome"]
        chrom_start_idx[current_chrom] = 0
        for i in range(1, n_records):
            chrom = df.iloc[i]["chromosome"]
            if chrom != current_chrom:
                chrom_start_idx[chrom] = i
                current_chrom = chrom

        chrom_base_pointer: Dict[str, int] = {}
        row_indices: List[int] = []
        names: List[str] = []
        region_starts: List[int] = []
        region_ends: List[int] = []

        for _, region in atlas.iterrows():
            chromosome = region["chr"]
            start = region["start"] - 1  # 1-based → 0-based
            end = region["end"]  # exclusive upper bound
            name = region["name"]

            if chromosome not in chrom_base_pointer:
                chrom_base_pointer[chromosome] = chrom_start_idx.get(
                    chromosome, n_records
                )

            base_ptr = chrom_base_pointer[chromosome]

            # Advance past reads that can no longer overlap this or later regions
            while base_ptr < n_records:
                record = df.iloc[base_ptr]
                if record["chromosome"] != chromosome:
                    break
                if record["read_end"] < start:
                    base_ptr += 1
                else:
                    break

            chrom_base_pointer[chromosome] = base_ptr

            scan_ptr = base_ptr
            while scan_ptr < n_records:
                record = df.iloc[scan_ptr]
                if record["chromosome"] != chromosome:
                    break
                if record["read_start"] >= end:
                    break
                if record["read_end"] >= start:
                    row_indices.append(scan_ptr)
                    names.append(name)
                    region_starts.append(region["start"])
                    region_ends.append(region["end"])
                scan_ptr += 1

        if not row_indices:
            empty = df.iloc[0:0].copy()
            empty["name"] = pd.Series(dtype=str)
            empty["region_start"] = pd.Series(dtype=int)
            empty["region_end"] = pd.Series(dtype=int)
            return empty

        overlapped = df.iloc[row_indices].copy().reset_index(drop=True)
        overlapped["name"] = names
        overlapped["region_start"] = region_starts
        overlapped["region_end"] = region_ends
        return overlapped

    def trim_reads(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Clip ``read_start`` and ``read_end`` to the overlapping region.

        Operates row-wise on a DataFrame that has already been passed through
        :meth:`overlap_reads` (so ``region_start`` and ``region_end`` are
        present).  Coordinates only; sequences are left unchanged.

        Subclasses may override to additionally clip sequence-level columns
        (e.g. ``seq`` and ``pattern`` in :class:`UXMMethylationAtlas`).
        """
        df = df.copy()
        region_start_0based = df["region_start"] - 1
        df["read_start"] = df["read_start"].clip(lower=region_start_0based)
        df["read_end"] = df["read_end"].clip(upper=df["region_end"] - 1)
        return df

    @abstractmethod
    def prepare_reads(
        self,
        df: pd.DataFrame,
        *,
        trim: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Prepare reads for atlas-based analysis.

        Implementations must at minimum call :meth:`overlap_reads` to annotate
        reads with atlas region identifiers, and optionally call
        :meth:`trim_reads` (when ``trim=True``) before returning.

        Parameters
        ----------
        df : pd.DataFrame
            Input reads sorted by ``["chromosome", "read_start", "read_end"]``.
        trim : bool
            Whether to clip coordinates (and subclass-specific columns)
            to region boundaries.  Default ``True``.
        seq_column, methylation_pattern_column : str
            Column names for sequence and methylation pattern.
        **kwargs
            Subclass-specific parameters.
        """

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @classmethod
    def _check_atlas_format(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check if the atlas DataFrame has the required format."""
        cls._check_required_columns(candidate_atlas)
        cls._check_chromosomes(candidate_atlas)

    @classmethod
    def _check_required_columns(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check if the atlas DataFrame has the required columns."""
        missing_columns = cls.REQUIRED_COLUMNS - set(candidate_atlas.columns)
        if missing_columns:
            raise ValueError(
                f"Atlas DataFrame is missing required columns: {missing_columns}"
            )

    @classmethod
    def _check_reference_genome(cls, reference_genome: str) -> None:
        """Check if the reference genome is valid."""
        if reference_genome not in cls.VALID_REFERENCE_GENOMES:
            raise ValueError(
                f"Invalid reference genome: {reference_genome}. "
                f"Valid options are: {cls.VALID_REFERENCE_GENOMES}"
            )

    @classmethod
    def _check_chromosomes(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check if the chromosome names in the atlas DataFrame are valid."""
        invalid_chromosomes = set(candidate_atlas["chr"]) - cls.VALID_CHROMOSOMES
        if invalid_chromosomes:
            raise ValueError(
                f"Atlas DataFrame contains invalid chromosome names: {invalid_chromosomes}. "
                f"Valid options are: {cls.VALID_CHROMOSOMES}"
            )


class AbstractMethylationAtlas(AbstractAtlas):
    """Abstract base class for per-CpG methylation atlases.

    Provides concrete implementations of :meth:`trim_reads` (coordinate +
    string slicing) and :meth:`prepare_reads` (overlap → trim) that are
    shared by all methylation atlas subclasses.  Subclasses that need DMR
    label annotation (e.g. :class:`UXMMethylationAtlas`) wrap
    ``super().prepare_reads()`` rather than re-implementing the core logic.

    ``REQUIRED_COLUMNS`` intentionally stays the same as
    :class:`AbstractAtlas` — subclasses add their own structural requirements
    (e.g. ``startCpG``/``endCpG`` for UXM, or just ``name`` for CpG-count
    atlases).
    """

    @property
    @abstractmethod
    def atlas(self) -> pd.DataFrame:
        """Return the atlas as a pandas DataFrame."""

    # ------------------------------------------------------------------
    # Shared reads-preparation logic
    # ------------------------------------------------------------------

    def trim_reads(
        self,
        df: pd.DataFrame,
        seq_column: str = "seq",
        methylation_pattern_column: str = "pattern",
        **kwargs,
    ) -> pd.DataFrame:
        """Clip coordinates **and** re-slice ``seq`` / ``pattern`` to the region.

        Extends the base-class coordinate clipping with methylation-atlas
        specific string trimming: the methylation pattern and DNA sequence
        strings are sliced to match the trimmed genomic span.

        Expects ``df`` to carry ``region_start`` and ``region_end`` columns
        (i.e. :meth:`overlap_reads` must have been called first).
        """
        orig_start = df["read_start"].copy()
        df = super().trim_reads(df, **kwargs)  # clips read_start, read_end

        offsets = (df["read_start"] - orig_start).values
        lengths = (df["read_end"] - df["read_start"] + 1).values

        df[seq_column] = [
            s[o : o + l] for s, o, l in zip(df[seq_column].tolist(), offsets, lengths)
        ]
        df[methylation_pattern_column] = [
            s[o : o + l]
            for s, o, l in zip(
                df[methylation_pattern_column].tolist(), offsets, lengths
            )
        ]
        return df

    def prepare_reads(
        self,
        df: pd.DataFrame,
        *,
        trim: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Overlap reads with atlas regions and optionally trim to boundaries.

        This concrete implementation is shared by all methylation atlas
        subclasses.  It performs:

        1. :meth:`overlap_reads` — annotate reads with ``name``,
           ``region_start``, ``region_end``.
        2. :meth:`trim_reads` *(if trim=True)* — clip ``read_start``,
           ``read_end``, and the sequence / pattern strings to the region.

        Subclasses that need DMR label annotation should override
        ``prepare_reads`` to call ``super().prepare_reads(df, ...)`` first
        and then apply their annotation step.

        Parameters
        ----------
        df : pd.DataFrame
            Input reads sorted by ``["chromosome", "read_start", "read_end"]``.
        trim : bool
            Clip reads to region boundaries.  Default ``True``.
        seq_column, methylation_pattern_column : str
            Column names for DNA sequence and CpG methylation pattern.
        **kwargs
            Forwarded to :meth:`trim_reads`.
        """
        dna_col = resolve_column(df.columns, "input_ids")
        meth_col = resolve_column(df.columns, "methylation_ids")
        df = self.overlap_reads(df)
        if trim:
            df = self.trim_reads(
                df,
                seq_column=dna_col,
                methylation_pattern_column=meth_col,
                **kwargs,
            )
        return df
