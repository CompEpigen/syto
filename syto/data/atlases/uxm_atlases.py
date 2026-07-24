""" """

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from syto.data.atlases.abstract_atlas import AbstractMethylationAtlas
from syto.data.dataset import resolve_column

_module_logger = logging.getLogger(__name__)


def mark_records_methyl_state(
    reads_data: pd.DataFrame,
    methyl_tr: float = 0.75,
    unmethyl_tr: float = 0.25,
) -> pd.DataFrame:
    """Classify each CpG read as methylated (M), unmethylated (U), or ambiguous (X).

    A read is classified by its methylation rate ``M / (M + U)`` over the
    called CpGs in its methylation pattern (``'1'`` = methylated, ``'0'`` =
    unmethylated; other characters are ignored):

    * **U** if ``rate <= unmethyl_tr``
    * **M** if ``rate >= methyl_tr``
    * **X** otherwise

    This reproduces the ``wgbstools homog`` classification used to build the
    UXM reference atlases; the defaults ``unmethyl_tr=0.25`` / ``methyl_tr=0.75``
    correspond to the ``l4`` atlases (read length rlen = 4).

    Adds columns ``M``, ``U``, ``NCPGS``, ``M_rate``, ``record_M``,
    ``record_U``, ``record_X`` to ``reads_data`` in-place and returns it.
    """
    meth_col = resolve_column(reads_data.columns, "methylation_ids")
    pat = reads_data[meth_col]
    reads_data["M"] = pat.apply(lambda x: x.count("1"))
    reads_data["U"] = pat.apply(lambda x: x.count("0"))
    reads_data["NCPGS"] = reads_data["M"] + reads_data["U"]
    reads_data["M_rate"] = reads_data["M"] / reads_data["NCPGS"].replace(0, np.nan)

    is_M = reads_data["M_rate"] >= methyl_tr
    is_U = reads_data["M_rate"] <= unmethyl_tr
    reads_data["record_M"] = is_M.astype(int)
    reads_data["record_U"] = is_U.astype(int)
    reads_data["record_X"] = (~is_M & ~is_U).astype(int)
    return reads_data


class UXMMethylationAtlas(AbstractMethylationAtlas):
    """Methylation atlases of the UXM paper."""

    REQUIRED_COLUMNS = AbstractMethylationAtlas.REQUIRED_COLUMNS.union(
        {"startCpG", "endCpG", "direction"}
    )
    VALID_DIRECTIONS = {"U", "M"}
    # Region-defining columns of a UXM marker set (output of wgbstools
    # find_markers).  These are *not* derivable from reads and must be supplied
    # to :meth:`from_reads`; only the cell-type fraction columns are populated
    # from reads.  Order matches the canonical Atlas.*.tsv layout.
    _MARKER_COLUMNS = [
        "chr",
        "start",
        "end",
        "startCpG",
        "endCpG",
        "target",
        "name",
        "direction",
    ]
    EXPECTED_CTYPE_COLUMNS = [
        "Adipocytes",
        "Bladder-Ep",
        "Blood-B",
        "Blood-Granul",
        "Blood-Mono+Macro",
        "Blood-NK",
        "Blood-T",
        "Bone-Osteob",
        "Breast-Basal-Ep",
        "Breast-Luminal-Ep",
        "Colon-Ep",
        "Colon-Fibro",
        "Dermal-Fibro",
        "Endothel",
        "Epid-Kerat",
        "Eryth-prog",
        "Fallopian-Ep",
        "Gallbladder",
        "Gastric-Ep",
        "Head-Neck-Ep",
        "Heart-Cardio",
        "Heart-Fibro",
        "Kidney-Ep",
        "Liver-Hep",
        "Lung-Ep-Alveo",
        "Lung-Ep-Bron",
        "Neuron",
        "Oligodend",
        "Ovary-Ep",
        "Pancreas-Acinar",
        "Pancreas-Alpha",
        "Pancreas-Beta",
        "Pancreas-Delta",
        "Pancreas-Duct",
        "Prostate-Ep",
        "Skeletal-Musc",
        "Small-Int-Ep",
        "Smooth-Musc",
        "Thyroid-Ep",
    ]

    def __init__(
        self,
        atlas_name: str,
        reference_genome: str,
        atlas_path: Optional[str] = None,
        atlas_df: Optional[pd.DataFrame] = None,
        sep: str = "\t",
        ignore: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ):
        """Initialize the UXM methylation atlas.

        Parameters
        ----------
        atlas_name, reference_genome, atlas_path / atlas_df, sep
            Same as before.
        ignore : list of str, optional
            Cell-type column names to exclude from ``ref_cells``.
        include : list of str, optional
            When provided, only these cell-type columns are kept in ``ref_cells``.
        """
        if atlas_path is not None and atlas_df is not None:
            raise ValueError("Only one of atlas_path or atlas_df should be provided.")
        elif atlas_path is not None:
            self._atlas = pd.read_csv(atlas_path, sep=sep)
        elif atlas_df is not None:
            self._atlas = atlas_df
        else:
            raise ValueError("Either atlas_path or atlas_df must be provided.")

        super().__init__()
        self.atlas_name = atlas_name
        self._reference_genome = reference_genome
        self._check_atlas_format(self._atlas)

        # Keep the atlas sorted by genomic position for efficient sweep queries
        self._atlas = self._atlas.sort_values(["chr", "start", "end"]).reset_index(
            drop=True
        )

        # Compute filtered ref_cells once at construction time
        all_cells = [c for c in self._atlas.columns if c in self.EXPECTED_CTYPE_COLUMNS]
        if ignore:
            all_cells = [c for c in all_cells if c not in ignore]
        if include:
            all_cells = [c for c in all_cells if c in include]
        self._ref_cells: List[str] = all_cells
        self._atlas = self._atlas[self._atlas["target"].apply(lambda x: x in self._ref_cells)]

    # ------------------------------------------------------------------
    # Factory: build atlas from labeled reads
    # ------------------------------------------------------------------

    @classmethod
    def from_reads(
        cls,
        reads: pd.DataFrame,
        atlas_name: str,
        reference_genome: str,
        labels_dict: Dict[Union[int, str], str],
        markers: Union[pd.DataFrame, str, "UXMMethylationAtlas"],
        *,
        min_cpgs: int = 4,
        methyl_tr: float = 0.75,
        unmethyl_tr: float = 0.25,
        output_path: Optional[str] = None,
        sep: str = "\t",
        ignore: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ) -> "UXMMethylationAtlas":
        """Populate a UXM atlas' cell-type columns by aggregating reads per region.

        This mirrors the UXM reference workflow (``wgbstools find_markers`` +
        ``UXM_deconv/src/build.py``): the marker **regions** — including their
        ``startCpG``/``endCpG``, ``target`` and ``direction`` — are found by
        comparing target vs. background and therefore cannot be derived from a
        single cell type's reads.  They are supplied via ``markers``.  Only the
        per-cell-type fraction columns are computed here, from ``reads``.

        For every atlas region and cell type the value is the fraction of that
        cell type's reads whose methylation state matches the region's
        ``direction`` — the fraction of **U** (unmethylated) reads for
        ``direction == "U"`` regions, or **M** (methylated) reads for
        ``direction == "M"``:

            value = n_matching_reads / (n_U + n_X + n_M)

        Reads are classified into U/X/M by :func:`mark_records_methyl_state`
        (which reproduces the ``wgbstools homog`` logic), and reads covering
        fewer than ``min_cpgs`` called CpGs in the region are discarded.  The
        defaults ``min_cpgs=4``, ``unmethyl_tr=0.25``, ``methyl_tr=0.75``
        correspond to the ``l4`` atlases (rlen = 4).  Regions with no coverage
        for a cell type are left as ``NaN`` (as in ``build.py``).

        Parameters
        ----------
        reads : pd.DataFrame
            Reads **already prepared** via :meth:`prepare_reads` (overlapped
            with the atlas and trimmed to region boundaries), so each row
            carries a ``name`` (atlas region) and a methylation-pattern column
            restricted to that region.  A read overlapping several regions is
            expected to appear once per region.  Must also contain
            ``original_label``.
        atlas_name, reference_genome
            Passed through to the constructor.
        labels_dict : dict
            ``{label_id -> cell_type_name}`` used to resolve ``original_label``
            to a cell-type string.  Keys may be ints or strings.  The atlas will
            contain one column per cell type in this mapping.
        markers : UXMMethylationAtlas | pd.DataFrame | str
            Source of the region metadata (``chr``, ``start``, ``end``,
            ``startCpG``, ``endCpG``, ``target``, ``name``, ``direction``).
            Accepts an existing atlas object, a DataFrame, or a path to a TSV.
            Any pre-existing cell-type columns are ignored and recomputed.
        min_cpgs : int
            Minimum called CpGs a read must cover in the region to be counted.
        methyl_tr, unmethyl_tr : float
            Methylation-rate thresholds forwarded to
            :func:`mark_records_methyl_state`.
        output_path : str, optional
            If given, write the resulting atlas TSV to this path.
        sep : str
            Column separator for reading ``markers`` (if a path) and for the
            optional output file.
        ignore, include : list of str, optional
            Forwarded to the constructor to filter ``ref_cells``.

        Returns
        -------
        UXMMethylationAtlas
        """
        # ---- Resolve region metadata from the marker source --------------
        if isinstance(markers, cls):
            marker_df = markers.atlas.copy()
        elif isinstance(markers, pd.DataFrame):
            marker_df = markers.copy()
        elif isinstance(markers, str):
            marker_df = pd.read_csv(markers, sep=sep)
        else:
            raise TypeError(
                "markers must be a UXMMethylationAtlas, a DataFrame, or a path; "
                f"got {type(markers).__name__}."
            )

        missing_marker_cols = set(cls._MARKER_COLUMNS) - set(marker_df.columns)
        if missing_marker_cols:
            raise ValueError(
                f"markers is missing required region columns: {missing_marker_cols}"
            )
        region_df = (
            marker_df[cls._MARKER_COLUMNS]
            .drop_duplicates(subset="name")
            .reset_index(drop=True)
        )
        cls._check_direction(region_df)

        # ---- Label resolution --------------------------------------------
        required = {"name", "original_label"}
        missing = required - set(reads.columns)
        if missing:
            raise ValueError(f"reads DataFrame is missing required columns: {missing}")

        int_labels: Dict[int, str] = {int(k): v for k, v in labels_dict.items()}
        cell_type_names = list(dict.fromkeys(int_labels.values()))

        _module_logger.info(
            "Building UXM atlas '%s' from %d reads over %d region(s) · "
            "%d cell type(s): %s",
            atlas_name,
            len(reads),
            len(region_df),
            len(cell_type_names),
            ", ".join(cell_type_names),
        )

        reads = reads.copy()
        reads["cell_type"] = reads["original_label"].map(int_labels)
        unmapped = reads["cell_type"].isna()
        if unmapped.any():
            _module_logger.warning(
                "Dropping %d / %d reads with labels not in labels_dict: %s",
                int(unmapped.sum()),
                len(reads),
                reads.loc[unmapped, "original_label"].unique().tolist(),
            )
            reads = reads[~unmapped]
        if reads.empty:
            raise ValueError("No reads remain after label mapping.")

        # ---- Per-read U/X/M classification (shared with the UXM baseline) -
        reads = mark_records_methyl_state(reads, methyl_tr, unmethyl_tr)
        reads = reads[reads["NCPGS"] >= min_cpgs]
        if reads.empty:
            raise ValueError(
                f"No reads cover at least min_cpgs={min_cpgs} called CpGs."
            )

        # ---- Aggregate counts per (region, cell type) --------------------
        agg = (
            reads.groupby(["name", "cell_type"], sort=False)[
                ["record_U", "record_X", "record_M"]
            ]
            .sum()
            .reset_index()
        )
        # inner-join drops reads whose region is not part of the marker set
        agg = agg.merge(region_df[["name", "direction"]], on="name", how="inner")
        total = (
            agg["record_U"] + agg["record_X"] + agg["record_M"]
        ).to_numpy(dtype=float)
        matching = np.where(
            agg["direction"].to_numpy() == "U",
            agg["record_U"].to_numpy(),
            agg["record_M"].to_numpy(),
        ).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            agg["value"] = np.round(np.where(total > 0, matching / total, np.nan), 3)

        # ---- Pivot to one column per cell type ---------------------------
        wide = agg.pivot_table(
            index="name", columns="cell_type", values="value", dropna=False
        )
        # Reindex to *all* labels_dict cell types (columns with no coverage
        # become all-NaN) so the atlas schema is complete and stable.
        wide = wide.reindex(columns=cell_type_names)
        wide.columns.name = None
        wide = wide.reset_index()

        atlas_df = region_df.merge(wide, on="name", how="left")
        atlas_df = atlas_df[cls._MARKER_COLUMNS + cell_type_names]

        n_covered = int(agg["value"].notna().sum())
        _module_logger.info(
            "UXM atlas ready: %d region(s) × %d cell type(s); "
            "%d region×cell-type cell(s) covered.",
            len(atlas_df),
            len(cell_type_names),
            n_covered,
        )

        if output_path is not None:
            _module_logger.info("Saving atlas to %s …", output_path)
            atlas_df.to_csv(output_path, sep=sep, index=False, na_rep="NA")
            _module_logger.info("Saved.")

        return cls(
            atlas_name=atlas_name,
            reference_genome=reference_genome,
            atlas_df=atlas_df,
            ignore=ignore,
            include=include,
        )

    # ------------------------------------------------------------------
    # AbstractAtlas interface
    # ------------------------------------------------------------------

    @classmethod
    def _check_direction(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check that the 'direction' column only contains valid values."""
        invalid_directions = set(candidate_atlas["direction"]) - cls.VALID_DIRECTIONS
        if invalid_directions:
            raise ValueError(
                f"Atlas DataFrame contains invalid direction values: {invalid_directions}. "
                f"Valid options are: {cls.VALID_DIRECTIONS}"
            )

    @classmethod
    def _check_ctype_columns(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check that all expected cell-type columns are present."""
        missing_ctype_columns = set(cls.EXPECTED_CTYPE_COLUMNS) - set(
            candidate_atlas.columns
        )
        if missing_ctype_columns:
            raise ValueError(
                f"Atlas DataFrame is missing expected cell type columns: {missing_ctype_columns}. "
                f"Expected cell type columns are: {cls.EXPECTED_CTYPE_COLUMNS}"
            )

    @classmethod
    def _check_atlas_format(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check format including direction and cell-type columns."""
        super()._check_atlas_format(candidate_atlas)
        cls._check_direction(candidate_atlas)
        cls._check_ctype_columns(candidate_atlas)

    @property
    def reference_genome(self) -> str:
        """Return the reference genome associated with the atlas."""
        return self._reference_genome

    @property
    def atlas(self) -> pd.DataFrame:
        """Return the atlas DataFrame.

        Columns include those from AbstractMethylationAtlas plus:
        - "direction": "U" (hypomethylated ratio) or "M" (hypermethylated ratio)
        - One column per cell type in EXPECTED_CTYPE_COLUMNS
        """
        return self._atlas

    @property
    def ref_cells(self) -> List[str]:
        """Cell-type columns available in this atlas (filtered by ignore/include)."""
        return self._ref_cells

    # ------------------------------------------------------------------
    # Reads preparation
    # ------------------------------------------------------------------

    def prepare_reads(
        self,
        df: pd.DataFrame,
        *,
        trim: bool = True,
        labels_dict: Optional[Dict] = None,
        cell_type_match_dict: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Overlap, trim, and optionally annotate reads with DMR labels.

        Delegates overlap and trimming to
        :meth:`AbstractMethylationAtlas.prepare_reads`, then applies DMR
        label annotation when ``labels_dict`` is provided.

        Note: M/U/X methylation state scoring (:func:`mark_records_methyl_state`)
        is UXM-algorithm-specific and must be applied separately by the caller.

        Parameters
        ----------
        df : pd.DataFrame
            Input reads sorted by ``["chromosome", "read_start", "read_end"]``.
        trim : bool
            Clip reads to region boundaries.  Default ``True``.
        seq_column, methylation_pattern_column : str
            Column names for DNA sequence and CpG methylation pattern.
        labels_dict : dict, optional
            ``label_id → cell_type_name``.  When provided, adds
            ``dmr_ctype``, ``dmr_ctype_matched``, ``dmr_ctype_label``.
        cell_type_match_dict : dict, optional
            Atlas cell-type name → project cell-type name alias mapping.
        """
        df = super().prepare_reads(df, trim=trim)
        if labels_dict is not None:
            df = self._annotate_dmr_labels(df, labels_dict, cell_type_match_dict or {})
        return df

    def _annotate_dmr_labels(
        self,
        df: pd.DataFrame,
        labels_dict: Dict,
        cell_type_match_dict: Dict[str, str],
    ) -> pd.DataFrame:
        """Join atlas targets and map to integer label indices."""
        labels_dict_reversed: Dict[str, int] = {
            v: int(k) for k, v in labels_dict.items()
        }
        # Drop any pre-existing annotation columns so the merge below cannot
        # produce duplicate column names (e.g. when the input reads were
        # previously annotated and already carry these columns).
        stale_cols = [
            c
            for c in ("target", "dmr_ctype", "dmr_ctype_matched", "dmr_ctype_label")
            if c in df.columns
        ]
        if stale_cols:
            df = df.drop(columns=stale_cols)
        df = pd.merge(df, self._atlas[["name", "target"]], on="name", how="left")
        df.rename(columns={"target": "dmr_ctype"}, inplace=True)

        df["dmr_ctype_matched"] = df["dmr_ctype"].apply(
            lambda x: cell_type_match_dict.get(x, x)
        )
        df["dmr_ctype_label"] = df["dmr_ctype_matched"].apply(
            lambda x: labels_dict_reversed.get(x, None)
        )
        df.dropna(subset=["dmr_ctype_label"], inplace=True)
        df["dmr_ctype_label"] = df["dmr_ctype_label"].astype(int)
        return df
