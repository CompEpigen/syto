""" """

from typing import Dict, List, Optional

import pandas as pd

from syto.data.atlases.abstract_atlas import AbstractMethylationAtlas


class UXMMethylationAtlas(AbstractMethylationAtlas):
    """Methylation atlases of the UXM paper."""

    REQUIRED_COLUMNS = AbstractMethylationAtlas.REQUIRED_COLUMNS.union(
        {"startCpG", "endCpG", "direction"}
    )
    VALID_DIRECTIONS = {"U", "M"}
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

        Note: M/U/X methylation state scoring (``mark_records_methyl_state``)
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
