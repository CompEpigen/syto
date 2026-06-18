""" """

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from syto.data.atlases.abstract_atlas import AbstractMethylationAtlas

_module_logger = logging.getLogger(__name__)


class CpGBetaCountsMethylationAtlas(AbstractMethylationAtlas):
    """Per-CpG methylation atlas storing both beta values and raw counts.

    Loads a tab-separated meth/cov file where each row is one CpG site:

        CHROM  START  END  name  {CellType}_METH  {CellType}_COV  ...

    ``START`` and ``END`` follow BED convention (0-based, half-open).
    ``name`` identifies the genomic region this CpG belongs to.

    At load time the class:

    * derives region boundaries (``chr``, ``start``, ``end``) for
      :meth:`overlap_reads` by grouping on ``name``;
    * builds a per-region CpG position lookup ``{START → column_index}``
      used by :func:`build_celfieish_input` and :func:`build_celfie_input`;
    * stores beta matrices ``ndarray(T, n_CpGs)`` as ``METH / COV``
      (NaN where coverage is zero) for CelFiE-ISH;
    * stores raw ``METH`` and ``COV`` count matrices for CelFiE.

    Overlap, trimming, and base read preparation are inherited from
    :class:`AbstractMethylationAtlas`.
    """

    # Does not include 'target' — added to the raw TSV in a future pass.
    REQUIRED_COLUMNS = {"chr", "start", "end", "name"}

    _RAW_REQUIRED = {"CHROM", "START", "END", "name"}

    def __init__(
        self,
        atlas_name: str,
        reference_genome: str,
        atlas_path: Optional[str] = None,
        atlas_df: Optional[pd.DataFrame] = None,
        sep: str = "\t",
    ):
        if atlas_path is not None and atlas_df is not None:
            raise ValueError("Only one of atlas_path or atlas_df should be provided.")
        elif atlas_path is not None:
            raw = pd.read_csv(atlas_path, sep=sep)
        elif atlas_df is not None:
            raw = atlas_df.copy()
        else:
            raise ValueError("Either atlas_path or atlas_df must be provided.")

        self._check_raw_format(raw)

        # Extract ordered cell type names from *_METH column suffixes.
        meth_cols = [c for c in raw.columns if c.endswith("_METH")]
        self._cell_types: List[str] = [c[:-5] for c in meth_cols]

        # Build region-level DataFrame for overlap_reads.
        # atlas uses 1-based start convention (overlap_reads does start - 1).
        region_df = (
            raw.groupby("name", sort=False)
            .agg(chr=("CHROM", "first"), start=("START", "min"), end=("END", "max"))
            .reset_index()
        )
        region_df["start"] = region_df["start"] + 1  # BED 0-based → 1-based
        # END is START+1 in BED; max(END) = last_cpg_START + 1, which is
        # already correct as the 1-based inclusive last position.
        self._atlas = (
            region_df.sort_values(["chr", "start", "end"]).reset_index(drop=True)
        )

        # Per-region CpG lookups, beta matrices, and raw count matrices.
        self._cpg_lookup: Dict[str, Dict[int, int]] = {}
        self._n_cpgs: Dict[str, int] = {}
        self._beta_matrices: Dict[str, np.ndarray] = {}
        self._meth_matrices: Dict[str, np.ndarray] = {}
        self._cov_matrices:  Dict[str, np.ndarray] = {}

        T = len(self._cell_types)
        for region_name, group in raw.groupby("name", sort=False):
            group_sorted = group.sort_values("START").reset_index(drop=True)
            positions = group_sorted["START"].tolist()
            n_cpgs = len(positions)

            self._cpg_lookup[region_name] = {pos: idx for idx, pos in enumerate(positions)}
            self._n_cpgs[region_name] = n_cpgs

            meth_mat = np.full((T, n_cpgs), np.nan)
            cov_mat  = np.full((T, n_cpgs), np.nan)
            beta     = np.full((T, n_cpgs), np.nan)
            for t, cell_type in enumerate(self._cell_types):
                meth = group_sorted[f"{cell_type}_METH"].values.astype(float)
                cov  = group_sorted[f"{cell_type}_COV"].values.astype(float)
                meth_mat[t, :] = meth
                cov_mat[t, :]  = cov
                with np.errstate(invalid="ignore", divide="ignore"):
                    beta[t, :] = np.where(cov > 0, meth / cov, np.nan)
            self._meth_matrices[region_name] = meth_mat
            self._cov_matrices[region_name]  = cov_mat
            self._beta_matrices[region_name] = beta

        super().__init__()
        self.atlas_name = atlas_name
        self._reference_genome = reference_genome
        self._check_atlas_format(self._atlas)

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
        output_path: Optional[str] = None,
        sep: str = "\t",
    ) -> "CpGBetaCountsMethylationAtlas":
        """Build a CelFiE-ISH atlas by aggregating methylation counts from reads.

        For every CpG position covered by a read, records one (methylated /
        total) observation keyed by ``(CHROM, START, name, cell_type)``.
        Accumulating across all reads for each cell type produces the per-CpG
        ``METH`` and ``COV`` counts that define the atlas.

        Parameters
        ----------
        reads : pd.DataFrame
            Must contain at least ``pattern``, ``read_start``,
            ``chromosome``, ``name``, ``original_label``.

            ``pattern`` uses **Syto encoding** per base:
            ``'0'`` = unmethylated CpG, ``'1'`` = methylated CpG,
            ``'2'`` (or any other character) = non-CpG / no data.
        atlas_name : str
        reference_genome : str
        labels_dict : dict
            ``{label_id → cell_type_name}`` used to resolve
            ``original_label`` to a cell-type string.  Keys may be
            integers or strings (e.g. from a JSON file).
        output_path : str, optional
            If given, write the resulting meth/cov TSV to this path so
            the atlas can be reloaded with :meth:`__init__`.
        sep : str
            Column separator for the optional output file.

        Returns
        -------
        CpGBetaCountsMethylationAtlas
        """
        from tqdm import tqdm

        required = {"pattern", "read_start", "chromosome", "name", "original_label"}
        missing = required - set(reads.columns)
        if missing:
            raise ValueError(f"reads DataFrame is missing required columns: {missing}")

        # ---- Label resolution --------------------------------------------
        # Normalise labels_dict keys to int for lookup against original_label.
        int_labels: Dict[int, str] = {int(k): v for k, v in labels_dict.items()}
        cell_type_names = list(dict.fromkeys(int_labels.values()))

        _module_logger.info(
            "Building atlas '%s' from %d reads · %d cell type(s): %s",
            atlas_name,
            len(reads),
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
            reads = reads[~unmapped].copy()

        if reads.empty:
            raise ValueError("No reads remain after label mapping.")

        reads_per_ctype = reads.groupby("cell_type").size().to_dict()
        _module_logger.info(
            "Reads per cell type: %s",
            "  ".join(f"{ct}={n:,}" for ct, n in reads_per_ctype.items()),
        )

        # ---- Group-first aggregation -------------------------------------
        # Key insight: building a 40M-row intermediate DataFrame and running
        # groupby on it causes the hang (memory + time).  Instead, sort reads
        # by (chromosome, name, cell_type) so the pandas groupby is a free
        # label-split of a sorted array, then aggregate each small group
        # in-place with np.bincount — a vectorised O(k) operation.
        # The large intermediate DataFrame is never materialised.
        #
        # Two extraction paths:
        #   cpg_sig  — pre-computed [(abs_pos, meth_state), …] per read,
        #              already present in the pipeline output; fastest path.
        #   pattern  — base-level string scan via numpy ASCII comparison;
        #              fallback when cpg_sig is absent.

        use_cpg_sig = "cpg_sig" in reads.columns
        if use_cpg_sig:
            _module_logger.info(
                "Using pre-computed 'cpg_sig' column — skipping pattern scan."
            )
        else:
            _module_logger.info(
                "No 'cpg_sig' column found; scanning 'pattern' strings "
                "(ASCII '0'=unmethylated, '1'=methylated)."
            )

        reads = reads.sort_values(
            ["chromosome", "name", "cell_type"]
        ).reset_index(drop=True)

        groups = list(reads.groupby(["chromosome", "name", "cell_type"], sort=False))
        _module_logger.info(
            "Processing %d (region × cell-type) group(s) …", len(groups)
        )

        out_chroms: list = []
        out_starts: list = []
        out_names:  list = []
        out_cts:    list = []
        out_meths:  list = []
        out_covs:   list = []

        for (chrom, name, ct), group in tqdm(
            groups,
            desc="Aggregating groups",
            unit="group",
        ):
            # -- Extract raw (positions, methylation) arrays for this group --
            if use_cpg_sig:
                pair_arrays = []
                for sig in group["cpg_sig"]:
                    if sig is not None and len(sig) > 0:
                        pair_arrays.append(np.asarray(sig.tolist(), dtype=np.int64))
                if not pair_arrays:
                    continue
                all_pairs = np.vstack(pair_arrays)   # (total_obs, 2)
                positions = all_pairs[:, 0]
                meths     = all_pairs[:, 1].astype(np.int32)
            else:
                pos_list:  list = []
                meth_list: list = []
                for pattern, rs in zip(
                    group["pattern"].tolist(), group["read_start"].tolist()
                ):
                    arr = np.frombuffer(pattern.encode("ascii"), dtype=np.uint8)
                    cpg_mask = (arr == 48) | (arr == 49)
                    offsets = np.where(cpg_mask)[0]
                    if offsets.size == 0:
                        continue
                    pos_list.append(int(rs) + offsets)
                    meth_list.append((arr[cpg_mask] == 49).astype(np.int32))
                if not pos_list:
                    continue
                positions = np.concatenate(pos_list)
                meths     = np.concatenate(meth_list)

            # -- Vectorised aggregation: np.unique + np.bincount -----------
            # np.unique returns sorted unique positions and an inverse index;
            # np.bincount accumulates meth and coverage counts in one pass.
            unique_pos, inv = np.unique(positions, return_inverse=True)
            n = len(unique_pos)
            meth_counts = np.bincount(
                inv, weights=meths.astype(float), minlength=n
            ).astype(np.int32)
            cov_counts = np.bincount(inv, minlength=n).astype(np.int32)

            out_chroms.extend([chrom] * n)
            out_starts.append(unique_pos)
            out_names.extend([name] * n)
            out_cts.extend([ct] * n)
            out_meths.append(meth_counts)
            out_covs.append(cov_counts)

        if not out_starts:
            raise ValueError("No CpG observations found in reads.")

        starts_cat = np.concatenate(out_starts)
        agg = pd.DataFrame(
            {
                "CHROM":     out_chroms,
                "START":     starts_cat,
                "END":       starts_cat + 1,
                "name":      out_names,
                "cell_type": out_cts,
                "METH":      np.concatenate(out_meths),
                "COV":       np.concatenate(out_covs),
            }
        )

        n_regions   = agg["name"].nunique()
        n_cpg_total = agg[["CHROM", "START"]].drop_duplicates().shape[0]
        _module_logger.info(
            "Aggregated: %d region(s) · %s CpG site(s) · mean coverage %.1f×",
            n_regions,
            f"{n_cpg_total:,}",
            agg["COV"].mean(),
        )

        # ---- Pivot to wide format ----------------------------------------
        # Cell types ordered as they appear in labels_dict for stable output.
        cell_types = [ct for ct in cell_type_names if ct in agg["cell_type"].unique()]
        missing_ct = set(cell_type_names) - set(cell_types)
        if missing_ct:
            _module_logger.warning(
                "No reads found for cell type(s): %s — excluded from atlas.",
                ", ".join(sorted(missing_ct)),
            )

        _module_logger.info("Pivoting to wide format (%d cell type(s)) …", len(cell_types))
        index_cols = ["CHROM", "START", "END", "name"]

        meth_wide = (
            agg.pivot_table(
                index=index_cols, columns="cell_type", values="METH", fill_value=0
            )
            .reindex(columns=cell_types, fill_value=0)
            .reset_index()
        )
        meth_wide.columns.name = None
        meth_wide = meth_wide.rename(columns={ct: f"{ct}_METH" for ct in cell_types})

        cov_wide = (
            agg.pivot_table(
                index=index_cols, columns="cell_type", values="COV", fill_value=0
            )
            .reindex(columns=cell_types, fill_value=0)
            .reset_index()
        )
        cov_wide.columns.name = None
        cov_wide = cov_wide.rename(columns={ct: f"{ct}_COV" for ct in cell_types})

        atlas_df = meth_wide.merge(
            cov_wide[index_cols + [f"{ct}_COV" for ct in cell_types]], on=index_cols
        )

        # Interleave _METH / _COV columns per cell type for readability.
        ordered_cols = index_cols + [
            col for ct in cell_types for col in (f"{ct}_METH", f"{ct}_COV")
        ]
        atlas_df = (
            atlas_df[ordered_cols]
            .sort_values(["CHROM", "START"])
            .reset_index(drop=True)
        )

        # pivot_table promotes integer values to float64; restore integer counts.
        count_cols = [c for c in atlas_df.columns if c.endswith(("_METH", "_COV"))]
        atlas_df[count_cols] = atlas_df[count_cols].astype(np.int32)

        _module_logger.info(
            "Atlas ready: %d row(s) × %d column(s).",
            len(atlas_df),
            len(atlas_df.columns),
        )

        if output_path is not None:
            _module_logger.info("Saving atlas to %s …", output_path)
            atlas_df.to_csv(output_path, sep=sep, index=False)
            _module_logger.info("Saved.")

        return cls(
            atlas_name=atlas_name,
            reference_genome=reference_genome,
            atlas_df=atlas_df,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _check_raw_format(cls, raw: pd.DataFrame) -> None:
        missing = cls._RAW_REQUIRED - set(raw.columns)
        if missing:
            raise ValueError(
                f"Atlas DataFrame is missing required columns: {missing}"
            )
        meth_cell_types = {c[:-5] for c in raw.columns if c.endswith("_METH")}
        cov_cell_types = {c[:-4] for c in raw.columns if c.endswith("_COV")}
        if not meth_cell_types:
            raise ValueError("No *_METH columns found in atlas.")
        unpaired_meth = meth_cell_types - cov_cell_types
        unpaired_cov = cov_cell_types - meth_cell_types
        if unpaired_meth:
            raise ValueError(f"Missing _COV columns for: {unpaired_meth}")
        if unpaired_cov:
            raise ValueError(f"Missing _METH columns for: {unpaired_cov}")

    # ------------------------------------------------------------------
    # AbstractAtlas interface
    # ------------------------------------------------------------------

    @property
    def reference_genome(self) -> str:
        return self._reference_genome

    @property
    def atlas(self) -> pd.DataFrame:
        """Region-level DataFrame with columns ``chr``, ``start``, ``end``, ``name``.

        ``start`` is 1-based (as required by :meth:`overlap_reads`).
        ``end`` is the 1-based inclusive last CpG position of the region.
        """
        return self._atlas

    @property
    def ref_cells(self) -> List[str]:
        """Ordered list of cell-type names (axis-0 of all beta matrices)."""
        return list(self._cell_types)

    # ------------------------------------------------------------------
    # CpG-level accessors (used by build_celfieish_input)
    # ------------------------------------------------------------------

    def __contains__(self, region_name: str) -> bool:
        return region_name in self._beta_matrices

    def get_cpg_lookup(self, region_name: str) -> Dict[int, int]:
        """Return ``{START_0based → column_index}`` for a region."""
        return self._cpg_lookup[region_name]

    def get_n_cpgs(self, region_name: str) -> int:
        """Return the number of CpG sites in the region."""
        return self._n_cpgs[region_name]

    def get_beta_for_regions(self, region_names: List[str]) -> List[np.ndarray]:
        """Return beta (METH/COV) matrices in the requested order.

        Returns
        -------
        list of ndarray(T, n_CpGs)
        """
        return [self._beta_matrices[name] for name in region_names]

    def get_meth_cov_for_regions(
        self, region_names: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return raw methylated-count and coverage matrices in the requested order.

        Used by CelFiE, which needs raw counts rather than pre-computed beta
        values because its M-step jointly re-estimates gamma from counts.

        Returns
        -------
        meth_list : list of ndarray(T, n_CpGs)  — raw methylated counts
        cov_list  : list of ndarray(T, n_CpGs)  — raw coverage counts
        """
        return (
            [self._meth_matrices[name] for name in region_names],
            [self._cov_matrices[name]  for name in region_names],
        )
