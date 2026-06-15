"""
Read Preparation Utilities

Adds derived columns required by the pseudo-bulk generator and optionally
constrains reads to the genomic span of overlapping atlas regions.
"""

from typing import Dict, List, Optional,Union
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas

import numpy as np
import pandas as pd


def _trim_reads_by_atlas_overlap(
    df: pd.DataFrame,
    atlas: pd.DataFrame,
    seq_column: str, 
    methylation_pattern_column: str,
) -> pd.DataFrame:
    """
    Clip each read to the genomic span of the atlas region(s) it overlaps.

    Reads that do not overlap any atlas region are dropped. A read overlapping
    several atlas regions is duplicated, once per overlapping region, with
    ``read_start``, ``read_end``, ``seq_column`` and
    ``methylation_pattern_column`` clipped to that region's span.

    ``df`` must be sorted by ``["chromosome", "read_start", "read_end"]`` and
    ``atlas`` by ``["chr", "start", "end"]``.
    """
    n_records = len(df)
    if n_records == 0 or len(atlas) == 0:
        return df.iloc[0:0].copy()

    # Build chromosome start index mapping
    chrom_start_idx = {}
    current_chrom = df.iloc[0]["chromosome"]
    chrom_start_idx[current_chrom] = 0
    for i in range(1, n_records):
        chrom = df.iloc[i]["chromosome"]
        if chrom != current_chrom:
            chrom_start_idx[chrom] = i
            current_chrom = chrom

    chrom_base_pointer: Dict[str, int] = {}
    row_indices: List[int] = []
    trimmed_starts: List[int] = []
    trimmed_ends: List[int] = []
    trimmed_seqs: List[str] = []
    trimmed_patterns: List[str] = []

    for _, region in atlas.iterrows():
        chromosome = region["chr"]
        start = region["start"] - 1  # 1-based to 0-based
        end = region["end"]  # EXCLUSIVE

        if chromosome not in chrom_base_pointer:
            chrom_base_pointer[chromosome] = chrom_start_idx.get(chromosome, n_records)

        base_ptr = chrom_base_pointer[chromosome]

        # Advance past records that can't overlap with this or future regions
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
                pattern = record[methylation_pattern_column]
                seq = record[seq_column]

                trimmed_start = max(start, record["read_start"])
                trimmed_end_inclusive = min(end - 1, record["read_end"])

                offset_start = trimmed_start - record["read_start"]
                offset_end = (
                    len(pattern) - (record["read_end"] - trimmed_end_inclusive) + 1
                )

                row_indices.append(scan_ptr)
                trimmed_starts.append(trimmed_start)
                trimmed_ends.append(trimmed_end_inclusive)
                trimmed_patterns.append(pattern[offset_start:offset_end])
                trimmed_seqs.append(seq[offset_start:offset_end])

            scan_ptr += 1

    if not row_indices:
        return df.iloc[0:0].copy()

    trimmed = df.iloc[row_indices].copy().reset_index(drop=True)
    trimmed["read_start"] = trimmed_starts
    trimmed["read_end"] = trimmed_ends
    trimmed[seq_column] = trimmed_seqs
    trimmed[methylation_pattern_column] = trimmed_patterns
    return trimmed


def prepare_splits_for_pseudobulk(
    splits: Dict[str, pd.DataFrame],
    target_columns: Optional[List[str]] = None,
    num_labels: int = 39,
    trim_by_atlas_regions_overlap: bool = True,
    atlas_path: Optional[str] = None,
    seq_column: str = "seq",
    methylation_pattern_column: str = "pattern",
) -> Dict[str, pd.DataFrame]:
    """
    Add derived columns needed by ``generate_pseudo_bulk*`` and optionally
    constrain reads to the genomic span of overlapping atlas regions.

    The following columns are added to every split DataFrame:

    * ``is_cell_informative_region`` — whether ``original_label == dmr_ctype_label``
    * ``prediction`` — argmax of the prediction columns
    * ``confidence`` — max of the prediction columns
    * ``total_marked_cpgs`` — ``methylated_CpGs + unmethylated_CpGs``
    * ``direction`` — set to ``"U"`` (placeholder)
    * ``read_start`` — alias for ``trimmed_start``
    * ``read_end`` — alias for ``trimmed_end``
    * ``chromosome`` — rename of ``chr`` if present

    Parameters
    ----------
    splits : Dict[str, pd.DataFrame]
        Input split DataFrames.
    target_columns : list of str, optional
        Prediction column names.  Default: ``[prediction_0 .. prediction_{num_labels-1}]``.
    num_labels : int
        Number of labels.  Default: 39.
    trim_by_atlas_regions_overlap : bool
        If ``True`` (default), clip each read's ``read_start``, ``read_end``,
        ``seq_column`` and ``methylation_pattern_column`` to the genomic span
        of the atlas region(s) it overlaps, dropping reads that overlap no
        atlas region. A read overlapping multiple atlas regions is duplicated
        once per overlapping region. Requires ``atlas_path``.
    atlas_path : str, optional
        Path to the atlas TSV (must contain ``chr``, ``start`` and ``end``
        columns).  Required when ``trim_by_atlas_regions_overlap=True``.
    seq_column, methylation_pattern_column : str
        Column names of the read sequence and methylation pattern, clipped
        when ``trim_by_atlas_regions_overlap=True``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        ``{"train": df_train, "valid": df_valid, ...}`` — enriched DataFrames.
    """
    if target_columns is None:
        target_columns = [f"prediction_{i}" for i in range(num_labels)]

    splits_copy = {name: df.copy() for name, df in splits.items()}

    for split_name, df in splits_copy.items():
        # Derived columns from predictions
        if "original_label" in df.columns and "dmr_ctype_label" in df.columns:
            df["is_cell_informative_region"] = (
                df["original_label"] == df["dmr_ctype_label"]
            )

        available_pred_cols = [c for c in target_columns if c in df.columns]
        if available_pred_cols:
            pred_array = df[available_pred_cols].to_numpy()
            df["prediction"] = np.argmax(pred_array, axis=1)
            df["confidence"] = np.max(pred_array, axis=1)

        # CpG count
        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df["total_marked_cpgs"] = df["methylated_CpGs"] + df["unmethylated_CpGs"]

        # Direction placeholder
        df["direction"] = "U"

        # Positional aliases
        if "trimmed_start" in df.columns:
            df["read_start"] = df["trimmed_start"]
        if "trimmed_end" in df.columns:
            df["read_end"] = df["trimmed_end"]

        # Chromosome column rename
        if "chr" in df.columns and "chromosome" not in df.columns:
            df.rename(columns={"chr": "chromosome"}, inplace=True)

    if trim_by_atlas_regions_overlap:
        if atlas_path is None:
            raise ValueError(
                "atlas_path is required when trim_by_atlas_regions_overlap=True"
            )

        # atlas = pd.read_csv(atlas_path, sep="\t")
        UXMatlas = UXMMethylationAtlas(atlas_name="U25l4", 
                                    reference_genome="hg38" if "hg38" in atlas_path else "hg19",
                                    atlas_path=atlas_path,
                                    sep="\t")
        UXMatlas._atlas = UXMatlas._atlas.sort_values(
            by=["chr", "start", "end"], ascending=True
        ).reset_index(drop=True)

        for split_name, df in splits_copy.items():
            df_sorted = df.sort_values(
                by=["chromosome", "read_start", "read_end"], ascending=True
            ).reset_index(drop=True)

            splits_copy[split_name] = _trim_reads_by_atlas_overlap(
                df_sorted,
                UXMatlas._atlas,
                seq_column=seq_column,
                methylation_pattern_column=methylation_pattern_column,
            )

    return splits_copy
