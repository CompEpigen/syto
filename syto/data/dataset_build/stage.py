""" """

import logging
from pathlib import Path

import pandas as pd
import pysam

from syto.data.dataset_build.schema import (
    adapt_recovered_reads,
    bam_reads_to_recovered,
)
from syto.data.dataset_build.buckets import assign_region_bucket
from syto.data.sequencing.bam_processing import read_bam_regions

logger = logging.getLogger(__name__)


def stage_dataframe(
    df, atlas, region_index, labels_dict, sample_id, *, cell_type_match_dict=None
):
    """Adapt, overlap+trim, label, and bucket one sample's reads."""
    adapted = adapt_recovered_reads(df, labels_dict, cell_type_match_dict)
    adapted = adapted.sort_values(["chromosome", "read_start", "read_end"]).reset_index(
        drop=True
    )
    prepared = atlas.prepare_reads(
        adapted,
        trim=True,
        labels_dict=labels_dict,
        cell_type_match_dict=cell_type_match_dict or {},
    )
    prepared["file"] = sample_id
    staged = assign_region_bucket(prepared, region_index)

    counts = adapted.groupby("original_label").size().reset_index(name="n_reads")
    counts["file"] = sample_id
    counts = counts[["file", "original_label", "n_reads"]]
    return staged, counts


def stage_file(
    csv_path,
    atlas,
    region_index,
    labels_dict,
    staged_dir,
    counts_dir,
    *,
    sep="\t",
    cell_type_match_dict=None,
):
    """Read one CSV, stage it, and write per-bucket parquet + counts sidecar."""
    sample = Path(csv_path).stem
    # Sequence / methylation patterns are numeric-looking strings (e.g.
    # "0101..." or "2222...") and must not be type-inferred to int/float.
    df = pd.read_csv(csv_path, sep=sep, dtype={"original_seq": str, "methyl_seq": str})
    staged, counts = stage_dataframe(
        df,
        atlas,
        region_index,
        labels_dict,
        sample,
        cell_type_match_dict=cell_type_match_dict,
    )
    return _write_staged(staged, counts, sample, staged_dir, counts_dir, len(df))


def stage_bam_file(
    bam_path,
    ctype,
    atlas,
    region_index,
    labels_dict,
    staged_dir,
    counts_dir,
    *,
    bam_params,
    intervals,
    cell_type_match_dict=None,
):
    """Parse one BAM over the atlas intervals, stage it, and write its outputs.

    Every read is labelled with the sample-level ``ctype``.  ``intervals`` use
    the atlas' ``chrN`` names; when the BAM header has no ``chr`` prefix
    (e.g. hs37d5) the prefix is stripped for fetching and restored on the reads.
    """
    sample = Path(bam_path).stem
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        references = set(bam.references)
    chrom_prefix = "" if any(r.startswith("chr") for r in references) else "chr"
    if chrom_prefix:
        intervals = [
            (chromosome.removeprefix(chrom_prefix), start, end)
            for chromosome, start, end in intervals
        ]

    reads = read_bam_regions({**bam_params, "bam_path": str(bam_path)}, intervals)
    if reads is None or reads.empty:
        logger.warning("No reads parsed from %s; nothing staged.", bam_path)
        return {"sample": sample, "n_in": 0, "n_out": 0, "duplication_factor": 0.0}

    df = bam_reads_to_recovered(reads, ctype, chrom_prefix=chrom_prefix)
    staged, counts = stage_dataframe(
        df,
        atlas,
        region_index,
        labels_dict,
        sample,
        cell_type_match_dict=cell_type_match_dict,
    )
    return _write_staged(staged, counts, sample, staged_dir, counts_dir, len(df))


def _write_staged(staged, counts, sample, staged_dir, counts_dir, n_in):
    """Write per-bucket shards and the counts sidecar; return staging stats."""
    n_out = len(staged)
    for bucket, group in staged.groupby("region_bucket"):
        out_dir = Path(staged_dir) / f"region_bucket={bucket}"
        out_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["region_bucket"]).to_parquet(
            out_dir / f"{sample}.parquet", index=False
        )

    Path(counts_dir).mkdir(parents=True, exist_ok=True)
    counts.to_parquet(Path(counts_dir) / f"{sample}.parquet", index=False)

    return {
        "sample": sample,
        "n_in": n_in,
        "n_out": n_out,
        "duplication_factor": (n_out / n_in) if n_in else 0.0,
    }
