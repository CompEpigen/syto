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
from syto.data.sequencing.bam_processing import merge_fetch_intervals, read_bam_regions

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
    padding=0,
    cell_type_match_dict=None,
):
    """Takes the BAM file and analyzes it over the different regions of the atlas, carries out the staging process, and then writes out the results.
    Each read is given a sample-level ``ctype''. The BAM is processed one region bucket at a time: since the buckets are contiguous genomic intervals, 
    each fetch remains local and only the reads from one bucket are kept in memory. 
    
    For a large atlas a complete pass through the file would require tens of GB, a amount that wouldn't fit alongside several parallel workers. 
    ``padding'' extends every region so that mates which fall outside it are still fetched and can then be merged. 
    The regions used in the atlas have ``chrN'' names; if the BAM header lacks a ``chr'' prefix (for example, hs37d5) 
    then the prefix is removed when fetching but put back on the reads.
    
    In contrast to :func:`stage_file`, this `n_reads` value counts the staged rows (those reads that overlap a region)
    since reads are only ever encountered on a per-bucket basis. Both of them are used for the same split planning.
    """
    sample = Path(bam_path).stem
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        references = set(bam.references)
    chrom_prefix = "" if any(r.startswith("chr") for r in references) else "chr"
    params = {**bam_params, "bam_path": str(bam_path)}

    n_in = 0
    label_counts = []
    for bucket, names in region_index.groupby("region_bucket")["name"]:
        bucket_atlas = atlas.subset_regions(names)
        intervals = merge_fetch_intervals(
            bucket_atlas.atlas, params["chromosomes"], padding
        )
        if chrom_prefix:
            intervals = [
                (chromosome.removeprefix(chrom_prefix), start, end)
                for chromosome, start, end in intervals
            ]
        if not intervals:
            continue

        reads = read_bam_regions(params, intervals, verbose=False)
        if reads is None or reads.empty:
            continue
        n_in += len(reads)

        df = bam_reads_to_recovered(reads, ctype, chrom_prefix=chrom_prefix)
        staged, _ = stage_dataframe(
            df,
            bucket_atlas,
            region_index,
            labels_dict,
            sample,
            cell_type_match_dict=cell_type_match_dict,
        )
        if staged.empty:
            continue
        _write_bucket(staged, bucket, sample, staged_dir)
        label_counts.append(staged.groupby("original_label").size())

    if not label_counts:
        logger.warning("No reads parsed from %s; nothing staged.", bam_path)
        return {"sample": sample, "n_in": 0, "n_out": 0, "duplication_factor": 0.0}

    counts = (
        pd.concat(label_counts)
        .groupby(level=0)
        .sum()
        .reset_index(name="n_reads")
        .assign(file=sample)[["file", "original_label", "n_reads"]]
    )
    Path(counts_dir).mkdir(parents=True, exist_ok=True)
    counts.to_parquet(Path(counts_dir) / f"{sample}.parquet", index=False)

    n_out = int(counts["n_reads"].sum())
    return {
        "sample": sample,
        "n_in": n_in,
        "n_out": n_out,
        "duplication_factor": (n_out / n_in) if n_in else 0.0,
    }


def _write_bucket(staged, bucket, sample, staged_dir):
    """Write one bucket's shard for *sample* (the partition key is the path)."""
    out_dir = Path(staged_dir) / f"region_bucket={bucket}"
    out_dir.mkdir(parents=True, exist_ok=True)
    staged.drop(columns=["region_bucket"]).to_parquet(
        out_dir / f"{sample}.parquet", index=False
    )


def _write_staged(staged, counts, sample, staged_dir, counts_dir, n_in):
    """Write per-bucket shards and the counts sidecar; return staging stats."""
    n_out = len(staged)
    for bucket, group in staged.groupby("region_bucket"):
        _write_bucket(group, bucket, sample, staged_dir)

    Path(counts_dir).mkdir(parents=True, exist_ok=True)
    counts.to_parquet(Path(counts_dir) / f"{sample}.parquet", index=False)

    return {
        "sample": sample,
        "n_in": n_in,
        "n_out": n_out,
        "duplication_factor": (n_out / n_in) if n_in else 0.0,
    }
