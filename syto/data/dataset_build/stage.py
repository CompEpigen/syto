""" """

from pathlib import Path

import pandas as pd

from syto.data.dataset_build.schema import adapt_recovered_reads
from syto.data.dataset_build.buckets import assign_region_bucket


def stage_dataframe(
    df, atlas, region_index, labels_dict, sample_id, *, cell_type_match_dict=None
):
    """Adapt, overlap+trim, label, and bucket one sample's reads."""
    adapted = adapt_recovered_reads(df, labels_dict,cell_type_match_dict)
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
    n_in = len(df)
    staged, counts = stage_dataframe(
        df,
        atlas,
        region_index,
        labels_dict,
        sample,
        cell_type_match_dict=cell_type_match_dict,
    )
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
