""" """

from pathlib import Path

import pandas as pd

from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)
from syto.data.labelers.labeler_factory import LabelerContext, apply_labelers
from syto.data.dataset_build.splits import apply_splits
from syto.data.dataset_build.filters import (
    filter_by_atlas_regions,
    filter_by_pattern_length,
)


def label_and_split(
    df,
    split_plan,
    *,
    labelers_config,
    context,
    fit_splits=None,
    fallback="uniform",
    remap=None,
):
    """Stamp the split column first, then apply all configured labelers.

    Splitting first lets signature labelers fit on ``fit_splits`` only, avoiding
    validation/test information leaking into their own targets. ``remap`` (an
    old->new split-name map) is applied as the final stamping step.
    """
    df = apply_splits(df, split_plan, remap=remap)
    df = apply_labelers(
        df, labelers_config, context, fit_splits=fit_splits, fallback=fallback
    )
    return df


def finalize_bucket(
    staged_dir,
    bucket,
    out_dir,
    split_plan,
    *,
    num_classes,
    labelers_config,
    signature_config,
    global_prior=None,
    region_names=None,
    min_pattern_length=None,
    pattern_column="pattern",
    fit_splits=None,
    fallback="uniform",
    split_remap=None,
):
    """Gather one bucket's staged shards, filter, label+split, sort, and compact.

    Returns the written path, or ``None`` when the bucket is empty after filtering.
    """
    bucket_dir = Path(staged_dir) / f"region_bucket={bucket}"
    shards = sorted(bucket_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)

    if region_names is not None:
        df = filter_by_atlas_regions(df, region_names)
    if min_pattern_length:
        df = filter_by_pattern_length(df, pattern_column, min_pattern_length)
    if df.empty:
        return None
    df = df.reset_index(drop=True)

    handler = BinaryCpGSignatureHandler(
        start_column=signature_config["start_column"],
        methylation_pattern_column=signature_config["methylation_pattern_column"],
    )
    context = LabelerContext(
        num_classes=num_classes,
        signature_handler=handler,
        global_prior=global_prior,
    )

    df = label_and_split(
        df,
        split_plan,
        labelers_config=labelers_config,
        context=context,
        fit_splits=fit_splits,
        fallback=fallback,
        remap=split_remap,
    )
    # "signature" is a tuple-of-tuples the soft/archetype labelers add for
    # grouping; pyarrow cannot serialize it, so drop before writing.
    df = df.drop(columns=["signature"], errors="ignore")
    df = df.sort_values("name").reset_index(drop=True)
    # Re-attach the partition key as a column (staging dropped it before writing
    # shards) so consumers reading individual part files still see it.
    df["region_bucket"] = bucket

    out_bucket = Path(out_dir) / f"region_bucket={bucket}"
    out_bucket.mkdir(parents=True, exist_ok=True)
    out_path = out_bucket / "part-0.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)
