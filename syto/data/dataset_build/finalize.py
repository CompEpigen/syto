""" """
from pathlib import Path

import pandas as pd

from syto.data.labelers.hard_with_background_labeler import HardWithBackgroundLabeler
from syto.data.labelers.data_driven_soft_labeler import DataDrivenSoftLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)
from syto.data.dataset_build.splits import apply_splits


def label_and_split(df, split_plan, *, num_classes, soft_labeler,
                    max_distance=0.5, min_reads=30):
    """Apply soft labels, hard background labels, and the split column."""
    df = soft_labeler.compute_labels(
        df, perform_pooling=True, min_reads=min_reads,
        max_distance=max_distance, num_classes=num_classes,
        keep_intermediate_values=False,
    )
    df = HardWithBackgroundLabeler().compute_labels(
        df, num_original_classes=num_classes,
        class_label_column="original_label",
        grg_class_label_column="dmr_ctype_label",
    )
    df = apply_splits(df, split_plan)
    return df


def finalize_bucket(staged_dir, bucket, out_dir, split_plan, *,
                    num_classes, max_distance=0.5, min_reads=30):
    """Gather one bucket's staged shards, label+split, sort, and compact."""
    bucket_dir = Path(staged_dir) / f"region_bucket={bucket}"
    shards = sorted(bucket_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)

    handler = BinaryCpGSignatureHandler(
        start_column="read_start", methylation_pattern_column="methylation_ids"
    )
    soft_labeler = DataDrivenSoftLabeler(distance_name="jaccard", signature_handler=handler)

    df = label_and_split(df, split_plan, num_classes=num_classes,
                         soft_labeler=soft_labeler, max_distance=max_distance,
                         min_reads=min_reads)
    # "signature" is a tuple-of-tuples the soft labeler adds for grouping;
    # pyarrow cannot serialize it, so drop before writing.
    df = df.drop(columns=["signature"], errors="ignore")
    df = df.sort_values("name").reset_index(drop=True)

    out_bucket = Path(out_dir) / f"region_bucket={bucket}"
    out_bucket.mkdir(parents=True, exist_ok=True)
    out_path = out_bucket / "part-0.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)
