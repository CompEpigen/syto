"""Data loading for classifier fitting.

Supports two on-disk layouts per dataset directory:

* **legacy**   -- separate ``train``/``valid``/``test`` files
                  (``{parquet,csv,txt}``).
* **columnar** -- a partitioned parquet dataset (``region_bucket=*/part-*.parquet``
                  or loose parquet parts) with the split stored in a ``split``
                  column and one or more label columns.

The columnar reader projects to only the columns a classifier needs and pushes
the split filter down into ``pyarrow.dataset``, so RAM stays bounded even when
the dataset carries many label columns.
"""

from pathlib import Path

import pandas as pd

from syto.data.dataset import resolve_column


def detect_format(dataset_dir, *, override="auto", split_column="split"):
    """Return ``"legacy"`` or ``"columnar"`` for a dataset directory."""
    dataset_dir = Path(dataset_dir)
    if override in ("legacy", "columnar"):
        return override
    if override != "auto":
        raise ValueError(f"Unknown data_format override: {override!r}")

    if any(dataset_dir.glob("region_bucket=*")):
        return "columnar"
    for ext in ("parquet", "csv", "txt"):
        if (dataset_dir / f"train.{ext}").exists():
            return "legacy"

    import pyarrow.parquet as pq  # local import: only needed for the schema peek

    parquets = sorted(dataset_dir.rglob("*.parquet"))
    if parquets:
        schema = pq.read_schema(parquets[0])
        if split_column in schema.names:
            return "columnar"

    raise ValueError(
        f"Could not detect data format for {dataset_dir}: no train.* files, no "
        f"region_bucket=* partitions, and no parquet carrying a "
        f"{split_column!r} column."
    )


def resolve_fit_columns(declared, available):
    """Map declared canonical/literal column names onto actual dataset columns.

    Uses ``resolve_column`` so canonical names (e.g. ``input_ids``) match any of
    their aliases (e.g. ``seq``). Order-preserving and de-duplicated. Raises if
    any declared column has no match.
    """
    available = list(available)
    resolved, missing = [], []
    for col in declared:
        hit = resolve_column(available, col)
        if hit is None:
            missing.append(col)
        elif hit not in resolved:
            resolved.append(hit)
    if missing:
        raise ValueError(
            f"Columns required for fitting are absent from the dataset: {missing}. "
            f"Available columns: {available}"
        )
    return resolved


def load_legacy_split(dataset_dir, split):
    """Load a legacy split file (try parquet, csv, txt)."""
    dataset_dir = Path(dataset_dir)
    if (dataset_dir / f"{split}.parquet").exists():
        return pd.read_parquet(dataset_dir / f"{split}.parquet")
    if (dataset_dir / f"{split}.csv").exists():
        return pd.read_csv(dataset_dir / f"{split}.csv")
    if (dataset_dir / f"{split}.txt").exists():
        return pd.read_csv(dataset_dir / f"{split}.txt", sep="\t")
    raise FileNotFoundError(f"Could not find {split} split in {dataset_dir}")


def apply_label_rename(df, label_column, soft_labels):
    """Rename the chosen ``label_column`` to the canonical name the classifiers read."""
    canonical = "soft_label" if soft_labels else "label"
    if label_column == canonical:
        return df
    if label_column not in df.columns:
        raise ValueError(
            f"label_column {label_column!r} not found in data; "
            f"available columns: {list(df.columns)}"
        )
    return df.rename(columns={label_column: canonical})


def apply_pattern_length_filter(df, min_pattern_length, *, pattern_column="methylation_ids"):
    """Drop reads whose methylation pattern has fewer than ``min_pattern_length`` marked CpGs.

    No-op when ``min_pattern_length`` is falsy or ``<= 1``. The pattern column is
    alias-resolved (e.g. ``methylation_ids`` matches ``pattern``); if the data
    carries no such column while filtering is requested, this raises so a
    misconfiguration is loud rather than silently training on unfiltered reads.
    """
    if not min_pattern_length or min_pattern_length <= 1:
        return df

    from syto.data.dataset_build.filters import filter_by_pattern_length

    col = resolve_column(list(df.columns), pattern_column)
    if col is None:
        raise ValueError(
            f"min_pattern_length={min_pattern_length} is set but no methylation "
            f"pattern column ({pattern_column!r} or an alias) was found in the "
            f"data; available columns: {list(df.columns)}"
        )
    # Reset the index: dropping rows leaves a gappy index, but downstream
    # classifiers index the per-read Series positionally (``series[i]``), which
    # is label-based and only correct on a contiguous 0..n-1 index.
    return filter_by_pattern_length(df, col, min_pattern_length).reset_index(drop=True)


def load_columnar_split(dataset_dir, split, *, declared_columns, split_column="split"):
    """Read one split from a columnar (split-as-column) parquet dataset.

    Projects to only the resolved ``declared_columns`` and pushes the
    ``split_column == split`` filter down into the scan, so neither unused label
    columns nor other splits are materialised. ``partitioning=None`` makes the
    reader ignore ``region_bucket=*`` directory names (the value is also present
    as a real column in each part file), avoiding an ambiguous-field clash.
    """
    import pyarrow.dataset as pads  # local import keeps module import cheap

    dataset = pads.dataset(str(dataset_dir), format="parquet", partitioning=None)
    available = dataset.schema.names
    if split_column not in available:
        raise ValueError(
            f"split column {split_column!r} not found in columnar dataset "
            f"{dataset_dir}; available columns: {available}"
        )

    columns = resolve_fit_columns(declared_columns, available)
    read_columns = columns + ([split_column] if split_column not in columns else [])

    table = dataset.to_table(
        columns=read_columns,
        filter=pads.field(split_column) == split,
    )
    df = table.to_pandas()
    if split_column not in columns:
        df = df.drop(columns=[split_column])
    return df
