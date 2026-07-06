"""Raw-split loading for the pseudobulk pipeline.

Reads training splits from either the legacy per-file layout or the new
columnar (split-as-column) parquet dataset, reusing the columnar readers in
``syto.classification.fit_data``. Projects only the columns the pseudobulk
pipeline needs (classifier features + generation/atlas columns) and derives
``NCPGS`` from the methylation pattern when the source lacks it.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from syto.data.dataset import resolve_column
from syto.classification.fit_data import (
    detect_format,
    load_legacy_split,
    load_columnar_split,
    apply_pattern_length_filter,
)

# Columns the pseudobulk pipeline always needs from a raw split; a columnar
# dataset missing any of these hard-fails in load_columnar_split. They drive
# atlas trimming (chr/read_start/read_end), grouping and labels.
REQUIRED_COLUMNS = [
    "original_label",
    "dmr_ctype_label",
    "chr",
    "read_start",
    "read_end",
    "methylation_ids",
]

# Columns projected only when present. ``NCPGS`` is derived downstream when
# missing (see ``ensure_ncpgs``); the rest are pipeline-optional metadata.
OPTIONAL_COLUMNS = [
    "NCPGS",
    "dmr_ctype",
    "trimmed_start",
    "trimmed_end",
    "methylated_CpGs",
    "unmethylated_CpGs",
    "file",
    "name",
]


def build_declared_columns(
    classifier_required: List[str], available: List[str]
) -> List[str]:
    """Columns to project from a columnar split.

    ``REQUIRED_COLUMNS`` plus the classifier's own feature columns are always
    declared (``load_columnar_split`` raises if a required one is absent).
    Optional columns are included only when present in ``available`` (alias-
    resolved), so genuinely-optional metadata does not trigger a hard failure.
    Order-preserving and de-duplicated.
    """
    declared: List[str] = []
    for col in list(REQUIRED_COLUMNS) + list(classifier_required):
        if col not in declared:
            declared.append(col)
    for col in OPTIONAL_COLUMNS:
        if resolve_column(available, col) is not None and col not in declared:
            declared.append(col)
    return declared


def ensure_ncpgs(
    df: pd.DataFrame, pattern_column: str = "methylation_ids"
) -> pd.DataFrame:
    """Add an ``NCPGS`` column (marked-CpG count) when the split lacks one.

    ``PseudobulkGenerator`` reads ``NCPGS`` unconditionally as the aggregation
    weight, so the column must exist by generation time. It equals the
    methylation-pattern length, so it is derived here from the (alias-resolved)
    pattern column when absent. Existing values are never overwritten.
    """
    if "NCPGS" in df.columns:
        return df

    from syto.data.dataset_build.filters import pattern_length

    col = resolve_column(list(df.columns), pattern_column)
    if col is None:
        raise ValueError(
            f"Cannot derive NCPGS: no methylation pattern column "
            f"({pattern_column!r} or an alias) in {list(df.columns)}"
        )
    df["NCPGS"] = df[col].map(pattern_length)
    return df


def load_raw_splits(
    data_path,
    split_names: List[str],
    *,
    classifier_required_columns: List[str],
    data_format: str = "auto",
    split_column: str = "split",
    min_pattern_length: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load raw training splits from a legacy or columnar dataset directory.

    Detects the on-disk layout once, then per split reads it (projecting the
    declared columns for columnar), applies the optional ``min_pattern_length``
    filter, and derives ``NCPGS`` when absent.
    """
    data_path = Path(data_path)
    fmt = detect_format(data_path, override=data_format, split_column=split_column)

    declared: Optional[List[str]] = None
    if fmt == "columnar":
        # Local import: columnar_schema_names is added to fit_data in Task 2, and
        # is only needed on the columnar path, so importing it lazily keeps this
        # module importable (and Task 1's helpers testable) before Task 2 lands.
        from syto.classification.fit_data import columnar_schema_names

        available = columnar_schema_names(data_path)
        declared = build_declared_columns(classifier_required_columns, available)

    splits: Dict[str, pd.DataFrame] = {}
    for split in split_names:
        if fmt == "legacy":
            df = load_legacy_split(data_path, split)
        else:
            df = load_columnar_split(
                data_path,
                split,
                declared_columns=declared,
                split_column=split_column,
            )
            if df.empty:
                raise FileNotFoundError(
                    f"No rows for split {split!r} in columnar dataset {data_path}"
                )
        if min_pattern_length:
            df = apply_pattern_length_filter(df, min_pattern_length)
        df = ensure_ncpgs(df)
        splits[split] = df
    return splits
