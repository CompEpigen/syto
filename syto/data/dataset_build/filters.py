"""Finalize-time read filters and staged-counts recompute."""

from pathlib import Path

import pandas as pd


def pattern_length(pattern: str) -> int:
    """Number of marked CpGs in a methylation pattern (count of '0'/'1')."""
    return pattern.count("0") + pattern.count("1")


def filter_by_pattern_length(
    df: pd.DataFrame, pattern_column: str, min_length: int
) -> pd.DataFrame:
    """Keep reads whose pattern has at least ``min_length`` marked CpGs."""
    lengths = df[pattern_column].map(pattern_length)
    return df[lengths >= min_length]


def filter_by_atlas_regions(df: pd.DataFrame, region_names: set) -> pd.DataFrame:
    """Keep reads whose region ``name`` is in ``region_names``."""
    return df[df["name"].isin(region_names)]


def load_region_names(atlas_path: str, sep: str = "\t") -> set:
    """Load the set of region names from an atlas TSV (only the 'name' column)."""
    names = pd.read_csv(atlas_path, sep=sep, usecols=["name"])["name"]
    return set(names)


def staged_counts(
    staged_dir: str,
    *,
    region_names: set | None = None,
    min_pattern_length: int | None = None,
    pattern_column: str,
) -> pd.DataFrame:
    """Recompute per-(file, original_label) read counts from filtered staged shards."""
    columns = ["file", "original_label", "name", pattern_column]
    frames = []
    for shard in sorted(Path(staged_dir).glob("region_bucket=*/*.parquet")):
        df = pd.read_parquet(shard, columns=columns)
        if region_names is not None:
            df = filter_by_atlas_regions(df, region_names)
        if min_pattern_length:
            df = filter_by_pattern_length(df, pattern_column, min_pattern_length)
        if not df.empty:
            frames.append(df[["file", "original_label"]])
    if not frames:
        return pd.DataFrame(columns=["file", "original_label", "n_reads"])
    allrows = pd.concat(frames, ignore_index=True)
    return (
        allrows.groupby(["file", "original_label"])
        .size()
        .reset_index(name="n_reads")
    )
