"""Finalize-time read filters and staged-counts recompute."""

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
