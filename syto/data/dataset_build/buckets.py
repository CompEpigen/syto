""" """
import numpy as np
import pandas as pd


def build_region_index(atlas_df: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    """Map each atlas region name to a contiguous genomic-range bucket.

    Regions are ordered by (chr, start, end) and cut into ``n_buckets``
    near-equal-count contiguous ranges. ``region_bucket`` is an int in
    ``[0, n_buckets)``. If there are fewer regions than buckets, only as many
    buckets as regions are used.
    """
    ordered = atlas_df.sort_values(["chr", "start", "end"]).reset_index(drop=True)
    n = len(ordered)
    if n == 0:
        return pd.DataFrame({"name": pd.Series(dtype=str), "region_bucket": pd.Series(dtype=int)})
    buckets = np.minimum((np.arange(n) * n_buckets) // n, n_buckets - 1)
    return pd.DataFrame({"name": ordered["name"].to_numpy(), "region_bucket": buckets.astype(int)})


def assign_region_bucket(df: pd.DataFrame, region_index: pd.DataFrame) -> pd.DataFrame:
    """Inner-join ``region_bucket`` onto ``df`` by ``name`` (drops unknown names)."""
    return df.merge(region_index, on="name", how="inner")
