"""Equal-size clustering of atlas regions within each target cell type.

Splits each cell type's marker regions into groups of (at most) ``group_size``
regions with similar methylation profiles across the atlas cell types. The
groups are meant as GR groups for an attention head: on a large atlas (e.g.
U250) one group per cell type mixes regions of very different specificity,
while tighter groups let the head learn a weight per kind of region.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

_module_logger = logging.getLogger(__name__)


class BalancedRegionClusterer:
    """Balanced k-means of atlas regions, run separately per target cell type.

    A target with ``n`` regions gets ``ceil(n / group_size)`` clusters whose
    sizes differ by at most one (250 regions -> 10 x 25; 151 -> 7 clusters of
    21-22). Clusters start from k-means++ centroids; each iteration then
    assigns regions to centroids under the size constraint (an exact minimum-
    cost assignment) and recomputes the centroids, until the assignment no
    longer changes.

    Features are the region's values in the atlas cell-type columns. Missing
    values are imputed with the column median over the same target's regions
    (over all regions if the whole column is missing for that target); a
    column missing everywhere is dropped.

    Parameters
    ----------
    group_size : int
        Maximum regions per cluster.
    seed : int
        Seed for the k-means++ initialisation.
    max_iter : int
        Maximum constrained-assignment iterations per target.
    """

    def __init__(self, group_size: int = 25, seed: int = 42, max_iter: int = 100):
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        self.group_size = group_size
        self.seed = seed
        self.max_iter = max_iter

    def fit_predict(
        self,
        atlas: pd.DataFrame,
        cell_type_columns: List[str],
        target_column: str = "target",
    ) -> pd.Series:
        """Return each region's cluster name, ``"<target>__<k>"``, indexed like ``atlas``.

        ``k`` runs from 0 within each target, ordered by descending mean
        value in the target's own column (for UXM U-fraction atlases, ``__0``
        holds the regions most unmethylated in the target).
        """
        features = self._impute(atlas, cell_type_columns, target_column)
        labels = pd.Series(index=atlas.index, dtype=object)
        for target, idx in atlas.groupby(target_column, sort=True).groups.items():
            X = features.loc[idx].to_numpy(dtype=np.float64)
            assignment = self._cluster(X)
            order = self._order_clusters(atlas.loc[idx], assignment, target)
            labels.loc[idx] = [f"{target}__{order[c]}" for c in assignment]
        return labels

    def _impute(self, atlas, cell_type_columns, target_column):
        features = atlas[cell_type_columns].astype(np.float64)
        per_target = features.groupby(atlas[target_column]).transform("median")
        features = features.fillna(per_target).fillna(features.median())
        # A column with no values at all carries no information.
        return features.dropna(axis=1, how="all")

    def _sizes(self, n: int) -> np.ndarray:
        k = -(-n // self.group_size)
        sizes = np.full(k, n // k)
        sizes[: n % k] += 1
        return sizes

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        sizes = self._sizes(len(X))
        if len(sizes) == 1:
            return np.zeros(len(X), dtype=int)
        centers = (
            KMeans(n_clusters=len(sizes), n_init=10, random_state=self.seed)
            .fit(X)
            .cluster_centers_
        )
        # Column j of the expanded cost matrix is one slot of cluster slot_of[j].
        slot_of = np.repeat(np.arange(len(sizes)), sizes)
        assignment: Optional[np.ndarray] = None
        for _ in range(self.max_iter):
            cost = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            rows, cols = linear_sum_assignment(cost[:, slot_of])
            new = np.empty(len(X), dtype=int)
            new[rows] = slot_of[cols]
            if assignment is not None and np.array_equal(new, assignment):
                break
            assignment = new
            centers = np.stack([X[assignment == c].mean(axis=0) for c in range(len(sizes))])
        return assignment

    @staticmethod
    def _order_clusters(regions, assignment, target):
        """Map raw cluster ids to 0..k-1 by descending mean value in the target column.

        Gives stable, interpretable names. Falls back to raw ids when the atlas
        has no column named after the target.
        """
        k = assignment.max() + 1
        if target not in regions.columns:
            return {c: c for c in range(k)}
        means = [regions[target].to_numpy()[assignment == c].mean() for c in range(k)]
        ranked = np.argsort(-np.nan_to_num(np.asarray(means), nan=-np.inf), kind="stable")
        return {int(c): rank for rank, c in enumerate(ranked)}
