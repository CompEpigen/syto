"""Equal-size clustering of atlas regions.

Splits a set of regions into groups of (at most) ``group_size`` regions with
similar feature vectors. :meth:`UXMMethylationAtlas.with_region_groups` runs it
once per target cell type on the regions' atlas values to build GR groups for
an attention head: on a large atlas (e.g. U250) one group per cell type mixes
regions of very different specificity, while tighter groups let the head learn
a weight per kind of region.
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

_module_logger = logging.getLogger(__name__)


class BalancedRegionClusterer:
    """Balanced k-means: ``ceil(n / group_size)`` clusters whose sizes differ by at most one.

    250 rows -> 10 x 25; 151 -> 7 clusters of 21-22. Clusters start from
    k-means++ centroids; each iteration then assigns rows to centroids under
    the size constraint (an exact minimum-cost assignment) and recomputes the
    centroids, until the assignment no longer changes.

    The assignment solves an ``n x n`` problem per iteration (one column per
    cluster slot), so time grows as ``n^3`` and memory as ``n^2``: fine for
    the few hundred regions a UXM target has, slow beyond a few thousand.

    Parameters
    ----------
    group_size : int
        Maximum rows per cluster.
    seed : int
        Seed for the k-means++ initialisation.
    max_iter : int
        Maximum constrained-assignment iterations.
    """

    def __init__(self, group_size: int = 25, seed: int = 42, max_iter: int = 100):
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        self.group_size = group_size
        self.seed = seed
        self.max_iter = max_iter

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Return a cluster id in ``0..k-1`` for each row of ``X`` (``n x d``, no NaN)."""
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("X must not contain missing values; impute them first")
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
            centers = np.stack(
                [X[assignment == c].mean(axis=0) for c in range(len(sizes))]
            )
        return assignment

    def _sizes(self, n: int) -> np.ndarray:
        k = -(-n // self.group_size)
        sizes = np.full(k, n // k)
        sizes[: n % k] += 1
        return sizes
