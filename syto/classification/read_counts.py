"""Read-count support shared by the read classifiers.

A de-duplicated split keeps one row per distinct read with a ``read_count``
multiplicity. These helpers let a classifier train and evaluate on such rows as
if every duplicate were present: samplers that draw rows in proportion to their
counts, and per-row losses that can be weighted by them. Dismir uses the
sampler and the per-row loss; the HF-Trainer classifiers (MethylBERT,
EpigenBERT2) use all of it through ``syto.classification.hf_training``.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler

_module_logger = logging.getLogger(__name__)


def _validated_counts(counts) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or len(counts) == 0 or (counts <= 0).any():
        raise ValueError("counts must be a non-empty 1-D array of positive values")
    return counts


def _torch_seeded_rng() -> np.random.Generator:
    """A NumPy generator seeded from torch's global RNG.

    Seeding torch (as the fit pipeline does) therefore makes the draws
    reproducible.
    """
    return np.random.default_rng(int(torch.randint(0, 2**63 - 1, (1,)).item()))


def normalised_weights(counts) -> np.ndarray:
    """Per-row weights ``count / mean(count)``, averaging to 1.

    Averaging ``loss * weight`` over all rows gives ``sum(c * l) / sum(c)``,
    the mean loss over every duplicated read.
    """
    counts = _validated_counts(counts)
    return counts / counts.mean()


def per_row_loss(loss_fct, *args) -> torch.Tensor:
    """Unreduced loss, one value per row (averaged over any trailing dims).

    ``loss_fct``'s reduction is switched to ``"none"`` for the call and
    restored afterwards; its ``"mean"`` equals the mean of these values.
    """
    reduction = loss_fct.reduction
    loss_fct.reduction = "none"
    try:
        per_row = loss_fct(*args)
    finally:
        loss_fct.reduction = reduction
    if per_row.ndim == 0:  # a batch of one squeezed to a scalar
        per_row = per_row.reshape(1)
    return per_row.reshape(per_row.shape[0], -1).mean(dim=1)


class CountWeightedSampler(Sampler):
    """Draw row indices with replacement, proportionally to read counts.

    Over de-duplicated reads this reproduces the distribution of the original,
    duplicated dataset: each draw is a read picked uniformly from the original
    data. Unlike ``WeightedRandomSampler`` it has no 2**24-category limit. The
    per-epoch seed is drawn from torch's global RNG, so seeding torch makes the
    draw order reproducible.
    """

    def __init__(self, counts, num_samples=None):
        counts = _validated_counts(counts)
        self.probabilities = counts / counts.sum()
        self.num_samples = int(num_samples or round(counts.sum()))

    def __iter__(self):
        draws = _torch_seeded_rng().choice(
            len(self.probabilities), size=self.num_samples, p=self.probabilities
        )
        return iter(draws.tolist())

    def __len__(self):
        return self.num_samples


class CountWeightedBalancedBatchSampler(Sampler):
    """Balanced batches over de-duplicated rows.

    Count-weighted counterpart of ``BalancedBackgroundBatchSampler``: every
    batch holds ``int(batch_size * bg_ratio)`` background rows and the rest
    on-target rows, each group drawn with replacement in proportion to its
    rows' read counts. By default an epoch draws as many on-target reads as
    the on-target rows stand for, i.e. as many batches as the uniform sampler
    yields on the duplicated data; ``num_samples`` instead sets the total draws
    per epoch. Batches are always full.
    """

    def __init__(
        self, signal_mask, counts, batch_size, bg_ratio=0.3, num_samples=None
    ):
        mask = np.asarray(signal_mask, dtype=bool)
        counts = _validated_counts(counts)
        if len(mask) != len(counts):
            raise ValueError("signal_mask and counts must have the same length")
        self.n_bg_per_batch = int(batch_size * bg_ratio)
        self.n_signal_per_batch = batch_size - self.n_bg_per_batch
        self.signal_indices = np.where(mask)[0]
        self.bg_indices = np.where(~mask)[0]
        if self.n_signal_per_batch < 1 or len(self.signal_indices) == 0:
            raise ValueError("balanced sampling needs on-target rows in every batch")
        if self.n_bg_per_batch and len(self.bg_indices) == 0:
            raise ValueError("balanced sampling needs background rows")
        signal_counts = counts[self.signal_indices]
        self.signal_p = signal_counts / signal_counts.sum()
        self.bg_p = (
            counts[self.bg_indices] / counts[self.bg_indices].sum()
            if len(self.bg_indices)
            else None
        )
        if num_samples:
            self.num_batches = int(np.ceil(num_samples / batch_size))
        else:
            self.num_batches = int(
                np.ceil(signal_counts.sum() / self.n_signal_per_batch)
            )

    def __iter__(self):
        rng = _torch_seeded_rng()
        for _ in range(self.num_batches):
            batch = rng.choice(
                self.signal_indices, size=self.n_signal_per_batch, p=self.signal_p
            )
            if self.n_bg_per_batch:
                batch = np.concatenate(
                    [
                        batch,
                        rng.choice(
                            self.bg_indices, size=self.n_bg_per_batch, p=self.bg_p
                        ),
                    ]
                )
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


class ReadWeightedLossMixin:
    """Per-read loss weighting for models that compute their loss in ``forward``.

    Every loss call goes through :meth:`_loss`. Given ``sample_weight`` (each
    row's read count over its dataset's mean count, see
    :class:`ReadCountDatasetMixin`), the batch loss is the mean of the weighted
    per-row losses: an unbiased estimate of the duplicated data's mean loss in
    training and, because the HF Trainer averages batch losses by row, the
    exact per-read mean over an evaluation pass. Without a weight the loss is
    ``loss_fct(*args)`` unchanged.
    """

    @staticmethod
    def _loss(loss_fct, *args, sample_weight=None):
        if sample_weight is None:
            return loss_fct(*args)
        per_row = per_row_loss(loss_fct, *args)
        return (per_row * sample_weight.to(per_row.dtype)).mean()


class ReadCountDatasetMixin:
    """Read counts of a dataset of de-duplicated rows.

    ``read_counts`` holds each row's multiplicity (``None`` when the rows are
    not de-duplicated). Items then carry ``sample_weight``, the row's count over
    the dataset's mean count, which :class:`ReadWeightedLossMixin` multiplies
    into the loss; trainers read ``read_counts`` for sampling and metric
    weights.
    """

    read_counts: Optional[np.ndarray] = None
    _sample_weights: Optional[np.ndarray] = None

    def _set_read_counts(self, counts) -> None:
        if counts is None:
            self.read_counts = None
            self._sample_weights = None
            return
        self.read_counts = _validated_counts(counts)
        self._sample_weights = normalised_weights(self.read_counts)

    def _with_sample_weight(self, item: dict, index: int) -> dict:
        if self._sample_weights is not None:
            item["sample_weight"] = torch.tensor(
                self._sample_weights[index], dtype=torch.float32
            )
        return item
