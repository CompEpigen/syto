"""Read-count helpers shared by Dismir and the HF-Trainer classifiers."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from syto.classification.read_counts import (
    CountWeightedBalancedBatchSampler,
    CountWeightedSampler,
    normalised_weights,
    per_row_loss,
)


class TestCountWeightedSamplerMove(unittest.TestCase):
    def test_dismir_reexports_the_shared_sampler(self):
        from syto.classification.classifiers import dismir

        self.assertIs(dismir.CountWeightedSampler, CountWeightedSampler)

    def test_epoch_length_defaults_to_total_reads(self):
        self.assertEqual(len(CountWeightedSampler([1, 4, 5])), 10)


class TestCountWeightedBalancedBatchSampler(unittest.TestCase):
    MASK = np.array([True, True, False, False])
    COUNTS = np.array([3, 5, 2, 2])

    def _sampler(self, **kwargs):
        params = dict(
            signal_mask=self.MASK, counts=self.COUNTS, batch_size=4, bg_ratio=0.5
        )
        params.update(kwargs)
        return CountWeightedBalancedBatchSampler(**params)

    def test_every_batch_keeps_bg_ratio(self):
        torch.manual_seed(0)
        for batch in self._sampler():
            self.assertEqual(len(batch), 4)
            self.assertEqual(sum(i in (0, 1) for i in batch), 2)
            self.assertEqual(sum(i in (2, 3) for i in batch), 2)

    def test_epoch_covers_the_on_target_read_total(self):
        # 8 on-target reads, 2 on-target rows per batch -> 4 batches.
        self.assertEqual(len(self._sampler()), 4)
        self.assertEqual(len(list(self._sampler())), 4)

    def test_num_samples_sets_total_draws(self):
        self.assertEqual(len(self._sampler(num_samples=10)), 3)  # ceil(10 / 4)

    def test_draws_follow_counts_within_each_group(self):
        torch.manual_seed(0)
        draws = np.concatenate(list(self._sampler(num_samples=80_000)))
        freq = np.bincount(draws, minlength=4).astype(float)
        np.testing.assert_allclose(
            freq[:2] / freq[:2].sum(), [3 / 8, 5 / 8], atol=0.01
        )
        np.testing.assert_allclose(freq[2:] / freq[2:].sum(), [0.5, 0.5], atol=0.01)

    def test_seeded_torch_makes_batches_reproducible(self):
        sampler = self._sampler()
        torch.manual_seed(3)
        first = list(sampler)
        torch.manual_seed(3)
        self.assertEqual(first, list(sampler))

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self._sampler(counts=np.array([1, 2, 3]))

    def test_rejects_missing_background(self):
        with self.assertRaises(ValueError):
            self._sampler(signal_mask=np.array([True, True, True, True]))


class TestPerRowLoss(unittest.TestCase):
    def test_returns_one_value_per_row_and_restores_reduction(self):
        loss_fct = nn.BCEWithLogitsLoss()
        logits = torch.randn(3, 4)
        targets = torch.rand(3, 4)
        rows = per_row_loss(loss_fct, logits, targets)
        self.assertEqual(rows.shape, (3,))
        self.assertEqual(loss_fct.reduction, "mean")
        self.assertTrue(torch.isclose(rows.mean(), loss_fct(logits, targets)))

    def test_scalar_unreduced_loss_becomes_one_row(self):
        rows = per_row_loss(nn.MSELoss(), torch.tensor(0.5), torch.tensor(1.0))
        self.assertEqual(rows.shape, (1,))


class TestNormalisedWeights(unittest.TestCase):
    def test_weights_average_to_one(self):
        np.testing.assert_allclose(normalised_weights([2, 1, 3]), [1.0, 0.5, 1.5])

    def test_rejects_non_positive_counts(self):
        with self.assertRaises(ValueError):
            normalised_weights([1, 0])


if __name__ == "__main__":
    unittest.main()
