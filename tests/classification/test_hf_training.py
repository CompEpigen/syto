import unittest

import numpy as np

from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedBackgroundBatchSampler,
    BalancedTrainer,
)
# Back-compat: the old name must still import from methylbert.
from syto.classification.classifiers.methylbert import (
    MethylBertTrainer as MethylBertTrainerReexport,
    BalancedTrainer as BalancedTrainerReexport,
    BalancedBackgroundBatchSampler as SamplerReexport,
)


class TestBalancedBackgroundBatchSampler(unittest.TestCase):
    def test_even_split_covers_all_signal_and_caps_background(self):
        np.random.seed(0)
        # 8 signal (indices 0-7), 4 background (indices 8-11)
        mask = np.array([True] * 8 + [False] * 4)
        sampler = BalancedBackgroundBatchSampler(
            mask, batch_size=4, bg_ratio=0.5, shuffle=True
        )

        self.assertEqual(len(sampler), 4)  # 8 signal / 2 signal-per-batch

        batches = list(sampler)
        self.assertEqual(len(batches), 4)

        signal_seen = []
        for batch in batches:
            self.assertEqual(len(batch), 4)
            sig = [i for i in batch if i < 8]
            bg = [i for i in batch if i >= 8]
            self.assertEqual(len(sig), 2)
            self.assertEqual(len(bg), 2)
            signal_seen.extend(sig)

        # Every signal index used exactly once (even division).
        self.assertCountEqual(signal_seen, list(range(8)))

    def test_remainder_batch_emitted_when_drop_last_false(self):
        np.random.seed(0)
        mask = np.array([True] * 5 + [False] * 5)  # signal 0-4, bg 5-9
        sampler = BalancedBackgroundBatchSampler(
            mask, batch_size=4, bg_ratio=0.5, shuffle=False, drop_last=False
        )
        self.assertEqual(len(sampler), 3)  # 5 // 2 == 2 full + 1 remainder
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        # Last (remainder) batch carries the leftover signal index 4.
        self.assertIn(4, batches[-1])

    def test_drop_last_true_omits_remainder(self):
        np.random.seed(0)
        mask = np.array([True] * 5 + [False] * 5)
        sampler = BalancedBackgroundBatchSampler(
            mask, batch_size=4, bg_ratio=0.5, shuffle=False, drop_last=True
        )
        # __len__ still reports the ceil count, but iteration drops remainder.
        emitted = list(sampler)
        for batch in emitted:
            sig = [i for i in batch if i < 5]
            self.assertEqual(len(sig), 2)


class TestReexports(unittest.TestCase):
    def test_methylbert_reexports_are_the_shared_classes(self):
        self.assertIs(MethylBertTrainerReexport, AuxLossLoggingTrainer)
        self.assertIs(BalancedTrainerReexport, BalancedTrainer)
        self.assertIs(SamplerReexport, BalancedBackgroundBatchSampler)


if __name__ == "__main__":
    unittest.main()
