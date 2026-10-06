"""De-duplicated (read-count weighted) and compact-input training for Dismir."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from syto.classification.classifiers.dismir import CountWeightedSampler, Dismir
from syto.classification.loss import ConfidenceWeightedCrossEntropy

NUM_LABELS = 3
SEQ_LEN = 16


def _model(**overrides):
    kwargs = dict(
        max_sequence_length=SEQ_LEN,
        train_data_path="",
        test_data_path="",
        valid_data_path="",
        device=torch.device("cpu"),
        num_labels=NUM_LABELS,
        classifier_type="grg_attention_based",
        num_grg_labels=4,
        grg_label_column="dmr_ctype_label",
        soft_labels=True,
    )
    kwargs.update(overrides)
    return Dismir(**kwargs)


def _random_reads(n, rng, min_len=4, max_len=24):
    """Reads of varying length over ACGTN + a lowercase base, with a pattern."""
    seqs, patterns = [], []
    for _ in range(n):
        length = int(rng.integers(min_len, max_len + 1))
        seqs.append("".join(rng.choice(list("ACGTNa"), size=length)))
        patterns.append("".join(rng.choice(list("0122"), size=length)))
    return seqs, patterns


def _reads_df(n_unique, rng, counts=None):
    """Unique soft-labelled reads, optionally carrying a ``read_count``."""
    seqs, patterns = _random_reads(n_unique, rng)
    labels = rng.dirichlet(np.ones(NUM_LABELS) * 0.5, size=n_unique)
    df = pd.DataFrame(
        {
            "input_ids": seqs,
            "methylation_ids": patterns,
            "dmr_ctype_label": rng.integers(0, 4, size=n_unique),
            "soft_label": list(labels),
        }
    )
    if counts is not None:
        df["read_count"] = counts
    return df


def _expand(df):
    """Undo de-duplication: repeat each row ``read_count`` times."""
    return (
        df.loc[df.index.repeat(df["read_count"])]
        .drop(columns="read_count")
        .reset_index(drop=True)
    )


class TestCompactInputs(unittest.TestCase):
    def test_expanded_codes_equal_conv_onehot(self):
        rng = np.random.default_rng(0)
        model = _model()
        seqs, patterns = _random_reads(200, rng, min_len=1, max_len=40)
        dna, meth = pd.Series(seqs), pd.Series(patterns)

        codes = model.encode_codes(dna, meth)
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(codes.shape, (200, SEQ_LEN))
        np.testing.assert_array_equal(
            model.expand_codes(torch.from_numpy(codes)).numpy(),
            model.conv_onehot(dna, meth),
        )

    def test_compact_training_matches_dense_training(self):
        rng = np.random.default_rng(1)
        train, valid = _reads_df(40, rng), _reads_df(12, rng)

        def fit(compact):
            torch.manual_seed(0)
            model = _model()
            with tempfile.TemporaryDirectory() as d:
                model.train(
                    train_dir=d,
                    epochs=2,
                    batch_size=8,
                    optimizer_type="ADAM",
                    lr=1e-3,
                    verbose=0,
                    train_df=train,
                    valid_df=valid,
                    compact_inputs=compact,
                )
            return model.history

        dense, compact = fit(False), fit(True)
        for a, b in zip(dense, compact):
            self.assertAlmostEqual(a["train_loss"], b["train_loss"], places=5)
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], places=5)


class TestCountWeightedLoss(unittest.TestCase):
    def test_weighted_loss_and_gradient_equal_the_duplicated_batch(self):
        rng = np.random.default_rng(2)
        counts = np.array([1, 5, 2, 9, 1, 3])
        dedup = _reads_df(len(counts), rng, counts=counts)
        full = _expand(dedup)

        def loss_and_grad(df, with_counts):
            torch.manual_seed(0)
            model = _model()
            model.model.eval()  # no dropout: both passes see the same network
            x = torch.from_numpy(
                model.conv_onehot(df["input_ids"], df["methylation_ids"])
            )
            y = torch.tensor(np.stack(df["soft_label"]), dtype=torch.float32)
            grg = torch.tensor(df["dmr_ctype_label"].to_numpy(), dtype=torch.long)
            criterion = ConfidenceWeightedCrossEntropy(NUM_LABELS, penalty_scale=0.5)
            outputs, _ = model.model(x, grg_ids=grg)
            if with_counts:
                c = torch.tensor(df["read_count"].to_numpy(), dtype=torch.float32)
                model._mean_train_count = float(c.mean())
                loss = model._batch_loss(criterion, outputs, y, c)
            else:
                loss = model._batch_loss(criterion, outputs, y)
            loss.backward()
            self.assertEqual(criterion.reduction, "mean")  # restored
            grad = torch.cat(
                [p.grad.flatten() for p in model.model.parameters() if p.grad is not None]
            )
            return loss.item(), grad

        full_loss, full_grad = loss_and_grad(full, with_counts=False)
        dedup_loss, dedup_grad = loss_and_grad(dedup, with_counts=True)
        self.assertAlmostEqual(full_loss, dedup_loss, places=5)
        torch.testing.assert_close(full_grad, dedup_grad, rtol=1e-4, atol=1e-6)


class TestDeduplicatedValidation(unittest.TestCase):
    def _history(self, valid):
        rng = np.random.default_rng(6)
        train = _reads_df(40, rng)
        torch.manual_seed(0)
        model = _model()
        with tempfile.TemporaryDirectory() as d:
            model.train(
                train_dir=d,
                epochs=2,
                batch_size=8,
                optimizer_type="ADAM",
                lr=1e-3,
                verbose=0,
                train_df=train,
                valid_df=valid,
                compact_inputs=True,
            )
        return model.history

    def test_count_weighted_validation_equals_expanded_validation(self):
        rng = np.random.default_rng(7)
        valid = _reads_df(15, rng, counts=rng.integers(1, 7, size=15))
        dedup, expanded = self._history(valid), self._history(_expand(valid))
        # Average precision is excluded: in the expanded pass identical reads
        # get probabilities differing by float noise (~1e-7, depending on batch
        # position), which splits tied thresholds and moves AP. With exact ties
        # restored, the expanded AP equals the count-weighted one (see
        # TestSampleWeightedMetrics in test_evaluation.py).
        val_keys = [
            k
            for k in expanded[0]
            if k.startswith("val_") and k != "val_average_precision"
        ]
        self.assertIn("val_f1", val_keys)
        for a, b in zip(dedup, expanded):
            for key in val_keys:
                self.assertAlmostEqual(a[key], b[key], places=5, msg=key)
            self.assertAlmostEqual(a["train_loss"], b["train_loss"], places=6)


class TestCountWeightedSampler(unittest.TestCase):
    def test_epoch_length_defaults_to_total_reads(self):
        self.assertEqual(len(CountWeightedSampler([1, 4, 5])), 10)
        self.assertEqual(len(CountWeightedSampler([1, 4, 5], num_samples=3)), 3)

    def test_draws_follow_counts(self):
        torch.manual_seed(0)
        counts = np.array([1, 3, 6])
        draws = list(CountWeightedSampler(counts, num_samples=60_000))
        freq = np.bincount(draws, minlength=3) / len(draws)
        np.testing.assert_allclose(freq, counts / counts.sum(), atol=0.01)

    def test_seeded_torch_makes_draws_reproducible(self):
        sampler = CountWeightedSampler([2, 7, 1, 4])
        torch.manual_seed(5)
        first = list(sampler)
        torch.manual_seed(5)
        self.assertEqual(first, list(sampler))

    def test_rejects_non_positive_counts(self):
        with self.assertRaises(ValueError):
            CountWeightedSampler([1, 0, 2])


class TestTrainingWithReadCounts(unittest.TestCase):
    def _fit(self, **train_kwargs):
        rng = np.random.default_rng(3)
        counts = rng.integers(1, 6, size=30)
        train, valid = _reads_df(30, rng, counts=counts), _reads_df(10, rng)
        torch.manual_seed(0)
        model = _model()
        with tempfile.TemporaryDirectory() as d:
            model.train(
                train_dir=d,
                epochs=1,
                batch_size=8,
                optimizer_type="ADAM",
                lr=1e-3,
                verbose=0,
                train_df=train,
                valid_df=valid,
                **train_kwargs,
            )
        return model, int(counts.sum())

    def _samples_seen(self, **train_kwargs):
        """Fit, returning the model, total reads and train samples in epoch 1."""
        seen = []
        original = Dismir._compute_epoch_metrics

        def spy(model, all_outputs, all_labels, prefix, all_weights=None):
            if prefix == "train":
                seen.append(sum(len(o) for o in all_outputs))
            return original(model, all_outputs, all_labels, prefix, all_weights)

        with patch.object(Dismir, "_compute_epoch_metrics", autospec=True) as m:
            m.side_effect = spy
            model, total = self._fit(**train_kwargs)
        return model, total, seen[0]

    def test_sampler_draws_one_epoch_of_original_reads(self):
        model, total, seen = self._samples_seen(
            count_weighting="sampler", compact_inputs=True
        )
        self.assertEqual(len(model.train_y), 30)  # stored de-duplicated
        self.assertEqual(seen, total)  # but an epoch covers every read
        self.assertTrue(np.isfinite(model.history[0]["train_loss"]))

    def test_samples_per_epoch_caps_the_draw(self):
        _, _, seen = self._samples_seen(count_weighting="sampler", samples_per_epoch=16)
        self.assertEqual(seen, 16)

    def test_loss_weighting_visits_each_row_once(self):
        model, _, seen = self._samples_seen(count_weighting="loss")
        self.assertEqual(seen, 30)
        self.assertTrue(np.isfinite(model.history[0]["train_loss"]))

    def test_unknown_count_weighting_is_rejected(self):
        with self.assertRaises(ValueError):
            self._fit(count_weighting="bogus")

    def test_variable_length_rejects_read_counts(self):
        rng = np.random.default_rng(4)
        model = _model(classifier_type="vanilla")
        with self.assertRaises(NotImplementedError):
            model.train(
                variable_length=True,
                verbose=0,
                train_df=_reads_df(5, rng, counts=[1, 2, 3, 4, 5]),
                valid_df=_reads_df(3, rng),
            )


if __name__ == "__main__":
    unittest.main()
