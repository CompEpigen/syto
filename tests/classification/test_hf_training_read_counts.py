"""HF trainers on de-duplicated rows behave as on the duplicated data."""

import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, RandomSampler
from transformers import TrainingArguments

from syto.classification.evaluation import (
    compute_metrics,
    preprocess_logits_for_prediction,
)
from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedBackgroundBatchSampler,
    BalancedTrainer,
)
from syto.classification.read_counts import (
    CountWeightedBalancedBatchSampler,
    CountWeightedSampler,
    ReadCountDatasetMixin,
    ReadWeightedLossMixin,
)

FEATURES = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
LABELS = np.array([0, 1, 2, 0, 1])
COUNTS = np.array([3, 1, 2, 4, 1])


class _Rows(ReadCountDatasetMixin, Dataset):
    def __init__(self, features, labels, counts=None):
        self.features = torch.as_tensor(features)
        self.labels = [int(x) for x in labels]
        self._set_read_counts(counts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self._with_sample_weight(
            {"features": self.features[i], "labels": self.labels[i]}, i
        )


def _collate(features):
    batch = {
        "features": torch.stack([f["features"] for f in features]),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }
    if "sample_weight" in features[0]:
        batch["sample_weight"] = torch.stack([f["sample_weight"] for f in features])
    return batch


class _WeightedModel(ReadWeightedLossMixin, nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(4, 3)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features=None, labels=None, sample_weight=None):
        logits = self.linear(features)
        loss = self._loss(self.loss_fct, logits, labels, sample_weight=sample_weight)
        return {"loss": loss, "logits": logits}


def _deduplicated():
    return _Rows(FEATURES, LABELS, COUNTS)


def _duplicated():
    return _Rows(np.repeat(FEATURES, COUNTS, axis=0), np.repeat(LABELS, COUNTS))


class _TrainerCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _trainer(self, cls=AuxLossLoggingTrainer, train_dataset=None, **kwargs):
        return cls(
            model=_WeightedModel(),
            args=TrainingArguments(
                output_dir=self._tmp.name,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=3,
                report_to=[],
                remove_unused_columns=False,
                use_cpu=True,
            ),
            train_dataset=train_dataset,
            data_collator=_collate,
            preprocess_logits_for_metrics=preprocess_logits_for_prediction,
            compute_metrics=compute_metrics,
            **kwargs,
        )


def _comparable(metrics):
    return {
        k: v for k, v in metrics.items() if "runtime" not in k and "per_second" not in k
    }


class TestEvaluationOnDeduplicatedRows(_TrainerCase):
    def test_eval_loss_and_metrics_equal_the_duplicated_data(self):
        dedup = _comparable(self._trainer().evaluate(eval_dataset=_deduplicated()))
        full = _comparable(self._trainer().evaluate(eval_dataset=_duplicated()))
        self.assertEqual(dedup.keys(), full.keys())
        for key, value in full.items():
            self.assertAlmostEqual(dedup[key], value, places=5, msg=key)

    def test_train_metrics_pass_is_weighted_in_sampler_mode(self):
        trainer = self._trainer(
            train_dataset=_deduplicated(), count_weighting="sampler"
        )
        dedup = trainer.evaluate(
            eval_dataset=trainer.train_dataset, metric_key_prefix="train"
        )
        full = self._trainer().evaluate(eval_dataset=_duplicated())
        self.assertAlmostEqual(dedup["train_loss"], full["eval_loss"], places=5)

    def test_compute_metrics_is_restored_after_evaluation(self):
        trainer = self._trainer()
        trainer.evaluate(eval_dataset=_deduplicated())
        self.assertIs(trainer.compute_metrics, compute_metrics)


class TestTrainingSampler(_TrainerCase):
    def test_sampler_mode_draws_by_counts(self):
        trainer = self._trainer(train_dataset=_deduplicated(), samples_per_epoch=7)
        sampler = trainer._get_train_sampler()
        self.assertIsInstance(sampler, CountWeightedSampler)
        self.assertEqual(len(sampler), 7)

    def test_sampler_epoch_defaults_to_total_reads(self):
        sampler = self._trainer(train_dataset=_deduplicated())._get_train_sampler()
        self.assertEqual(len(sampler), int(COUNTS.sum()))

    def test_loss_mode_and_plain_data_shuffle_rows(self):
        for trainer in (
            self._trainer(train_dataset=_deduplicated(), count_weighting="loss"),
            self._trainer(train_dataset=_duplicated()),
        ):
            self.assertIsInstance(trainer._get_train_sampler(), RandomSampler)

    def test_invalid_count_weighting_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "count_weighting must be 'sampler' or 'loss'"
        ):
            self._trainer(count_weighting="both")

    def test_counts_need_a_single_process(self):
        trainer = self._trainer(train_dataset=_deduplicated())
        with mock.patch.object(
            TrainingArguments,
            "world_size",
            new_callable=mock.PropertyMock,
            return_value=2,
        ):
            with self.assertRaisesRegex(ValueError, "single process"):
                trainer._get_train_sampler()


class TestTrainingLoss(_TrainerCase):
    def _batch(self):
        dataset = _deduplicated()
        return _collate([dataset[i] for i in range(len(dataset))])

    def test_sampler_mode_drops_the_weights_in_training(self):
        trainer = self._trainer(
            train_dataset=_deduplicated(), count_weighting="sampler"
        )
        trainer.model.train()
        batch = self._batch()
        unweighted = {k: v for k, v in batch.items() if k != "sample_weight"}
        self.assertTrue(
            torch.isclose(
                trainer.compute_loss(trainer.model, batch),
                trainer.compute_loss(trainer.model, unweighted),
            )
        )

    def test_loss_mode_keeps_the_weights(self):
        trainer = self._trainer(train_dataset=_deduplicated(), count_weighting="loss")
        trainer.model.train()
        batch = self._batch()
        expected = trainer.model(**batch)["loss"]
        self.assertTrue(
            torch.isclose(trainer.compute_loss(trainer.model, batch), expected)
        )


class TestBalancedTrainerWithCounts(_TrainerCase):
    MASK = np.array([True, False, True, False, True])

    def _balanced(self, **kwargs):
        return self._trainer(
            cls=BalancedTrainer,
            train_dataset=_deduplicated(),
            signal_mask=self.MASK,
            bg_ratio=0.5,
            **kwargs,
        )

    def test_sampler_mode_uses_the_count_weighted_balanced_sampler(self):
        loader = self._balanced(count_weighting="sampler").get_train_dataloader()
        self.assertIsInstance(loader.batch_sampler, CountWeightedBalancedBatchSampler)

    def test_loss_mode_keeps_the_uniform_balanced_sampler(self):
        loader = self._balanced(count_weighting="loss").get_train_dataloader()
        self.assertIsInstance(loader.batch_sampler, BalancedBackgroundBatchSampler)


if __name__ == "__main__":
    unittest.main()
