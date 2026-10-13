"""EpigenBERT2 trains and evaluates on de-duplicated reads like on duplicated ones."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments

import syto.classification.classifiers.dnabert2 as dnabert2_module
from syto.classification.classifiers.dnabert2 import (
    DataCollatorForFineTunedDataset,
    DNABERT2FineTuneDataset,
    EpigenDnabert2,
)


class _FakeTokenizer:
    model_max_length = 8
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[0, 1]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


def _frame():
    return pd.DataFrame(
        {
            "input_ids": ["ACGT", "ACGA", "ACGC"],
            "methylation_ids": ["2222"] * 3,
            "label": [0, 1, 0],
            "read_count": [2, 1, 3],
        }
    )


def _dataset(read_count_column="read_count"):
    ds = DNABERT2FineTuneDataset(
        data_path_or_list=_frame(),
        tokenizer=_FakeTokenizer(),
        data_interface="pandas",
        lazy_tokenization=True,
        read_count_column=read_count_column,
    )
    ds.cpg_methylation = None  # skip methylation tokenization in getitem
    return ds


class TestDatasetCounts(unittest.TestCase):
    def test_rows_carry_counts_and_normalised_weights(self):
        ds = _dataset()
        np.testing.assert_array_equal(ds.read_counts, [2.0, 1.0, 3.0])
        self.assertAlmostEqual(float(ds[2]["sample_weight"]), 1.5)

    def test_counts_are_opt_in(self):
        ds = _dataset(read_count_column=None)
        self.assertIsNone(ds.read_counts)
        self.assertNotIn("sample_weight", ds[0])

    def test_collator_stacks_sample_weight(self):
        ds = _dataset()
        batch = DataCollatorForFineTunedDataset(tokenizer=_FakeTokenizer())(
            [ds[i] for i in range(3)]
        )
        np.testing.assert_allclose(batch["sample_weight"].numpy(), [1.0, 0.5, 1.5])


class _FixedBackbone(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, *args, **kwargs):
        return self.outputs


class _FixedClassifier(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, pooled_output):
        return self.logits


def _stub_model(logits, soft_labels=False):
    model = object.__new__(dnabert2_module.BertForSequenceClassification)
    torch.nn.Module.__init__(model)
    n = logits.shape[0]
    model.num_labels = 1 if logits.ndim == 1 else logits.shape[-1]
    model.num_grg_labels = None
    model.config = SimpleNamespace(problem_type=None, use_return_dict=True)
    model.bert = _FixedBackbone((torch.zeros(n, 2, 3), torch.zeros(n, 3), ()))
    model.dropout = torch.nn.Identity()
    model.soft_labels = soft_labels
    model.classifier = _FixedClassifier(logits)
    return model


class TestForwardWeighting(unittest.TestCase):
    COUNTS = torch.tensor([2, 1, 3])

    def _cases(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 3)
        soft = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
        multi = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        return [
            ("ce", logits, torch.tensor([0, 2, 1]), False),
            ("cwce", logits, soft, True),
            ("bce", logits, multi, False),
            ("mse", torch.randn(3, 1), torch.randn(3, 1), False),
        ]

    def test_weighted_rows_match_the_duplicated_batch(self):
        weights = self.COUNTS.float() / self.COUNTS.float().mean()
        for name, logits, labels, soft in self._cases():
            with self.subTest(name):
                weighted = _stub_model(logits, soft)(
                    input_ids=torch.zeros(3, 2, dtype=torch.long),
                    labels=labels,
                    sample_weight=weights,
                )
                duplicated = _stub_model(
                    logits.repeat_interleave(self.COUNTS, dim=0), soft
                )(
                    input_ids=torch.zeros(6, 2, dtype=torch.long),
                    labels=labels.repeat_interleave(self.COUNTS, dim=0),
                )
                self.assertTrue(
                    torch.isclose(weighted.loss, duplicated.loss, atol=1e-6)
                )


class TestWrapperWiring(unittest.TestCase):
    def _stub(self):
        model = object.__new__(EpigenDnabert2)
        model.num_labels = 2
        model.num_grg_labels = None
        model.soft_labels = False
        model.tokenizer = _FakeTokenizer()
        model.training_args = TrainingArguments(output_dir="/tmp/eb-out", report_to=[])
        model.fine_tune = MagicMock(name="fine_tune")
        model.trainer = MagicMock()
        model.history = []
        model.save = MagicMock()
        model.model = object()
        model.data_collator = object()
        return model

    def test_declares_read_count_support(self):
        self.assertTrue(EpigenDnabert2.supports_read_counts)

    def test_fit_feeds_counts_and_options_to_fine_tune(self):
        model = self._stub()
        with patch.object(dnabert2_module, "evaluate_train_metrics"), patch.object(
            dnabert2_module, "log_fit_completion"
        ), patch.object(dnabert2_module, "extract_trainer_metrics", return_value={}):
            EpigenDnabert2.fit_classificaton.__wrapped__(
                model,
                train_df=_frame(),
                val_df=_frame(),
                output_dir=None,
                count_weighting="loss",
                samples_per_epoch=9,
                compute_train_metrics=False,
            )
        kwargs = model.fine_tune.call_args.kwargs
        self.assertEqual(kwargs["count_weighting"], "loss")
        self.assertEqual(kwargs["samples_per_epoch"], 9)
        np.testing.assert_array_equal(
            kwargs["train_dataset"].read_counts, [2.0, 1.0, 3.0]
        )
        np.testing.assert_array_equal(
            kwargs["val_dataset"].read_counts, [2.0, 1.0, 3.0]
        )

    def test_init_trainer_forwards_the_options(self):
        model = self._stub()
        for mask, trainer_name in (
            (None, "AuxLossLoggingTrainer"),
            (np.array([True, False]), "BalancedTrainer"),
        ):
            with self.subTest(trainer_name), patch.object(
                dnabert2_module, trainer_name
            ) as trainer_cls, patch.object(
                dnabert2_module, "use_table_progress_callback"
            ):
                model._init_trainer(
                    args=MagicMock(),
                    signal_mask=mask,
                    count_weighting="loss",
                    samples_per_epoch=5,
                )
            kwargs = trainer_cls.call_args.kwargs
            self.assertEqual(kwargs["count_weighting"], "loss")
            self.assertEqual(kwargs["samples_per_epoch"], 5)


if __name__ == "__main__":
    unittest.main()
