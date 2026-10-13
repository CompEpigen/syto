"""MethylBERT trains and evaluates on de-duplicated reads like on duplicated ones."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, TrainingArguments

import syto.classification.classifiers.methylbert as methylbert_module
from syto.classification.classifiers.methylbert import (
    MethylBert,
    MethylBertEmbeddedGRG,
    MethylBertFinetuneDataset,
    MethylVocab,
    methylbert_finetune_collator,
    prepare_methylbert_list,
)

HEADER = [
    "dna_seq",
    "methyl_seq",
    "grg_ctype",
    "grg_label",
    "ctype",
    "original_label",
    "read_name",
    "ncpgs_marked",
    "on_target_mask",
]


def _reads():
    return pd.DataFrame(
        [
            {
                "read_name": "r0",
                "input_ids": "ACGTACGTAC",
                "methylation_ids": "0101010101",
                "label": 1,
                "dmr_ctype_label": 1,
                "original_label": 1,
                "read_count": 4,
            },
            {
                "read_name": "r1",
                "input_ids": "ACGTAC",
                "methylation_ids": "010101",
                "label": 0,
                "dmr_ctype_label": 1,
                "original_label": 0,
                "read_count": 2,
            },
        ]
    )


def _chunks(read_count_column="read_count"):
    return prepare_methylbert_list(
        _reads(),
        grg_label_column="dmr_ctype_label",
        seq_length=3,
        stride=1,
        read_count_column=read_count_column,
    )


class TestPrepareListCounts(unittest.TestCase):
    def test_every_chunk_inherits_its_reads_count(self):
        data = _chunks()
        self.assertEqual(data[0], HEADER + ["read_count"])
        name, count = HEADER.index("read_name"), len(HEADER)
        by_read = {}
        for row in data[1:]:
            by_read.setdefault(row[name], set()).add(row[count])
        self.assertEqual(by_read, {"r0": {4}, "r1": {2}})
        self.assertGreater(sum(row[name] == "r0" for row in data[1:]), 1)

    def test_without_the_option_the_layout_is_unchanged(self):
        data = _chunks(read_count_column=None)
        self.assertEqual(data[0], HEADER)
        self.assertEqual(data[1][-3], "r0")  # predict_split's positional read_name


class TestDatasetCounts(unittest.TestCase):
    def _dataset(self, data):
        return MethylBertFinetuneDataset(
            data_source=data, vocab=MethylVocab(k=3), seq_len=10, lazy_tokenization=True
        )

    def test_read_counts_and_weights_are_per_chunk(self):
        data = _chunks()
        dataset = self._dataset(data)
        counts = np.array([row[-1] for row in data[1:]], dtype=float)
        np.testing.assert_array_equal(dataset.read_counts, counts)
        self.assertAlmostEqual(
            float(dataset[0]["sample_weight"]), counts[0] / counts.mean(), places=6
        )

    def test_plain_rows_carry_no_weight(self):
        dataset = self._dataset(_chunks(read_count_column=None))
        self.assertIsNone(dataset.read_counts)
        self.assertNotIn("sample_weight", dataset[0])

    def test_subset_renormalises_the_weights(self):
        dataset = self._dataset(_chunks())
        dataset.subset_data(2)
        self.assertEqual(len(dataset.read_counts), 2)
        self.assertAlmostEqual(float(dataset[0]["sample_weight"]), 1.0)

    def test_collator_stacks_sample_weight(self):
        features = [
            {
                "input_ids": torch.zeros(6, dtype=torch.long),
                "token_type_ids": torch.zeros(6, dtype=torch.long),
                "grg_ids": 0,
                "labels": 1,
                "sample_weight": torch.tensor(w),
            }
            for w in (0.5, 1.5)
        ]
        batch = methylbert_finetune_collator(features)
        self.assertEqual(batch["sample_weight"].tolist(), [0.5, 1.5])


def _model(loss, num_labels):
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=80,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=24,
        max_position_embeddings=32,
        type_vocab_size=3,
    )
    config.num_labels = num_labels
    config.num_grg_labels = 4
    config.loss = loss
    return MethylBertEmbeddedGRG(config, seq_len=5).eval()


class TestForwardWeighting(unittest.TestCase):
    COUNTS = torch.tensor([2, 1, 3])
    SOFT = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    CASES = [
        ("ce", 3, torch.tensor([0, 2, 1])),
        ("bce", 3, torch.tensor([0, 2, 1])),
        ("focal_bce", 3, torch.tensor([0, 2, 1])),
        ("cwce", 3, SOFT),
        ("bce", 1, torch.tensor([1.0, 0.0, 1.0])),
    ]

    def _inputs(self):
        torch.manual_seed(1)
        return {
            "input_ids": torch.randint(5, 20, (3, 6)),
            "token_type_ids": torch.randint(0, 3, (3, 6)),
            "grg_ids": torch.tensor([0, 1, 2]),
        }

    def test_weighted_rows_match_the_duplicated_batch(self):
        weights = self.COUNTS.float() / self.COUNTS.float().mean()
        for loss, num_labels, labels in self.CASES:
            with self.subTest(loss=loss, num_labels=num_labels):
                model = _model(loss, num_labels)
                inputs = self._inputs()
                with torch.no_grad():
                    weighted = model(**inputs, labels=labels, sample_weight=weights)
                    duplicated = model(
                        **{
                            k: v.repeat_interleave(self.COUNTS, dim=0)
                            for k, v in inputs.items()
                        },
                        labels=labels.repeat_interleave(self.COUNTS, dim=0),
                    )
                self.assertTrue(
                    torch.isclose(weighted.loss, duplicated.loss, atol=1e-5)
                )
                if duplicated.loss_ce is not None:
                    self.assertTrue(
                        torch.isclose(weighted.loss_ce, duplicated.loss_ce, atol=1e-5)
                    )

    def test_without_weight_the_loss_is_unchanged(self):
        model = _model("ce", 3)
        with torch.no_grad():
            a = model(**self._inputs(), labels=torch.tensor([0, 2, 1]))
            b = model(
                **self._inputs(), labels=torch.tensor([0, 2, 1]), sample_weight=None
            )
        self.assertTrue(torch.equal(a.loss, b.loss))


class TestWrapperWiring(unittest.TestCase):
    def _stub(self):
        model = object.__new__(MethylBert)
        model.seq_len = 4
        model.num_labels = 2
        model.soft_labels = False
        model.training_args = TrainingArguments(output_dir="/tmp/mb-out", report_to=[])
        model.fine_tune = MagicMock(name="fine_tune")
        model.trainer = MagicMock()
        model.history = []
        model.save = MagicMock()
        model.model = object()
        model.tokenizer = None
        return model

    def test_declares_read_count_support(self):
        self.assertTrue(MethylBert.supports_read_counts)

    def test_fit_feeds_counts_and_options_to_fine_tune(self):
        model = self._stub()
        with patch.object(methylbert_module, "evaluate_train_metrics"), patch.object(
            methylbert_module, "log_fit_completion"
        ), patch.object(methylbert_module, "extract_trainer_metrics", return_value={}):
            MethylBert.fit_classificaton.__wrapped__(
                model,
                train_df=_reads(),
                val_df=_reads(),
                output_dir=None,
                grg_label_column="dmr_ctype_label",
                count_weighting="loss",
                samples_per_epoch=7,
                compute_train_metrics=False,
            )
        kwargs = model.fine_tune.call_args.kwargs
        self.assertEqual(kwargs["count_weighting"], "loss")
        self.assertEqual(kwargs["samples_per_epoch"], 7)
        self.assertEqual(set(kwargs["train_dataset"].read_counts), {4.0, 2.0})
        self.assertEqual(set(kwargs["val_dataset"].read_counts), {4.0, 2.0})

    def test_init_trainer_forwards_the_options(self):
        model = self._stub()
        for mask, trainer_name in (
            (None, "MethylBertTrainer"),
            (np.array([True, False]), "BalancedTrainer"),
        ):
            with self.subTest(trainer_name), patch.object(
                methylbert_module, trainer_name
            ) as trainer_cls, patch.object(
                methylbert_module, "use_table_progress_callback"
            ):
                model._init_trainer(
                    signal_mask=mask, count_weighting="loss", samples_per_epoch=5
                )
            kwargs = trainer_cls.call_args.kwargs
            self.assertEqual(kwargs["count_weighting"], "loss")
            self.assertEqual(kwargs["samples_per_epoch"], 5)

    def test_predict_split_ignores_read_counts(self):
        """Pseudobulk feeds de-duplicated splits; prediction stays read-for-read."""
        split_df = pd.DataFrame(
            [
                {
                    "read_name": "read-1",
                    "seq": "ATCGATCG",
                    "pattern": "01010101",
                    "dmr_ctype_label": 3,
                    "original_label": 775,
                    "read_count": 9,
                }
            ]
        )
        clf = object.__new__(MethylBert)
        clf.seq_len = 4
        clf.soft_labels = True
        clf.num_labels = 2
        captured = {}

        def fake_predict(dataset, batch_size=None):
            batch = methylbert_finetune_collator(
                [dataset[i] for i in range(len(dataset))]
            )
            captured["keys"] = set(batch)
            return (np.tile([0.25, 0.75], (len(dataset), 1)),)

        clf.predict = fake_predict
        result = clf.predict_split(split_df, grg_label_column="dmr_ctype_label")
        self.assertNotIn("sample_weight", captured["keys"])
        self.assertEqual(list(result["read_name"]), ["read-1"])
        self.assertEqual(list(result["read_count"]), [9])


if __name__ == "__main__":
    unittest.main()
