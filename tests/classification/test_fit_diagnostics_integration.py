"""End-to-end check that a real Trainer evaluation emits an on-target plot.

Uses a tiny stub model rather than MethylBERT: the point under test is the
hook wiring plus prediction/metadata alignment through the genuine HF
evaluation loop, not the architecture.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from syto.classification.evaluation import preprocess_logits_for_prediction
from syto.classification.fit_diagnostics import OnTargetScoreRecorder
from syto.classification.hf_training import AuxLossLoggingTrainer


class _TinyDataset(Dataset):
    def __init__(self, n=8, num_labels=3):
        self.n = n
        self.num_labels = num_labels

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "features": torch.full((4,), float(i), dtype=torch.float),
            "labels": i % self.num_labels,
        }


def _collate(features):
    return {
        "features": torch.stack([f["features"] for f in features]),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }


class _TinyModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.linear = nn.Linear(4, num_labels)

    def forward(self, features=None, labels=None):
        logits = self.linear(features)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class TestEvaluationEmitsOnTargetPlot(unittest.TestCase):
    def test_real_evaluation_writes_one_png_per_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            dataset = _TinyDataset(n=8, num_labels=3)
            recorder = OnTargetScoreRecorder(out / "on_target_score_plots")
            # Half the chunks are on target; labels stay < num_labels.
            recorder.register(
                dataset,
                dmr_labels=np.array([0, 1, 2, 0, 1, 2, 0, 1]),
                on_target_mask=np.array(
                    [True, False, True, False, True, False, True, False]
                ),
            )

            trainer = AuxLossLoggingTrainer(
                model=_TinyModel(),
                args=TrainingArguments(
                    output_dir=str(out),
                    per_device_eval_batch_size=3,
                    report_to=[],
                    remove_unused_columns=False,
                    use_cpu=True,
                ),
                eval_dataset=dataset,
                data_collator=_collate,
                preprocess_logits_for_metrics=preprocess_logits_for_prediction,
                # Required, and not incidental: Trainer.evaluate forces
                # prediction_loss_only=True when compute_metrics is None, which
                # discards output.predictions and leaves the recorder nothing to
                # plot. MethylBert._init_trainer always supplies one
                # (compute_metrics / compute_metrics_soft_labels), so this
                # mirrors the production configuration.
                compute_metrics=lambda eval_pred: {},
                eval_diagnostics=recorder,
            )

            trainer.evaluate()

            plots = sorted((out / "on_target_score_plots").glob("*.png"))
            self.assertEqual(len(plots), 1, f"expected 1 plot, got {plots}")
            self.assertTrue(plots[0].name.startswith("eval_step_"))
            self.assertGreater(plots[0].stat().st_size, 0)

            # A second pass with a different prefix adds a second file, the way
            # evaluate_train_metrics does at the end of a fit.
            trainer.evaluate(eval_dataset=dataset, metric_key_prefix="train")
            self.assertEqual(
                len(sorted((out / "on_target_score_plots").glob("train_*.png"))), 1
            )
