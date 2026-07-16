import logging
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import numpy as np

from transformers import EarlyStoppingCallback

from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedBackgroundBatchSampler,
    BalancedTrainer,
    apply_early_stopping,
    evaluate_train_metrics,
    log_fit_completion,
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


@dataclass
class _DummyArgs:
    """Minimal stand-in for TrainingArguments (a dataclass) so we can exercise
    the dataclasses.replace-based prerequisite auto-setting without building a
    full transformers.TrainingArguments."""

    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None


class TestApplyEarlyStopping(unittest.TestCase):
    def _ready_args(self):
        return _DummyArgs(
            load_best_model_at_end=True, metric_for_best_model="eval_loss"
        )

    def test_none_cfg_is_noop(self):
        args = _DummyArgs(load_best_model_at_end=False)
        out_args, cbs = apply_early_stopping(args, None, None)
        self.assertIs(out_args, args)
        self.assertIsNone(cbs)

    def test_empty_cfg_is_noop(self):
        args = _DummyArgs()
        out_args, cbs = apply_early_stopping(args, {}, [])
        self.assertIs(out_args, args)
        self.assertEqual(cbs, [])

    def test_disabled_flag_is_noop(self):
        args = self._ready_args()
        out_args, cbs = apply_early_stopping(
            args, {"enabled": False, "early_stopping_patience": 5}, []
        )
        self.assertIs(out_args, args)
        self.assertEqual(cbs, [])

    def test_enabled_appends_callback_with_params(self):
        args = self._ready_args()
        out_args, cbs = apply_early_stopping(
            args,
            {"early_stopping_patience": 7, "early_stopping_threshold": 0.01},
            None,
        )
        self.assertEqual(len(cbs), 1)
        cb = cbs[0]
        self.assertIsInstance(cb, EarlyStoppingCallback)
        self.assertEqual(cb.early_stopping_patience, 7)
        self.assertAlmostEqual(cb.early_stopping_threshold, 0.01)
        # prerequisites already satisfied -> args returned unchanged
        self.assertIs(out_args, args)

    def test_defaults_patience_and_threshold(self):
        args = self._ready_args()
        _, cbs = apply_early_stopping(args, {"enabled": True}, None)
        cb = cbs[0]
        self.assertEqual(cb.early_stopping_patience, 1)
        self.assertEqual(cb.early_stopping_threshold, 0.0)

    def test_preserves_existing_callbacks(self):
        args = self._ready_args()
        sentinel = object()
        _, cbs = apply_early_stopping(args, {"early_stopping_patience": 1}, [sentinel])
        self.assertEqual(len(cbs), 2)
        self.assertIs(cbs[0], sentinel)
        self.assertIsInstance(cbs[1], EarlyStoppingCallback)

    def test_autosets_prereqs_with_defaults(self):
        args = _DummyArgs(load_best_model_at_end=False, metric_for_best_model=None)
        out_args, cbs = apply_early_stopping(args, {"early_stopping_patience": 3}, None)
        self.assertIsNot(out_args, args)  # replaced, not mutated
        self.assertTrue(out_args.load_best_model_at_end)
        self.assertEqual(out_args.metric_for_best_model, "eval_loss")
        # original left untouched
        self.assertFalse(args.load_best_model_at_end)
        self.assertIsNone(args.metric_for_best_model)
        self.assertEqual(len(cbs), 1)

    def test_keeps_existing_prereqs(self):
        args = _DummyArgs(
            load_best_model_at_end=True, metric_for_best_model="eval_accuracy"
        )
        out_args, _ = apply_early_stopping(args, {"early_stopping_patience": 2}, None)
        # nothing to fix -> same instance, metric preserved
        self.assertIs(out_args, args)
        self.assertEqual(out_args.metric_for_best_model, "eval_accuracy")


class TestEvaluateTrainMetrics(unittest.TestCase):
    def test_runs_evaluate_over_train_dataset(self):
        trainer = mock.Mock()
        trainer.train_dataset = [1, 2, 3]
        trainer.callback_handler.callbacks = []
        evaluate_train_metrics(trainer)
        trainer.evaluate.assert_called_once()
        _, kwargs = trainer.evaluate.call_args
        self.assertEqual(kwargs.get("metric_key_prefix"), "train")
        self.assertIs(kwargs.get("eval_dataset"), trainer.train_dataset)

    def test_skips_when_no_train_dataset(self):
        trainer = mock.Mock()
        trainer.train_dataset = None
        evaluate_train_metrics(trainer)
        trainer.evaluate.assert_not_called()

    def test_detaches_and_restores_early_stopping_callback(self):
        trainer = mock.Mock()
        trainer.train_dataset = [1]
        es = EarlyStoppingCallback(early_stopping_patience=2)
        trainer.callback_handler.callbacks = [es]
        evaluate_train_metrics(trainer)
        trainer.remove_callback.assert_called_once_with(es)
        trainer.add_callback.assert_called_once_with(es)


class TestLogFitCompletion(unittest.TestCase):
    def _trainer(self, best_ckpt, best_metric, metric_name="eval_loss"):
        return SimpleNamespace(
            state=SimpleNamespace(
                best_model_checkpoint=best_ckpt, best_metric=best_metric
            ),
            args=SimpleNamespace(metric_for_best_model=metric_name),
        )

    def test_logs_finished_with_best_checkpoint_and_metric(self):
        trainer = self._trainer("/out/checkpoint-400", 0.2612345)
        with self.assertLogs("syto.classification.hf_training", level="INFO") as cm:
            log_fit_completion(trainer)
        text = "\n".join(cm.output)
        self.assertIn("finished", text.lower())
        self.assertIn("checkpoint-400", text)
        self.assertIn("eval_loss", text)
        self.assertIn("0.261", text)

    def test_handles_missing_best_info(self):
        trainer = self._trainer(None, None)
        with self.assertLogs("syto.classification.hf_training", level="INFO") as cm:
            log_fit_completion(trainer)  # must not raise
        self.assertIn("finished", "\n".join(cm.output).lower())


class TestReexports(unittest.TestCase):
    def test_methylbert_reexports_are_the_shared_classes(self):
        self.assertIs(MethylBertTrainerReexport, AuxLossLoggingTrainer)
        self.assertIs(BalancedTrainerReexport, BalancedTrainer)
        self.assertIs(SamplerReexport, BalancedBackgroundBatchSampler)


if __name__ == "__main__":
    unittest.main()
