import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from methyldl.deconvolution.deep_deconvolvers.training import (
    EarlyStopping,
    train_matrix_deconvolver,
)
from methyldl.deconvolution.history import DeconvolutionHistory


class TinySoftmaxRegressor(nn.Module):
    """Minimal differentiable model used to exercise the training loop cheaply."""

    def __init__(self, input_dim: int, output_dim: int):
        """Build a single linear layer followed by a softmax output head."""
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the matrix input and return normalized class proportions."""
        logits = self.linear(x.flatten(start_dim=1))
        return torch.softmax(logits, dim=-1)


class RecordingPlateauScheduler:
    """Lightweight stand-in that records metric values passed to step."""

    def __init__(self):
        """Initialize an empty list of metric values."""
        self.step_args = []

    def step(self, metric):
        """Record each plateau metric value for later assertions."""
        self.step_args.append(metric)


class RecordingCosineScheduler:
    """Lightweight scheduler stub that records how often it advances."""

    def __init__(self):
        """Initialize the scheduler call counter."""
        self.step_calls = 0

    def step(self):
        """Record a scheduler advance without touching optimizer state."""
        self.step_calls += 1


class TestTrainingDataclasses(unittest.TestCase):
    """Tests for helper dataclasses defined in the training module."""

    def test_training_history_to_dict_returns_all_fields(self):
        """The history helper should expose metric series and scalar metadata in dictionary form."""
        history = DeconvolutionHistory(
            train_loss=[0.4],
            val_loss=[0.3],
            val_metrics={
                "mae": [0.2],
                "mse": [0.1],
                "kl": [0.05],
                "max_error": [0.3],
                "cosine_sim": [0.8],
            },
            learning_rates=[1e-3],
            best_epoch=2,
            stopped_early=True,
        )

        as_dict = history.to_dict()

        self.assertEqual(
            set(as_dict),
            {
                "train_loss",
                "val_loss",
                "val_mae",
                "val_mse",
                "val_kl",
                "val_max_error",
                "val_cosine_sim",
                "learning_rates",
                "best_epoch",
                "stopped_early",
            },
        )
        self.assertEqual(as_dict["best_epoch"], 2)
        self.assertTrue(as_dict["stopped_early"])


class TestEarlyStopping(unittest.TestCase):
    """Tests for the early stopping utility."""

    def test_min_mode_tracks_improvements_and_stops_after_patience(self):
        """Min-mode early stopping should reset on improvement and stop after repeated regressions."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.05, mode="min")

        self.assertFalse(early_stopping(1.0, epoch=0))
        self.assertEqual(early_stopping.best_score, 1.0)
        self.assertEqual(early_stopping.best_epoch, 0)
        self.assertFalse(early_stopping(0.96, epoch=1))
        self.assertEqual(early_stopping.counter, 1)
        self.assertFalse(early_stopping(0.7, epoch=2))
        self.assertEqual(early_stopping.best_score, 0.7)
        self.assertEqual(early_stopping.counter, 0)
        self.assertFalse(early_stopping(0.68, epoch=3))
        self.assertTrue(early_stopping(0.67, epoch=4))
        self.assertTrue(early_stopping.should_stop)

    def test_max_mode_uses_positive_improvements(self):
        """Max-mode early stopping should stop after patience is exhausted without a large enough gain."""
        early_stopping = EarlyStopping(patience=1, min_delta=0.05, mode="max")

        self.assertFalse(early_stopping(0.4, epoch=0))
        self.assertTrue(early_stopping(0.44, epoch=1))
        self.assertEqual(early_stopping.counter, 1)
        self.assertTrue(early_stopping.should_stop)

    def test_max_mode_resets_counter_after_clear_improvement(self):
        """A sufficiently large score increase should update the best score and clear prior regressions."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.05, mode="max")

        self.assertFalse(early_stopping(0.4, epoch=0))
        self.assertFalse(early_stopping(0.44, epoch=1))
        self.assertEqual(early_stopping.counter, 1)
        self.assertFalse(early_stopping(0.6, epoch=2))
        self.assertEqual(early_stopping.best_score, 0.6)
        self.assertEqual(early_stopping.best_epoch, 2)
        self.assertEqual(early_stopping.counter, 0)


class TestTrainMatrixDeconvolver(unittest.TestCase):
    """Tests for the high-level matrix deconvolver training loop."""

    def setUp(self):
        """Create compact synthetic train and validation fixtures for the training-loop tests."""
        self.X_train = np.array(
            [
                [[0.1, 0.9], [0.8, 0.2]],
                [[0.3, 0.7], [0.6, 0.4]],
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.5, 0.5], [0.4, 0.6]],
            ],
            dtype=np.float32,
        )
        self.y_train = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.3, 0.3, 0.4],
            ],
            dtype=np.float32,
        )
        self.X_val = self.X_train[:2]
        self.y_val = self.y_train[:2]

    def _build_model(self):
        """Create a fresh differentiable model so tests do not share optimizer state."""
        torch.manual_seed(31)
        return TinySoftmaxRegressor(input_dim=4, output_dim=3)

    def test_train_with_plateau_scheduler_saves_best_model_and_stops_early(self):
        """Plateau scheduling should use metric-aware steps, save new best states, and stop early."""
        model = self._build_model()
        scheduler = RecordingPlateauScheduler()
        metrics = [
            {
                "mae": 0.30,
                "mse": 0.21,
                "kl": 0.11,
                "max_error": 0.40,
                "cosine_sim": 0.70,
            },
            {
                "mae": 0.20,
                "mse": 0.18,
                "kl": 0.10,
                "max_error": 0.35,
                "cosine_sim": 0.75,
            },
            {
                "mae": 0.21,
                "mse": 0.19,
                "kl": 0.11,
                "max_error": 0.37,
                "cosine_sim": 0.74,
            },
            {
                "mae": 0.22,
                "mse": 0.20,
                "kl": 0.12,
                "max_error": 0.38,
                "cosine_sim": 0.73,
            },
        ]

        with patch(
            "methyldl.deconvolution.deep_deconvolvers.training.compute_deconvolution_metrics",
            side_effect=metrics,
        ) as mocked_metrics, patch(
            "methyldl.deconvolution.deep_deconvolvers.training.torch.optim.lr_scheduler.ReduceLROnPlateau",
            return_value=scheduler,
        ) as mocked_scheduler_ctor, patch(
            "methyldl.deconvolution.deep_deconvolvers.training.torch.save"
        ) as mocked_save, patch(
            "builtins.print"
        ) as mocked_print:
            trained_model, history = train_matrix_deconvolver(
                model=model,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                n_epochs=4,
                batch_size=2,
                lr=1e-2,
                device="cpu",
                early_stopping_metric="val_mae",
                early_stopping_patience=2,
                scheduler_type="plateau",
                verbose=2,
                save_path="/tmp/deconvolver-best.pt",
            )

        self.assertIs(trained_model, model)
        mocked_scheduler_ctor.assert_called_once()
        self.assertEqual(mocked_metrics.call_count, 4)
        self.assertEqual(scheduler.step_args, [0.30, 0.20, 0.21, 0.22])
        self.assertEqual(mocked_save.call_count, 2)
        self.assertGreaterEqual(mocked_print.call_count, 6)
        self.assertTrue(history.stopped_early)
        self.assertEqual(history.best_epoch, 1)
        self.assertEqual(len(history.train_loss), 4)
        self.assertEqual(len(history.val_loss), 4)
        self.assertEqual(history.val_metrics["mae"], [0.30, 0.20, 0.21, 0.22])

    def test_train_with_cosine_scheduler_uses_max_mode_metric(self):
        """Cosine scheduling should advance once per epoch and maximize cosine similarity when requested."""
        model = self._build_model()
        scheduler = RecordingCosineScheduler()
        metrics = [
            {
                "mae": 0.25,
                "mse": 0.16,
                "kl": 0.08,
                "max_error": 0.33,
                "cosine_sim": 0.55,
            },
            {
                "mae": 0.20,
                "mse": 0.14,
                "kl": 0.07,
                "max_error": 0.30,
                "cosine_sim": 0.72,
            },
        ]

        with patch(
            "methyldl.deconvolution.deep_deconvolvers.training.compute_deconvolution_metrics",
            side_effect=metrics,
        ), patch(
            "methyldl.deconvolution.deep_deconvolvers.training.torch.optim.lr_scheduler.CosineAnnealingLR",
            return_value=scheduler,
        ) as mocked_scheduler_ctor:
            _, history = train_matrix_deconvolver(
                model=model,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                n_epochs=2,
                batch_size=2,
                lr=1e-2,
                device="cpu",
                early_stopping_metric="val_cosine_sim",
                early_stopping_patience=5,
                scheduler_type="cosine",
                verbose=0,
            )

        mocked_scheduler_ctor.assert_called_once()
        self.assertEqual(scheduler.step_calls, 2)
        self.assertFalse(history.stopped_early)
        self.assertEqual(history.best_epoch, 1)
        self.assertEqual(history.val_metrics["cosine_sim"], [0.55, 0.72])

    def test_train_without_scheduler_accepts_custom_loss_weights(self):
        """Disabling the scheduler should still train and record history with custom loss weights."""
        model = self._build_model()
        metrics = [
            {
                "mae": 0.15,
                "mse": 0.09,
                "kl": 0.04,
                "max_error": 0.20,
                "cosine_sim": 0.88,
            }
        ]

        with patch(
            "methyldl.deconvolution.deep_deconvolvers.training.compute_deconvolution_metrics",
            side_effect=metrics,
        ), patch(
            "methyldl.deconvolution.deep_deconvolvers.training.torch.optim.lr_scheduler.ReduceLROnPlateau"
        ) as mocked_plateau_ctor, patch(
            "methyldl.deconvolution.deep_deconvolvers.training.torch.optim.lr_scheduler.CosineAnnealingLR"
        ) as mocked_cosine_ctor:
            trained_model, history = train_matrix_deconvolver(
                model=model,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                n_epochs=1,
                batch_size=2,
                lr=1e-2,
                device="cpu",
                scheduler_type="none",
                loss_weights={"mse": 2.0, "kl": 0.0},
                verbose=0,
            )

        self.assertIs(trained_model, model)
        mocked_plateau_ctor.assert_not_called()
        mocked_cosine_ctor.assert_not_called()
        self.assertFalse(history.stopped_early)
        self.assertEqual(history.best_epoch, 0)
        self.assertEqual(len(history.learning_rates), 1)
        self.assertEqual(history.val_metrics["mse"], [0.09])
