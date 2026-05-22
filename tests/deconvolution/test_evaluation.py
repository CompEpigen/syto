import unittest

import numpy as np
import torch
from sklearn.metrics import r2_score

from syto.deconvolution.evaluation import (
    compute_combined_loss,
    compute_deconvolution_metrics,
)


def _manual_numpy_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8):
    """Compute expected metric values independently for test assertions."""
    pred_2d = pred[np.newaxis, :] if pred.ndim == 1 else pred
    target_2d = target[np.newaxis, :] if target.ndim == 1 else target

    mae = np.abs(pred_2d - target_2d).mean()
    mse = ((pred_2d - target_2d) ** 2).mean()
    target_safe = np.clip(target_2d, eps, None)
    pred_safe = np.clip(pred_2d, eps, None)
    kl = (target_safe * (np.log(target_safe) - np.log(pred_safe))).sum(axis=-1).mean()
    max_error = np.abs(pred_2d - target_2d).max()
    dot_product = (pred_2d * target_2d).sum(axis=-1)
    pred_norm = np.linalg.norm(pred_2d, axis=-1)
    target_norm = np.linalg.norm(target_2d, axis=-1)
    cosine_sim = (dot_product / (pred_norm * target_norm + eps)).mean()

    overall_r2 = r2_score(target_2d, pred_2d, multioutput="variance_weighted")

    return {
        "mae": float(mae),
        "mse": float(mse),
        "kl": float(kl),
        "max_error": float(max_error),
        "cosine_sim": float(cosine_sim),
        "r2": float(overall_r2),
    }


def _manual_combined_loss(
    pred: np.ndarray,
    target: np.ndarray,
    mse_weight: float = 1.0,
    kl_weight: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute the combined loss formula expected from the NumPy helper."""
    metrics = _manual_numpy_metrics(pred=pred, target=target, eps=eps)
    return mse_weight * metrics["mse"] + kl_weight * metrics["kl"]


class TestComputeDeconvolutionMetricsNp(unittest.TestCase):
    """Tests for the NumPy deconvolution metric helper."""

    def test_returns_expected_metrics_for_two_dimensional_inputs(self):
        """The NumPy helper should match independently computed batch metrics."""
        pred = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
        target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        # This fixture intentionally contains zeros in the target so the KL path
        # exercises the epsilon-based clipping instead of relying on log(0).
        result = compute_deconvolution_metrics(pred=pred, target=target)
        expected = _manual_numpy_metrics(pred=pred, target=target)

        for key in (
            "mae",
            "mse",
            "kl",
            "max_error",
            "cosine_sim",
            "r2",
            "loa_lower",
            "loa_upper",
            "loa_width",
            "worst_class_idx",
            "worst_class_name",
            "worst_class_loa_lower",
            "worst_class_loa_upper",
            "worst_class_loa_width",
            "per_class_loa",
        ):
            self.assertIn(key, result)
        for key, expected_value in expected.items():
            self.assertAlmostEqual(result[key], expected_value, places=5)

    def test_promotes_one_dimensional_inputs_to_single_sample_batch(self):
        """One-dimensional vectors should be evaluated as a batch of size one."""
        pred = np.array([0.25, 0.75], dtype=float)
        target = np.array([0.0, 1.0], dtype=float)

        # The implementation adds a leading batch axis for 1D inputs, so this test
        # validates that branch without forcing callers to reshape inputs manually.
        result = compute_deconvolution_metrics(pred=pred, target=target)
        expected = _manual_numpy_metrics(pred=pred, target=target)

        for key, expected_value in expected.items():
            if np.isnan(expected_value):
                self.assertTrue(np.isnan(result[key]), msg=f"{key} should be nan")
            else:
                self.assertAlmostEqual(result[key], expected_value, places=5)

    def test_raises_assertion_error_for_shape_mismatch(self):
        """Mismatched NumPy shapes should fail fast with an assertion error."""
        pred = np.array([0.2, 0.8], dtype=float)
        target = np.array([[0.2, 0.8]], dtype=float)

        with self.assertRaisesRegex(AssertionError, "same shape"):
            compute_deconvolution_metrics(pred=pred, target=target)


class TestComputeDeconvolutionMetricsTorch(unittest.TestCase):
    """Tests for the PyTorch deconvolution metric helper."""

    def test_returns_expected_metrics_for_tensor_inputs(self):
        """The PyTorch helper should agree with the reference NumPy equations."""
        pred = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

        result = compute_deconvolution_metrics(pred=pred, target=target)
        expected = _manual_numpy_metrics(pred=pred.numpy(), target=target.numpy())

        for key, expected_value in expected.items():
            self.assertIsInstance(result[key], float)
            self.assertAlmostEqual(result[key], expected_value, places=6)

    def test_accepts_tensors_that_require_grad_and_returns_plain_floats(self):
        """Metric computation should be inference-only even for grad-enabled tensors."""
        pred = torch.tensor([0.25, 0.75], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)

        result = compute_deconvolution_metrics(pred=pred, target=target)

        for key, value in result.items():
            if key in ("per_class_loa", "worst_class_idx", "worst_class_name"):
                continue
            self.assertIsInstance(value, float, msg=f"{key} is not float")

    def test_raises_assertion_error_for_shape_mismatch(self):
        """Mismatched tensor shapes should fail before metric computation starts."""
        pred = torch.tensor([0.2, 0.8], dtype=torch.float32)
        target = torch.tensor([[0.2, 0.8]], dtype=torch.float32)

        with self.assertRaisesRegex(AssertionError, "same shape"):
            compute_deconvolution_metrics(pred=pred, target=target)


class TestComputeCombinedLoss(unittest.TestCase):
    """Tests for the combined MSE-plus-KL loss helper."""

    def test_returns_expected_weighted_sum(self):
        """The combined loss should match the weighted reference computation."""
        pred = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
        target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        result = compute_combined_loss(
            pred=pred,
            target=target,
            mse_weight=1.5,
            kl_weight=0.25,
        )
        expected = _manual_combined_loss(
            pred=pred,
            target=target,
            mse_weight=1.5,
            kl_weight=0.25,
        )

        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected, places=7)

    def test_handles_zero_entries_via_epsilon_clipping(self):
        """Zeros in the targets or predictions should still produce a finite loss."""
        pred = np.array([[0.0, 1.0], [0.3, 0.7]], dtype=float)
        target = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

        result = compute_combined_loss(pred=pred, target=target, eps=1e-6)

        self.assertTrue(np.isfinite(result))

    def test_raises_assertion_error_for_shape_mismatch(self):
        """Mismatched NumPy shapes should be rejected for the loss helper too."""
        pred = np.array([[0.2, 0.8]], dtype=float)
        target = np.array([0.2, 0.8], dtype=float)

        with self.assertRaisesRegex(AssertionError, "same shape"):
            compute_combined_loss(pred=pred, target=target)
