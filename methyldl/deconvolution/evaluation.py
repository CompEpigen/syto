from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def compute_deconvolution_metrics_np(
    pred: np.ndarray, target: np.ndarray, eps: float = 1e-8
) -> Dict[str, float]:
    """
    Compute all evaluation metrics (NumPy version).

    Matches the PyTorch `compute_deconvolution_metrics` function.
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"

    # Ensure 2D
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
        target = target[np.newaxis, :]

    # MAE
    mae = np.abs(pred - target).mean()

    # MSE
    mse = ((pred - target) ** 2).mean()

    # KL Divergence: sum over classes, mean over samples
    target_safe = np.clip(target, eps, None)
    pred_safe = np.clip(pred, eps, None)
    kl = (target_safe * (np.log(target_safe) - np.log(pred_safe))).sum(axis=-1).mean()

    # Max error (worst case across all samples and classes)
    max_error = np.abs(pred - target).max()

    # Cosine similarity (average across batch)
    dot_product = (pred * target).sum(axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cosine_sim = (dot_product / (pred_norm * target_norm + eps)).mean()

    return {
        "mae": float(mae),
        "mse": float(mse),
        "kl": float(kl),
        "max_error": float(max_error),
        "cosine_sim": float(cosine_sim),
    }


def compute_deconvolution_metrics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> dict:
    """Compute all evaluation metrics."""
    assert pred.shape == target.shape, "pred and target must have the same shape"

    with torch.no_grad():
        # MAE
        mae = (pred - target).abs().mean().item()

        # MSE
        mse = ((pred - target) ** 2).mean().item()

        # KL Divergence
        kl = (
            (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
            .sum(dim=-1)
            .mean()
            .item()
        )

        # Max error (worst case)
        max_error = (pred - target).abs().max().item()

        # Cosine similarity (average across batch)
        cosine_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()

    return {
        "mae": mae,
        "mse": mse,
        "kl": kl,
        "max_error": max_error,
        "cosine_sim": cosine_sim,
    }


def compute_combined_loss(
    pred: np.ndarray,
    target: np.ndarray,
    mse_weight: float = 1.0,
    kl_weight: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute combined MSE + KL loss matching PyTorch version."""
    assert pred.shape == target.shape, "pred and target must have the same shape"

    mse = ((pred - target) ** 2).mean()
    target_safe = np.clip(target, eps, None)
    pred_safe = np.clip(pred, eps, None)
    kl = (target_safe * (np.log(target_safe) - np.log(pred_safe))).sum(axis=-1).mean()
    return mse_weight * mse + kl_weight * kl
