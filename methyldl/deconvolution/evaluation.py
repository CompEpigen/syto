from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from typing import Union
import numpy as np
import torch
import torch.nn as nn

def compute_deconvolution_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
) -> dict:
    """Compute all evaluation metrics.

    Supports both torch.Tensor and numpy.ndarray inputs.
    NumPy inputs are converted to tensors internally.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    assert pred.shape == target.shape, "pred and target must have the same shape"

    with torch.no_grad():
        mae = (pred - target).abs().mean().item()
        mse = ((pred - target) ** 2).mean().item()
        kl = (
            (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
            .sum(dim=-1)
            .mean()
            .item()
        )
        max_error = (pred - target).abs().max().item()
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
