from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from typing import Union,Optional,List
import numpy as np
import torch
import torch.nn as nn


# def compute_deconvolution_metrics(
#     pred: Union[torch.Tensor, np.ndarray],
#     target: Union[torch.Tensor, np.ndarray],
#     eps: float = 1e-8,
# ) -> dict:
#     """Compute all evaluation metrics.

#     Supports both torch.Tensor and numpy.ndarray inputs.
#     NumPy inputs are converted to tensors internally.
#     """
#     if isinstance(pred, np.ndarray):
#         pred = torch.from_numpy(pred)
#     if isinstance(target, np.ndarray):
#         target = torch.from_numpy(target)

#     assert pred.shape == target.shape, "pred and target must have the same shape"

#     with torch.no_grad():
#         mae = (pred - target).abs().mean().item()
#         mse = ((pred - target) ** 2).mean().item()
#         kl = (
#             (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
#             .sum(dim=-1)
#             .mean()
#             .item()
#         )
#         max_error = (pred - target).abs().max().item()
#         cosine_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()

#     return {
#         "mae": mae,
#         "mse": mse,
#         "kl": kl,
#         "max_error": max_error,
#         "cosine_sim": cosine_sim,
#     }


def compute_deconvolution_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
    class_names: Optional[List[str]] = None,
) -> dict:
    """Compute all evaluation metrics.
    Supports both torch.Tensor and numpy.ndarray inputs.
    NumPy inputs are converted to tensors internally.

    Limits of Agreement (LoA) are computed per Bland-Altman:
        bias ± 1.96 * std(differences)
    Overall LoA uses all elements; per-class LoA uses each column.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert pred.shape == target.shape, "pred and target must have the same shape"

    with torch.no_grad():
        diff = pred - target

        mae = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()
        kl = (
            (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
            .sum(dim=-1)
            .mean()
            .item()
        )
        max_error = diff.abs().max().item()
        cosine_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()

        # --- Overall Limits of Agreement (Bland-Altman) ---
        flat_diff = diff.reshape(-1)
        bias = flat_diff.mean().item()
        std = flat_diff.std().item()
        loa_lower = bias - 1.96 * std
        loa_upper = bias + 1.96 * std

        # --- Per-class LoA (column-wise) ---
        # diff shape: (n_samples, n_classes)
        per_class_bias = diff.mean(dim=0)       # (n_classes,)
        per_class_std = diff.std(dim=0)          # (n_classes,)
        per_class_loa_lower = per_class_bias - 1.96 * per_class_std
        per_class_loa_upper = per_class_bias + 1.96 * per_class_std

        # LoA width per class; worst = widest interval
        per_class_loa_width = 2 * 1.96 * per_class_std  # (n_classes,)
        worst_idx = per_class_loa_width.argmax().item()

        if class_names is not None:
            worst_class_name = class_names[worst_idx]
        else:
            worst_class_name = worst_idx

    return {
        "mae": mae,
        "mse": mse,
        "kl": kl,
        "max_error": max_error,
        "cosine_sim": cosine_sim,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "loa_width": loa_upper - loa_lower,
        "worst_class_idx": worst_idx,
        "worst_class_name": worst_class_name,
        "worst_class_loa_lower": per_class_loa_lower[worst_idx].item(),
        "worst_class_loa_upper": per_class_loa_upper[worst_idx].item(),
        "worst_class_loa_width": per_class_loa_width[worst_idx].item(),
        "per_class_loa": {
            "bias": per_class_bias.cpu().numpy(),
            "std": per_class_std.cpu().numpy(),
            "lower": per_class_loa_lower.cpu().numpy(),
            "upper": per_class_loa_upper.cpu().numpy(),
            "width": per_class_loa_width.cpu().numpy(),
        },
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
