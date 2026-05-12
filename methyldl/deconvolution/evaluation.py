from typing import Union, Optional, List

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
    n_resamples=10000,
    confidence_level=0.95,
    random_state=42,
    compute_ci: bool = False,
    return_per_sample: bool = False,
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
    assert (
        pred.shape == target.shape
    ), f"pred and target must have the same shape but got {pred.shape=} vs {target.shape=}"
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    with torch.inference_mode():
        diff = pred - target

        if compute_ci:
            metrics = compute_metrics_with_bootstrap_bca_ci(
                pred.numpy(force=True),
                target.numpy(force=True),
                n_resamples,
                confidence_level,
                random_state,
                eps,
                return_per_sample=return_per_sample,
            )
        else:
            metrics = {}
            metrics["mae"] = diff.abs().mean().item()
            metrics["mse"] = (diff**2).mean().item()
            kl = (
                (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
                .sum(dim=-1)
                .mean()
                .item()
            )
            metrics["kl"] = kl
            metrics["r2"] = r2_score(
                target.numpy(force=True),
                pred.numpy(force=True),
                multioutput="variance_weighted",
            )

            if return_per_sample:
                metrics["mse_per_sample"] = (diff**2).mean(dim=-1).numpy(force=True)
                metrics["mae_per_sample"] = diff.abs().mean(dim=-1).numpy(force=True)
                metrics["kl_per_sample"] = (
                    (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
                    .sum(dim=-1)
                    .numpy(force=True)
                )

        max_error = diff.abs().max().item()
        metrics["max_error"] = max_error
        # pylint: disable-next=not-callable
        cosine_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
        metrics["cosine_sim"] = cosine_sim

        # --- Overall Limits of Agreement (Bland-Altman) ---
        flat_diff = diff.reshape(-1)
        bias = flat_diff.mean().item()
        std = flat_diff.std().item()
        loa_lower = bias - 1.96 * std
        loa_upper = bias + 1.96 * std

        # --- Per-class LoA (column-wise) ---
        # diff shape: (n_samples, n_classes)
        per_class_bias = diff.mean(dim=0)  # (n_classes,)
        per_class_std = diff.std(dim=0)  # (n_classes,)
        per_class_loa_lower = per_class_bias - 1.96 * per_class_std
        per_class_loa_upper = per_class_bias + 1.96 * per_class_std

        # LoA width per class; worst = widest interval
        per_class_loa_width = 2 * 1.96 * per_class_std  # (n_classes,)
        worst_idx = per_class_loa_width.argmax().item()

        if class_names is not None:
            worst_class_name = class_names[worst_idx]
        else:
            worst_class_name = worst_idx

    metrics.update(
        {
            "loa_lower": loa_lower,
            "loa_upper": loa_upper,
            "loa_width": loa_upper - loa_lower,
            "worst_class_idx": worst_idx,
            "worst_class_name": worst_class_name,
            "worst_class_loa_lower": per_class_loa_lower[worst_idx].item(),
            "worst_class_loa_upper": per_class_loa_upper[worst_idx].item(),
            "worst_class_loa_width": per_class_loa_width[worst_idx].item(),
            "per_class_loa": {
                "bias": per_class_bias.numpy(force=True),
                "std": per_class_std.numpy(force=True),
                "lower": per_class_loa_lower.numpy(force=True),
                "upper": per_class_loa_upper.numpy(force=True),
                "width": per_class_loa_width.numpy(force=True),
            },
        }
    )

    return metrics


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


# CONFIDENCE INTERVALS USING BOOTSTRAP


def r2_variance_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute variance-weighted R² in a vectorized way.
    (same as sklearn's multioutput="variance_weighted").

    Args:
        y_true: (n_samples, n_celltypes) array of true proportions
        y_pred: (n_samples, n_celltypes) array of predicted proportions
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return 1.0 - ss_res / ss_tot


def bca_ci_r2(
    preds: np.ndarray,
    targets: np.ndarray,
    n_resamples=10_000,
    confidence_level=0.95,
    random_state=42,
):
    """BCa bootstrap Confidence Interval for R2 (variance weighted).

    This custom implementation is optimized for R2
    and avoids memory issues stemming from scipy bootstrap.

    Args:
        preds: (n_samples, n_celltypes) array of predicted proportions
        targets: (n_samples, n_celltypes) array of true proportions
        n_resamples: number of bootstrap resamples to perform
        confidence_level: desired confidence level for the interval (e.g., 0.95)
        random_state: seed for reproducibility

    Returns:
        r2_hat: the R² computed on the original dataset
        ci_lower: lower bound of the confidence interval
        ci_upper: upper bound of the confidence interval
    """
    rng = np.random.default_rng(random_state)
    n = len(preds)

    # pre compute per-sample ss_res and ss_tot for the full dataset
    target_avg_over_ctypes = targets.mean(axis=0)  # (n_ctypes, )
    ss_tot_per_sample = np.sum(
        (targets - target_avg_over_ctypes) ** 2, axis=1
    )  # (n_samples,)
    ss_res_per_sample = np.sum((targets - preds) ** 2, axis=1)  # (n_samples,)
    ss_res_sum = np.sum(ss_res_per_sample)
    ss_tot_sum = np.sum(ss_tot_per_sample)

    # Bootstrap distribution (full n-out-of-n resamples)
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_stats[i] = 1 - np.sum(ss_res_per_sample[indices]) / np.sum(
            ss_tot_per_sample[indices]
        )

    # Bias correction (z0)
    r2_hat = 1 - ss_res_sum / ss_tot_sum
    z0 = ndtri(np.mean(boot_stats < r2_hat))

    # Acceleration (a) — analytical jackknife for r2

    jack_stats = 1 - (ss_res_sum - ss_res_per_sample) / (
        ss_tot_sum - ss_tot_per_sample
    )  # 1-D array, O(n) memory
    jack_mean = np.mean(jack_stats)
    diff = jack_mean - jack_stats
    a = np.sum(diff**3) / (6.0 * np.sum(diff**2) ** 1.5)

    # Adjusted percentiles
    alpha = (1 - confidence_level) / 2
    z_lo, z_hi = ndtri(alpha), ndtri(1 - alpha)
    q_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    q_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    ci_lower, ci_upper = np.percentile(boot_stats, [q_lo * 100, q_hi * 100])
    return r2_hat, ci_lower, ci_upper


def bca_ci_mean(data, n_resamples=10_000, confidence_level=0.95, random_state=42):
    """BCa bootstrap Confidence Interval for np.mean on the full dataset.

    The jackknife is computed analytically for the mean:
        jack_i = (sum(data) - data[i]) / (n - 1)

    Args:
        data: 1-D array of data points (for example MSE, MAE or KL values across samples)
        n_resamples: number of bootstrap resamples to perform
        confidence_level: desired confidence level for the interval (e.g., 0.95)
        random_state: seed for reproducibility

    Returns:
        theta_hat: the mean computed on the original dataset
        ci_lower: lower bound of the confidence interval
        ci_upper: upper bound of the confidence interval
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    theta_hat = np.mean(data)

    # Bootstrap distribution (full n-out-of-n resamples)
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        boot_stats[i] = np.mean(data[rng.integers(0, n, size=n)])

    # Bias correction (z0)
    z0 = ndtri(np.mean(boot_stats < theta_hat))

    # Acceleration (a) — analytical jackknife for the mean
    total = np.sum(data)
    jack_stats = (total - data) / (n - 1)  # 1-D array, O(n) memory
    jack_mean = np.mean(jack_stats)
    diff = jack_mean - jack_stats
    a = np.sum(diff**3) / (6.0 * np.sum(diff**2) ** 1.5)

    # Adjusted percentiles
    alpha = (1 - confidence_level) / 2
    z_lo, z_hi = ndtri(alpha), ndtri(1 - alpha)
    q_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    q_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    ci_lower, ci_upper = np.percentile(boot_stats, [q_lo * 100, q_hi * 100])
    return theta_hat, ci_lower, ci_upper


def compute_metrics_with_bootstrap_bca_ci(
    pred: np.ndarray,
    target: np.ndarray,
    n_resamples=10000,
    confidence_level=0.95,
    random_state=42,
    eps: float = 1e-8,
    return_per_sample: bool = False,
) -> dict:
    """
    Compute MSE, MAE, KL and R2 along with their BCa bootstrap CIs for R2 and mean MSE.

    Args:
        pred: (n_samples, n_celltypes) array of predicted proportions
        target: (n_samples, n_celltypes) array of true proportions
        n_resamples: number of bootstrap resamples to perform
        confidence_level: desired confidence level for the interval (e.g., 0.95)
        random_state: seed for reproducibility
        eps: small constant to avoid log(0) in KL divergence
    """
    # MSE
    mse_per_sample = ((pred - target) ** 2).mean(axis=1)
    mse, mse_ci_lower, mse_ci_upper = bca_ci_mean(
        mse_per_sample, n_resamples, confidence_level, random_state
    )

    # MAE
    mae_per_sample = (np.abs(pred - target)).mean(axis=1)
    mae, mae_ci_lower, mae_ci_upper = bca_ci_mean(
        mae_per_sample, n_resamples, confidence_level, random_state
    )

    # KL
    kl_per_sample = np.sum(
        target
        * (np.log(np.clip(target, eps, None)) - np.log(np.clip(pred, eps, None))),
        axis=1,
    )
    kl, kl_ci_lower, kl_ci_upper = bca_ci_mean(
        kl_per_sample, n_resamples, confidence_level, random_state
    )

    # R2
    r2, r2_ci_lower, r2_ci_upper = bca_ci_r2(
        pred, target, n_resamples, confidence_level, random_state
    )

    result = {
        "mse": mse,
        "mse_ci_lower": mse_ci_lower,
        "mse_ci_upper": mse_ci_upper,
        "mae": mae,
        "mae_ci_lower": mae_ci_lower,
        "mae_ci_upper": mae_ci_upper,
        "kl": kl,
        "kl_ci_lower": kl_ci_lower,
        "kl_ci_upper": kl_ci_upper,
        "r2": r2,
        "r2_ci_lower": r2_ci_lower,
        "r2_ci_upper": r2_ci_upper,
    }

    if return_per_sample:
        result["mse_per_sample"] = mse_per_sample
        result["mae_per_sample"] = mae_per_sample
        result["kl_per_sample"] = kl_per_sample

    return result
