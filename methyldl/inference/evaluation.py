from typing import Dict

import torch
import torch.nn as nn
import numpy as np


def compute_deconvolution_metrics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> dict:
    """Compute all evaluation metrics."""
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


def print_deconvolution_metrics_summary(metrics: dict):
    """Print a formatted summary of the metrics."""
    print("=" * 60)
    print("DECONVOLUTION RESULTS SUMMARY")
    print("=" * 60)

    if "celltype_metrics" in metrics:
        ct_metrics = metrics["celltype_metrics"]
        print(f"\n📊 Overall R²: {ct_metrics['overall_r2']:.4f}")

        print("\n📋 Per Cell Type R²:")
        print("-" * 40)

        # Sort by R² descending
        sorted_ct = sorted(
            ct_metrics["per_celltype_r2"].items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -1,
            reverse=True,
        )

        for ct_name, r2 in sorted_ct:
            if not np.isnan(r2):
                bar = "█" * int(r2 * 20) + "░" * (20 - int(r2 * 20))
                print(f"  {ct_name:20s} │ {bar} │ {r2:.4f}")
            else:
                print(f"  {ct_name:20s} │ {'N/A':^20s} │ N/A")

    if "complexity_metrics" in metrics:
        cx_metrics = metrics["complexity_metrics"]

        print("\n📈 Performance by Mixture Complexity:")
        print("-" * 50)
        print(f"  {'# Cell Types':^15s} │ {'N Samples':^10s} │ {'R²':^10s}")
        print("-" * 50)

        for complexity in sorted(cx_metrics["per_complexity_r2"].keys()):
            r2 = cx_metrics["per_complexity_r2"][complexity]
            n_samples = cx_metrics["per_complexity_n_samples"][complexity]

            if not np.isnan(r2):
                print(f"  {complexity:^15d} │ {n_samples:^10d} │ {r2:^10.4f}")
            else:
                print(f"  {complexity:^15d} │ {n_samples:^10d} │ {'N/A':^10s}")

    print("\n" + "=" * 60)