"""matplotlib rendering of the deconvolution ReportModel to a multipage PDF.

Page A — report: consensus bar (± range) + agreement heatmap + method divergence.
Page B — focus: top-N cell types, per-method clustered bars with consensus ticks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .deconvolution_data import ReportModel

_GREY_SHADES = ["#4d4d4d", "#6b6b6b", "#8a8a8a", "#a6a6a6", "#c2c2c2"]


def _method_colors(model: ReportModel) -> Dict[str, tuple]:
    """Baseline → greys; syto → hue per deconvolver (tab10)."""
    palette = plt.get_cmap("tab10")
    syto_deconv: list[str] = []
    for m in model.methods:
        if not m.is_baseline and m.deconvolver not in syto_deconv:
            syto_deconv.append(m.deconvolver)

    colors: Dict[str, tuple] = {}
    gi = 0
    for m in model.methods:
        if m.is_baseline:
            colors[m.key] = _GREY_SHADES[gi % len(_GREY_SHADES)]
            gi += 1
        else:
            colors[m.key] = palette(syto_deconv.index(m.deconvolver) % 10)
    return colors


def _n_baseline(model: ReportModel) -> int:
    return sum(1 for m in model.methods if m.is_baseline)


def _page_report(model: ReportModel, colors: Dict[str, tuple], pdf: PdfPages) -> None:
    n_ct = model.n_cell_types
    n_m = len(model.methods)
    fig = plt.figure(figsize=(14, max(7.0, n_ct * 0.3)))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 2.2], height_ratios=[max(4, n_ct), max(3, n_m * 0.5)],
        hspace=0.25, wspace=0.06,
    )
    y = np.arange(n_ct)

    # consensus horizontal bar + min/max whiskers (row 0 = highest consensus at top)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.barh(y, model.consensus, color="#3b6ea5", height=0.7)
    ax0.errorbar(
        model.consensus, y,
        xerr=[model.consensus - model.spread_min, model.spread_max - model.consensus],
        fmt="none", ecolor="#b03a2e", elinewidth=1.0, capsize=2.0,
    )
    ax0.set_yticks(y)
    ax0.set_yticklabels(model.cell_types, fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xlabel("Proportion", fontsize=8)
    ax0.set_title("Consensus (median) ± range", fontsize=9)
    if model.known_truth in model.cell_types:
        ti = model.cell_types.index(model.known_truth)
        ax0.get_yticklabels()[ti].set_color("goldenrod")
        ax0.get_yticklabels()[ti].set_fontweight("bold")

    # agreement heatmap: rows = cell types (same order), cols = methods
    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(model.matrix, aspect="auto", cmap="viridis", origin="upper")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(n_m))
    ax1.set_xticklabels([m.key for m in model.methods], rotation=90, fontsize=6)
    for tick, m in zip(ax1.get_xticklabels(), model.methods):
        tick.set_color("grey" if m.is_baseline else "teal")
    split = _n_baseline(model)
    if 0 < split < n_m:
        ax1.axvline(split - 0.5, color="black", linewidth=1.2)
    ax1.set_title("Per-method proportions (grey = baseline, teal = syto)", fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.01)

    # method divergence bar (full width)
    ax2 = fig.add_subplot(gs[1, :])
    xs = np.arange(n_m)
    ax2.bar(xs, [m.divergence for m in model.methods],
            color=[colors[m.key] for m in model.methods])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([m.key for m in model.methods], rotation=90, fontsize=6)
    ax2.set_ylabel("L1 divergence\nfrom consensus", fontsize=8)
    ax2.set_title("Method disagreement (higher = more of an outlier)", fontsize=9)

    fig.suptitle(f"Deconvolution report — {model.file_name}", fontsize=12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_focus(model: ReportModel, colors: Dict[str, tuple], pdf: PdfPages) -> None:
    top = model.top_n
    cts = model.cell_types[:top]
    methods = model.methods
    n_m = len(methods)
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(top)
    width = 0.8 / max(1, n_m)
    for j, m in enumerate(methods):
        vals = model.matrix[:top, j]
        ax.bar(x + j * width - 0.4 + width / 2.0, vals, width=width,
               color=colors[m.key], label=m.key)
    ax.plot(x, model.consensus[:top], "k_", markersize=18, markeredgewidth=2,
            label="consensus")
    if model.known_truth in cts:
        ti = cts.index(model.known_truth)
        ax.axvspan(ti - 0.5, ti + 0.5, color="gold", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion")
    ax.set_title(f"Top {top} cell types — per-method comparison")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_pdf(model: ReportModel, out_path) -> Path:
    out_path = Path(out_path)
    colors = _method_colors(model)
    with PdfPages(out_path) as pdf:
        _page_report(model, colors, pdf)
        _page_focus(model, colors, pdf)
    return out_path
