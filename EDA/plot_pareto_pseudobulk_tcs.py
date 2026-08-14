"""Pareto plot: pseudobulk rank vs Tissue Concordance Score rank.

Reads `pseudobulk_vs_tcs_ranks.csv` (produced by
`pseudobulk_vs_tcs_rank_correlation.py`) and draws all 376 method
configurations in rank space, with:

  * the non-dominated (Pareto-optimal) methods given individual markers,
  * the four baselines as originally published (uncalibrated), and
  * each baseline's best Syto-calibrated variant, joined to its published
    counterpart by an arrow showing what calibration buys.

Both axes are inverted so that better is up and to the right.

Density contours
----------------
`density_contour` thresholds at 50% of the *fitted density's* mass, not at
50% of a group's points. Because the kernel is wider than Scott's rule and
the grid truncates the tails, the region drawn is larger than half the group.
Measured point coverage, for captions:

    pareto_level_contours     (n = 40-88,  bw 0.70):  60-83%, mean 73%
    pareto_pooling_and_priors (n = 20,     bw 0.70):  70-90%, mean 76%
    pareto_pooling_and_priors (n = 12,     bw 0.95):  75-100%, mean 88%

Describe them as smoothed cores of each group, not as "50% regions".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

EDA = Path(__file__).resolve().parent
FIGDIR = EDA / "figures"

#: Top edge of the legend band, in axes coordinates, for the A4 Pareto plot,
#: and the space reserved below it for the two column headers.
LEGEND_TOP = 1.345
LEGEND_HEADER_GAP = 0.078

#: Left edge of the first legend column, and the shift that puts the marker's
#: left edge on it. A legend centres each handle inside a box `handlelength`
#: ems wide, so the column is nudged left by half that box plus half a marker
#: to align the marker's corner -- not its centre -- with the header text.
LEGEND_X = -0.075
LEGEND_HANDLE_SHIFT = 0.012

#: Clear space between the two legend columns, in axes fractions.
LEGEND_COL_GAP = 0.055

#: Every text size in this module is expressed at its base value and scaled
#: here, so the whole figure set can be resized in one place.
FONT_SCALE = 1.22


def fs(size):
    """Scaled point size."""
    return round(size * FONT_SCALE, 1)

# Palette: categorical slots 1-3 of the validated default palette. Validated
# all-pairs, light mode (worst CVD dE 9.2, worst normal-vision dE 24.0).
C_PARETO = "#2a78d6"  # blue
C_PUBLISHED = "#eb6834"  # orange
C_CALIBRATED = "#1baf7a"  # aqua -- below 3:1 on the surface, so it is
# always direct-labeled (the relief rule).
C_BG = "#c9c8c3"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

#: One marker per Pareto-front method, in front order. Shapes carry identity so
#: the front does not depend on colour alone.
PARETO_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SHORT_CLASSIFIER = {
    "cancerdetector": "CancerDetector",
    "dismir": "Dismir",
    "lookup": "Lookup",
    "methylbert": "MethylBERT",
}
SHORT_SCHEME = {
    "hard_labels": "hard",
    "soft_labels_with_pooling": "SL w/ pool",
    "soft_labels_without_pooling": "SL w/o pool",
    "canonical_soft_labels": "Canonical",
    "": "",
}
SHORT_PRIOR = {"train_freq_prior": "Train-Freq", "uniform_prior": "Uniform", "": ""}
SHORT_CALIB = {
    "uncalibrated": "None",
    "linear_clip_normalize": "Clip",
    "linear_simplex_projection": "Simplex",
    "vector_scaling": "Vector",
}

#: Published-baseline labels that need wrapping to stay clear of the axis.
BASELINE_DISPLAY_WRAPPED = {"Houseman_ineq": "Houseman\nCP"}

#: Display names for the baseline column stems carried in the ranks table.
BASELINE_DISPLAY = {
    "Celfie": "Celfie",
    "EpiDISH": "EpiDISH",
    "Houseman_ineq": "Houseman CP",
    "UXM U25": "UXM U25",
}


def label_syto(row):
    scheme = SHORT_SCHEME[row["Labeling Scheme"]] or SHORT_PRIOR[row["Prior"]]
    return (
        f"{SHORT_CLASSIFIER[row['Classifier']]} · {scheme} · "
        f"{row['Deconvolver'].upper()} · {SHORT_CALIB[row['Calibrator']]}"
    )


def _nondominated_mask(pts):
    return ~np.array(
        [
            (
                (pts[:, 0] <= pts[i, 0])
                & (pts[:, 1] <= pts[i, 1])
                & ((pts[:, 0] < pts[i, 0]) | (pts[:, 1] < pts[i, 1]))
            ).any()
            for i in range(len(pts))
        ]
    )


def pareto_front(df):
    """Rows not dominated on (rank_pseudobulk, rank_tcs); lower rank is better."""
    pts = df[["rank_pseudobulk", "rank_tcs"]].to_numpy()
    return df.iloc[np.flatnonzero(_nondominated_mask(pts))].sort_values(
        "rank_pseudobulk"
    )


def pareto_layers(df):
    """Non-dominated sorting: peel the front, then the front of what remains.

    Returns a list of DataFrames, outermost (best) first.
    """
    pts = df[["rank_pseudobulk", "rank_tcs"]].to_numpy()
    remaining = np.arange(len(pts))
    layers = []
    while len(remaining):
        nd = _nondominated_mask(pts[remaining])
        layers.append(df.iloc[remaining[nd]].sort_values("rank_pseudobulk"))
        remaining = remaining[~nd]
    return layers


def depth_quantile_layers(df, quantiles=(0.25, 0.50, 0.75)):
    """The Pareto layer at each depth quantile.

    The layer returned for `q` is the shallowest one at which the cumulative
    count of methods reaches `q` of the total -- i.e. the staircase beyond
    which the best q of all methods lie. Unlike the strict front, these
    contours are set by many points each, so no single outlier moves them.
    """
    layers = pareto_layers(df)
    cum = np.cumsum([len(x) for x in layers]) / len(df)
    out = []
    for q in quantiles:
        k = int(np.searchsorted(cum, q))
        out.append((q, layers[k], cum[k]))
    return out


def staircase(layer):
    """x, y of the step boundary through a Pareto layer (sorted by rank_pb)."""
    fx = layer["rank_pseudobulk"].to_numpy()
    fy = layer["rank_tcs"].to_numpy()
    return np.repeat(fx, 2)[1:], np.repeat(fy, 2)[:-1]


C_BAD = "#e34948"  # categorical slot 8; >= 3:1 on the surface

#: Configuration levels to highlight, ordered by enrichment in the worst
#: quartile. (factor column, level, panel title).
BAD_LEVELS = [
    ("Classifier", "lookup", "Lookup classifier"),
    ("Labeling Scheme", "hard_labels", "Hard labels"),
    ("Deconvolver", "xgb", "XGBoost deconvolver"),
    ("Feature Scheme", "diagbckg", "Diag + background features"),
    ("Calibrator", "uncalibrated", "No calibration"),
    ("Deconvolver", "mlp", "MLP deconvolver"),
]


def plot_worst_signature(r):
    """Small multiples: where each configuration choice lands in rank space."""
    syto = r[r["Deconvolver"] != ""].copy()
    cut = syto["avg_rank"].quantile(0.75)
    syto["worst"] = syto["avg_rank"] >= cut
    base = syto["worst"].mean()

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.068, right=0.985, top=0.965, bottom=0.075, wspace=0.11, hspace=0.17
    )

    for ax, (factor, level, title) in zip(axes.ravel(), BAD_LEVELS):
        sel = syto[factor] == level
        # worst-quartile band, drawn once per panel as context
        ax.axhspan(0, 0, color="none")
        ax.scatter(
            syto.loc[~sel, "rank_pseudobulk"],
            syto.loc[~sel, "rank_tcs"],
            s=16,
            c=C_BG,
            alpha=0.5,
            linewidths=0,
            zorder=1,
        )
        ax.scatter(
            syto.loc[sel, "rank_pseudobulk"],
            syto.loc[sel, "rank_tcs"],
            s=26,
            c=C_BAD,
            alpha=0.9,
            linewidths=0,
            zorder=2,
        )

        p_worst = syto.loc[sel, "worst"].mean()
        enrich = p_worst / base
        ax.set_title(
            f"{title}   (n = {int(sel.sum())})",
            fontsize=fs(11),
            color=INK,
            loc="left",
            pad=7,
        )
        ax.text(
            0.03,
            0.965,
            f"{p_worst:.0%} of these land in the worst quartile\n"
            f"{enrich:.1f}× the {base:.0%} base rate",
            transform=ax.transAxes,
            fontsize=fs(9.2),
            color=C_BAD if enrich >= 1.5 else INK_2,
            va="top",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=3.5),
            zorder=4,
        )
        ax.set_xlim(390, -14)
        ax.set_ylim(390, -14)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=fs(9))

    fig.text(
        0.5,
        0.018,
        "Pseudobulk rank  (R²)  →  better",
        ha="center",
        fontsize=fs(11),
        color=INK,
    )
    fig.text(
        0.012,
        0.46,
        "Tissue Concordance Score rank  →  better",
        rotation=90,
        va="center",
        fontsize=fs(11),
        color=INK,
    )
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_worst_signature.{ext}", dpi=220)
    print(f"wrote {FIGDIR}/pareto_worst_signature.png / .pdf")


C_BEST = "#2a78d6"  # blue -- best level. Blue/red validates all-pairs
# (CVD dE 21.6, normal-vision 32.6); green/red would not (deutan dE 6.9).

#: Dimensions to contrast. `drop` removes a level from the panel entirely;
#: `not_candidate` keeps its points on screen but bars it from being named
#: best or worst, for levels that share no configuration cells with the rest.
DIMENSIONS = [
    ("Classifier", "Classifier", (), ("cancerdetector",)),
    # CancerDetector carries no labeling scheme, so its blank level is excluded
    # rather than shown as a category.
    ("Labeling Scheme", "Labeling scheme", ("",), ()),
    ("Feature Scheme", "Feature scheme", (), ()),
    ("Deconvolver", "Deconvolver", (), ()),
    ("Calibrator", "Calibrator", (), ()),
]

PRETTY_LEVEL = {
    "cancerdetector": "CancerDetector",
    "dismir": "Dismir",
    "lookup": "Lookup",
    "methylbert": "MethylBERT",
    # Two lines: at one line these run wide enough to collide with their
    # neighbours in the labeling-scheme panel.
    "hard_labels": "Hard Labels",
    "soft_labels_with_pooling": "Data Driven\nSoft Labels, Pooled",
    "soft_labels_without_pooling": "Data Driven\nSoft Labels, Unpooled",
    "canonical_soft_labels": "Canonical\nSoft Labels",
    "top156": "Top-156",
    "diagbckg": "Diag + Background",
    "uncalibrated": "Uncalibrated",
    "linear_clip_normalize": "Clip",
    "linear_simplex_projection": "Simplex",
    "vector_scaling": "Vector",
    "nnls": "NNLS",
    "psls": "PSLS",
    "swn": "SWN",
    "mlp": "MLP",
    "xgb": "XGB",
}


def plot_best_worst_dichotomy(r):
    """Per dimension, the best and the worst level in one rank-space panel."""
    syto = r[r["Deconvolver"] != ""].copy()
    lo, hi = syto["avg_rank"].quantile(0.25), syto["avg_rank"].quantile(0.75)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.068, right=0.985, top=0.965, bottom=0.075, wspace=0.11, hspace=0.17
    )

    for ax, (factor, title, drop, not_candidate) in zip(axes.ravel(), DIMENSIONS):
        pool = syto[~syto[factor].isin(drop)]
        cand = pool[~pool[factor].isin(not_candidate)]
        stats = cand.groupby(factor).agg(
            med=("avg_rank", "median"), n=("avg_rank", "size")
        )
        # Best / worst level by median combined rank; ties break on the share
        # of configurations reaching the best quartile.
        share_best = cand.groupby(factor)["avg_rank"].apply(lambda x: (x <= lo).mean())
        order = stats.assign(tie=-share_best).sort_values(["med", "tie"])
        best_lvl, worst_lvl = order.index[0], order.index[-1]

        ax.scatter(
            pool["rank_pseudobulk"],
            pool["rank_tcs"],
            s=16,
            c=C_BG,
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )
        lines = []
        for lvl, color, quart, qlabel in (
            (best_lvl, C_BEST, lo, "best"),
            (worst_lvl, C_BAD, hi, "worst"),
        ):
            sel = pool[factor] == lvl
            ax.scatter(
                pool.loc[sel, "rank_pseudobulk"],
                pool.loc[sel, "rank_tcs"],
                s=28,
                c=color,
                alpha=0.9,
                linewidths=0,
                zorder=2,
            )
            frac = (
                (pool.loc[sel, "avg_rank"] <= quart).mean()
                if qlabel == "best"
                else (pool.loc[sel, "avg_rank"] >= quart).mean()
            )
            lines.append(
                (
                    color,
                    f"{PRETTY_LEVEL.get(lvl, lvl)}  (n = {int(sel.sum())})\n"
                    f"median rank {stats.loc[lvl, 'med']:.0f} · "
                    f"{frac:.0%} in the {qlabel} quartile",
                )
            )

        for i, (color, text) in enumerate(lines):
            ax.text(
                0.03,
                0.965 - i * 0.135,
                text,
                transform=ax.transAxes,
                fontsize=fs(9),
                color=color,
                va="top",
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=3.0),
                zorder=4,
            )

        ax.set_title(title, fontsize=fs(11.5), color=INK, loc="left", pad=7)
        ax.set_xlim(390, -14)
        ax.set_ylim(390, -14)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=fs(9))

    # The third column has no bottom panel, so sharex would leave it with no
    # tick labels at all -- restore them on the panel that ends that column.
    last_in_column = axes[0, 2]
    last_in_column.tick_params(labelbottom=True)

    # Sixth cell: encoding key and caveats instead of a panel.
    key = axes.ravel()[-1]
    key.set_axis_off()
    key.legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor=C_BEST,
                   markersize=9, label="best level on this dimension"),
            Line2D([], [], marker="o", color="none", markerfacecolor=C_BAD,
                   markersize=9, label="worst level on this dimension"),
            Line2D([], [], marker="o", color="none", markerfacecolor=C_BG,
                   markersize=8, label="every other configuration"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.72),
        frameon=False,
        fontsize=fs(9.6),
        labelspacing=0.8,
    )
    fig.text(0.5, 0.018, "Pseudobulk rank  (R²)  →  better", ha="center",
             fontsize=fs(11), color=INK)
    fig.text(0.016, 0.46, "Tissue Concordance Score rank  →  better", rotation=90,
             va="center", fontsize=fs(11), color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_best_worst_dichotomy.{ext}", dpi=220)
    print(f"wrote {FIGDIR}/pareto_best_worst_dichotomy.png / .pdf")


# Four categorical slots, validated all-pairs in light mode (worst CVD dE 9.2,
# worst normal-vision dE 16.3). Four is the hard cap here -- no five-hue set
# clears the normal-vision floor, which is why Deconvolver merges NNLS/PSLS.
C_LEVELS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

#: Levels merged before contouring, because they are statistically
#: indistinguishable on both axes (within-config-group mean ranks 8.72 vs 9.33;
#: identical worst-quartile rate).
MERGE_LEVELS = {"Deconvolver": {"nnls": "NNLS / PSLS", "psls": "NNLS / PSLS"}}

#: Classifier hues are pinned globally, not derived from rank order within a
#: figure -- colour must follow the entity, or Dismir changes colour between
#: panels the moment another classifier overtakes it.
CLASSIFIER_COLOR = {
    "cancerdetector": C_LEVELS[0],
    "dismir": C_LEVELS[1],
    "methylbert": C_LEVELS[2],
    "lookup": C_LEVELS[3],
}

#: Label offsets, in points, for peaks that land on top of each other.
LABEL_NUDGE = {
    "soft_labels_without_pooling": (0, 16),
    "soft_labels_with_pooling": (0, -16),
    "canonical_soft_labels": (0, -6),
    "hard_labels": (0, -10),
    "cancerdetector": (0, 16),
    "dismir": (0, -16),
    "methylbert": (0, 16),
    "lookup": (0, -16),
    "top156": (0, 15),
    "diagbckg": (0, -15),
    "NNLS / PSLS": (0, 15),
    "swn": (0, -15),
    "mlp": (0, 15),
    "xgb": (0, -15),
    "linear_simplex_projection": (0, 15),
    "vector_scaling": (0, -15),
    "linear_clip_normalize": (0, 15),
    "uncalibrated": (0, -15),
}


def density_contour(ax, x, y, color, mass=0.50, grid=140, bw=0.55, ls="solid", **kw):
    """Draw the smallest KDE region containing `mass` of a level's points.

`mass` is a fraction of the group's observations, not of the fitted
    density -- see the level computation below.

    `bw` is passed to scipy as a scalar, so it becomes ``kde.factor``
    directly rather than scaling Scott's rule. Scott's factor for these group
    sizes (n = 12-120, d = 2) is 0.45-0.66, so bw = 0.70 is roughly 1.1-1.6x
    *wider* than Scott and bw = 0.95 is 1.4-2.1x wider. The oversmoothing is
    deliberate: at these sample sizes Scott's bandwidth resolves clumps of a
    handful of points into separate islands, which is sampling noise rather
    than structure. It only sets the smoothness of the outline; the fraction
    of points enclosed is fixed by `mass` regardless.

    Returns the (x, y) of the density peak, for direct labelling.
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    gx = np.linspace(-14, 390, grid)
    gy = np.linspace(-14, 390, grid)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    # Highest-density region: threshold where the cumulative mass hits `mass`.
    # NOTE: `mass` is a fraction of the *fitted density*, not of the points.
    # The region drawn is larger than a `mass` fraction of the group, for two
    # reasons: the kernel is wider than Scott's rule, and the grid truncates
    # the tails (it holds 76-97% of the total mass), so the absolute-0.50
    # level corresponds to a larger share of the in-grid mass. Measured point
    # coverage is reported in the module docstring; quote that, not `mass`,
    # when describing these contours.
    flat = np.sort(Z.ravel())[::-1]
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    cum = np.cumsum(flat) * cell
    level = flat[np.searchsorted(cum, mass)]

    ax.contourf(GX, GY, Z, levels=[level, Z.max()], colors=[color], alpha=0.09, **kw)
    ax.contour(
        GX, GY, Z, levels=[level], colors=[color], linewidths=2.0,
        linestyles=ls, **kw,
    )
    peak = np.unravel_index(np.argmax(Z), Z.shape)
    return GX[peak], GY[peak]


def _top156_no_canonical(d):
    """Comparable subset: one feature scheme, distribution-derived labels only.

    Diag+background is dropped because only some classifiers were run in it,
    and canonical soft labels because they exist as a contrast to the
    distribution-derived schemes rather than as a candidate configuration.
    """
    return d[
        (d["Feature Scheme"] == "top156")
        & (d["Labeling Scheme"] != "canonical_soft_labels")
    ]


#: (column, panel title, levels to drop, subsetting rule). The feature-scheme
#: panel is the one place diag+background belongs, and hard labels are the only
#: scheme run in both feature sets, so that panel is built on them alone.
LEVEL_PANELS = [
    ("Classifier", "Classifier", (), _top156_no_canonical),
    (
        "Labeling Scheme",
        "Labeling scheme",
        ("",),
        lambda d: d[d["Feature Scheme"] == "top156"],
    ),
    (
        "Feature Scheme",
        "Feature scheme",
        (),
        lambda d: d[d["Labeling Scheme"] == "hard_labels"],
    ),
    ("Deconvolver", "Deconvolver", (), _top156_no_canonical),
    ("Calibrator", "Calibrator", (), _top156_no_canonical),
]


def plot_level_contours(r):
    """Every level of every dimension as overlapping density contours."""
    syto = r[r["Deconvolver"] != ""].copy()

    fig, axes = plt.subplots(3, 2, figsize=(8.6, 11.4), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.115, right=0.985, top=0.955, bottom=0.072, wspace=0.09, hspace=0.155
    )

    for panel_i, (ax, (factor, title, drop, subset)) in enumerate(
        zip(axes.ravel(), LEVEL_PANELS)
    ):
        pool = subset(syto)
        pool = pool[~pool[factor].isin(drop)].copy()
        pool["_lvl"] = pool[factor].replace(MERGE_LEVELS.get(factor, {}))

        ax.scatter(
            pool["rank_pseudobulk"],
            pool["rank_tcs"],
            s=9,
            c=C_BG,
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )

        # Fixed hue order: best median rank takes slot 1, so colour follows the
        # level, never its position in the figure.
        order = pool.groupby("_lvl")["avg_rank"].median().sort_values().index
        for slot, lvl in enumerate(order):
            g = pool[pool["_lvl"] == lvl]
            color = CLASSIFIER_COLOR.get(lvl, C_LEVELS[slot])
            density_contour(
                ax, g["rank_pseudobulk"].to_numpy(), g["rank_tcs"].to_numpy(),
                color, bw=0.70, zorder=2 + slot,
            )
            # Anchor at the group median rather than the KDE peak: once the
            # panels are restricted to comparable subsets the cores overlap
            # heavily, and neighbouring peaks land on top of one another.
            px = float(np.clip(g["rank_pseudobulk"].median(), 88, 320))
            py = float(np.clip(g["rank_tcs"].median(), 30, 350))
            ax.annotate(
                PRETTY_LEVEL.get(lvl, lvl),
                (px, py),
                textcoords="offset points",
                xytext=LABEL_NUDGE.get(lvl, (0, 0)),
                fontsize=fs(9),
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                zorder=10,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.78, pad=1.6),
            )

        # Letter prefixes so captions can address panels individually.
        ax.set_title(
            f"{chr(ord('A') + panel_i)}   {title}",
            fontsize=fs(11.5), color=INK, loc="left", pad=7,
        )
        ax.set_xlim(390, -14)
        ax.set_ylim(390, -14)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=fs(9))

    # Five dimensions in a six-cell grid; the spare cell carries the x labels
    # for its column instead of a panel.
    axes[2, 0].tick_params(labelbottom=True)
    axes[1, 1].tick_params(labelbottom=True)
    axes[2, 1].set_axis_off()

    fig.text(0.5, 0.022, "Pseudobulk rank  (R²)  →  better", ha="center",
             fontsize=fs(11), color=INK)
    fig.text(0.022, 0.53, "Tissue Concordance Score rank  →  better", rotation=90,
             va="center", fontsize=fs(11), color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_level_contours.{ext}", dpi=220)
    print(f"wrote {FIGDIR}/pareto_level_contours.png / .pdf")


SOFT_SCHEMES = (
    "soft_labels_with_pooling",
    "soft_labels_without_pooling",
    "canonical_soft_labels",
)


def plot_soft_classifier_contours(r):
    """Classifier contours with hard labels excluded, and a matched control.

    Hard labels are the worst scheme by a wide margin and are not evenly
    distributed across classifiers, so the all-scheme classifier comparison
    partly measures the schemes each classifier happened to be run in. The
    right panel removes that: only the two schemes all three classifiers
    share, at a single feature set.
    """
    syto = r[r["Deconvolver"] != ""].copy()

    panels = [
        (
            syto[syto["Labeling Scheme"].isin(SOFT_SCHEMES)],
            "All soft labeling schemes",
            "canonical, pooled and unpooled · both feature sets",
        ),
        (
            syto[
                syto["Labeling Scheme"].isin(SOFT_SCHEMES[:2])
                & (syto["Feature Scheme"] == "top156")
            ],
            "Matched subset",
            "pooled + unpooled only · top-156 only · 40 configs each",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.082, right=0.99, top=0.93, bottom=0.135, wspace=0.09
    )

    for ax, (pool, title, subtitle) in zip(axes, panels):
        ax.scatter(
            pool["rank_pseudobulk"], pool["rank_tcs"],
            s=13, c=C_BG, alpha=0.55, linewidths=0, zorder=1,
        )
        order = pool.groupby("Classifier")["avg_rank"].median().sort_values().index
        for slot, lvl in enumerate(order):
            g = pool[pool["Classifier"] == lvl]
            color = CLASSIFIER_COLOR[lvl]
            density_contour(
                ax, g["rank_pseudobulk"].to_numpy(), g["rank_tcs"].to_numpy(),
                color, zorder=2 + slot,
            )
            # Anchor each contour at its median, not its density peak: with
            # 40 points a KDE peak wanders, the median does not.
            mx = g["rank_pseudobulk"].median()
            my = g["rank_tcs"].median()
            # Above the labels: where two medians nearly coincide a label
            # box would otherwise bury the marker it belongs to.
            ax.scatter([mx], [my], s=120, marker="X", facecolor=color,
                       edgecolor=SURFACE, linewidths=1.8, zorder=12)
            # Stagger above/below and hold the anchor clear of the panel
            # edge: in the matched panel two medians nearly coincide, and a
            # label sitting on the midpoint buries the other marker.
            ax.annotate(
                f"{SHORT_CLASSIFIER[lvl]}  (n = {len(g)})",
                (float(np.clip(mx, 88, 318)), my),
                textcoords="offset points",
                xytext=(0, 16) if slot % 2 == 0 else (0, -27),
                fontsize=fs(9.4), fontweight="bold", color=color,
                ha="center", zorder=10,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1.8),
            )

        ax.set_title(title, fontsize=fs(11.5), color=INK, loc="left", pad=7)
        ax.set_xlim(390, -14)
        ax.set_ylim(390, -14)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=fs(9))

    fig.text(0.5, 0.028, "Pseudobulk rank  (R²)  →  better", ha="center",
             fontsize=fs(11), color=INK)
    fig.text(0.016, 0.53, "Tissue Concordance Score rank  →  better", rotation=90,
             va="center", fontsize=fs(11), color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_soft_classifiers.{ext}", dpi=220)
    print(f"wrote {FIGDIR}/pareto_soft_classifiers.png / .pdf")


#: Deconvolvers kept in the restricted row. MLP and XGB are dropped: both are
#: enriched in the worst quartile and neither reaches the Pareto front except
#: through CancerDetector.
STRONG_DECONVOLVERS = ("nnls", "psls", "swn")


def plot_pooling_and_priors(r):
    """Pooled vs unpooled soft labels, and CancerDetector's two priors.

    Canonical soft labels are excluded throughout -- they exist as a contrast
    to the distribution-derived schemes, not as a candidate configuration.

    Top row uses every deconvolver; bottom row keeps only NNLS, PSLS and SWN.
    """
    syto = r[r["Deconvolver"] != ""].copy()

    def columns(pool):
        return [
            (
                pool[
                    (pool["Labeling Scheme"] == "soft_labels_with_pooling")
                    & (pool["Feature Scheme"] == "top156")
                ],
                "Classifier",
                "Data Driven Soft Labels, Pooled",
            ),
            (
                pool[
                    (pool["Labeling Scheme"] == "soft_labels_without_pooling")
                    & (pool["Feature Scheme"] == "top156")
                ],
                "Classifier",
                "Data Driven Soft Labels, without Pooling",
            ),
            (
                pool[pool["Classifier"] == "cancerdetector"],
                "Prior",
                "CancerDetector, by prior",
            ),
        ]

    # Deconvolver set varies down the columns, condition down the rows.
    deconv_sets = [
        (syto, "All five deconvolvers"),
        (
            syto[syto["Deconvolver"].isin(STRONG_DECONVOLVERS)],
            "NNLS, PSLS and SWN only",
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8.6, 11.4), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.135, right=0.985, top=0.932, bottom=0.062, wspace=0.09, hspace=0.165
    )

    for col_i, (pool, col_label) in enumerate(deconv_sets):
        for row_i, (sub, factor, title) in enumerate(columns(pool)):
            ax = axes[row_i, col_i]
            ax.scatter(
                sub["rank_pseudobulk"], sub["rank_tcs"],
                s=20, c=C_BG, alpha=0.75, linewidths=0, zorder=1,
            )
            order = sub.groupby(factor)["avg_rank"].median().sort_values().index
            for slot, lvl in enumerate(order):
                g = sub[sub[factor] == lvl]
                if factor == "Prior":
                    # Both priors are the same model, so they share the
                    # CancerDetector hue and separate on line style instead.
                    color = CLASSIFIER_COLOR["cancerdetector"]
                    ls = "solid" if lvl == "train_freq_prior" else "dashed"
                    name = SHORT_PRIOR[lvl]
                else:
                    color, ls = CLASSIFIER_COLOR[lvl], "solid"
                    name = SHORT_CLASSIFIER[lvl]
                # Widen the kernel when a group is small: 12 points cannot
                # support the bandwidth that 20 or more can.
                density_contour(
                    ax, g["rank_pseudobulk"].to_numpy(), g["rank_tcs"].to_numpy(),
                    color, bw=0.70 if len(g) >= 20 else 0.95, ls=ls,
                    zorder=2 + slot,
                )
                mx, my = g["rank_pseudobulk"].median(), g["rank_tcs"].median()
                lx = float(np.clip(mx, 55, 345))
                ax.scatter([mx], [my], s=110, marker="X", facecolor=color,
                           edgecolor=SURFACE, linewidths=1.7, zorder=12)
                ax.annotate(
                    f"{name}  {g['avg_rank'].median():.0f}",
                    (lx, my),
                    textcoords="offset points",
                    xytext=(0, 15 if slot % 2 == 0 else -25),
                    fontsize=fs(9.2), fontweight="bold", color=color,
                    ha="center", zorder=10,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1.8),
                )

            # Each row is one lettered panel; its title sits above the left
            # column and runs across the row's empty title space.
            if col_i == 0:
                ax.set_title(
                    f"{chr(ord('A') + row_i)}   {title}",
                    fontsize=fs(11.5), color=INK, loc="left", pad=8,
                )
            ax.text(
                0.975, 0.03, f"n = {len(sub) // len(order)} each",
                transform=ax.transAxes, fontsize=fs(8.6), color=INK_2,
                ha="right", va="bottom",
            )
            ax.set_xlim(390, -14)
            ax.set_ylim(390, -14)
            ax.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=fs(9))

        pos = axes[0, col_i].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2, 0.972, col_label,
            va="bottom", ha="center",
            fontsize=fs(10.5), color=INK, fontweight="bold",
        )

    fig.text(0.5, 0.018, "Pseudobulk rank  (R²)  →  better", ha="center",
             fontsize=fs(11), color=INK)
    fig.text(0.022, 0.5, "Tissue Concordance Score rank  →  better", rotation=90,
             va="center", fontsize=fs(11), color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_pooling_and_priors.{ext}", dpi=220)
    print(f"wrote {FIGDIR}/pareto_pooling_and_priors.png / .pdf")


def main():
    FIGDIR.mkdir(exist_ok=True)
    r = pd.read_csv(EDA / "pseudobulk_vs_tcs_ranks.csv", keep_default_na=False)

    syto = r[r["Deconvolver"] != ""]
    published = r[(r["Deconvolver"] == "") & (r["Calibrator"] == "uncalibrated")]
    calibrated = r[(r["Deconvolver"] == "") & (r["Calibrator"] != "uncalibrated")]
    # Best calibrated variant per baseline, by average of the two ranks.
    best_cal = calibrated.loc[calibrated.groupby("Classifier")["avg_rank"].idxmin()]

    front = pareto_front(r)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    # A4 portrait minus ~2 cm margins. The legends sit in a band above the
    # axes rather than beside them, so the plot keeps the full page width.
    fig, ax = plt.subplots(figsize=(7.2, 9.3))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.745, bottom=0.062)

    # --- background: every method configuration ---------------------------
    ax.scatter(
        r["rank_pseudobulk"],
        r["rank_tcs"],
        s=26,
        c=C_BG,
        alpha=0.55,
        linewidths=0,
        zorder=1,
        label="_nolegend_",
    )

    # --- depth contours: best 25% / 50% / 75% ------------------------------
    # Depth is a magnitude, so the three contours are one hue stepped
    # light -> dark, with the strict front (below) darkest of all.
    contour_handles = []
    contour_styles = [
        ("#5b9be0", (0, (7, 2.5)), 1.5),
        ("#8fbcea", (0, (3.5, 2.5)), 1.4),
        ("#bcd7f2", (0, (1.5, 2.2)), 1.4),
    ]
    for (q, layer, _), (color, dash, lw) in zip(
        depth_quantile_layers(r), contour_styles
    ):
        cx, cy = staircase(layer)
        ax.plot(cx, cy, color=color, lw=lw, ls=dash, zorder=2)
        # Inline label at the right-hand (best-pseudobulk) end of the contour.
        tip = layer.loc[layer["rank_pseudobulk"].idxmin()]
        ax.annotate(
            f"{q:.0%}",
            (tip["rank_pseudobulk"], tip["rank_tcs"]),
            textcoords="offset points",
            xytext=(-6, -13),
            fontsize=fs(9),
            color=color,
            fontweight="bold",
            ha="center",
            zorder=6,
        )
        contour_handles.append(
            Line2D([], [], color=color, lw=lw, ls=dash, label=f"best {q:.0%} beyond")
        )

    # --- Pareto staircase --------------------------------------------------
    step_x, step_y = staircase(front)
    ax.plot(step_x, step_y, color=C_PARETO, lw=1.8, alpha=0.6, zorder=3)

    # --- baselines: published -> best calibrated ---------------------------
    for _, pub in published.iterrows():
        cal = best_cal[best_cal["Classifier"] == pub["Classifier"]]
        if cal.empty:
            continue
        cal = cal.iloc[0]
        ax.annotate(
            "",
            xy=(cal["rank_pseudobulk"], cal["rank_tcs"]),
            xytext=(pub["rank_pseudobulk"], pub["rank_tcs"]),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.22,head_length=0.5",
                color=C_CALIBRATED,
                lw=1.5,
                alpha=0.85,
                shrinkA=7,
                shrinkB=7,
            ),
            zorder=3,
        )

    for df, color in ((published, C_PUBLISHED), (best_cal, C_CALIBRATED)):
        ax.scatter(
            df["rank_pseudobulk"],
            df["rank_tcs"],
            s=110,
            facecolor=color,
            edgecolor=SURFACE,
            linewidths=2,
            zorder=5,
            label="_nolegend_",
        )

    # Direct labels on every baseline point (relief rule for the aqua slot,
    # and identity for the orange one).
    pub_offsets = {
        "Celfie": (10, -15),
        "Houseman_ineq": (-10, -15),
        "UXM U25": (9, -15),
        "EpiDISH": (-10, -15),
    }
    for _, row in published.iterrows():
        dx, dy = pub_offsets.get(row["Classifier"], (8, 8))
        ax.annotate(
            BASELINE_DISPLAY_WRAPPED.get(
                row["Classifier"],
                BASELINE_DISPLAY.get(row["Classifier"], row["Classifier"]),
            ),
            (row["rank_pseudobulk"], row["rank_tcs"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=fs(9),
            color=INK,
            fontweight="bold",
            ha="right" if dx < 0 else "left",
            zorder=6,
        )
    cal_offsets = {
        "Celfie": (10, 8),
        "Houseman_ineq": (10, -14),
        "UXM U25": (10, -14),
        "EpiDISH": (10, -14),
    }
    for _, row in best_cal.iterrows():
        dx, dy = cal_offsets.get(row["Classifier"], (8, 8))
        ax.annotate(
            f"{BASELINE_DISPLAY.get(row['Classifier'], row['Classifier'])}"
            f" + {SHORT_CALIB[row['Calibrator']]}",
            (row["rank_pseudobulk"], row["rank_tcs"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=fs(9),
            color=INK_2,
            ha="right" if dx < 0 else "left",
            zorder=6,
        )

    # --- Pareto-front methods, one marker each -----------------------------
    front_handles = []
    for marker, (_, row) in zip(PARETO_MARKERS, front.iterrows()):
        is_baseline = row["Deconvolver"] == ""
        name = row["Classifier"] if is_baseline else label_syto(row)
        ax.scatter(
            row["rank_pseudobulk"],
            row["rank_tcs"],
            s=190 if marker == "*" else 130,
            marker=marker,
            facecolor=C_PARETO,
            edgecolor=SURFACE,
            linewidths=1.8,
            zorder=7,
        )
        front_handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                color="none",
                markerfacecolor=C_PARETO,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                markersize=11 if marker == "*" else 9,
                label=f"{name}  ({row['rank_pseudobulk']:.0f},{row['rank_tcs']:.0f})",
            )
        )

    # --- axes --------------------------------------------------------------
    ax.set_xlim(390, -14)
    ax.set_ylim(390, -14)
    ax.set_xlabel("Pseudobulk rank  (R²)  →  better", fontsize=fs(10), color=INK)
    ax.set_ylabel("Tissue Concordance Score rank  →  better", fontsize=fs(10), color=INK)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=fs(9))

    # --- legends -----------------------------------------------------------
    class_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=C_BG,
            markersize=7,
            label=f"all methods (n = {len(r)})",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=C_PUBLISHED,
            markeredgecolor=SURFACE,
            markersize=9,
            label="baseline, as published",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=C_CALIBRATED,
            markeredgecolor=SURFACE,
            markersize=9,
            label="baseline + Syto calibration",
        ),
        Line2D([], [], color=C_PARETO, lw=1.8, alpha=0.6, label="Pareto frontier"),
    ] + contour_handles
    leg1 = ax.legend(
        handles=front_handles,
        # Both columns hang from a fixed top edge, below a shared header band.
        loc="upper left",
        bbox_to_anchor=(
            LEGEND_X - LEGEND_HANDLE_SHIFT,
            LEGEND_TOP - LEGEND_HEADER_GAP,
        ),
        frameon=False,
        fontsize=fs(7.9),
        labelspacing=0.42,
        handletextpad=0.7,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    # Second column: the field, the baselines and the depth contours --
    # everything that is not a point on the frontier.
    leg2 = ax.legend(
        handles=class_handles,
        loc="upper left",
        bbox_to_anchor=(0.475, LEGEND_TOP - LEGEND_HEADER_GAP),
        frameon=False,
        fontsize=fs(7.9),
        labelspacing=0.42,
        handletextpad=0.7,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    leg2.get_title().set_color(INK)

    # Park the second column immediately right of the first and level with
    # it, so both headers sit on one line. The frontier labels vary in width
    # and the columns hold different numbers of entries, so measure the first
    # legend rather than guessing an offset.
    fig.canvas.draw()
    leg1_box = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    col2_x = leg1_box.x1 + LEGEND_COL_GAP
    leg2.set_bbox_to_anchor(
        (col2_x, LEGEND_TOP - LEGEND_HEADER_GAP), transform=ax.transAxes
    )

    # Headers drawn as plain text at one shared y. A legend's own title is
    # laid out inside its box, so a one-line and a two-line title do not end
    # up on the same line however the boxes are anchored.
    for x, header in (
        (LEGEND_X, "Pareto-optimal methods\n(pseudobulk rank, TCS rank)"),
        (col2_x + LEGEND_HANDLE_SHIFT, "Reference points\nand contours"),
    ):
        ax.text(
            x, LEGEND_TOP, header,
            transform=ax.transAxes, fontsize=fs(9), color=INK,
            va="top", ha="left",
        )
    leg2._legend_box.align = "left"

    # Park the second column immediately right of the first and level with
    # it, so both headers sit on one line. The frontier labels vary in width
    # and the columns hold different numbers of entries, so measure the first
    # legend rather than guessing an offset.
    fig.canvas.draw()
    leg1_box = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    col2_x = leg1_box.x1 + LEGEND_COL_GAP
    leg2.set_bbox_to_anchor(
        (col2_x, LEGEND_TOP - LEGEND_HEADER_GAP), transform=ax.transAxes
    )

    # Headers drawn as plain text at one shared y. A legend's own title is
    # laid out inside its box, so a one-line and a two-line title do not end
    # up on the same line however the boxes are anchored.
    for x, header in (
        (LEGEND_X, "Pareto-optimal methods\n(pseudobulk rank, TCS rank)"),
        (col2_x + LEGEND_HANDLE_SHIFT, "Reference points\nand contours"),
    ):
        ax.text(
            x, LEGEND_TOP, header,
            transform=ax.transAxes, fontsize=fs(9), color=INK,
            va="top", ha="left",
        )
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_pseudobulk_vs_tcs.{ext}", dpi=220)
    print(f"wrote {FIGDIR}/pareto_pseudobulk_vs_tcs.png / .pdf")
    print(f"\nPareto front ({len(front)} methods):")
    cols = ["Classifier", "Labeling Scheme", "Prior", "Feature Scheme",
            "Deconvolver", "Calibrator", "rank_pseudobulk", "rank_tcs"]
    print(front[cols].to_string(index=False))
    print("\nBest calibrated variant per baseline:")
    print(best_cal[["Classifier", "Calibrator", "rank_pseudobulk", "rank_tcs"]].to_string(index=False))

    print("\nDepth contours:")
    for q, layer, reached in depth_quantile_layers(r):
        print(f"  q={q:.0%}: Pareto layer of {len(layer):2d} methods, cumulative {reached:.1%}")

    plot_worst_signature(r)
    plot_best_worst_dichotomy(r)
    plot_level_contours(r)
    plot_soft_classifier_contours(r)
    plot_pooling_and_priors(r)


if __name__ == "__main__":
    main()
