# ---------------------------------------------------------------- setup
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

CSV = "EDA/cfsort_results_v2/OtherBaselines/calibrated_all_baselines.csv"
REF_JSON = "EDA/cfsort_full_map_dict_v2.json"
OUT = "EDA/cfsort_results_v2/OtherBaselines/figs"

METHODS = ["Celfie", "EpiDISH", "Houseman_ineq", "UXM U25"]
METHOD_LABEL = {"Celfie": "CelFiE", "EpiDISH": "EpiDISH",
                "Houseman_ineq": "Houseman Ineq.", "UXM U25": "UXM U25"}
STAGES = ["none", "clip0", "simplex", "vector"]          # fixed order
STAGE_LABEL = {"none": "Uncalibrated", "clip0": "Clip",
               "simplex": "Simplex", "vector": "Vector"}
TCS = "Average Tissue Concordance Score"

# One distinct hue + marker per calibration stage. Validated all-pairs on a
# light surface (worst normal-vision dE 20.8); green<->red sits in the 6-8 CVD
# band, which is why each stage also gets its own marker shape.
STAGE_COLOR = dict(zip(STAGES, ["#e34948", "#eda100", "#008300", "#4a3aa7"]))
STAGE_MARKER = dict(zip(STAGES, ["o", "s", "^", "D"]))

# reference cell-type names that differ from the prediction vocabulary
REF_RENAME = {"Endothelium": "Endothel", "Ovary+Endom-Ep": "Ovary-Ep"}

LOYFER_COLORS = {
    "Blood-T": "#4e79a7",
    "Blood-B": "#59a14f",
    "Blood-Granul": "#e15759",
    "Blood-Mono+Macro": "#f28e2b",
    "Blood-NK": "#76b7b2",
    "Eryth-prog": "#b07aa1",
    "Endothelium": "#ff9da7",
    "Liver-Hep": "#9c755f",
    "Heart-Cardio": "#bab0ac",
    "Heart-Fibro": "#d4a373",
    "Kidney-Ep": "#edc948",
    "Lung-Ep-Alveo": "#aec7e8",
    "Lung-Ep-Bron": "#98d8c8",
    "Pancreas-Acinar": "#c7b198",
    "Pancreas-Duct": "#d4c5a9",
    "Pancreas-Alpha": "#e8d5b7",
    "Pancreas-Beta": "#c9ada7",
    "Pancreas-Delta": "#a8a39d",
    "Bladder-Ep": "#b5c99a",
    "Prostate-Ep": "#87bba2",
    "Small-Int-Ep": "#c49792",
    "Colon-Ep": "#a7c4bc",
    "Colon-Fibro": "#8fbc8f",
    "Dermal-Fibro": "#dda0dd",
    "Epid-Kerat": "#f0c987",
    "Breast-Basal-Ep": "#e6b0aa",
    "Breast-Luminal-Ep": "#d5a6bd",
    "Adipocytes": "#ffe0b2",
    "Head-Neck-Ep": "#bcaaa4",
    "Ovary+Endom-Ep": "#ce93d8",
    "Fallopian-Ep": "#9fa8da",
    "Smooth-Musc": "#a1887f",
    "Skeletal-Musc": "#90a4ae",
    "Thyroid-Ep": "#80cbc4",
    "Neuron": "#fff59d",
    "Oligodend": "#c5e1a5",
    "Ovary-Ep": "#ce93d8",
    "unmapped": "#3a3a3a",
}
# the atlas spells two types differently from the prediction columns
LOYFER_COLORS.update({new: LOYFER_COLORS[old] for old, new in REF_RENAME.items()
                      if old in LOYFER_COLORS})
# still not in LOYFER_COLORS: Bone-Osteob, Gallbladder, Gastric-Ep
UNCOLORED = "#d9d8d3"


def loyfer_color(cell_type):
    return LOYFER_COLORS.get(cell_type, UNCOLORED)

# categorical slots 1,2,3,7 of the reference palette: validated all-pairs on a
# light surface (worst CVD dE 9.2, worst normal-vision dE 16.3). Light-surface
# only -- violet/blue collide if re-used on a dark background.
SERIES = {"Celfie": "#2a78d6", "EpiDISH": "#eb6834",
          "Houseman_ineq": "#1baf7a", "UXM U25": "#4a3aa7"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                      "#f4a3a2", "#e34948", "#8f1f1e"])

SAVE_FORMATS = ("pdf", "png")          # pdf for LaTeX, png to eyeball

mpl.rcParams.update({
    # TrueType (42) rather than Type-3 -- keeps text selectable/searchable in the
    # PDF and is what most journal pipelines require
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    # crop hard: the LaTeX float supplies its own surrounding space
    "savefig.pad_inches": 0.01,
    "figure.constrained_layout.w_pad": 0.01, "figure.constrained_layout.h_pad": 0.01,
    "figure.constrained_layout.wspace": 0.01,
    "figure.constrained_layout.hspace": 0.01,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK3, "axes.linewidth": 0.6, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.facecolor": "white",
})


def col(method, stage):
    """Column name for a (method, calibration stage) pair."""
    return method if stage == "none" else f"{method} {stage}"


# Figures go into the document at a common rendered width, so a point size only
# says how big the text *looks* once you also know how far the figure was
# shrunk. PAGE_W is the figure width that renders at 1x; PAGE_FS are the sizes
# in that frame, and _fs() converts them to the points a wider figure needs to
# come out the same size on the page.
PAGE_W = 9.6
PAGE_FS = {"tick": 11.25, "label": 11.25, "title": 12.5,
           "legend": 10.0, "small": 9.4}


def _fs(figw):
    """On-page font sizes, in points, for a figure `figw` inches wide."""
    k = figw / PAGE_W
    return {name: round(size * k, 1) for name, size in PAGE_FS.items()}


TISSUE_EXPECTED = {
    "skin": ["Blood-T", "Dermal-Fibro", "Blood-Mono+Macro", "Blood-Granul",
             "Endothel", "Smooth-Musc"],
    "salivary gland": ["Head-Neck-Ep", "Blood-T", "Blood-B", "Dermal-Fibro",
                       "Blood-Mono+Macro", "Endothel", "Epid-Kerat", "Smooth-Musc"],
    "liver": ["Liver-Hep", "Blood-Mono+Macro", "Blood-T", "Endothel",
              "Eryth-prog", "Blood-B", "Blood-Granul", "Blood-NK"],
    "pancreas": ["Pancreas-Acinar", "Pancreas-Duct", "Endothel",
                 "Blood-Mono+Macro", "Blood-T", "Dermal-Fibro"],
    "small intestine": ["Small-Int-Ep", "Blood-T", "Blood-B", "Dermal-Fibro",
                        "Blood-Mono+Macro"],
    "stomach": ["Blood-Mono+Macro", "Blood-T", "Dermal-Fibro", "Blood-B",
                "Blood-Granul", "Gastric-Ep", "Endothel"],
    "lung": ["Blood-Mono+Macro", "Lung-Ep-Alveo", "Endothel", "Blood-T",
             "Lung-Ep-Bron", "Epid-Kerat", "Blood-Granul", "Blood-NK", "Blood-B"],
    "adipose tissue": ["Dermal-Fibro", "Blood-T", "Blood-Mono+Macro", "Blood-B",
                       "Endothel", "Adipocytes", "Blood-Granul", "Smooth-Musc"],
    "bladder": ["Bladder-Ep", "Dermal-Fibro", "Blood-Mono+Macro", "Blood-T",
                "Blood-Granul", "Blood-B", "Smooth-Musc", "Endothel"],
    "kidney": ["Kidney-Ep", "Blood-T", "Blood-B", "Blood-Mono+Macro", "Blood-NK"],
    "ovary": ["Ovary-Ep", "Smooth-Musc", "Endothel"],
    "prostate": ["Prostate-Ep", "Blood-T", "Endothel", "Blood-Mono+Macro",
                 "Dermal-Fibro", "Smooth-Musc", "Blood-Granul"],
    "uterus": ["Dermal-Fibro", "Endothel", "Smooth-Musc", "Blood-T", "Ovary-Ep",
               "Blood-Mono+Macro", "Fallopian-Ep"],
    "spleen": ["Blood-B", "Blood-T", "Blood-Granul", "Blood-Mono+Macro",
               "Blood-NK", "Endothel"],
    "blood vessel": ["Dermal-Fibro", "Smooth-Musc", "Blood-Mono+Macro",
                     "Endothel", "Blood-T", "Blood-Granul"],
    "heart": ["Endothel", "Heart-Cardio", "Heart-Fibro", "Smooth-Musc",
              "Blood-Mono+Macro", "Blood-T"],
}
MAPPING_RATE = {
    "skin": 0.9655, "salivary gland": 0.8725, "liver": 0.9463, "pancreas": 0.9488,
    "small intestine": 0.976, "stomach": 0.9926, "lung": 0.9569,
    "adipose tissue": 0.9617, "bladder": 0.9308, "kidney": 0.9892, "ovary": 0.9381,
    "prostate": 0.9886, "uterus": 0.9321, "spleen": 0.9897, "blood vessel": 0.9457,
    "heart": 0.8971,
}


def load(csv=CSV):
    """Long frame restricted to tissues with an expected-cell-type mapping."""
    df = pd.read_csv(csv)
    df = df[df["Biosample term name"].isin(TISSUE_EXPECTED)].copy()
    df = df.rename(columns={"Biosample term name": "tissue"})
    df["on_target"] = [ct in TISSUE_EXPECTED[t]
                       for ct, t in zip(df["CellType"], df["tissue"])]
    return df


def load_reference(path=REF_JSON):
    """Per tissue: dominant reference cell type and its reference fraction.

    Fractions come from `loyfer_counts_strict` (Tabula Sapiens counts mapped onto
    the Loyfer atlas vocabulary), normalised over all counts including
    `unmapped` -- which reproduces the json's own `biggest_proportion_strict`.
    """
    raw = json.load(open(path))
    out = {}
    for tissue, rec in raw.items():
        counts = rec.get("loyfer_counts_strict")
        if not counts:
            continue
        total = sum(counts.values())
        frac = {REF_RENAME.get(k, k): v / total
                for k, v in counts.items() if k != "unmapped"}
        dom = max(frac, key=frac.get)
        out[tissue] = {"dominant": dom, "ref_frac": frac[dom], "all": frac}
    return out


def per_sample(df):
    """(file, tissue, method, stage) -> TCS, off-target mass, zero fraction."""
    rows = []
    for m in METHODS:
        for s in STAGES:
            c = col(m, s)
            g = df.groupby(["file", "tissue", "on_target"])[c].sum().unstack("on_target")
            z = df.groupby(["file", "tissue"])[c].apply(lambda v: float((v == 0).mean()))
            top1 = df.groupby(["file", "tissue"])[c].max()
            n_off = (df[~df["on_target"]].groupby("file")[c]
                     .apply(lambda v: int((v > 0.05).sum())))
            part = pd.DataFrame({"tcs": g[True], "off": g[False],
                                 "zero_frac": z, "max_frac": top1}).reset_index()
            part["n_off_gt5"] = part["file"].map(n_off).astype(int)
            part["method"], part["stage"] = m, s
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def dominant_fractions(df, ref):
    """(tissue, method, stage) -> mean predicted fraction of the tissue's
    dominant reference cell type, plus that cell type's reference fraction."""
    rows = []
    for tissue, rec in ref.items():
        sub = df[(df["tissue"] == tissue) & (df["CellType"] == rec["dominant"])]
        if sub.empty:
            continue
        for m in METHODS:
            for s in STAGES:
                rows.append(dict(tissue=tissue, method=m, stage=s,
                                 cell_type=rec["dominant"],
                                 ref_frac=rec["ref_frac"],
                                 pred=sub[col(m, s)].mean()))
    return pd.DataFrame(rows)


def _save(fig, path):
    """A path with a suffix is honoured as-is; a bare stem writes every format
    in SAVE_FORMATS (so LaTeX gets the pdf and you get a png to look at)."""
    from pathlib import Path
    p = Path(path)
    if p.suffix:
        fig.savefig(p)
    else:
        for ext in SAVE_FORMATS:
            fig.savefig(p.with_suffix(f".{ext}"))


def _seg_dist(pt, a, b):
    """Distance from pt to segment a-b, all in display coordinates."""
    ab = b - a
    n2 = float(ab @ ab)
    t = 0.0 if n2 == 0 else float(np.clip((pt - a) @ ab / n2, 0, 1))
    return float(np.linalg.norm(pt - (a + t * ab)))


def _perp_labels(ax, xs, ys, labels, avoid=(), pad=None, fontsize=7, color=None,
                 anchors=None):
    """Label each vertex of a path just off the path, perpendicular to it.

    Offsets are computed in display space (after a draw, so the axes geometry is
    final), and for each vertex the side with more clearance wins -- measured
    against this path *and* any `avoid` paths (the ghosted context lines), which
    is what keeps 'simplex' off the arrow it belongs to and off its neighbours.

    `anchors`, if given, is one entry per label: an (dx, dy) offset in points
    that is used verbatim, or None to place that label automatically.
    """
    pad = fontsize if pad is None else pad     # clearance tracks the text height
    p = ax.transData.transform(np.c_[xs, ys])
    segs = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
    for axs, ays in avoid:
        q = ax.transData.transform(np.c_[axs, ays])
        segs += [(q[i], q[i + 1]) for i in range(len(q) - 1)]
    bbox = ax.get_window_extent()
    placed = []                                    # anchors of labels already set
    for i, lab in enumerate(labels):
        prev = p[i] - p[i - 1] if i > 0 else None
        nxt = p[i + 1] - p[i] if i < len(p) - 1 else None
        t = sum(v / (np.linalg.norm(v) or 1) for v in (prev, nxt) if v is not None)
        if np.linalg.norm(t) < 1e-9:
            t = np.array([1.0, 0.0])
        t = t / np.linalg.norm(t)
        n = np.array([t[1], -t[0]])
        scale = ax.figure.dpi / 72                      # points -> display px
        # rough text extent, enough to tell whether a side runs off the panel
        w = 0.55 * fontsize * len(lab) * scale
        h = 1.2 * fontsize * scale
        manual = None if anchors is None else anchors[i]
        if manual is not None:
            best_off = np.asarray(manual, float)
            best = best_off / (np.linalg.norm(best_off) or 1.0)
        else:
            best, best_off, best_score = None, None, -1e9
            for side in (n, -n):
                for mult in (1.0, 1.4):                 # nudge out only if needed
                    cand = p[i] + side * pad * mult * scale
                    clear = min(_seg_dist(cand, a, b) for a, b in segs)
                    # labels are wide, demand more horizontal than vertical room
                    for q in placed:
                        d = cand - q
                        clear = min(clear, float(np.hypot(d[0] / 2.5, d[1])))
                    # the text grows away from the anchor in the direction it is
                    # offset, so that corner is the one that can leave the panel
                    sx, sy = side
                    x1 = cand[0] + np.sign(sx) * w if abs(sx) > 0.15 else cand[0]
                    y1 = cand[1] + np.sign(sy) * h if abs(sy) > 0.15 else cand[1]
                    spill = (max(0.0, bbox.x0 - min(cand[0], x1))
                             + max(0.0, max(cand[0], x1) - bbox.x1)
                             + max(0.0, bbox.y0 - min(cand[1], y1))
                             + max(0.0, max(cand[1], y1) - bbox.y1))
                    # stay close: only a real collision justifies moving out
                    score = clear - 14.0 * (mult - 1.0) - 0.6 * spill
                    if score > best_score:
                        best, best_off, best_score = side, side * pad * mult, score
        ha = "left" if best[0] > 0.15 else ("right" if best[0] < -0.15 else "center")
        va = "bottom" if best[1] > 0.15 else ("top" if best[1] < -0.15 else "center")
        ax.annotate(lab, (xs[i], ys[i]), xytext=best_off, ha=ha, va=va,
                    textcoords="offset points", fontsize=fontsize,
                    color=color or INK2)
        placed.append(p[i] + best_off * scale)


def _spread(ys, min_sep):
    """Nudge label positions apart, preserving order, keeping the mean fixed."""
    ys = np.asarray(ys, float)
    idx = np.argsort(ys)
    out = ys[idx].copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + min_sep)
    out -= out.mean() - ys.mean()
    res = np.empty_like(out)
    res[idx] = out
    return res


def _end_labels(ax, x, ys, labels, colors, fontsize=8.5, pad=14):
    """Right-edge direct labels in ink, de-overlapped, with a series-colored
    leader so identity is carried by the mark and not by the text color."""
    lo, hi = ax.get_ylim()
    moved = _spread(ys, 0.052 * (hi - lo))
    for y0, y1, lab, c in zip(ys, moved, labels, colors):
        if abs(y1 - y0) > 1e-9:            # a zero-length leader divides by zero
            ax.annotate("", (x, y0), (x, y1), xycoords="data", textcoords="data",
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=c, lw=1.0,
                                        shrinkA=0, shrinkB=2,
                                        connectionstyle="angle,angleA=0,angleB=90,"
                                                        "rad=3"))
        ax.annotate(lab, (x, y1), xytext=(pad, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=fontsize,
                    annotation_clip=False)


def _boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    means = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return np.percentile(means, [2.5, 97.5])


# ------------------------------------------------- fig 1: on-target slopegraph
def fig_tcs_slopegraph(ps, metric="tcs", ylabel=TCS, path=None):
    """Left: absolute level per stage. Right: paired change vs uncalibrated.

    The stages are evaluated on the *same* samples, so the informative
    uncertainty is on the within-sample difference -- an across-sample CI on the
    absolute level is dominated by tissue heterogeneity and hides the effect.
    """
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                  gridspec_kw=dict(width_ratios=[1.05, 1]))
    x = np.arange(len(STAGES))
    wide = ps.pivot_table(index=["file", "method"], columns="stage", values=metric)
    ends, ends_d = [], []

    for m in METHODS:
        sub = ps[ps.method == m]
        mu = [sub.loc[sub.stage == s, metric].mean() for s in STAGES]
        ax.plot(x, mu, color=SERIES[m], lw=2, marker="o", ms=5,
                mec="white", mew=1.2, zorder=3, clip_on=False)
        ends.append((f"{METHOD_LABEL[m]}  {mu[-1]:.3f}", mu[-1]))

        w = wide.xs(m, level="method")
        d = [w[s] - w["none"] for s in STAGES[1:]]
        mud = [v.mean() for v in d]
        lo, hi = zip(*[_boot_ci(v) for v in d])
        xs = np.arange(1, len(STAGES))
        axd.errorbar(xs, mud, yerr=[np.array(mud) - lo, np.array(hi) - np.array(mud)],
                     color=SERIES[m], lw=2, marker="o", ms=5, mec="white", mew=1.2,
                     capsize=3, elinewidth=1.2, zorder=3)
        ends_d.append((f"{METHOD_LABEL[m]}  +{mud[-1]:.3f}", mud[-1]))

    ax.set_xticks(x, [STAGE_LABEL[s] for s in STAGES])
    ax.set_xlim(-0.15, len(STAGES) - 1)
    ax.set_ylabel(ylabel)

    axd.axhline(0, color=INK3, lw=0.8, zorder=1)
    axd.set_xticks(np.arange(1, len(STAGES)), [STAGE_LABEL[s] for s in STAGES[1:]])
    axd.set_xlim(0.85, len(STAGES) - 1)
    axd.set_ylabel("paired Δ TCS vs uncalibrated  (95% bootstrap CI)")

    for a in (ax, axd):
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.margins(y=0.10)
    _end_labels(ax, x[-1], [y for _, y in ends], [t for t, _ in ends],
                [SERIES[m] for m in METHODS])
    _end_labels(axd, len(STAGES) - 1, [y for _, y in ends_d],
                [t for t, _ in ends_d], [SERIES[m] for m in METHODS])
    fig.subplots_adjust(left=0.07, right=0.83, wspace=0.90)
    if path:
        _save(fig, path)
    return fig


# ------------------------------------- fig 2: where the off-target mass goes
def fig_leakage_destinations(df, top_n=8, path=None):
    figw = 10.4                          # widened to pay for the bigger legend
    fs = _fs(figw)
    off = df[~df["on_target"]]
    n_samples = df["file"].nunique()
    rank = (off[[col(m, s) for m in METHODS for s in STAGES]].sum(1)
            .groupby(off["CellType"]).sum().sort_values(ascending=False))
    top = list(rank.index[:top_n])
    print(f"top-{top_n} recipients hold "
          f"{rank.iloc[:top_n].sum() / rank.sum():.0%} of all off-target mass")
    missing = [c for c in top if c not in LOYFER_COLORS]
    if missing:
        print(f"no LOYFER_COLORS entry, drawn in {UNCOLORED}: {missing}")
    fill = {c: loyfer_color(c) for c in top}
    fill["Other"] = "#c9c8c2"

    fig, axes = plt.subplots(1, len(METHODS), figsize=(figw, 3.6),
                             sharey=True, constrained_layout=True)
    for ax, m in zip(axes, METHODS):
        bottoms = np.zeros(len(STAGES))
        for ct in top + ["Other"]:
            sel = off["CellType"].isin(top) if ct == "Other" else off["CellType"] == ct
            sel = ~sel if ct == "Other" else sel
            vals = np.array([off.loc[sel, col(m, s)].sum() / n_samples for s in STAGES])
            ax.bar(np.arange(len(STAGES)), vals, bottom=bottoms, width=0.62,
                   color=fill[ct], lw=1.4, edgecolor="white", zorder=2,
                   hatch="///" if ct in missing else None)
            bottoms += vals
        for i, tot in enumerate(bottoms):
            ax.annotate(f"{tot:.2f}", (i, tot), ha="center", va="bottom",
                        fontsize=fs["small"], color=INK2, xytext=(0, 2),
                        textcoords="offset points")
        ax.set_xticks(np.arange(len(STAGES)),
                      [STAGE_LABEL[s] for s in STAGES], rotation=45, ha="right")
        ax.set_title(METHOD_LABEL[m], fontsize=fs["title"])
        ax.tick_params(labelsize=fs["tick"])
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
    axes[0].set_ylabel("mean off-target mass", fontsize=fs["label"])
    handles = [Patch(facecolor=fill[c], edgecolor="white",
                     hatch="///" if c in missing else None,
                     label=c + (" (no color)" if c in missing else ""))
               for c in top + ["Other"]]
    fig.legend(handles=handles, loc="outside right upper", fontsize=fs["legend"],
               title="leakage recipient", title_fontsize=fs["legend"],
               alignment="left")
    if path:
        _save(fig, path)
    return fig


# ------------------------------------------- fig 3: trajectory in bias space
# Hand-placed stage labels, as an (dx, dy) offset in points from the dot, keyed
# by x-metric then (method, stage). The automatic placement only ever picks the
# two sides perpendicular to the path, which is the wrong family of choices
# where a trajectory doubles back on itself. Anything not listed stays
# automatic, so adding a key here is how you overrule one awkward label.
LABEL_ANCHORS = {
    "zero_frac": {
        ("Celfie", "none"): (-1.25, 5.5),      
        ("Celfie", "clip0"): (-1.25, 2),       
        ("Celfie", "simplex"): (3, 0),       
        ("Celfie", "vector"): (0, 2),       
        ("EpiDISH", "none"): (0, -2),     
        ("EpiDISH", "clip0"): (3, 0),
        ("EpiDISH", "simplex"): (3, 0),  
        ("EpiDISH", "vector"): (0, 2),
        ("Houseman_ineq", "none"): (0, -2),
        ("Houseman_ineq", "clip0"): (-3, 0),
        ("Houseman_ineq", "simplex"): (3, 0),  
        ("Houseman_ineq", "vector"): (0, 2),        
        ("UXM U25", "none"): (0, -2), 
        ("UXM U25", "clip0"): (4, 0),        
        ("UXM U25", "simplex"): (3, 0),       
        ("UXM U25", "vector"): (0, 2),
    },
    "max_frac": {
        ("Celfie", "none"): (-1.25, 7.5),      
        ("Celfie", "clip0"): (-1.25, 2),       
        ("Celfie", "simplex"): (5, -1),       
        ("Celfie", "vector"): (0, 2),       
        ("EpiDISH", "none"): (0, -2),     
        ("EpiDISH", "clip0"): (3, 0),
        ("EpiDISH", "simplex"): (5, 0),  
        ("EpiDISH", "vector"): (0, 2),
        ("Houseman_ineq", "none"): (0, -3),
        ("Houseman_ineq", "clip0"): (-3, 1),
        ("Houseman_ineq", "simplex"): (5, 0),  
        ("Houseman_ineq", "vector"): (0, 2),        
        ("UXM U25", "none"): (0, -2), 
        ("UXM U25", "clip0"): (3, 0),        
        ("UXM U25", "simplex"): (-6, 0),       
        ("UXM U25", "vector"): (0, 2),
    },
}

def fig_bias_trajectory(ps, xmetric="zero_frac",
                        xlabel="zero-fraction  →  sparser", path=None):
    """Small multiples -- overlaying all four paths in one panel collides
    (Houseman and UXM occupy the same region) and needs 16 stage labels."""
    anchors = LABEL_ANCHORS.get(xmetric, {})
    figw = 11.0
    fs = _fs(figw)
    tr = {m:(np.array([ps.loc[(ps.method == m) & (ps.stage == s), xmetric].mean()
                        for s in STAGES]),
              np.array([ps.loc[(ps.method == m) & (ps.stage == s), "tcs"].mean()
                        for s in STAGES]))
          for m in METHODS}

    fig, axes = plt.subplots(1, len(METHODS), figsize=(figw, 4.2), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, m in zip(axes, METHODS):
        for other in METHODS:                       # ghosted context
            if other != m:
                ox, oy = tr[other]
                ax.plot(ox, oy, color="#dcdbd6", lw=1.2, zorder=1)
        xs, ys = tr[m]
        for i in range(len(STAGES) - 1):
            ax.annotate("", (xs[i + 1], ys[i + 1]), (xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=SERIES[m], lw=1.8,
                                        shrinkA=5, shrinkB=5, mutation_scale=11))
        ax.scatter(xs, ys, s=[30, 30, 30, 66], color=SERIES[m],
                   ec="white", lw=1.2, zorder=3)
        ax.set_title(METHOD_LABEL[m], fontsize=fs["title"])
        ax.tick_params(labelsize=fs["tick"])
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        # the stage labels live inside the panel, and they are wide, so x needs
        # roughly a label's width of slack on each side
        ax.margins(x=0.40, y=0.22)
    axes[0].set_ylabel(f"{TCS}\n→  less leakage", fontsize=fs["label"])
    fig.supxlabel(xlabel, fontsize=fs["label"], color=INK2)

    # settle the layout first: the offsets are perpendicular in display space
    fig.canvas.draw()
    stage_names = ["Uncal." if s == "none" else STAGE_LABEL[s] for s in STAGES]
    for ax, m in zip(axes, METHODS):
        xs, ys = tr[m]
        # deliberately not avoiding the ghosted paths: staying off its own arrow
        # and off the neighbouring labels matters, brushing a gray line does not
        _perp_labels(ax, xs, ys, stage_names, fontsize=fs["small"],
                     anchors=[anchors.get((m, s)) for s in STAGES])
    if path:
        _save(fig, path)
    return fig


# --------------------------------- fig 4: signed per-cell-type delta heatmap
def fig_delta_heatmap(df, robust=0.98, path=None):
    """`robust` clips the color scale at that quantile of |delta| so one extreme
    cell (Endothel under Celfie-vector) doesn't wash out the other 38 rows."""
    n = df["file"].nunique()
    panels = {}
    for m in METHODS:
        base = df.groupby("CellType")[col(m, "none")].sum() / n
        mat = pd.DataFrame({
            s: df.groupby("CellType")[col(m, s)].sum() / n - base
            for s in STAGES[1:]})
        panels[m] = mat
    # order rows by how much off-target mass they carry when uncalibrated
    off = df[~df["on_target"]]
    order = (off.groupby("CellType")[[col(m, "none") for m in METHODS]].sum()
             .sum(1).sort_values(ascending=False).index.tolist())
    order += [c for c in panels[METHODS[0]].index if c not in order]
    allv = np.abs(np.concatenate([p.values.ravel() for p in panels.values()]))
    vmax = float(np.quantile(allv, robust)) if robust else float(allv.max())

    fig, axes = plt.subplots(1, len(METHODS), figsize=(9.2, 8.4), sharey=True,
                             constrained_layout=True)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    for ax, m in zip(axes, METHODS):
        mat = panels[m].reindex(order)
        im = ax.imshow(np.clip(mat.values, -vmax, vmax), cmap=DIVERGING,
                       norm=norm, aspect="auto")
        ax.set_xticks(range(len(STAGES) - 1), [STAGE_LABEL[s] for s in STAGES[1:]],
                      rotation=45, ha="right")
        ax.set_title(METHOD_LABEL[m], fontsize=9)
        ax.set_yticks(range(len(order)), order, fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.32, pad=0.02, aspect=14, extend="both",
                 label="Δ mean predicted fraction vs uncalibrated")
    if path:
        _save(fig, path)
    return fig


# ------------------------------------------- fig 5: per-tissue dumbbells
def _stage_strip(ax, vals, y, label=False):
    """One row per y: a gray track spanning the four stages plus one
    colour+shape-coded dot per stage. `vals` maps stage -> array over y."""
    lo = np.min([vals[s] for s in STAGES], axis=0)
    hi = np.max([vals[s] for s in STAGES], axis=0)
    ax.hlines(y, lo, hi, color=GRID, lw=3, zorder=1)
    for s in STAGES:
        ax.scatter(vals[s], y, s=32, color=STAGE_COLOR[s],
                   marker=STAGE_MARKER[s], ec="white", lw=0.8, zorder=3,
                   label=STAGE_LABEL[s] if label else None)


def fig_tissue_bumpbells(ps, path=None):
    """All four stages per tissue row, one hue+marker per stage."""
    piv = (ps.groupby(["method", "stage", "tissue"])["tcs"].mean()
           .unstack("stage"))
    # order by the mean vector-minus-uncalibrated gain across all four methods
    delta = (piv["vector"] - piv["none"]).groupby("tissue").mean()
    order = delta.sort_values().index.tolist()
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, len(METHODS), figsize=(11.6, 5.0), sharey=True,
                             sharex=True, constrained_layout=True)
    for ax, m in zip(axes, METHODS):
        sub = piv.xs(m, level="method").reindex(order)
        _stage_strip(ax, {s: sub[s].to_numpy() for s in STAGES}, y,
                     label=ax is axes[0])
        ax.set_title(METHOD_LABEL[m], fontsize=9)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(order) - 0.3)
    axes[0].set_yticks(y, [f"{t}  ({MAPPING_RATE[t]:.2f})" for t in order],
                       fontsize=8)
    fig.supxlabel(TCS, fontsize=9, color=INK2)
    fig.legend(loc="outside upper center", ncols=4, fontsize=8)
    if path:
        _save(fig, path)
    return fig


# ------------------- fig 6: dominant reference cell type, all four stages
def fig_dominant_celltype(dom, relative=False, path=None):
    """Per tissue, the predicted fraction of that tissue's dominant reference
    cell type at all four stages, against the reference fraction.

    `relative=True` plots pred - ref instead, so the target collapses to a
    single zero line and over/under-shoot is read off one axis.
    """
    tissues = (dom[dom.stage == "vector"].groupby("tissue")["pred"].mean()
               - dom[dom.stage == "none"].groupby("tissue")["pred"].mean())
    order = tissues.sort_values().index.tolist()
    ref = dom.groupby("tissue")[["ref_frac"]].first()
    ct = dom.groupby("tissue")["cell_type"].first()

    fig, axes = plt.subplots(1, len(METHODS), figsize=(11.6, 5.0), sharey=True,
                             sharex=True, constrained_layout=True)
    for ax, m in zip(axes, METHODS):
        sub = (dom[dom.method == m].pivot(index="tissue", columns="stage",
                                          values="pred").reindex(order))
        y = np.arange(len(order))
        base = ref.loc[order, "ref_frac"].to_numpy() if relative else 0.0
        vals = {s: sub[s].to_numpy() - base for s in STAGES}
        _stage_strip(ax, vals, y, label=ax is axes[0])
        if relative:
            ax.axvline(0, color=INK, lw=1.0, zorder=2)
        else:                                           # reference as a target tick
            ax.scatter(ref.loc[order, "ref_frac"], y, marker="|", s=190,
                       color=INK, lw=1.4, zorder=4,
                       label="reference fraction" if ax is axes[0] else None)
        ax.set_title(METHOD_LABEL[m], fontsize=9)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(order) - 0.3)
    axes[0].set_yticks(
        y, [f"{t}  ({ct[t]}, {ref.loc[t, 'ref_frac']:.2f})" for t in order],
        fontsize=8)
    fig.supxlabel("predicted − reference fraction of the dominant cell type"
                  if relative else
                  "predicted fraction of the dominant reference cell type",
                  fontsize=9, color=INK2)
    fig.legend(loc="outside upper center", ncols=5, fontsize=8)
    if path:
        _save(fig, path)
    return fig


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    df = load()
    ps = per_sample(df)
    ref = load_reference()
    dom = dominant_fractions(df, ref)
    print(ps.groupby(["method", "stage"])[["tcs", "off", "zero_frac", "n_off_gt5"]]
          .mean().round(3))
    print(dom.pivot_table(index="stage", values=["pred", "ref_frac"]).round(3))
    fig_tcs_slopegraph(ps, path=f"{OUT}/f1_tcs_slopegraph")
    fig_leakage_destinations(df, path=f"{OUT}/f2_leakage_destinations")
    fig_bias_trajectory(ps, path=f"{OUT}/f3_bias_trajectory_sparsity")
    fig_bias_trajectory(ps, xmetric="max_frac",
                        xlabel="Average Highest Mass Component\n→  more "
                               "concentrated on one cell type",
                        path=f"{OUT}/f3_bias_trajectory_top1")
    fig_delta_heatmap(df, path=f"{OUT}/f4_delta_heatmap")
    fig_tissue_bumpbells(ps, path=f"{OUT}/f5_tissue_bumpbells")
    fig_dominant_celltype(dom, relative=True,
                          path=f"{OUT}/f6_dominant_celltype_relative")
    print("wrote figures to", OUT)
