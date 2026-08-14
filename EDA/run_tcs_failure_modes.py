#!/usr/bin/env python
"""Reproduce every Tissue Concordance Score failure-mode result and figure.

Run from anywhere:

    python EDA/run_tcs_failure_modes.py
    python EDA/run_tcs_failure_modes.py --outdir /tmp/tcs --ext png

Writes to ``EDA/figures/tcs_failure_modes`` by default:

    tables/    the numbers behind every claim, as CSV
    *.pdf      the figures
    dominance/ one untitled dominance panel per group, for arranging by hand

Everything is derived from the 260 samples spanning the 16 tissues that pass
the mapping-rate filter; nothing here treats Tabula Sapiens proportions as
ground truth. See ``tcs_failure_modes`` for why that matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tcs_failure_modes as t  # noqa: E402

EDA = Path(__file__).resolve().parent

#: Saved width for every figure, in inches. Set this to the document's text
#: block so LaTeX includes each figure at scale 1 and the text sizes chosen in
#: `tcs_failure_modes.FONT` are the sizes that appear on the page.
FIGURE_WIDTH = t.FIGURE_WIDTH

# --------------------------------------------------------------------------
# Analysis configuration -- every judgement call in the pipeline, in one place
# --------------------------------------------------------------------------

#: Tabula Sapiens -> Loyfer mapping confidence level.
LEVEL = "loyfer_counts_strict"

#: A cell type is "expected" in a tissue at or above this reference proportion.
THRESHOLD = 0.01

#: Tissues whose expected set covers less than this fraction of the reference
#: are dropped as too poorly mapped to score against.
MIN_MAPPING_RATE = 0.7

#: Cell types expected in at least this many tissues count as ubiquitous.
UBIQUITY_MIN_TISSUES = 14

#: Exclusion levels for the ubiquity ladder.
LADDER_THRESHOLDS = (14, 8)

#: Different atlases rather than different deconvolution methods; keeping them
#: in a "baselines" group would compare atlas choice with method choice.
EXCLUDE_METHODS = ["UXM U250", "UXM U25 GSS Sorted"]

#: Configurations that describe their component rather than their group.
#: The baselines are restricted to the four uncalibrated methods -- Celfie,
#: EpiDISH, Houseman CP and UXM U25 -- because the calibrated variants are the
#: same four methods again and would trile-count them in any group-level
#: consensus.
FILTERS = {
    "Baseline": {"calibrator": ["none"]},
    "Hard Labels": {
        "feature_scheme": ["Top156 Features"],
        "classifier__not": ["Lookup"],
    },
    "Soft Labels Canonical": {"feature_scheme": ["Top156 Features"]},
}

#: The two Cancer Detector priors are one method, not two labelling schemes.
GROUP_MAP = {
    "Train Frequencies": "Cancer Detector",
    "uniform": "Cancer Detector",
}

#: Groups labelled directly on the ubiquity scatter rather than via the legend.
#: The baselines and Cancer Detector priors are not labelling schemes; DD SL
#: w/o pool sits alone in a clear part of the plot, so annotating it costs
#: nothing and buys a shorter legend. Annotated groups drop out of the legend
#: automatically.
SCATTER_ANNOTATE = [
    "Baseline",
    "Train Frequencies",
    "uniform",
    "Soft Labels No Pooling",
]

#: Groups labelled by their group name rather than their full configuration.
SCATTER_GROUP_LABELLED = ["Train Frequencies", "uniform", "Soft Labels No Pooling"]

#: The two constant predictors drawn on the ubiquity scatter: the score you
#: get for free, and the score the metric can be pushed to without using data.
BLIND_FLOOR = "uniform_all"
BLIND_CEILING = "constant_optimum"

#: Baselines pulled out of the shared "Baseline" marker and given their own,
#: so they can be named in the legend instead of on the plot: both sit inside
#: the crowded middle of the cluster, where another direct label costs more
#: than it explains.
SCATTER_OWN_MARKER = ["UXM U250", "UXM U25 GSS Sorted"]

#: The ubiquity scatter needs more width than the text block to keep its
#: broken panels legible. Its text is scaled by the same factor, so it still
#: lands on the page at the same size as every other figure once LaTeX has
#: scaled the wider figure down.
SCATTER_WIDTH = 8.8

#: Cells highlighted on the extremes grid regardless of how far they deviate.
#: Smooth muscle in ovary and uterus is where most methods place their largest
#: single share, taking the dominant-cell-type call away from the tissue's own
#: epithelium -- an identity error TCS charges nothing for, and one the
#: deviation ranking alone would not surface.
EXTRA_HIGHLIGHT_CELLS = [("ovary", "Smooth-Musc"), ("uterus", "Smooth-Musc")]

BASELINES_FOR_HEADLINE = [
    "Celfie",
    "EpiDISH",
    "Houseman_ineq",
    "UXM U25",
    "UXM U250",
    "UXM U25 GSS Sorted",
]


def _save(fig_or_ax, path, dpi=200, width=None, passes=12, tol=0.03,
          keep_height=False):
    """Save a figure at a fixed on-page width.

    Text sizes in matplotlib are absolute points in the *saved* figure. A
    figure saved 9.5 inches wide and included at a 6.3-inch text block is
    scaled to 66%, and its 10pt text lands on the page at 6.6pt -- which is why
    figures of different widths look like they use different fonts even when
    they do not. Rescaling every figure to one saved width, with text sizes
    held fixed, makes a point in the figure a point on the page.

    ``bbox_inches="tight"`` crops to the drawn content, so the final width is
    not the requested `figsize`; it is found by iterating, since shrinking the
    canvas changes how much room the fixed-size text takes.
    """
    fig = getattr(fig_or_ax, "figure", fig_or_ax)
    target = FIGURE_WIDTH if width is None else width
    if target:
        renderer = fig.canvas.get_renderer()
        for _ in range(passes):
            actual = fig.get_tightbbox(renderer).width
            if abs(actual - target) <= tol:
                break
            scale = target / actual
            w, h = fig.get_size_inches()
            # Wide, short figures lose too much height if the aspect is held:
            # a 13-row ladder squeezed to 2.6 inches is unreadable. Their
            # height is set deliberately, so leave it alone.
            fig.set_size_inches(w * scale, h if keep_height else h * scale)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def main():
    global FIGURE_WIDTH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(EDA / "cfsort_results_v2"))
    parser.add_argument("--mapping", default=str(EDA / "cfsort_full_map_dict_v2.json"))
    parser.add_argument("--outdir", default=str(EDA / "figures" / "tcs_failure_modes"))
    parser.add_argument(
        "--ext", default="pdf", choices=["pdf", "png", "svg"],
        help="figure format (default pdf)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--figure-width", type=float, default=FIGURE_WIDTH,
        help="saved width of every figure, inches (match your \\textwidth)",
    )
    parser.add_argument(
        "--font-scale", type=float, default=1.0,
        help="scale every text size relative to the module defaults",
    )
    args = parser.parse_args()

    FIGURE_WIDTH = args.figure_width
    t.set_font_scale(args.font_scale)

    # Embed fonts as TrueType rather than Type 3, which many journals reject.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    out = Path(args.outdir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (out / "dominance").mkdir(parents=True, exist_ok=True)
    ext = args.ext

    def table(df, name, **kwargs):
        path = tables / f"{name}.csv"
        df.to_csv(path, **kwargs)
        print(f"  wrote tables/{path.name}")

    # ---------------------------------------------------------------- inputs
    print("Loading results ...")
    deconv, meta, coverage = t.load_method_frame(args.results_root)
    with open(args.mapping) as fh:
        tissue_dict = json.load(fh)

    reference = t.reference_composition(tissue_dict, level=LEVEL)
    expected = t.expected_sets(
        reference, threshold=THRESHOLD, min_mapping_rate=MIN_MAPPING_RATE
    )
    tissues = set(expected["cfsort_tissue_name"])
    n_samples = deconv.loc[
        deconv["Biosample term name"].isin(tissues), "file"
    ].nunique()
    print(
        f"  {len(meta)} methods | {len(tissues)} tissues | {n_samples} samples "
        f"| {deconv['CellType'].nunique()} cell types"
    )
    table(coverage, "source_coverage", index=False)
    table(reference, "reference_composition", index=False)
    table(expected, "expected_sets", index=False)
    table(t.ubiquity(expected), "ubiquity")

    all_methods = list(meta.index)

    # -------------------------------------------------- selection per group
    print("\nSelection per group ...")
    best = t.select_best_per_group(deconv, meta, expected)
    table(best, "best_per_group", index=False)
    print(best.to_string(index=False))

    headline = list(best["best_method"])
    for name in BASELINES_FOR_HEADLINE:
        if name in meta.index and name not in headline:
            headline.append(name)
    labels = t.short_names(headline, meta)

    # ------------------------------------------------- tissue-blind ceiling
    print("\nTissue-blind predictors ...")
    predictors = t.tissue_blind_predictors(
        reference, expected, deconv["CellType"].unique()
    )
    blind_per_tissue, blind = t.tissue_blind_scores(predictors, expected)
    table(predictors, "tissue_blind_predictors")
    table(blind_per_tissue, "tissue_blind_per_tissue")
    print((blind * 100).round(2).to_string())

    # ------------------------------------------------------------- ubiquity
    print("\nUbiquity ...")
    profile = t.ubiquity_profile(
        deconv, expected, reference, headline, min_tissues=UBIQUITY_MIN_TISSUES
    )
    table(profile, "ubiquity_profile")
    ladder, excluded = t.ubiquity_ladder(
        deconv, expected, headline, thresholds=LADDER_THRESHOLDS
    )
    table(ladder, "ubiquity_ladder")
    table(
        pd.DataFrame(
            [{"min_tissues": k, "excluded_cell_types": ", ".join(v)}
             for k, v in excluded.items()]
        ),
        "ubiquity_ladder_excluded",
        index=False,
    )

    scatter_labels = dict(labels)
    for method in meta.index[meta["group"].isin(SCATTER_GROUP_LABELLED)]:
        if method in scatter_labels:
            scatter_labels[method] = t.group_label(meta.loc[method, "group"])

    blind_points = t.tissue_blind_profile(
        predictors.loc[[BLIND_FLOOR, BLIND_CEILING]], blind, reference, expected,
        min_tissues=UBIQUITY_MIN_TISSUES,
    )
    table(blind_points, "tissue_blind_profile")
    # Giving a method its own group name gives it its own marker, and the
    # legend picks it up automatically since only annotated groups drop out.
    scatter_meta = meta.copy()
    for method in SCATTER_OWN_MARKER:
        if method in scatter_meta.index:
            scatter_meta.loc[method, "group"] = t.short_names([method], meta)[method]

    t.set_font_scale(args.font_scale * SCATTER_WIDTH / FIGURE_WIDTH)
    fig, _ = t.plot_ubiquity_scatter_broken(
        profile, blind_points=blind_points, meta=scatter_meta,
        annotate=SCATTER_ANNOTATE, labels=scatter_labels,
    )
    t.set_font_scale(args.font_scale)
    _save(fig, out / f"ubiquity_scatter.{ext}", args.dpi,
          width=SCATTER_WIDTH, keep_height=True)

    ladder_labels = t.group_labels_for(headline, meta)
    for suffix, gains in (("", False), ("_gains", True)):
        _save(
            t.plot_ubiquity_arrows(
                ladder, excluded, labels=ladder_labels,
                n_tissues=len(expected), show_gains=gains, show_legend=False,
            ),
            out / f"ubiquity_ladder{suffix}.{ext}", args.dpi, keep_height=True,
        )

    # ------------------------------------------- reference-free divergence
    print("\nWithin-expected-set divergence (reference-free) ...")
    pairs = t.within_set_divergence(deconv, expected, headline)
    table(pairs, "within_set_divergence", index=False)
    close = pairs[pairs["abs_delta_tcs"] < 0.02]
    print(
        f"  {len(close)}/{len(pairs)} pairs within 2 TCS pts; "
        f"their within-set TVD: median {close['divergence'].median() * 100:.1f}%, "
        f"max {close['divergence'].max() * 100:.1f}%"
    )
    _save(
        t.plot_tcs_vs_divergence(pairs, labels=labels),
        out / f"tcs_vs_divergence.{ext}", args.dpi,
    )

    # ----------------------------------------------- shared vs specific
    print("\nShared vs method-specific deviation ...")
    err = t.error_tensor(deconv, expected, reference, headline)
    table(err, "error_tensor_headline", index=False)
    cells = t.shared_vs_specific(err)
    table(cells, "shared_vs_specific_headline", index=False)
    variance = t.variance_decomposition(err)
    print("  headline variance decomposition:", {
        k: (round(v, 3) if isinstance(v, float) else v) for k, v in variance.items()
    })

    grouped_meta = t.remap_groups(meta, GROUP_MAP)
    groups = t.group_error_tensors(
        deconv, grouped_meta, expected, reference,
        exclude_methods=EXCLUDE_METHODS, filters=FILTERS,
    )
    rows = []
    for group, res in groups.items():
        rows.append({"group": group, "n_methods": res["n_methods"], **res["variance"]})
        table(res["cells"], f"shared_vs_specific_{_slug(group)}", index=False)
    table(pd.DataFrame(rows), "variance_decomposition_by_group", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Two versions: each panel naming its own extremes, and one naming a single
    # shared set so the panels can be read against each other.
    fig, _ = t.plot_shared_vs_specific_grid(groups, ncols=2, label_top=4)
    _save(fig, out / f"shared_vs_specific_grid.{ext}", args.dpi)

    shared_cells = t.common_label_cells(groups, label_top=4, min_common=4)
    print("  shared labels:", ", ".join(f"{c} / {s}" for s, c in shared_cells))
    fig, _ = t.plot_shared_vs_specific_grid(
        groups, ncols=2, label_cells=shared_cells
    )
    _save(fig, out / f"shared_vs_specific_grid_common.{ext}", args.dpi)
    table(
        pd.DataFrame(shared_cells, columns=["tissue", "cell_type"]),
        "shared_vs_specific_common_labels",
        index=False,
    )

    # Third version: the union of each group's two most over- and two most
    # under-predicted cells, marked by shape and named in the legend.
    extreme_cells = t.extreme_label_cells(
        groups, per_side=2, always_include=EXTRA_HIGHLIGHT_CELLS
    )
    print("  extreme labels:", ", ".join(f"{c} / {s}" for s, c in extreme_cells))
    fig, _ = t.plot_shared_vs_specific_grid(
        groups, ncols=2, highlight_cells=extreme_cells
    )
    _save(fig, out / f"shared_vs_specific_grid_extremes.{ext}", args.dpi)
    table(
        pd.DataFrame(extreme_cells, columns=["tissue", "cell_type"]),
        "shared_vs_specific_extreme_labels",
        index=False,
    )
    _save(
        t.plot_shared_vs_specific(cells),
        out / f"shared_vs_specific_headline.{ext}", args.dpi,
    )

    fig, _ = t.plot_logratio_grid(
        err.assign(method=err["method"].map(lambda m: labels.get(m, m))),
        [labels[m] for m in headline], ncols=4,
    )
    _save(fig, out / f"logratio_grid.{ext}", args.dpi)

    # ----------------------------------------------------------- compartments
    print("\nCompartment-conditional comparison ...")
    within, between = t.compartment_profile(err)
    table(within, "compartment_within", index=False)
    table(between, "compartment_between", index=False)
    print(between.groupby("compartment")[["pred_total", "ref_total"]]
          .mean().round(3).to_string())

    # ------------------------------------------------------------- dominance
    print("\nDominant cell type ...")
    dom_groups = t.dominance_by_group(
        deconv, grouped_meta, expected, reference,
        exclude_methods=EXCLUDE_METHODS, filters=FILTERS,
    )
    dom_table, dom_detail = t.dominance_table(dom_groups)
    table(dom_table, "dominance_table")
    table(dom_detail, "dominance_detail", index=False)
    print(dom_table.to_string())

    for suffix, ranked in (("", False), ("_ranked", True)):
        tbl, detail = t.dominance_table(dom_groups, ranked=ranked)
        if ranked:
            table(tbl, "dominance_table_ranked")
            table(detail, "dominance_detail_ranked", index=False)
        latex = t.dominance_latex_table(
            detail,
            caption=(
                "Dominant cell type called per tissue by each labelling "
                "scheme, against the Tabula Sapiens reference."
            ),
            label=f"tab:dominance{suffix}",
        )
        (tables / f"dominance_table{suffix}.tex").write_text(latex)
        print(f"  wrote tables/dominance_table{suffix}.tex")

    for group, res in dom_groups.items():
        table(res["accuracy"], f"dominance_accuracy_{_slug(group)}")
        table(res["consensus"], f"dominance_consensus_{_slug(group)}", index=False)

    for group, res in dom_groups.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        t.plot_dominance(res["consensus"], ax=ax, title=None)
        _save(fig, out / "dominance" / f"dominance_{_slug(t.group_label(group))}.{ext}",
              args.dpi)

    dom_all = t.dominance_profile(deconv, expected, reference, headline)
    table(dom_all, "dominance_headline", index=False)
    _save(
        t.plot_dominance(t.dominance_consensus(dom_all)),
        out / f"dominance_headline.{ext}", args.dpi,
    )

    # Which configuration field explains a split, for every tissue where one
    # group disagrees with itself.
    breakdowns = []
    for group, res in dom_groups.items():
        dominance = res["dominance"]
        for tissue in sorted(dominance["tissue"].unique()):
            calls = dominance[dominance["tissue"] == tissue]["pred_dominant"]
            if calls.nunique() < 2:
                continue
            for field, tab in t.dominance_breakdown(
                dominance, grouped_meta, tissue
            ).items():
                melted = tab.reset_index().melt(
                    id_vars="pred_dominant", var_name="value", value_name="n"
                )
                melted = melted[melted["n"] > 0]
                melted.insert(0, "group", group)
                melted.insert(1, "tissue", tissue)
                melted.insert(2, "field", field)
                breakdowns.append(melted)
    if breakdowns:
        table(pd.concat(breakdowns, ignore_index=True), "dominance_breakdown",
              index=False)

    # ----------------------------------------------------------- sensitivity
    print("\nThreshold sensitivity ...")
    sensitivity = t.threshold_sensitivity(
        deconv, tissue_dict, headline, min_mapping_rate=MIN_MAPPING_RATE
    )
    stability = t.rank_stability(sensitivity, reference_threshold=THRESHOLD)
    table(sensitivity, "threshold_sensitivity", index=False)
    table(stability, "rank_stability", index=False)
    print(stability.round(3).to_string(index=False))
    _save(
        # Same label style as the ladder: one name per scheme, not per config.
        t.plot_threshold_stability(sensitivity, labels=ladder_labels),
        out / f"threshold_stability.{ext}", args.dpi,
    )

    print(f"\nDone. Figures and tables under {out}")


def _slug(name):
    slug = "".join(c if c.isalnum() else "_" for c in str(name).lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


if __name__ == "__main__":
    main()
