"""Spearman rank correlation between pseudobulk metrics and Tissue Concordance Score.

Joins `all_pseudobulk_results.csv` (one row per method configuration, evaluated on
pseudobulk mixtures with known ground truth) to the cfSort RRBS TCS results in
`cfsort_results_v2/`, which are reference-free and scored per real sample.

Method identity is the full configuration tuple:
    (classifier | baseline, labeling scheme, prior, feature set, deconvolver, calibrator)

GSS-atlas Dismir runs (`GSS_ATLAS_Dismir_*.csv`) are excluded, as are the
UXM U250 and UXM-U25-GSS-Sorted baseline columns, which have no pseudobulk
counterpart.

Outputs (written next to this file):
    pseudobulk_vs_tcs_spearman.csv   -- one row per pseudobulk metric
    pseudobulk_vs_tcs_ranks.csv      -- method / rank_pseudobulk / rank_tcs / avg rank
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EDA = Path(__file__).resolve().parent
TCS_ROOT = EDA / "cfsort_results_v2"

# Pseudobulk metric -> +1 if higher is better, -1 if lower is better.
METRIC_DIRECTION = {
    "r2": +1,
    "cosine_sim": +1,
    "mse": -1,
    "mae": -1,
    "kl": -1,
    "max_error": -1,
    "loa_width": -1,
    "worst_class_loa_width": -1,
}

#: Metric used for the headline `rank_pseudobulk` column of the final table.
PRIMARY_METRIC = "r2"

# ---------------------------------------------------------------------------
# Expected cell-type sets (mirrors cfsort_results_analysis.ipynb)
# ---------------------------------------------------------------------------

LOYFER_NAME_CORRECTIONS = {
    "Endothelium": "Endothel",
    "Ovary+Endom-Ep": "Ovary-Ep",
}


def get_expected_cell_types(
    tissue_dict,
    threshold=0.01,
    exclude_keys=("unmapped",),
    loyfer_confidence_level="loyfer_counts_strict",
):
    """Expected cell types per cfSort tissue, from Tabula Sapiens composition."""
    rows = []
    for tissue_name, tissue_data in tissue_dict.items():
        loyfer_counts = tissue_data.get(loyfer_confidence_level, {}) or {}
        filtered = {k: v for k, v in loyfer_counts.items() if k not in exclude_keys}
        total = sum(loyfer_counts.values())

        if total == 0:
            rows.append(
                {
                    "cfsort_tissue_name": tissue_name,
                    "expected_cell_types": [],
                    "final_mapping_rate": 0.0,
                }
            )
            continue

        passing = [
            (ct, cnt / total)
            for ct, cnt in filtered.items()
            if (cnt / total) >= threshold
        ]
        passing.sort(key=lambda x: x[1], reverse=True)

        rows.append(
            {
                "cfsort_tissue_name": tissue_name,
                "expected_cell_types": [
                    LOYFER_NAME_CORRECTIONS.get(ct, ct) for ct, _ in passing
                ],
                "final_mapping_rate": round(sum(p for _, p in passing), 4),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["cfsort_tissue_name", "expected_cell_types", "final_mapping_rate"],
    )


def load_expected_sets():
    with open(EDA / "cfsort_full_map_dict_v2.json", "rb") as f:
        tissue_dict = json.load(f)
    df = get_expected_cell_types(tissue_dict)
    return df[df["final_mapping_rate"] > 0.7].reset_index(drop=True)


# ---------------------------------------------------------------------------
# TCS
# ---------------------------------------------------------------------------


def on_target_proportions(deconv_df, expected_df, method_columns):
    """Per (file, tissue) share of predicted signal landing on expected cell types."""
    expected_map = dict(
        zip(expected_df["cfsort_tissue_name"], expected_df["expected_cell_types"])
    )

    df = deconv_df.copy()
    df["_expected"] = df["Biosample term name"].map(expected_map)
    df = df[df["_expected"].notna()]

    on_target = [ct in exp for ct, exp in zip(df["CellType"], df["_expected"])]
    df = df[np.asarray(on_target)]

    tcs = (
        df.groupby(["file", "Biosample term name"], observed=True)[list(method_columns)]
        .sum()
        .reset_index()
    )
    return tcs.dropna()


def summarise_tcs(tcs, method_columns):
    """Sample-level mean and tissue-balanced mean TCS per method column."""
    per_sample = tcs[list(method_columns)].mean(axis=0)
    per_tissue = (
        tcs.groupby("Biosample term name", observed=True)[list(method_columns)]
        .mean()
        .mean(axis=0)
    )
    return per_sample, per_tissue


# ---------------------------------------------------------------------------
# Method-key normalisation
# ---------------------------------------------------------------------------

DECONVOLVER_MAP = {
    "xgboost": "xgb",
    "3Layer_MLP": "mlp",
    "Shallow_Wide_Network": "swn",
    "nnls": "nnls",
    "psls": "psls",
}

CALIBRATOR_MAP = {
    "linear_clip_normalize": "linear_clip_normalize",
    "linear_clip0_normalize": "linear_clip_normalize",
    "linear_simplex_projection": "linear_simplex_projection",
    "vector_scaling": "vector_scaling",
}

#: TCS calibrated-column suffix -> pseudobulk calibrator name (baselines).
BASELINE_CALIBRATOR_SUFFIX = {
    "": "uncalibrated",
    " clip0": "linear_clip_normalize",
    " simplex": "linear_simplex_projection",
    " vector": "vector_scaling",
}

#: pseudobulk `baseline` value -> baseline column stem in calibrated_all_baselines.csv
BASELINE_COLUMN = {
    "celfie": "Celfie",
    "epidish": "EpiDISH",
    "epidish_houseman": "Houseman_ineq",
    "uxm": "UXM U25",
}

#: (TCS subdirectory, feature subdirectory) -> (labeling, prior, feature_set)
#: as named in all_pseudobulk_results.csv.
SCHEME_MAP = {
    ("Hard Labels", "diagbackfeatures"): ("hard_labels", "", "diagbckg"),
    ("Hard Labels", "top156features"): ("hard_labels", "", "top156"),
    ("Soft Labels", ""): ("soft_labels_with_pooling", "", "top156"),
    ("Soft Labels No Pooling", ""): ("soft_labels_without_pooling", "", "top156"),
    ("Soft Labels Canonical", "diagbackfeatures"): (
        "canonical_soft_labels",
        "",
        "diagbckg",
    ),
    ("Soft Labels Canonical", "top156features"): (
        "canonical_soft_labels",
        "",
        "top156",
    ),
}

CLASSIFIER_MAP = {
    "dismir": "dismir",
    "lookup": "lookup",
    "methylbert": "methylbert",
    "cancer_detector": "cancerdetector",
}


def method_key(classifier, labeling, prior, feature_set, deconvolver, calibrator):
    return "|".join(
        [classifier, labeling, prior, feature_set, deconvolver, calibrator]
    )


def key_from_pseudobulk(row):
    def s(v):
        return "" if pd.isna(v) else str(v)

    classifier = s(row["baseline"]) or s(row["syto_classifier"])
    return method_key(
        classifier,
        s(row["syto_labeling"]),
        s(row["syto_prior"]),
        s(row["syto_feature_set"]),
        s(row["syto_deconvolver"]),
        s(row["syto_calibrator"]),
    )


# ---------------------------------------------------------------------------
# TCS assembly
# ---------------------------------------------------------------------------


def collect_tcs_files():
    """Syto TCS result files, with the scheme metadata implied by their path.

    Excludes GSS-atlas Dismir runs.
    """
    out = []
    for path in sorted(TCS_ROOT.rglob("*.csv")):
        rel = path.relative_to(TCS_ROOT)
        parts = rel.parts
        top = parts[0]

        if top in {"OtherBaselines", "UXM", "GSS"}:
            continue
        if path.name.startswith("GSS_ATLAS_"):
            continue  # GSS atlas Dismir variants -- excluded by request

        if top == "Cancer Detector":
            prior = (
                "uniform_prior" if "uniformprior" in path.name else "train_freq_prior"
            )
            out.append((path, ("", prior, "top156")))
            continue

        feature_dir = parts[1] if len(parts) > 2 else ""
        scheme = SCHEME_MAP.get((top, feature_dir))
        if scheme is None:
            raise ValueError(f"unmapped TCS result path: {rel}")
        out.append((path, scheme))
    return out


def syto_tcs(expected_df):
    """TCS per syto method key, as (per-sample mean, tissue-balanced mean)."""
    per_sample, per_tissue = {}, {}

    for path, (labeling, prior, feature_set) in collect_tcs_files():
        df = pd.read_csv(path)
        df["Biosample term name"] = df["Biosample term name"].replace(
            {"stom": "stomach"}
        )
        df = df[df["Calibrator"] != "linear_clip01_normalize"]

        df["_deconv"] = df["Deconvolver"].map(DECONVOLVER_MAP)
        df["_calib"] = df["Calibrator"].map(CALIBRATOR_MAP).fillna("uncalibrated")
        df["_clf"] = df["Classifier"].map(CLASSIFIER_MAP)
        if df["_deconv"].isna().any() or df["_clf"].isna().any():
            raise ValueError(f"unmapped deconvolver/classifier in {path}")

        df["_method"] = [
            method_key(clf, labeling, prior, feature_set, dec, cal)
            for clf, dec, cal in zip(df["_clf"], df["_deconv"], df["_calib"])
        ]

        wide = df.pivot(
            index=["CellType", "file", "Biosample term name"],
            columns="_method",
            values="PredictedProportion",
        ).reset_index()
        wide.columns.name = None

        methods = [c for c in wide.columns if "|" in c]
        tcs = on_target_proportions(wide, expected_df, methods)
        s, t = summarise_tcs(tcs, methods)
        per_sample.update(s.to_dict())
        per_tissue.update(t.to_dict())

    return per_sample, per_tissue


def baseline_tcs(expected_df):
    """TCS per baseline method key, from calibrated_all_baselines.csv."""
    df = pd.read_csv(TCS_ROOT / "OtherBaselines" / "calibrated_all_baselines.csv")
    df["Biosample term name"] = df["Biosample term name"].replace({"stom": "stomach"})

    rename, methods = {}, []
    for baseline, stem in BASELINE_COLUMN.items():
        for suffix, calibrator in BASELINE_CALIBRATOR_SUFFIX.items():
            col = f"{stem}{suffix}"
            if col not in df.columns:
                raise ValueError(f"missing baseline column {col!r}")
            key = method_key(baseline, "", "", "", "", calibrator)
            rename[col] = key
            methods.append(key)

    df = df.rename(columns=rename)
    tcs = on_target_proportions(df, expected_df, methods)
    s, t = summarise_tcs(tcs, methods)
    return s.to_dict(), t.to_dict()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_metrics(pb):
    """Add derived metrics and drop rows that are not method configurations."""
    pb = pb.copy()
    pb["loa_width"] = pb["loa_upper"] - pb["loa_lower"]
    pb["worst_class_loa_width"] = (
        pb["worst_class_loa_upper"] - pb["worst_class_loa_lower"]
    )
    return pb


def config_columns(pb):
    """Split the method configuration into one readable column per axis.

    Baselines carry their name in `Classifier`; the syto-only axes are blank
    for them, since they have no labeling scheme, prior, or feature set.
    """
    is_baseline = pb["baseline"].notna()

    def col(source, blank_for_baseline=True):
        out = pb[source].fillna("")
        return out.mask(is_baseline, "") if blank_for_baseline else out

    return pd.DataFrame(
        {
            "Classifier": col(
                "syto_classifier", blank_for_baseline=False
            ).mask(is_baseline, pb["baseline"].map(BASELINE_COLUMN)),
            "Labeling Scheme": col("syto_labeling"),
            "Prior": col("syto_prior"),
            "Feature Scheme": col("syto_feature_set"),
            "Deconvolver": col("syto_deconvolver"),
            "Calibrator": pb["syto_calibrator"].fillna(""),
        },
        index=pb.index,
    )


def pretty_name(row):
    def s(v):
        return "" if pd.isna(v) else str(v)

    if not pd.isna(row["baseline"]):
        stem = BASELINE_COLUMN[row["baseline"]]
        return f"{stem} / {row['syto_calibrator']}"

    bits = [s(row["syto_classifier"])]
    if s(row["syto_labeling"]):
        bits.append(s(row["syto_labeling"]))
    if s(row["syto_prior"]):
        bits.append(s(row["syto_prior"]))
    bits += [s(row["syto_feature_set"]), s(row["syto_deconvolver"])]
    bits.append(s(row["syto_calibrator"]))
    return " / ".join(bits)


def main():
    expected_df = load_expected_sets()

    pb = build_metrics(pd.read_csv(EDA / "all_pseudobulk_results.csv"))
    pb["method_key"] = pb.apply(key_from_pseudobulk, axis=1)
    pb["method"] = pb.apply(pretty_name, axis=1)
    pb = pd.concat([pb, config_columns(pb)], axis=1)
    if pb["method_key"].duplicated().any():
        raise ValueError("duplicate method keys in all_pseudobulk_results.csv")

    syto_s, syto_t = syto_tcs(expected_df)
    base_s, base_t = baseline_tcs(expected_df)
    tcs_sample = {**syto_s, **base_s}
    tcs_tissue = {**syto_t, **base_t}

    pb["tcs"] = pb["method_key"].map(tcs_tissue)
    pb["tcs_per_sample"] = pb["method_key"].map(tcs_sample)

    unmatched = pb[pb["tcs"].isna()]
    extra = set(tcs_tissue) - set(pb["method_key"])
    print(f"pseudobulk methods: {len(pb)}")
    print(f"TCS methods:        {len(tcs_tissue)}")
    print(f"matched:            {int(pb['tcs'].notna().sum())}")
    if len(unmatched):
        print("\nUnmatched pseudobulk methods:")
        print(unmatched["method_key"].to_string())
    if extra:
        print("\nTCS methods with no pseudobulk row:")
        for k in sorted(extra):
            print(" ", k)

    m = pb[pb["tcs"].notna()].copy()

    # --- Spearman: each pseudobulk metric vs TCS -----------------------------
    rows = []
    for metric, direction in METRIC_DIRECTION.items():
        for tcs_col, label in (("tcs", "tissue_balanced"), ("tcs_per_sample", "per_sample")):
            rho, p = stats.spearmanr(m[metric], m[tcs_col])
            rows.append(
                {
                    "metric": metric,
                    "tcs_aggregation": label,
                    "higher_is_better": direction > 0,
                    "spearman_rho": rho,
                    "p_value": p,
                    # rho re-signed so that positive == metric and TCS agree on
                    # which method is better
                    "concordant_rho": rho * direction,
                    "n_methods": len(m),
                }
            )
    spearman = pd.DataFrame(rows)

    # --- Rank table ---------------------------------------------------------
    config_cols = [
        "Classifier",
        "Labeling Scheme",
        "Prior",
        "Feature Scheme",
        "Deconvolver",
        "Calibrator",
    ]
    ranks = m[
        config_cols + ["method", "method_key", PRIMARY_METRIC, "tcs", "tcs_per_sample"]
    ].copy()
    ranks["rank_pseudobulk"] = ranks[PRIMARY_METRIC].rank(
        ascending=METRIC_DIRECTION[PRIMARY_METRIC] < 0, method="average"
    )
    ranks["rank_tcs"] = ranks["tcs"].rank(ascending=False, method="average")
    ranks["avg_rank"] = (ranks["rank_pseudobulk"] + ranks["rank_tcs"]) / 2
    ranks = ranks.sort_values("avg_rank").reset_index(drop=True)
    ranks = ranks.rename(columns={PRIMARY_METRIC: f"pseudobulk_{PRIMARY_METRIC}"})

    # Per-metric ranks, for reference.
    for metric, direction in METRIC_DIRECTION.items():
        ranks[f"rank_{metric}"] = m.set_index("method_key")[metric].rank(
            ascending=direction < 0, method="average"
        ).reindex(ranks["method_key"]).to_numpy()

    spearman.to_csv(EDA / "pseudobulk_vs_tcs_spearman.csv", index=False)
    ranks.to_csv(EDA / "pseudobulk_vs_tcs_ranks.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Spearman rank correlation: pseudobulk metric vs TCS ===")
    print(
        spearman[spearman.tcs_aggregation == "tissue_balanced"][
            ["metric", "higher_is_better", "spearman_rho", "concordant_rho", "p_value"]
        ].to_string(index=False)
    )
    print("\n=== Top 25 by average rank ===")
    print(
        ranks.head(25)[
            [
                "method",
                f"pseudobulk_{PRIMARY_METRIC}",
                "tcs",
                "rank_pseudobulk",
                "rank_tcs",
                "avg_rank",
            ]
        ].to_string(index=False)
    )
    print("\nWrote pseudobulk_vs_tcs_spearman.csv and pseudobulk_vs_tcs_ranks.csv")


if __name__ == "__main__":
    main()
