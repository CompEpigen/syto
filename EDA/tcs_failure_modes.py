"""Failure-mode diagnostics for the Tissue Concordance Score (TCS).

TCS is the share of predicted cfDNA signal that lands on the set of cell types
"expected" in a tissue, where the expected set is derived from Tabula Sapiens
composition at a proportion threshold. It is a set-membership score: it says
how much mass landed in the right bucket, never how that mass is distributed
inside the bucket, and it rewards any method biased toward cell types that are
expected in most tissues.

This module builds the evidence for those limitations. Its organising principle
is that **Tabula Sapiens proportions are not ground truth**. Tabula Sapiens 2.0
(PMC12407789) states that about two thirds of organs used a MACS-based
enrichment strategy "to balance cell types between four compartments;
epithelial, endothelial, immune, and stromal", and that e.g. stomach was
FAC-sorted and recombined at a fixed 70:30 CD45+:CD45- ratio. Between-compartment
proportions are therefore an imposed experimental design, not a measurement.

Consequences enforced throughout:

* The headline evidence is **reference-free**. `within_set_divergence` compares
  methods against each other, and `tissue_blind_scores` compares them against
  constant, input-ignoring predictors. Neither treats the reference as truth;
  the reference only defines *which* cell types are in the set, exactly as TCS
  already does.
* Reference-based comparison is **compartment-conditional**. `compartment_profile`
  renormalises within each of the four compartments and reports the
  between-compartment split separately, flagged as reference-limited.
* Comparisons against the reference are reported as a shared-versus-specific
  decomposition (`shared_vs_specific`), so a deviation common to every method
  is attributed to the reference, atlas, or data rather than to any method.

The functions consume the frames already built in
``EDA/cfsort_results_analysis.ipynb``:

``deconv_df``
    Long frame, one row per (CellType, file, Biosample term name), with one
    column of predicted proportions per method. Proportions sum to ~1 per file.
``tissue_dict``
    ``cfsort_full_map_dict_v2.json`` loaded as a dict.
``expected_df``
    Output of `expected_sets` (drop-in compatible with the notebook's
    ``get_expected_cell_types``): columns ``cfsort_tissue_name``,
    ``expected_cell_types``, ``final_mapping_rate``.
"""

from __future__ import annotations

import textwrap
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Cell-type naming and compartments
# --------------------------------------------------------------------------

#: Reference (Tabula Sapiens -> Loyfer) label names that differ from the atlas
#: label names. Mirrors ``cell_types_names_correction_map`` in the notebook.
LOYFER_NAME_CORRECTIONS = {
    "Endothelium": "Endothel",
    "Ovary+Endom-Ep": "Ovary-Ep",
}

#: Loyfer L4 label -> Tabula Sapiens compartment. The four compartment names are
#: those used by Tabula Sapiens 2.0 (PMC12407789): epithelial, endothelial,
#: immune, stromal. Tabula Sapiens has no separate muscle, adipose, or neural
#: compartment, so those labels fall to stromal -- see UNCERTAIN_COMPARTMENTS.
COMPARTMENTS = {
    # endothelial
    "Endothel": "endothelial",
    # immune
    "Blood-B": "immune",
    "Blood-Granul": "immune",
    "Blood-Mono+Macro": "immune",
    "Blood-NK": "immune",
    "Blood-T": "immune",
    "Eryth-prog": "immune",
    "Megakaryocytes": "immune",
    # epithelial
    "Bladder-Ep": "epithelial",
    "Breast-Basal-Ep": "epithelial",
    "Breast-Luminal-Ep": "epithelial",
    "Colon-Ep": "epithelial",
    "Epid-Kerat": "epithelial",
    "Fallopian-Ep": "epithelial",
    "Gallbladder": "epithelial",
    "Gastric-Ep": "epithelial",
    "Head-Neck-Ep": "epithelial",
    "Kidney-Ep": "epithelial",
    "Liver-Hep": "epithelial",
    "Lung-Ep-Alveo": "epithelial",
    "Lung-Ep-Bron": "epithelial",
    "Ovary-Ep": "epithelial",
    "Pancreas-Acinar": "epithelial",
    "Pancreas-Alpha": "epithelial",
    "Pancreas-Beta": "epithelial",
    "Pancreas-Delta": "epithelial",
    "Pancreas-Duct": "epithelial",
    "Prostate-Ep": "epithelial",
    "Small-Int-Ep": "epithelial",
    "Thyroid-Ep": "epithelial",
    # stromal
    "Adipocytes": "stromal",
    "Bone-Osteob": "stromal",
    "Colon-Fibro": "stromal",
    "Dermal-Fibro": "stromal",
    "Heart-Cardio": "stromal",
    "Heart-Fibro": "stromal",
    "Neuron": "stromal",
    "Oligodend": "stromal",
    "Skeletal-Musc": "stromal",
    "Smooth-Musc": "stromal",
}

#: Assignments that are a consequence of Tabula Sapiens having only four
#: compartments rather than a positive claim about the lineage. Muscle, neural,
#: and adipose labels have no compartment of their own and are parked in
#: stromal; erythroid and megakaryocytic labels are grouped with immune. Report
#: within-compartment results for the stromal compartment with this in mind.
UNCERTAIN_COMPARTMENTS = frozenset(
    {
        "Adipocytes",
        "Eryth-prog",
        "Heart-Cardio",
        "Megakaryocytes",
        "Neuron",
        "Oligodend",
        "Skeletal-Musc",
        "Smooth-Musc",
    }
)

#: Cell types whose scRNA-seq capture bias has a direction known a priori:
#: +1 = over-recovered by dissociation-based scRNA-seq (small, robust cells),
#: -1 = under-recovered (large or fragile cells that do not survive
#: dissociation intact). Used by `shared_vs_specific` to test whether a
#: deviation shared by every method points the way scRNA-seq bias predicts,
#: which would attribute it to the reference rather than to the methods.
PRIOR_CAPTURE_BIAS = {
    "Blood-T": +1,
    "Blood-B": +1,
    "Blood-NK": +1,
    "Blood-Mono+Macro": +1,
    "Blood-Granul": -1,  # granulocytes are lost to dissociation and freezing
    "Adipocytes": -1,
    "Heart-Cardio": -1,
    "Neuron": -1,
    "Skeletal-Musc": -1,
    "Liver-Hep": -1,
}

# --------------------------------------------------------------------------
# Figure palette (validated: dataviz skill, light surface, all-pairs, 3 slots)
# --------------------------------------------------------------------------

PALETTE = {
    "series_1": "#2a78d6",  # blue
    "series_2": "#eb6834",  # orange
    "series_3": "#1baf7a",  # aqua -- sub-3:1 on light, always direct-labelled
    "diverging_low": "#2a78d6",
    "diverging_mid": "#f0efec",
    "diverging_high": "#d03b3b",
    "reference": "#52514e",  # text-secondary, for reference lines
    "grid": "#d8d7d3",
    "text": "#0b0b0b",
    "muted": "#52514e",
}


def diverging_cmap():
    """Blue -> neutral gray -> red, for signed log-ratio maps."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "tcs_diverging",
        [PALETTE["diverging_low"], PALETTE["diverging_mid"], PALETTE["diverging_high"]],
    )


# --------------------------------------------------------------------------
# Assembling one frame from the results tree
# --------------------------------------------------------------------------

SEP = "__"

#: The one file holding the finished baselines (Celfie, EpiDISH, Houseman, UXM
#: and their calibrated variants), already in wide form.
BASELINES_FILE = "OtherBaselines/calibrated_all_baselines.csv"

DECONVOLVERS_MAP = {
    "xgboost": "XGB",
    "3Layer_MLP": "MLP",
    "Shallow_Wide_Network": "SWN",
    "nnls": "NNLS",
    "psls": "PSLS",
}

CALIBRATORS_MAP = {
    "linear_clip0_normalize": "clip0",
    "linear_clip_normalize": "clip0",
    "linear_simplex_projection": "simplex",
    "vector_scaling": "vector",
}

CLASSIFIERS_MAP = {
    "dismir": "Dismir",
    "cancer_detector": "Cancer Detector",
    "lookup": "Lookup",
    "methylbert": "MethylBERT",
    "dismir_u250infered": "Dismir U250Inference",
}

#: Calibrator suffixes appearing in the wide baselines file's column names.
_BASELINE_CALIBRATOR_SUFFIXES = ("clip0", "simplex", "vector")

ID_COLS = ["CellType", "file", "Biosample term name"]


def discover_result_files(
    root,
    skip_dirs=("UXM",),
    skip_prefixes=("GSS",),
    baselines_file=BASELINES_FILE,
):
    """Result CSVs worth loading, as paths relative to `root`.

    Everything under `skip_dirs` is dropped (the UXM directory is superseded by
    the baselines file), as is any basename starting with one of
    `skip_prefixes` (GSS-atlas runs belong to a separate analysis). Inside
    ``OtherBaselines`` only `baselines_file` survives -- the other files there
    are the uncalibrated inputs it was built from.
    """
    root = Path(root)
    baselines = (root / baselines_file).resolve()
    out = []
    for path in sorted(root.rglob("*.csv")):
        rel = path.relative_to(root)
        if rel.parts[0] in set(skip_dirs):
            continue
        if any(path.name.startswith(p) for p in skip_prefixes):
            continue
        if rel.parts[0] == Path(baselines_file).parts[0]:
            if path.resolve() != baselines:
                continue
        out.append(rel)
    return out


def _labelling_fields(rel):
    """Group / feature-scheme fields implied by a result file's location.

    Mirrors the notebook's per-directory logic: the top-level directory names
    the labelling scheme (or, for Cancer Detector, the prior lives in the file
    name), and a ``diagbackfeatures`` subdirectory names the feature scheme.
    """
    parts = rel.parts
    top = parts[0]
    subdir = parts[1] if len(parts) > 2 else ""
    feature_scheme = "Diag + Bckg" if subdir == "diagbackfeatures" else "Top156 Features"

    if top == "Cancer Detector":
        prior = "uniform" if "uniformprior" in rel.name else "Train Frequencies"
        return {"group": prior, "feature_scheme": feature_scheme, "is_prior": True}
    return {"group": top, "feature_scheme": feature_scheme, "is_prior": False}


def _load_syto_file(path, rel):
    """One long syto results CSV -> (wide frame, metadata rows).

    Metadata is captured here, from the config columns themselves, rather than
    parsed back out of the joined column name later. ``Cancer Detector`` and
    ``Dismir U250Inference`` would otherwise have to survive a round trip
    through a separator-joined string.
    """
    df = pd.read_csv(path)
    df["Biosample term name"] = df["Biosample term name"].replace("stom", "stomach")
    df = df[df["Calibrator"] != "linear_clip01_normalize"]

    fields = _labelling_fields(rel)
    df["_deconvolver"] = df["Deconvolver"].map(DECONVOLVERS_MAP).fillna(df["Deconvolver"])
    df["_calibrator"] = df["Calibrator"].map(CALIBRATORS_MAP).fillna("none")
    df["_classifier"] = df["Classifier"].map(CLASSIFIERS_MAP).fillna(df["Classifier"])

    name_parts = [
        pd.Series(fields["group"], index=df.index),
        pd.Series(fields["feature_scheme"], index=df.index),
        df["_classifier"],
        df["_deconvolver"],
        df["_calibrator"],
    ]
    df["_method"] = pd.concat(name_parts, axis=1).astype(str).agg(SEP.join, axis=1)

    dupes = df.duplicated(subset=ID_COLS + ["_method"])
    if dupes.any():
        raise ValueError(
            f"{int(dupes.sum())} duplicate (sample, method) rows in {rel} -- "
            "pivoting would silently drop them."
        )

    wide = (
        df.pivot(index=ID_COLS, columns="_method", values="PredictedProportion")
        .reset_index()
        .rename_axis(columns=None)
    )

    meta = (
        df[["_method", "_classifier", "_deconvolver", "_calibrator"]]
        .drop_duplicates()
        .rename(
            columns={
                "_method": "method",
                "_classifier": "classifier",
                "_deconvolver": "deconvolver",
                "_calibrator": "calibrator",
            }
        )
    )
    meta["group"] = fields["group"]
    meta["group_kind"] = "prior" if fields["is_prior"] else "labelling_scheme"
    meta["feature_scheme"] = fields["feature_scheme"]
    meta["is_baseline"] = False
    meta["source"] = str(rel)
    return wide, meta


def _load_baselines_file(path, rel):
    """The wide baselines CSV -> (wide frame, metadata rows)."""
    df = pd.read_csv(path)
    df["Biosample term name"] = df["Biosample term name"].replace("stom", "stomach")
    methods = [c for c in df.columns if c not in set(ID_COLS) | {
        "Biosample organism", "Biosample type"
    }]

    rows = []
    for method in methods:
        calibrator = "none"
        base = method
        for suffix in _BASELINE_CALIBRATOR_SUFFIXES:
            if method.endswith(" " + suffix):
                base, calibrator = method[: -(len(suffix) + 1)], suffix
                break
        rows.append(
            {
                "method": method,
                "classifier": base,
                "deconvolver": base,
                "calibrator": calibrator,
                "group": "Baseline",
                "group_kind": "baseline",
                "feature_scheme": "",
                "is_baseline": True,
                "source": str(rel),
            }
        )
    return df[ID_COLS + methods], pd.DataFrame(rows)


def load_method_frame(
    root,
    files=None,
    how="inner",
    baselines_file=BASELINES_FILE,
    **discover_kwargs,
):
    """Assemble every method into one wide frame keyed by (cell type, sample).

    This is the input `error_tensor`, `ubiquity_profile`, and
    `within_set_divergence` all expect: one row per (CellType, file, Biosample
    term name), one column per method, proportions summing to ~1 per sample.

    Parameters
    ----------
    root : str or Path
        The ``cfsort_results_v2`` directory.
    files : sequence of relative paths, or None
        Defaults to `discover_result_files(root, **discover_kwargs)`.
    how : {'inner', 'outer'}
        ``'inner'`` keeps only samples every file scored, which is what paired
        tests across methods require. ``'outer'`` keeps everything and leaves
        NaNs; check `coverage` before using it.

    Returns
    -------
    deconv_df : pd.DataFrame
        Wide frame, `ID_COLS` plus one column per method.
    meta : pd.DataFrame
        One row per method: classifier, deconvolver, calibrator, group,
        group_kind, feature_scheme, is_baseline, source.
    coverage : pd.DataFrame
        Per source file, the number of samples and methods contributed, and how
        many samples it lost to the join. Non-empty ``samples_dropped`` means
        that file did not score every sample.
    """
    root = Path(root)
    files = files if files is not None else discover_result_files(
        root, baselines_file=baselines_file, **discover_kwargs
    )

    frames, metas, coverage = [], [], []
    for rel in files:
        rel = Path(rel)
        path = root / rel
        if rel.as_posix() == Path(baselines_file).as_posix():
            wide, meta = _load_baselines_file(path, rel)
        else:
            wide, meta = _load_syto_file(path, rel)
        frames.append(wide)
        metas.append(meta)
        coverage.append(
            {
                "source": str(rel),
                "n_samples": wide["file"].nunique(),
                "n_methods": wide.shape[1] - len(ID_COLS),
            }
        )

    merged = frames[0]
    for frame in frames[1:]:
        overlap = (set(merged.columns) & set(frame.columns)) - set(ID_COLS)
        if overlap:
            raise ValueError(f"method name collision across files: {sorted(overlap)}")
        merged = merged.merge(frame, on=ID_COLS, how=how)

    meta = pd.concat(metas, ignore_index=True).set_index("method")
    cov = pd.DataFrame(coverage)
    cov["samples_dropped"] = cov["n_samples"] - merged["file"].nunique()
    return merged, meta, cov


#: Display names for the groups, following the paper's convention. "DD" is the
#: data-derived soft labelling; the pooling variants are distinguished
#: explicitly because that is the comparison the labels exist to support.
GROUP_LABELS = {
    "Soft Labels": "DD SL w/ pool",
    "Soft Labels No Pooling": "DD SL w/o pool",
    "Soft Labels Canonical": "Canonical SL",
    "Hard Labels": "HL",
    # The Cancer Detector priors are not labelling schemes; naming the
    # classifier in the label keeps them from being read as one.
    "Train Frequencies": "Cancer Detector Train Freq.",
    "uniform": "Cancer Detector Uniform",
    "Baseline": "Baselines",
}

#: Group names as they appear where there is room for the unabbreviated form
#: (the shared-vs-specific grid). Only entries that differ from the internal
#: key need listing.
GROUP_FULL_LABELS = {
    "Baseline": "Baselines",
}


def group_full_label(group):
    """Unabbreviated display name for a group; unknown groups pass through."""
    return GROUP_FULL_LABELS.get(group, group)

#: Groups that are not labelling schemes, and so do not belong in a legend
#: titled as such. Label these directly on the figure instead.
NON_SCHEME_GROUPS = frozenset({"Baseline", "Train Frequencies", "uniform"})

#: Reporting order for groups: external baselines first, then the syto schemes
#: from least to most derived. Both the merged "Cancer Detector" name and the
#: two unmerged prior names are listed, so the order holds whether or not
#: `remap_groups` was applied. Groups not named here keep their original order
#: and follow the ones that are.
GROUP_ORDER = (
    "Baseline",
    "Cancer Detector",
    "Train Frequencies",
    "uniform",
    "Hard Labels",
    "Soft Labels Canonical",
    "Soft Labels No Pooling",
    "Soft Labels",
)


def order_groups(groups):
    """Sort group names into `GROUP_ORDER`, keeping unknown ones at the end."""
    present = list(dict.fromkeys(groups))
    known = [g for g in GROUP_ORDER if g in present]
    return known + [g for g in present if g not in set(known)]

#: Baseline column names that differ from how the paper refers to them.
BASELINE_DISPLAY_NAMES = {
    "Celfie": "CelFiE",
    "Houseman_ineq": "Houseman CP",
}


def group_label(group):
    """Paper-convention display name for a group; unknown groups pass through."""
    return GROUP_LABELS.get(group, group)


def group_labels_for(methods, meta):
    """method -> its group's display name; baselines keep their own name.

    For figures that compare one configuration per scheme, the configuration is
    not what the reader is choosing between, and spelling it out costs more
    width than it returns. `short_names` remains the right choice wherever
    several configurations of one scheme appear together.
    """
    out = {}
    for method in methods:
        row = meta.loc[method]
        if row.get("is_baseline", False):
            out[method] = short_names([method], meta)[method]
        else:
            out[method] = group_label(row["group"])
    return out


def short_names(methods, meta, sep="·"):
    """Compact display labels -- full method names do not fit on a figure.

    A syto method becomes ``<group>[ (D+B)]<sep><classifier><sep><deconvolver>
    <sep><calibrator>``, e.g. ``DD SL w/o pool.Lookup.NNLS.vector``, where the
    group name follows `GROUP_LABELS`. ``(D+B)`` marks the Diag + Bckg feature
    scheme so the two feature schemes never collapse onto one label. Baselines
    keep their own names.

    Returns
    -------
    dict
        method -> label. Raises if two methods would share a label.
    """
    out = {}
    for method in methods:
        row = meta.loc[method]
        if row.get("is_baseline", False):
            base = BASELINE_DISPLAY_NAMES.get(method)
            if base is None:
                # Calibrated variants carry the suffix; rename the stem only.
                stem, _, suffix = method.rpartition(" ")
                base = (
                    f"{BASELINE_DISPLAY_NAMES[stem]} {suffix}"
                    if stem in BASELINE_DISPLAY_NAMES
                    else method
                )
            out[method] = base
            continue
        name = group_label(row["group"])
        if row.get("feature_scheme") == "Diag + Bckg":
            name += " (D+B)"
        parts = [name]
        # The Cancer Detector groups already name their classifier; repeating
        # it makes the label longer without making it more specific.
        if str(row["classifier"]).lower() not in name.lower():
            parts.append(str(row["classifier"]))
        parts += [str(row["deconvolver"]), str(row["calibrator"])]
        out[method] = sep.join(parts)
    clashes = pd.Series(list(out.values())).value_counts()
    clashes = clashes[clashes > 1]
    if not clashes.empty:
        raise ValueError(f"short labels collide: {list(clashes.index)}")
    return out


def remap_groups(meta, group_map):
    """Copy of `meta` with groups renamed, so several can be analysed as one.

    The two Cancer Detector priors are one method with a hyperparameter
    difference, not two labelling schemes, so a panel each says the same thing
    twice. Pass ``{"Train Frequencies": "Cancer Detector", "uniform":
    "Cancer Detector"}`` to fold them together. Note that `filters` and
    `groups` elsewhere then key off the *new* names.
    """
    out = meta.copy()
    out["group"] = out["group"].replace(dict(group_map))
    return out


def select_methods(meta, group=None, filters=None, exclude_methods=()):
    """Methods in a group, after applying per-group metadata filters.

    Some configurations are not comparable to the rest of their group and drag
    a group-level summary around without saying anything about the labelling
    scheme itself -- the Lookup classifier under Hard Labels, for instance,
    fails badly enough that including it describes Lookup rather than Hard
    Labels. `filters` lets those be excluded explicitly and visibly, rather
    than by silently hand-picking a method list.

    Parameters
    ----------
    filters : dict or None
        ``{group: {meta_column: [allowed, ...], meta_column + '__not':
        [disallowed, ...]}}``. Groups absent from the dict are unfiltered.

    Examples
    --------
    >>> filters = {
    ...     "Hard Labels": {
    ...         "feature_scheme": ["Top156 Features"],
    ...         "classifier__not": ["Lookup"],
    ...     },
    ...     "Soft Labels Canonical": {"feature_scheme": ["Top156 Features"]},
    ... }
    """
    sub = meta if group is None else meta[meta["group"] == group]
    for column, values in ((filters or {}).get(group, {}) or {}).items():
        negate = column.endswith("__not")
        column = column[: -len("__not")] if negate else column
        if column not in sub.columns:
            raise KeyError(f"filter column {column!r} not in meta")
        mask = sub[column].isin(list(values))
        sub = sub[~mask if negate else mask]
    return [m for m in sub.index if m not in set(exclude_methods)]


def select_best_per_group(deconv_df, meta, expected_df, methods=None, **score_kwargs):
    """Per group, the top-TCS configuration -- reported with its full range.

    Selecting the best configuration *by the metric under audit* biases the
    selection toward whatever exploits that metric, so the argmax alone is not
    reportable. The range over configurations within a group is, and it is what
    belongs in the paper beside the named winner.

    Returns
    -------
    pd.DataFrame
        One row per group: best_method, best_tcs, min_tcs, max_tcs, median_tcs,
        n_configs, group_kind.
    """
    methods = list(methods if methods is not None else meta.index)
    scores = on_target_scores(deconv_df, expected_df, methods, **score_kwargs)
    mean = tissue_weighted_mean(scores, methods)

    rows = []
    for group, grp in meta.loc[methods].groupby("group"):
        vals = mean.loc[grp.index].dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "group": group,
                "group_kind": grp["group_kind"].iloc[0],
                "best_method": vals.idxmax(),
                "best_tcs": float(vals.max()),
                "median_tcs": float(vals.median()),
                "min_tcs": float(vals.min()),
                "max_tcs": float(vals.max()),
                "n_configs": int(len(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values("best_tcs", ascending=False)


# --------------------------------------------------------------------------
# Reference composition and expected sets
# --------------------------------------------------------------------------


def reference_composition(
    tissue_dict,
    level="loyfer_counts_strict",
    exclude_keys=("unmapped",),
    rename=True,
):
    """Tidy per-tissue reference composition from the cfsort mapping dict.

    Returns one row per (tissue, cell_type) with two proportions, because the
    two downstream uses need different denominators:

    ``prop_all``
        count / total including ``unmapped``. This is the denominator the
        notebook's ``get_expected_cell_types`` uses, so expected sets built
        from it reproduce the existing analysis exactly.
    ``prop_mapped``
        count / total over mapped cell types only. This is the one comparable
        to predicted proportions, which sum to 1 over atlas cell types.

    Parameters
    ----------
    tissue_dict : dict
        ``cfsort_full_map_dict_v2.json`` contents.
    level : str
        Which ``loyfer_counts_*`` sub-dict to read.
    exclude_keys : tuple of str
        Keys dropped from the numerator (but kept in the ``prop_all``
        denominator), by default ``('unmapped',)``.
    rename : bool
        Apply `LOYFER_NAME_CORRECTIONS` so labels match the atlas.

    Returns
    -------
    pd.DataFrame
        Columns: tissue, cell_type, count, prop_all, prop_mapped, compartment.
    """
    rows = []
    for tissue, data in tissue_dict.items():
        counts = data.get(level, {}) or {}
        total_all = sum(counts.values())
        kept = {k: v for k, v in counts.items() if k not in exclude_keys}
        total_mapped = sum(kept.values())
        if total_all == 0 or total_mapped == 0:
            continue
        for cell_type, count in kept.items():
            name = (
                LOYFER_NAME_CORRECTIONS.get(cell_type, cell_type)
                if rename
                else cell_type
            )
            rows.append(
                {
                    "tissue": tissue,
                    "cell_type": name,
                    "count": count,
                    "prop_all": count / total_all,
                    "prop_mapped": count / total_mapped,
                }
            )

    ref = pd.DataFrame(rows)
    # A rename can merge two source labels onto one atlas label.
    ref = (
        ref.groupby(["tissue", "cell_type"], as_index=False)
        .agg({"count": "sum", "prop_all": "sum", "prop_mapped": "sum"})
    )
    ref["compartment"] = ref["cell_type"].map(COMPARTMENTS)
    return ref


def expected_sets(reference, threshold=0.01, min_mapping_rate=None):
    """Expected cell types per tissue, in the notebook's ``expected_df`` format.

    Drop-in replacement for ``get_expected_cell_types`` followed by the
    name-correction step: the threshold is applied to ``prop_all`` (denominator
    includes ``unmapped``), matching the original, but labels are already
    corrected to atlas names.

    Parameters
    ----------
    reference : pd.DataFrame
        Output of `reference_composition`.
    threshold : float
        Minimum ``prop_all`` for a cell type to be expected.
    min_mapping_rate : float or None
        If given, drop tissues whose ``final_mapping_rate`` falls at or below
        this (the notebook uses 0.7).

    Returns
    -------
    pd.DataFrame
        Columns: cfsort_tissue_name, expected_cell_types, final_mapping_rate.
    """
    rows = []
    for tissue, grp in reference.groupby("tissue"):
        passing = grp[grp["prop_all"] >= threshold].sort_values(
            "prop_all", ascending=False
        )
        rows.append(
            {
                "cfsort_tissue_name": tissue,
                "expected_cell_types": passing["cell_type"].tolist(),
                "final_mapping_rate": round(float(passing["prop_all"].sum()), 4),
            }
        )
    out = pd.DataFrame(
        rows, columns=["cfsort_tissue_name", "expected_cell_types", "final_mapping_rate"]
    )
    if min_mapping_rate is not None:
        out = out[out["final_mapping_rate"] > min_mapping_rate].reset_index(drop=True)
    return out


def _expected_lookup(expected_df):
    """tissue -> set of expected cell types."""
    return {
        row["cfsort_tissue_name"]: set(row["expected_cell_types"])
        for _, row in expected_df.iterrows()
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def on_target_scores(
    deconv_df,
    expected_df,
    methods,
    biosample_col="Biosample term name",
    file_col="file",
    celltype_col="CellType",
    restrict_to=None,
):
    """Per-sample TCS: predicted mass landing on the expected set.

    Vectorised equivalent of the notebook's ``compute_on_target_proportions``,
    with one addition: `restrict_to` limits the sum to a subset of cell types,
    which is what splits TCS into its ubiquitous and tissue-specific parts.

    Parameters
    ----------
    deconv_df : pd.DataFrame
        Long predictions frame.
    expected_df : pd.DataFrame
        Output of `expected_sets`.
    methods : sequence of str
        Prediction columns to score.
    restrict_to : set of str, or None
        If given, only cell types in this set contribute. The full TCS is the
        sum of the restricted score over any partition of the cell types.

    Returns
    -------
    pd.DataFrame
        One row per (file, tissue), one column per method.
    """
    methods = [methods] if isinstance(methods, str) else list(methods)
    expected = _expected_lookup(expected_df)

    df = deconv_df[deconv_df[biosample_col].isin(expected)].copy()
    on_target = [
        ct in expected[tissue]
        for ct, tissue in zip(df[celltype_col], df[biosample_col])
    ]
    df["_on_target"] = on_target
    if restrict_to is not None:
        df["_on_target"] &= df[celltype_col].isin(set(restrict_to))

    scored = df[df["_on_target"]].groupby([file_col, biosample_col])[methods].sum()
    # Samples with no on-target mass at all still belong in the output as zeros.
    index = pd.MultiIndex.from_frame(
        df[[file_col, biosample_col]].drop_duplicates()
    )
    return scored.reindex(index).fillna(0.0).reset_index()


def tissue_weighted_mean(scores, methods, biosample_col="Biosample term name"):
    """Mean per tissue, then across tissues -- so tissues weigh equally.

    Sample counts per tissue are unequal, so a plain mean over samples silently
    weights the analysis toward whichever tissues were sequenced most.
    """
    methods = list(methods)
    return scores.groupby(biosample_col)[methods].mean().mean()


def ubiquity(expected_df):
    """Per cell type: in how many tissues it is expected.

    Returns
    -------
    pd.DataFrame
        Index cell_type; columns n_tissues_expected, ubiquity, idf, compartment.
        ``idf = log(T / n_tissues_expected)`` is a specificity weight: a cell
        type expected everywhere carries no tissue-specific information.
    """
    lists = expected_df["expected_cell_types"]
    n_tissues = len(lists)
    counts = pd.Series(
        [ct for lst in lists for ct in set(lst)], dtype=object
    ).value_counts()
    out = pd.DataFrame({"n_tissues_expected": counts})
    out["ubiquity"] = out["n_tissues_expected"] / n_tissues
    out["idf"] = np.log(n_tissues / out["n_tissues_expected"])
    out["compartment"] = out.index.map(COMPARTMENTS)
    out.index.name = "cell_type"
    return out.sort_values("ubiquity", ascending=False)


def ubiquitous_types(expected_df, min_tissues=None, min_ubiquity=None):
    """Cell types expected in at least `min_tissues` (or `min_ubiquity`) tissues."""
    ub = ubiquity(expected_df)
    if min_tissues is not None:
        return set(ub.index[ub["n_tissues_expected"] >= min_tissues])
    if min_ubiquity is not None:
        return set(ub.index[ub["ubiquity"] >= min_ubiquity])
    raise ValueError("pass either min_tissues or min_ubiquity")


# --------------------------------------------------------------------------
# 1. Tissue-blind ceiling -- how much TCS needs no tissue information at all
# --------------------------------------------------------------------------


def tissue_blind_predictors(reference, expected_df, cell_types):
    """Constant predictors that ignore the input entirely.

    Each predicts the *same* proportion vector for every sample, so whatever
    TCS they reach is reachable without any tissue-specific signal. A method
    that does not clearly beat them carries no tissue information, however good
    its TCS looks in isolation.

    ``uniform_all``
        Uniform over every atlas cell type. The naive chance level.
    ``uniform_expected_union``
        Uniform over the union of all expected sets. Knows which cell types are
        ever expected, but not in which tissue.
    ``ubiquity_weighted``
        Mass proportional to the number of tissues a cell type is expected in.
        The best *smooth* exploit of failure mode (3) -- it spreads over the
        ubiquitous types rather than collapsing onto one. Its score has a
        closed form, ``sum(n_c^2) / (T * sum(n_c))`` with ``n_c`` the number of
        tissues in which `c` is expected.
    ``mean_reference``
        The reference composition averaged over tissues.
    ``constant_optimum``
        All mass on the single most-expected cell type. This is the true
        maximum over constant predictors, not merely a plausible one: TCS is
        linear in ``p``, so maximising ``sum_c n_c p_c`` over the simplex is a
        linear program whose optimum sits at a vertex. It scores
        ``max_c n_c / T`` -- the metric's ceiling, reached by a prediction that
        is biologically absurd. Report it whenever the ubiquity argument is
        made; a reader will otherwise construct it themselves.

    Returns
    -------
    pd.DataFrame
        Index predictor name, columns `cell_types`, rows sum to 1. The cell
        type carrying ``constant_optimum`` is recorded in
        ``.attrs['constant_optimum_cell_type']``.
    """
    cell_types = list(cell_types)
    ub = ubiquity(expected_df)
    union = [ct for ct in cell_types if ct in ub.index]

    def _vector(weights):
        vec = pd.Series(0.0, index=cell_types)
        vec.update(pd.Series(weights, dtype=float))
        total = vec.sum()
        return vec / total if total > 0 else vec

    mean_ref = (
        reference[reference["tissue"].isin(set(expected_df["cfsort_tissue_name"]))]
        .pivot_table(
            index="tissue", columns="cell_type", values="prop_mapped", fill_value=0.0
        )
        .mean(axis=0)
    )

    counts = ub.loc[union, "n_tissues_expected"].astype(float)
    # Ties broken alphabetically so the choice is reproducible.
    best = sorted(counts.index[counts == counts.max()])[0]

    predictors = {
        "uniform_all": _vector({ct: 1.0 for ct in cell_types}),
        "uniform_expected_union": _vector({ct: 1.0 for ct in union}),
        "ubiquity_weighted": _vector(counts.to_dict()),
        "mean_reference": _vector(mean_ref.to_dict()),
        "constant_optimum": _vector({best: 1.0}),
    }
    out = pd.DataFrame(predictors).T[cell_types]
    out.attrs["constant_optimum_cell_type"] = best
    return out


def tissue_blind_scores(predictors, expected_df):
    """TCS of each constant predictor, per tissue and averaged.

    Returns
    -------
    per_tissue : pd.DataFrame
        Index tissue, columns predictor names.
    summary : pd.Series
        Mean TCS per predictor across tissues (tissues weighted equally).
    """
    expected = _expected_lookup(expected_df)
    per_tissue = pd.DataFrame(
        {
            name: {
                tissue: float(vec[[c for c in vec.index if c in exp]].sum())
                for tissue, exp in expected.items()
            }
            for name, vec in predictors.iterrows()
        }
    )
    summary = per_tissue.mean(axis=0)
    # Carry the constant-optimum cell type through, so a figure can name it.
    summary.attrs.update(predictors.attrs)
    per_tissue.attrs.update(predictors.attrs)
    return per_tissue, summary


# --------------------------------------------------------------------------
# 2. Ubiquity profile -- is the TCS coming from cell types that say nothing?
# --------------------------------------------------------------------------


def ubiquity_profile(
    deconv_df,
    expected_df,
    reference,
    methods,
    min_tissues=14,
    biosample_col="Biosample term name",
    file_col="file",
    celltype_col="CellType",
):
    """Split each method's TCS into its ubiquitous and tissue-specific halves.

    TCS decomposes additively, since the ubiquitous and specific expected types
    partition the expected set::

        tcs = tcs_ubiquitous + tcs_specific

    A method whose advantage lives in ``tcs_ubiquitous`` is winning on cell
    types expected almost everywhere, which is failure mode (3).

    ``ubiquitous_excess`` compares the mass a method puts on ubiquitous cell
    types against the mass the reference puts there, per tissue, so the
    comparison is matched rather than against a grand average. It is a
    *relative* quantity: the reference's absolute level is not trusted (see
    module docstring), but every method is offset by the same amount, so the
    spread across methods is interpretable even if the zero point is not.

    Returns
    -------
    pd.DataFrame
        Index method; columns tcs, tcs_ubiquitous, tcs_specific,
        specific_share, ubiquitous_mass, ubiquitous_mass_reference,
        ubiquitous_excess.
    """
    methods = list(methods)
    ubiq = ubiquitous_types(expected_df, min_tissues=min_tissues)
    kwargs = dict(
        biosample_col=biosample_col, file_col=file_col, celltype_col=celltype_col
    )

    tcs = tissue_weighted_mean(
        on_target_scores(deconv_df, expected_df, methods, **kwargs),
        methods,
        biosample_col,
    )
    tcs_ubiq = tissue_weighted_mean(
        on_target_scores(deconv_df, expected_df, methods, restrict_to=ubiq, **kwargs),
        methods,
        biosample_col,
    )

    # Total mass on ubiquitous types, expected or not, per tissue then averaged.
    tissues = set(expected_df["cfsort_tissue_name"])
    sub = deconv_df[
        deconv_df[biosample_col].isin(tissues) & deconv_df[celltype_col].isin(ubiq)
    ]
    ubiq_mass = (
        sub.groupby([file_col, biosample_col])[methods]
        .sum()
        .reset_index()
        .groupby(biosample_col)[methods]
        .mean()
        .mean()
    )

    ref_ubiq = (
        reference[
            reference["tissue"].isin(tissues) & reference["cell_type"].isin(ubiq)
        ]
        .groupby("tissue")["prop_mapped"]
        .sum()
        .mean()
    )

    out = pd.DataFrame(
        {
            "tcs": tcs,
            "tcs_ubiquitous": tcs_ubiq,
            "tcs_specific": tcs - tcs_ubiq,
            "ubiquitous_mass": ubiq_mass,
        }
    )
    out["specific_share"] = out["tcs_specific"] / out["tcs"].replace(0, np.nan)
    out["ubiquitous_mass_reference"] = ref_ubiq
    out["ubiquitous_excess"] = out["ubiquitous_mass"] - ref_ubiq
    return out.sort_values("tcs", ascending=False)


def ubiquity_ladder(
    deconv_df,
    expected_df,
    methods,
    thresholds=(14, 8),
    biosample_col="Biosample term name",
    file_col="file",
    celltype_col="CellType",
):
    """TCS recomputed with progressively more ubiquitous cell types excluded.

    At each threshold `k`, cell types expected in at least `k` of the tissues
    are struck from the expected set and the score recomputed. Reading down the
    ladder shows how much of a method's TCS rests on cell types that are
    expected almost everywhere and therefore say little about which tissue the
    sample came from.

    Returns
    -------
    ladder : pd.DataFrame
        Index method; column ``tcs`` plus one ``tcs_excl_<k>`` per threshold.
    excluded : dict
        threshold -> sorted list of cell types excluded at that level. Use it
        to name the excluded types in the figure rather than leaving the reader
        to infer what "ubiquitous" meant.
    """
    methods = list(methods)
    kwargs = dict(
        biosample_col=biosample_col, file_col=file_col, celltype_col=celltype_col
    )
    all_types = set(deconv_df[celltype_col].unique())

    out = {
        "tcs": tissue_weighted_mean(
            on_target_scores(deconv_df, expected_df, methods, **kwargs),
            methods,
            biosample_col,
        )
    }
    excluded = {}
    for k in sorted(thresholds, reverse=True):
        drop = ubiquitous_types(expected_df, min_tissues=k)
        excluded[k] = sorted(drop)
        keep = all_types - drop
        out[f"tcs_excl_{k}"] = tissue_weighted_mean(
            on_target_scores(
                deconv_df, expected_df, methods, restrict_to=keep, **kwargs
            ),
            methods,
            biosample_col,
        )
    return pd.DataFrame(out), excluded


# --------------------------------------------------------------------------
# 3. Dominant cell type -- the identity error TCS cannot see
# --------------------------------------------------------------------------


def dominance_profile(
    deconv_df,
    expected_df,
    reference,
    methods,
    biosample_col="Biosample term name",
    celltype_col="CellType",
):
    """Which cell type each method calls dominant, per tissue.

    TCS is invariant to *which* expected cell type carries the mass. A method
    can therefore name a plausible-but-wrong dominant cell type -- Smooth-Musc
    in ovary, say -- and pay no penalty at all, provided that type is in the
    expected set. ``pred_dominant_expected`` is the column that makes this
    explicit: where it is True, the identity error is entirely invisible to
    TCS.

    The reference dominant is reported for context, not as ground truth; the
    reference-free reading is ``dominance_consensus``, which asks whether the
    methods agree with *each other*.

    Returns
    -------
    pd.DataFrame
        One row per (method, tissue): pred_dominant, pred_share,
        ref_dominant, ref_share, matches_reference, pred_dominant_expected,
        pred_dominant_compartment.
    """
    methods = list(methods)
    tissues = set(expected_df["cfsort_tissue_name"])
    expected = _expected_lookup(expected_df)

    pred = (
        deconv_df[deconv_df[biosample_col].isin(tissues)]
        .groupby([biosample_col, celltype_col])[methods]
        .mean()
    )
    ref = reference[reference["tissue"].isin(tissues)]
    ref_dom = ref.loc[ref.groupby("tissue")["prop_mapped"].idxmax()].set_index("tissue")
    # Rank of every cell type within each tissue's reference, 1 = most
    # abundant. Rank 2 is worth separating out: calling the second-most
    # abundant type dominant is a different kind of error from calling an
    # unrelated one, and the reference's own top-two ordering is exactly the
    # part a balanced-compartment atlas is least able to pin down.
    ref_rank = (
        ref.assign(
            rank=ref.groupby("tissue")["prop_mapped"].rank(
                ascending=False, method="min"
            )
        )
        .set_index(["tissue", "cell_type"])["rank"]
    )

    rows = []
    for tissue, grp in pred.groupby(level=0):
        cells = grp.index.get_level_values(1)
        for method in methods:
            values = grp[method].to_numpy()
            winner = cells[int(np.argmax(values))]
            rows.append(
                {
                    "method": method,
                    "tissue": tissue,
                    "pred_dominant": winner,
                    "pred_share": float(values.max()),
                    "ref_dominant": ref_dom.loc[tissue, "cell_type"],
                    "ref_share": float(ref_dom.loc[tissue, "prop_mapped"]),
                    "pred_dominant_expected": winner in expected[tissue],
                    "pred_dominant_compartment": COMPARTMENTS.get(winner),
                    "pred_dominant_ref_rank": float(
                        ref_rank.get((tissue, winner), np.nan)
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out["matches_reference"] = out["pred_dominant"] == out["ref_dominant"]
    return out


def dominance_consensus(dominance):
    """Per tissue, how the methods split over the dominant cell type.

    A tissue where most methods agree on a dominant type that is not the
    reference's is a shared failure -- it points at the atlas or the data, not
    at any one method. ``invisible_to_tcs`` counts the methods whose dominant
    call is wrong yet still inside the expected set, so TCS charges them
    nothing for it.

    Returns
    -------
    pd.DataFrame
        One row per (tissue, pred_dominant): n_methods, frac_methods,
        ref_dominant, is_reference, invisible_to_tcs, methods.
    """
    rows = []
    for tissue, grp in dominance.groupby("tissue"):
        n_total = len(grp)
        for winner, sub in grp.groupby("pred_dominant"):
            rows.append(
                {
                    "tissue": tissue,
                    "pred_dominant": winner,
                    "n_methods": len(sub),
                    "frac_methods": len(sub) / n_total,
                    "ref_dominant": sub["ref_dominant"].iloc[0],
                    "ref_rank": float(sub["pred_dominant_ref_rank"].iloc[0]),
                    "is_reference": winner == sub["ref_dominant"].iloc[0],
                    "invisible_to_tcs": bool(
                        winner != sub["ref_dominant"].iloc[0]
                        and sub["pred_dominant_expected"].all()
                    ),
                    "mean_share": float(sub["pred_share"].mean()),
                    "methods": sorted(sub["method"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["tissue", "n_methods"], ascending=[True, False]
    )


def dominance_by_group(
    deconv_df,
    meta,
    expected_df,
    reference,
    groups=None,
    exclude_methods=(),
    filters=None,
    **kwargs,
):
    """`dominance_profile` per group, so labelling schemes can be compared.

    Returns
    -------
    dict
        group -> {'dominance': DataFrame, 'consensus': DataFrame,
        'accuracy': DataFrame, 'n_methods': int}.
    """
    groups = list(groups if groups is not None else meta["group"].unique())
    out = {}
    for group in groups:
        methods = select_methods(meta, group, filters, exclude_methods)
        methods = [m for m in methods if m in deconv_df.columns]
        if not methods:
            continue
        dom = dominance_profile(deconv_df, expected_df, reference, methods, **kwargs)
        out[group] = {
            "dominance": dom,
            "consensus": dominance_consensus(dom),
            "accuracy": dominance_accuracy(dom),
            "n_methods": len(methods),
        }
    return out


#: How a dominant-cell-type call relates to the reference. The middle case is
#: the one worth naming: the call is wrong, but the cell type is inside the
#: expected set, so TCS charges nothing for it.
DOMINANCE_STATUS = (
    "matches reference",
    "wrong, inside expected set",
    "wrong, outside expected set",
)

#: The finer vocabulary, splitting off the case where the called cell type is
#: the reference's *second* most abundant. That is the mildest possible miss --
#: and the one least safely attributed to the method, since a compartment-
#: balanced reference is weakest at ordering its own top two.
DOMINANCE_STATUS_RANKED = (
    "matches reference",
    "second most abundant in reference",
    "wrong, inside expected set",
    "wrong, outside expected set",
)


def dominance_status(row, ranked=False):
    """Which status a consensus row falls into.

    With `ranked`, a call that is the reference's second most abundant cell
    type and inside the expected set gets its own category.
    """
    if row["is_reference"]:
        return DOMINANCE_STATUS[0]
    if (
        ranked
        and row.get("ref_rank") == 2
        and row.get("invisible_to_tcs")
    ):
        return DOMINANCE_STATUS_RANKED[1]
    inside = DOMINANCE_STATUS_RANKED[2] if ranked else DOMINANCE_STATUS[1]
    outside = DOMINANCE_STATUS_RANKED[3] if ranked else DOMINANCE_STATUS[2]
    return inside if row["invisible_to_tcs"] else outside


def dominance_table(dominance_groups, min_fraction=0.0, ranked=False):
    """Modal dominant call per (tissue, group), as a reportable table.

    With seven groups and sixteen tissues a figure per group is unwieldy; this
    is the compact form. Each cell gives the cell type most methods in that
    group called dominant, the share of methods that agreed, and its status.

    Returns
    -------
    table : pd.DataFrame
        Rows tissue, columns group, values ``"Cell-Type (n/N)"``. A leading
        ``*`` marks a call that is wrong but inside the expected set, and so
        invisible to TCS; ``!`` marks wrong and outside it.
    detail : pd.DataFrame
        Long form with tissue, group, pred_dominant, ref_dominant, n_methods,
        n_total, frac_methods, status.
    """
    rows = []
    for group, res in dominance_groups.items():
        cons = res["consensus"]
        for tissue, grp in cons.groupby("tissue"):
            grp = grp[grp["frac_methods"] >= min_fraction]
            if grp.empty:
                continue
            top = grp.loc[grp["n_methods"].idxmax()]
            rows.append(
                {
                    "tissue": tissue,
                    "group": group,
                    "pred_dominant": top["pred_dominant"],
                    "ref_dominant": top["ref_dominant"],
                    "n_methods": int(top["n_methods"]),
                    "n_total": res["n_methods"],
                    "frac_methods": float(top["frac_methods"]),
                    "mean_share": float(top["mean_share"]),
                    "ref_rank": float(top.get("ref_rank", float("nan"))),
                    "status": dominance_status(top, ranked=ranked),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail

    marks = dict(DOMINANCE_MARKS_PLAIN)
    detail["_cell"] = [
        f"{marks[s]}{ct} ({n}/{tot})"
        for s, ct, n, tot in zip(
            detail["status"],
            detail["pred_dominant"],
            detail["n_methods"],
            detail["n_total"],
        )
    ]
    table = detail.pivot(index="tissue", columns="group", values="_cell")
    table = table[order_groups(table.columns)]
    ref = detail.groupby("tissue")["ref_dominant"].first()
    table.insert(0, "reference", ref)
    return table, detail.drop(columns="_cell")


#: Cell shading per status. Shared by both vocabularies, so the three-status
#: and four-status tables stay visually consistent with one another.
DOMINANCE_COLORS = {
    "matches reference": "teal!15",
    "second most abundant in reference": "olive!18",
    "wrong, inside expected set": "blue!12",
    "wrong, outside expected set": "magenta!15",
}

#: Symbols repeat the shading, so the distinction survives greyscale printing.
DOMINANCE_MARKS_TEX = {
    "matches reference": "",
    "second most abundant in reference": r"$\circ$\,",
    "wrong, inside expected set": r"$\ast$\,",
    "wrong, outside expected set": r"$\dagger$\,",
}

DOMINANCE_MARKS_PLAIN = {
    "matches reference": "",
    "second most abundant in reference": "o",
    "wrong, inside expected set": "*",
    "wrong, outside expected set": "!",
}

DOMINANCE_LEGEND_TEXT = {
    "matches reference": "matches Tabula Sapiens",
    "second most abundant in reference": (
        "the reference's second most abundant type, and in the expected set"
    ),
    "wrong, inside expected set": (
        "differs, but the called type is in the expected set, so TCS is unaffected"
    ),
    "wrong, outside expected set": "differs and lies outside the expected set",
}


def _latex_escape(text):
    """Escape LaTeX specials. Cell type names carry '+' and '_'."""
    out = str(text)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(a, b)
    return out


def dominance_latex_table(
    detail,
    caption=None,
    label=None,
    colors=None,
    font_size="footnotesize",
    group_order=None,
    column_format=None,
    show_counts=True,
    resizebox=True,
    tabcolsep="4pt",
):
    """LaTeX table of the dominant cell type called per tissue per group.

    Cells are shaded by `DOMINANCE_STATUS`, which is the point of the table:
    the reader needs to distinguish a call that is simply wrong from one that
    is wrong yet still inside the expected set, because TCS charges nothing for
    the second. Shading carries that alone, so a symbol is added too -- colour
    does not survive a greyscale print.

    Requires ``\\usepackage[table]{xcolor}`` and ``\\usepackage{booktabs}``.

    Parameters
    ----------
    detail : pd.DataFrame
        Second return value of `dominance_table`.
    colors : dict or None
        status -> xcolor spec. Defaults to teal / blue / magenta at 15--20%.
    group_order : sequence or None
        Column order; defaults to `GROUP_ORDER`.
    show_counts : bool
        Append ``(n/N)``, the number of configurations making that call.
    resizebox : bool
        Wrap the tabular in ``\\resizebox{\\textwidth}{!}{...}`` so it fits the
        text block. Needs ``\\usepackage{graphicx}``.
    tabcolsep : str or None
        Column padding; the default tightens it from LaTeX's 6pt.

    Returns
    -------
    str
        Requires ``\\usepackage[table]{xcolor}``, ``\\usepackage{booktabs}``
        and, with `resizebox`, ``\\usepackage{graphicx}``.
    """
    if detail.empty:
        return ""

    colors = {**DOMINANCE_COLORS, **(colors or {})}
    marks = DOMINANCE_MARKS_TEX

    groups = list(group_order) if group_order else order_groups(detail["group"])
    tissues = sorted(detail["tissue"].unique())
    indexed = detail.set_index(["tissue", "group"])

    lines = []
    if caption or label:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        if font_size:
            lines.append("\\" + font_size)

    if tabcolsep is not None:
        lines.append(r"\setlength{\tabcolsep}{" + str(tabcolsep) + "}")
    if resizebox:
        # Eight columns of cell-type names overrun \textwidth even in
        # landscape; scaling the box is the least disruptive fix for a
        # supplementary table.
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(
        r"\begin{tabular}{" + (column_format or "ll" + "l" * len(groups)) + "}"
    )
    lines.append(r"\toprule")
    header = ["Tissue", "Tabula Sapiens"] + [
        _latex_escape(group_label(g)) for g in groups
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for tissue in tissues:
        row = [_latex_escape(tissue)]
        ref = indexed.loc[(tissue, groups[0]), "ref_dominant"]
        row.append(_latex_escape(ref))
        for group in groups:
            try:
                cell = indexed.loc[(tissue, group)]
            except KeyError:
                row.append("--")
                continue
            body = marks[cell["status"]] + _latex_escape(cell["pred_dominant"])
            if show_counts:
                body += f" ({int(cell['n_methods'])}/{int(cell['n_total'])})"
            row.append(f"\\cellcolor{{{colors[cell['status']]}}}{body}")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resizebox:
        lines.append("}")

    present = [s for s in DOMINANCE_STATUS_RANKED if s in set(detail["status"])]
    swatches = "; ".join(
        r"\colorbox{%s}{%s%s}"
        % (colors[s], marks[s].replace(r"\,", " "), DOMINANCE_LEGEND_TEXT[s])
        for s in present
    )
    lines.append(
        r"\par\smallskip{\footnotesize Cell type most configurations in each "
        r"group call dominant, with the number that agree. " + swatches + ".}"
    )

    if caption or label:
        if caption:
            lines.append(r"\caption{" + caption + "}")
        if label:
            lines.append(r"\label{" + label + "}")
        lines.append(r"\end{table}")
    return "\n".join(lines)


def dominance_accuracy(dominance):
    """Per method: fraction of tissues whose dominant cell type it recovers."""
    return (
        dominance.groupby("method")
        .agg(
            top1_accuracy=("matches_reference", "mean"),
            n_tissues=("matches_reference", "size"),
            n_wrong_but_expected=(
                "pred_dominant_expected",
                lambda s: int(
                    (s & ~dominance.loc[s.index, "matches_reference"]).sum()
                ),
            ),
        )
        .sort_values("top1_accuracy", ascending=False)
    )


# --------------------------------------------------------------------------
# 4. Within-set divergence -- reference-free proof that TCS is blind
# --------------------------------------------------------------------------


def within_set_divergence(
    deconv_df,
    expected_df,
    methods,
    metric="tvd",
    biosample_col="Biosample term name",
    file_col="file",
    celltype_col="CellType",
):
    """How differently two methods distribute mass *inside* the expected set.

    This is the central reference-free result. For each sample, each method's
    predictions are renormalised over that tissue's expected set, and the two
    renormalised distributions are compared. The reference supplies only the
    membership of the set -- exactly what TCS already assumes -- so the
    conclusion survives any criticism of Tabula Sapiens' proportions.

    Two methods with indistinguishable TCS and a large divergence are direct
    evidence that TCS cannot see within-tissue composition error.

    Parameters
    ----------
    metric : {'tvd', 'jsd'}
        Total-variation distance (half the L1 distance, in [0, 1]) or
        Jensen-Shannon divergence in bits.

    Returns
    -------
    pd.DataFrame
        One row per unordered method pair: divergence (tissue-weighted mean),
        delta_tcs (signed, method_a - method_b), abs_delta_tcs.
    """
    methods = list(methods)
    expected = _expected_lookup(expected_df)
    df = deconv_df[deconv_df[biosample_col].isin(expected)].copy()
    keep = [
        ct in expected[t] for ct, t in zip(df[celltype_col], df[biosample_col])
    ]
    df = df[keep]

    # Renormalise each method within each sample's expected set.
    totals = df.groupby([file_col, biosample_col])[methods].transform("sum")
    norm = df[methods].div(totals.replace(0, np.nan))
    norm[[file_col, biosample_col, celltype_col]] = df[
        [file_col, biosample_col, celltype_col]
    ]

    tcs = on_target_scores(
        deconv_df,
        expected_df,
        methods,
        biosample_col=biosample_col,
        file_col=file_col,
        celltype_col=celltype_col,
    )
    tcs_mean = tissue_weighted_mean(tcs, methods, biosample_col)

    rows = []
    for a, b in combinations(methods, 2):
        pair = norm[[file_col, biosample_col, a, b]].dropna()
        if pair.empty:
            continue
        if metric == "tvd":
            per_sample = (
                pair.assign(_d=(pair[a] - pair[b]).abs())
                .groupby([file_col, biosample_col])["_d"]
                .sum()
                .mul(0.5)
            )
        elif metric == "jsd":
            m = 0.5 * (pair[a] + pair[b])
            terms = _kl_terms(pair[a], m) + _kl_terms(pair[b], m)
            per_sample = (
                pair.assign(_d=0.5 * terms)
                .groupby([file_col, biosample_col])["_d"]
                .sum()
            )
        else:
            raise ValueError("metric must be 'tvd' or 'jsd'")

        divergence = per_sample.reset_index().groupby(biosample_col)["_d"].mean().mean()
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "divergence": float(divergence),
                "delta_tcs": float(tcs_mean[a] - tcs_mean[b]),
                "abs_delta_tcs": float(abs(tcs_mean[a] - tcs_mean[b])),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_delta_tcs")


def _kl_terms(p, q):
    """Elementwise p*log2(p/q), with the 0*log0 = 0 convention."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0) & (q > 0)
    out[mask] = p[mask] * np.log2(p[mask] / q[mask])
    return out


# --------------------------------------------------------------------------
# 4. Error tensor -- shared versus method-specific deviation
# --------------------------------------------------------------------------


def error_tensor(
    deconv_df,
    expected_df,
    reference,
    methods,
    biosample_col="Biosample term name",
    file_col="file",
    celltype_col="CellType",
    eps=1e-3,
):
    """Long frame of predicted vs reference proportion per method/tissue/cell type.

    The reference column is *not* a ground truth (module docstring); it is a
    fixed yardstick. Only differences that vary across methods are attributable
    to the methods -- see `shared_vs_specific`.

    Returns
    -------
    pd.DataFrame
        Columns: method, tissue, cell_type, pred, ref, diff, log2_ratio,
        expected, compartment, ubiquity.
    """
    methods = list(methods)
    tissues = set(expected_df["cfsort_tissue_name"])
    expected = _expected_lookup(expected_df)
    ub = ubiquity(expected_df)

    pred = (
        deconv_df[deconv_df[biosample_col].isin(tissues)]
        .groupby([biosample_col, celltype_col])[methods]
        .mean()
        .reset_index()
        .melt(
            id_vars=[biosample_col, celltype_col],
            var_name="method",
            value_name="pred",
        )
        .rename(columns={biosample_col: "tissue", celltype_col: "cell_type"})
    )

    ref = reference[reference["tissue"].isin(tissues)][
        ["tissue", "cell_type", "prop_mapped"]
    ].rename(columns={"prop_mapped": "ref"})

    out = pred.merge(ref, on=["tissue", "cell_type"], how="left")
    out["ref"] = out["ref"].fillna(0.0)
    out["diff"] = out["pred"] - out["ref"]
    out["log2_ratio"] = np.log2((out["pred"] + eps) / (out["ref"] + eps))
    out["expected"] = [
        ct in expected.get(t, set()) for ct, t in zip(out["cell_type"], out["tissue"])
    ]
    out["compartment"] = out["cell_type"].map(COMPARTMENTS)
    out["ubiquity"] = out["cell_type"].map(ub["ubiquity"]).fillna(0.0)
    return out


def shared_vs_specific(err, value="diff"):
    """Split deviation from the reference into shared and method-specific parts.

    For each (tissue, cell type), a deviation that every method shows cannot
    discriminate between methods and points at the reference, the atlas, or the
    data. A deviation that varies across methods is the methods' own.

    ``prior_direction_match`` is the falsifiable check: for cell types with a
    known scRNA-seq capture-bias direction (`PRIOR_CAPTURE_BIAS`), does the
    shared deviation point the way that bias predicts? A cell type the
    reference over-counts should be under-predicted by every method. Where the
    signs agree, the shared deviation is evidence about the reference; where
    they disagree, attribution moves to the atlas or the data.

    Returns
    -------
    pd.DataFrame
        One row per (tissue, cell_type): mean_across_methods, sd_across_methods,
        frac_positive, consensus (all methods share a sign), expected,
        compartment, ubiquity, prior_bias, prior_direction_match.
    """
    grouped = err.groupby(["tissue", "cell_type"])
    out = grouped.agg(
        mean_across_methods=(value, "mean"),
        sd_across_methods=(value, "std"),
        n_methods=(value, "size"),
        expected=("expected", "first"),
        compartment=("compartment", "first"),
        ubiquity=("ubiquity", "first"),
    ).reset_index()
    out["frac_positive"] = grouped[value].apply(lambda s: float((s > 0).mean())).values
    out["consensus"] = (out["frac_positive"] == 1.0) | (out["frac_positive"] == 0.0)

    out["prior_bias"] = out["cell_type"].map(PRIOR_CAPTURE_BIAS)
    # A reference that over-counts a type (+1) should leave every method
    # under-predicting it relative to that reference, i.e. mean deviation < 0.
    out["prior_direction_match"] = np.where(
        out["prior_bias"].isna(),
        np.nan,
        (np.sign(out["mean_across_methods"]) == -np.sign(out["prior_bias"])).astype(
            float
        ),
    )
    return out.sort_values("sd_across_methods", ascending=False)


def group_error_tensors(
    deconv_df,
    meta,
    expected_df,
    reference,
    groups=None,
    exclude_methods=(),
    filters=None,
    value="diff",
    **tensor_kwargs,
):
    """Per group, the error tensor and shared-vs-specific cells over ALL configs.

    Running the shared-vs-specific split on a handful of hand-picked winners
    conflates two questions: whether a deviation is shared across *methods*,
    and whether it is shared across *configurations of one method*. Grouping by
    labelling scheme separates them -- Hard Labels spans a 58-point TCS range
    across its 120 configurations, so "all Hard Labels configs agree" is a much
    weaker statement than "all labelling schemes agree", and the two should not
    be reported as if they were the same evidence.

    Parameters
    ----------
    groups : sequence of str, or None
        Group names to build tensors for; defaults to every group in `meta`.
    exclude_methods : sequence of str
        Methods to drop before grouping (e.g. UXM U250 and UXM U25 GSS Sorted
        from the baselines, which are different atlases rather than different
        deconvolution methods).
    filters : dict or None
        Per-group metadata filters; see `select_methods`.

    Returns
    -------
    dict
        group -> {'err': DataFrame, 'cells': DataFrame, 'n_methods': int,
        'variance': dict}.
    """
    groups = list(groups if groups is not None else meta["group"].unique())

    out = {}
    for group in groups:
        methods = select_methods(meta, group, filters, exclude_methods)
        methods = [m for m in methods if m in deconv_df.columns]
        if len(methods) < 2:
            continue
        err = error_tensor(
            deconv_df, expected_df, reference, methods, **tensor_kwargs
        )
        out[group] = {
            "err": err,
            "cells": shared_vs_specific(err, value=value),
            "n_methods": len(methods),
            "variance": variance_decomposition(err, value=value),
        }
    return out


def variance_decomposition(err, value="diff"):
    """Fraction of deviation variance that is shared rather than method-specific.

    Fits the two-way decomposition ``x[m, c] = mu + a[c] + b[m] + resid`` over
    (tissue, cell type) cells `c` and methods `m`. ``cell_effect`` is the share
    of variance explained by the cell being off for everyone -- the part no
    method choice can fix.

    ``method_effect`` is identically zero when `value` is ``'diff'`` and every
    tissue is complete: predictions and reference both sum to 1 over cell
    types, so each method's mean deviation over all cells is exactly 0 and the
    method main effect vanishes by construction. Read a 0.000 there as an
    algebraic identity, not as evidence that methods do not differ -- the
    method-specific signal lives entirely in ``interaction``.

    Returns
    -------
    dict
        cell_effect, method_effect, interaction (fractions summing to 1),
        n_cells, n_methods.
    """
    wide = err.pivot_table(
        index=["tissue", "cell_type"], columns="method", values=value
    ).dropna()
    x = wide.to_numpy()
    grand = x.mean()
    cell = x.mean(axis=1, keepdims=True) - grand
    method = x.mean(axis=0, keepdims=True) - grand
    resid = x - grand - cell - method
    total = ((x - grand) ** 2).sum()
    if total == 0:
        return {"cell_effect": np.nan, "method_effect": np.nan, "interaction": np.nan}
    return {
        "cell_effect": float((cell**2).sum() * x.shape[1] / total),
        "method_effect": float((method**2).sum() * x.shape[0] / total),
        "interaction": float((resid**2).sum() / total),
        "n_cells": int(x.shape[0]),
        "n_methods": int(x.shape[1]),
    }


# --------------------------------------------------------------------------
# 5. Compartment-conditional comparison
# --------------------------------------------------------------------------


def compartment_profile(err):
    """Compare within compartments; report the between-compartment split apart.

    Tabula Sapiens 2.0 balanced cell types between the four compartments by
    MACS enrichment in about two thirds of organs, and fixed the CD45+:CD45-
    ratio by FACS in others. Compartment *totals* are therefore an experimental
    design choice and are reported here only as ``between``, flagged
    reference-limited and not to be read as accuracy. Relative composition
    *within* a compartment is not renormalised by that design and is the part
    that retains biological signal.

    Returns
    -------
    within : pd.DataFrame
        One row per (method, tissue, compartment) with the total-variation
        distance between predicted and reference composition renormalised
        inside that compartment, plus the number of cell types compared.
        Compartments with fewer than two cell types are dropped -- there is no
        composition to compare.
    between : pd.DataFrame
        One row per (method, tissue, compartment) with predicted and reference
        compartment totals. Reference-limited: interpret across methods only.
    """
    err = err[err["compartment"].notna()].copy()

    totals = err.groupby(["method", "tissue", "compartment"])[["pred", "ref"]].sum()
    between = totals.reset_index().rename(
        columns={"pred": "pred_total", "ref": "ref_total"}
    )
    between["reference_limited"] = True

    joined = err.merge(
        totals.rename(columns={"pred": "_pred_tot", "ref": "_ref_tot"}),
        on=["method", "tissue", "compartment"],
    )
    joined["pred_within"] = joined["pred"] / joined["_pred_tot"].replace(0, np.nan)
    joined["ref_within"] = joined["ref"] / joined["_ref_tot"].replace(0, np.nan)
    joined = joined.dropna(subset=["pred_within", "ref_within"])

    within = (
        joined.assign(_d=(joined["pred_within"] - joined["ref_within"]).abs())
        .groupby(["method", "tissue", "compartment"])
        .agg(tvd=("_d", lambda s: 0.5 * s.sum()), n_cell_types=("_d", "size"))
        .reset_index()
    )
    within = within[within["n_cell_types"] >= 2]
    within["has_uncertain_assignment"] = within["compartment"].map(
        lambda comp: any(
            COMPARTMENTS.get(ct) == comp for ct in UNCERTAIN_COMPARTMENTS
        )
    )
    return within, between


# --------------------------------------------------------------------------
# 6. Threshold sensitivity -- is the 1% cutoff doing the work?
# --------------------------------------------------------------------------


def threshold_sensitivity(
    deconv_df,
    tissue_dict,
    methods,
    thresholds=(0.0, 0.005, 0.01, 0.02, 0.05),
    levels=("loyfer_counts_strict", "loyfer_counts_manual_legacy"),
    min_mapping_rate=0.7,
    biosample_col="Biosample term name",
    file_col="file",
    celltype_col="CellType",
):
    """Method TCS and rank across expected-set thresholds and mapping levels.

    The 1% threshold and the choice of ``loyfer_counts_*`` level are analyst
    choices. If the method ordering is stable across them, neither is doing the
    work; if it is not, the ranking is an artefact and must be reported as one.

    Returns
    -------
    pd.DataFrame
        Columns: level, threshold, n_tissues, method, tcs, rank.
    """
    methods = list(methods)
    rows = []
    for level in levels:
        ref = reference_composition(tissue_dict, level=level)
        for threshold in thresholds:
            exp = expected_sets(ref, threshold, min_mapping_rate=min_mapping_rate)
            if exp.empty:
                continue
            scores = on_target_scores(
                deconv_df,
                exp,
                methods,
                biosample_col=biosample_col,
                file_col=file_col,
                celltype_col=celltype_col,
            )
            mean = tissue_weighted_mean(scores, methods, biosample_col)
            ranks = mean.rank(ascending=False)
            for method in methods:
                rows.append(
                    {
                        "level": level,
                        "threshold": threshold,
                        "n_tissues": len(exp),
                        "method": method,
                        "tcs": float(mean[method]),
                        "rank": float(ranks[method]),
                    }
                )
    return pd.DataFrame(rows)


def rank_stability(sensitivity, reference_threshold=0.01, reference_level=None):
    """Spearman correlation of each setting's ranking against a reference setting."""
    from scipy.stats import spearmanr

    if reference_level is None:
        reference_level = sensitivity["level"].iloc[0]
    base = sensitivity[
        (sensitivity["threshold"] == reference_threshold)
        & (sensitivity["level"] == reference_level)
    ].set_index("method")["tcs"]

    rows = []
    for (level, threshold), grp in sensitivity.groupby(["level", "threshold"]):
        other = grp.set_index("method")["tcs"].reindex(base.index)
        rho, p = spearmanr(base.to_numpy(), other.to_numpy())
        rows.append(
            {"level": level, "threshold": threshold, "spearman": rho, "pvalue": p}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=FONT["tick"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return ax


def tissue_blind_profile(
    predictors, blind_summary, reference, expected_df, min_tissues=14
):
    """Put the constant predictors on the same axes as `ubiquity_profile`.

    A constant predictor has both coordinates the scatter uses: its TCS, and
    the mass it places on ubiquitous cell types. Plotting them as points rather
    than as horizontal rules shows *why* they score what they score -- the
    ceiling reaches 94% precisely by putting everything on a ubiquitous type,
    which a horizontal line cannot say.

    Returns
    -------
    pd.DataFrame
        Index predictor name; columns tcs, ubiquitous_mass,
        ubiquitous_mass_reference, ubiquitous_excess -- named to match
        `ubiquity_profile` so the same plotting code handles both.
    """
    ubiq = ubiquitous_types(expected_df, min_tissues=min_tissues)
    tissues = set(expected_df["cfsort_tissue_name"])
    ref_ubiq = (
        reference[
            reference["tissue"].isin(tissues) & reference["cell_type"].isin(ubiq)
        ]
        .groupby("tissue")["prop_mapped"]
        .sum()
        .mean()
    )
    columns = [c for c in predictors.columns if c in ubiq]
    out = pd.DataFrame(
        {
            "tcs": blind_summary.reindex(predictors.index),
            "ubiquitous_mass": predictors[columns].sum(axis=1),
        }
    )
    out["ubiquitous_mass_reference"] = ref_ubiq
    out["ubiquitous_excess"] = out["ubiquitous_mass"] - ref_ubiq
    out.attrs.update(getattr(predictors, "attrs", {}))
    return out


#: Distinct shapes for the constant predictors, so the legend identifies each
#: without a label beside the point.
BLIND_MARKERS = {"uniform_all": "o", "constant_optimum": "D"}

#: How the constant predictors are named on a figure.
BLIND_DISPLAY_NAMES = {
    "uniform_all": "Uniform prediction",
    "uniform_expected_union": "uniform over expected types",
    "ubiquity_weighted": "mass by ubiquity",
    "mean_reference": "mean reference composition",
    # Wrapped: the legend sits beside the plot, so its width is the figure's.
    "constant_optimum": "Predict {cell_type}\neverywhere",
}


def blind_display_name(predictor, attrs=None):
    """Readable label for a constant predictor, filling in its cell type."""
    text = BLIND_DISPLAY_NAMES.get(predictor, predictor)
    cell_type = (attrs or {}).get("constant_optimum_cell_type", "one cell type")
    return text.format(cell_type=cell_type)


def _reference_caption(ax, profile, rotate=True):
    """Name the vertical reference line, set along it rather than across it."""
    text = "Reference level"
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        0.0, 0.03, text,
        transform=trans,
        fontsize=FONT["note"],
        color=PALETTE["muted"],
        ha="right" if rotate else "center",
        va="bottom",
        rotation=90 if rotate else 0,
        clip_on=False,
    )


def _blind_line(ax, blind_summary, predictor, kind, annotate=True):
    """Draw one input-independent reference level on `ax`.

    Both lines share a hue and differ only in dash, because they are two
    members of one family -- predictions that never look at the data -- rather
    than two unrelated thresholds.

    Returns the level in percent, or None if the predictor is absent.
    """
    if blind_summary is None or predictor not in getattr(
        blind_summary, "index", []
    ):
        return None

    level = float(blind_summary[predictor]) * 100
    if kind == "ceiling":
        cell_type = blind_summary.attrs.get("constant_optimum_cell_type")
        what = f"{cell_type} everywhere" if cell_type else "one cell type everywhere"
        text = f"predict {what} ({level:.1f}%)"
        style, width = (0, (5, 2)), 1.6
    else:
        text = f"uniform over all cell types ({level:.1f}%)"
        style, width = (0, (2, 2)), 1.4

    ax.axhline(
        level, color=PALETTE["series_3"], linewidth=width, linestyle=style, zorder=2
    )
    if annotate:
        # Anchored at the x = 0 reference line and running leftwards, so the
        # text never lies across it.
        ax.annotate(
            text,
            (min(0.0, ax.get_xlim()[1]), level),
            textcoords="offset points",
            xytext=(-5, 5),
            fontsize=FONT["note"],
            color=PALETTE["muted"],
            ha="right",
        )
    return level


#: Marker shapes carry group identity on the ubiquity scatter. Shape is a
#: secondary encoding, so it can run past the palette's three all-pairs
#: categorical slots without the groups becoming indistinguishable.
#: No "*" here: at a given point size a star reads much smaller than the solid
#: shapes beside it, so a group landing on it looks like a stray tick.
GROUP_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p")


def plot_ubiquity_scatter(
    profile,
    meta=None,
    blind_summary=None,
    blind_predictor="uniform_all",
    ceiling_predictor="constant_optimum",
    ax=None,
    figsize=(6.6, 4.6),
    annotate=False,
    labels=None,
    legend_title="Syto best per\nlabelling scheme",
    legend_groups=None,
    xlim=None,
    ylim=None,
    label_safety=1.8,
):
    """TCS against excess mass on ubiquitous cell types.

    The x axis is a *component* of the y axis -- mass on ubiquitous types is
    on-target mass in nearly every tissue -- so this figure shows association,
    not attribution. Read it with `plot_ubiquity_arrows`, which removes the
    ubiquitous contribution and shows what survives.

    Group identity is carried by marker shape and a legend rather than by
    per-point annotation: with a dozen methods the direct labels collide, and
    the group is what the reader is comparing anyway.

    Parameters
    ----------
    profile : pd.DataFrame
        Output of `ubiquity_profile`.
    meta : pd.DataFrame or None
        Method metadata; supplies the group each method belongs to. Without it
        every method is drawn with one marker and no legend.
    blind_summary : pd.Series or None
        Output of `tissue_blind_scores`. Two reference lines are drawn from it.
    blind_predictor : str
        The uninformed constant predictor, drawn as the lower line. Defaults to
        ``uniform_all`` -- uniform over every atlas cell type. It is the only
        constant predictor constructible at test time, when the tissue and its
        composition are exactly what is unknown; ``ubiquity_weighted`` needs
        the expected sets and so could not actually be deployed as a default.
    ceiling_predictor : str or None
        The constant predictor that maximises TCS outright -- all mass on one
        cell type. Drawn as the upper line and labelled with that cell type.
        Pass None to omit it, though omitting it invites the reader to
        construct it themselves and wonder why it was left out.
    annotate : bool, str, or iterable
        Which points to label directly. ``False`` labels none, ``True`` labels
        all, a group name (or iterable of method names / group names) labels
        just those -- pass ``"Baseline"`` to name the baselines individually
        while the syto methods stay identified by marker and legend.
    legend_groups : iterable or None
        Groups to show in the legend. Defaults to the labelling schemes only:
        the baselines and the Cancer Detector priors are not labelling schemes
        and would be misread as one, so label those directly via `annotate`.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _style(ax)
    labels = labels or {}

    if meta is not None:
        groups = meta.loc[list(profile.index), "group"]
    else:
        groups = pd.Series("all", index=profile.index)
    order = list(dict.fromkeys(groups))

    if annotate is True:
        annotated_groups = set(order)
    elif not annotate:
        annotated_groups = set()
    else:
        wanted = {annotate} if isinstance(annotate, str) else set(annotate)
        annotated_groups = wanted & set(order)

    if legend_groups is None:
        # A directly annotated group names itself on the plot; repeating it in
        # the legend costs a row and tells the reader nothing new.
        legend_groups = [
            g for g in order
            if g not in NON_SCHEME_GROUPS and g not in annotated_groups
        ]
    legend_groups = set(legend_groups)

    for i, group in enumerate(order):
        members = [m for m in profile.index if groups[m] == group]
        is_baseline = (
            bool(meta.loc[members[0], "is_baseline"]) if meta is not None else False
        )
        ax.scatter(
            profile.loc[members, "ubiquitous_excess"] * 100,
            profile.loc[members, "tcs"] * 100,
            s=54,
            marker=GROUP_MARKERS[i % len(GROUP_MARKERS)],
            color=PALETTE["series_2"] if is_baseline else PALETTE["series_1"],
            zorder=3,
            edgecolor="#fcfcfb",
            linewidth=1.0,
            label=group_label(group) if group in legend_groups else "_nolegend_",
        )

    ax.margins(x=0.14, y=0.20)
    # Limits before labels: the de-collision reasons in normalised coordinates.
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if annotate is True:
        to_label = list(profile.index)
    elif not annotate:
        to_label = []
    else:
        to_label = [
            m for m in profile.index if m in wanted or groups[m] in wanted
        ]

    if to_label:
        halo = [pe.withStroke(linewidth=2.2, foreground="#fcfcfb")]
        # Stagger labels that sit at nearly the same height, and hang them off
        # a leader line so each stays attached to its own point.
        y0, y1 = ax.get_ylim()
        x0, x1 = ax.get_xlim()
        ygap = 0.04 * (y1 - y0)
        xgap = 0.30 * (x1 - x0)

        ordered = profile.loc[to_label].sort_values("tcs")
        rows, run = [], []
        for method, row in ordered.iterrows():
            y = row["tcs"] * 100
            if run and y - run[-1][1] >= ygap:
                rows.append(run)
                run = []
            run.append((method, y, row["ubiquitous_excess"] * 100))
        if run:
            rows.append(run)

        # Two points at the same height only need pulling apart if their labels
        # would actually meet -- labels run rightwards from the marker, so a
        # wide gap in x is separation enough and the leader line is just clutter.
        runs = []
        for band in rows:
            band = sorted(band, key=lambda item: item[2])
            group = [band[0]]
            for item in band[1:]:
                if item[2] - group[-1][2] < xgap:
                    group.append(item)
                else:
                    runs.append(group)
                    group = [item]
            runs.append(group)

        # Data units per inch, so a label's width can be compared against the
        # room left beside its point. A fixed fraction-of-axes rule breaks as
        # soon as the figure is rescaled to a fixed page width.
        axes_width_in = (
            ax.get_position().width * ax.get_figure().get_size_inches()[0]
        )
        per_inch = (x1 - x0) / max(axes_width_in, 1e-6)

        for run in runs:
            for i, (method, y, x) in enumerate(run):
                dy = (i - (len(run) - 1) / 2) * 10.0
                # Labels run rightwards from the marker, so one that would
                # not fit takes its label on the other side instead of being
                # clipped. 0.55 em per character estimates a proportional
                # face; `label_safety` covers the rest, since the figure is
                # rescaled to a fixed page width *after* this runs and text
                # then occupies more data units than it does here.
                text = labels.get(method, method)
                width = (
                    len(text) * 0.55 * FONT["annot"] / 72.0
                    * per_inch * label_safety
                )
                to_right = x + 0.14 * per_inch + width <= x1
                ax.annotate(
                    text,
                    (x, y),
                    textcoords="offset points",
                    xytext=(9 if to_right else -9, dy),
                    fontsize=FONT["annot"],
                    color=PALETTE["text"],
                    va="center",
                    ha="left" if to_right else "right",
                    path_effects=halo,
                    arrowprops=(
                        None
                        if dy == 0
                        else dict(
                            arrowstyle="-", color=PALETTE["muted"], linewidth=0.6,
                            shrinkA=0, shrinkB=3,
                        )
                    ),
                )

    ax.axvline(
        0.0, color=PALETTE["reference"], linewidth=1.4, linestyle="--", zorder=2
    )
    _reference_caption(ax, profile)

    # Two members of one family: predictions that ignore the input entirely.
    # Same hue, different dash, so they read as related rather than as two
    # unrelated thresholds.
    if blind_summary is not None:
        _blind_line(ax, blind_summary, blind_predictor, kind="floor")
        _blind_line(ax, blind_summary, ceiling_predictor, kind="ceiling")

    ax.set_xlabel(
        "Excess mass on ubiquitous cell types vs reference (pts)", fontsize=FONT["axis"]
    )
    ax.set_ylabel("Tissue Concordance Score (%)", fontsize=FONT["axis"])
    if meta is not None:
        ax.legend(
            frameon=False,
            fontsize=FONT["note"],
            title=legend_title,
            title_fontsize=FONT["legend"],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
    return ax


def plot_ubiquity_scatter_broken(
    profile,
    blind_points=None,
    meta=None,
    figsize=(7.6, 5.4),
    pad=2.0,
    label_room=0.30,
    band=0.20,
    band_pad=4.0,
    hspace=0.07,
    wspace=0.07,
    **kwargs,
):
    """The ubiquity scatter with both axes broken around the methods.

    The constant predictors are drawn as points, not as rules: each has both
    coordinates the scatter uses, and plotting them shows *why* they score what
    they score. The ceiling reaches 94% by placing all of its mass on a
    ubiquitous cell type -- it sits far to the right, +63 points of excess
    ubiquitous mass -- which a horizontal line cannot express. Uniform-over-all
    sits far to the left and low.

    Those positions are so far outside the methods' own range that a single
    pair of axes squashes the comparison to nothing, so the figure is a 3x3
    grid of panels: the methods occupy the centre, and each constant predictor
    its own corner band. Empty panels are genuinely empty regions of the
    coordinate space.

    Parameters
    ----------
    blind_points : pd.DataFrame or None
        Output of `tissue_blind_profile`, restricted to the predictors to
        show. With None this degenerates to a single unbroken panel.
    pad : float
        Headroom around the methods, in points, in the centre panel.
    label_room : float
        Extra width added to the right of the centre panel, as a fraction of
        its span, so direct labels have somewhere to go.
    band : float
        Size of each outer band relative to the centre panel.
    band_pad : float
        Half-width of an outer band around the point it holds.

    Returns
    -------
    (fig, axes)
        ``axes`` is the 3x3 array; ``axes[1][1]`` holds the methods.
    """
    import matplotlib.pyplot as plt

    if blind_points is None or blind_points.empty:
        ax = plot_ubiquity_scatter(profile, meta=meta, **kwargs)
        return ax.get_figure(), np.array([[None, None, None],
                                          [None, ax, None],
                                          [None, None, None]], dtype=object)

    xs = profile["ubiquitous_excess"] * 100
    ys = profile["tcs"] * 100
    xlo, xhi = float(xs.min()) - pad, float(xs.max()) + pad
    ylo, yhi = float(ys.min()) - pad, float(ys.max()) + pad
    # The x = 0 reference line has to stay inside the centre panel.
    xhi = max(xhi, pad)
    # Method labels run rightwards from their marker, and the figure is
    # rescaled to a fixed page width after they are placed -- so the room they
    # need cannot be predicted at draw time. Reserve it explicitly instead.
    xhi += label_room * (xhi - xlo)

    placed = {}
    for name, row in blind_points.iterrows():
        bx, by = float(row["ubiquitous_excess"]) * 100, float(row["tcs"]) * 100
        col = 0 if bx < xlo else (2 if bx > xhi else 1)
        r = 0 if by > yhi else (2 if by < ylo else 1)
        placed.setdefault((r, col), []).append((name, bx, by))

    used_rows = {1} | {r for r, _ in placed}
    used_cols = {1} | {c for _, c in placed}
    heights = [band if r != 1 else 1.0 for r in range(3)]
    widths = [band if c != 1 else 1.0 for c in range(3)]

    fig, axes = plt.subplots(
        3, 3, figsize=figsize, squeeze=False,
        gridspec_kw={"height_ratios": heights, "width_ratios": widths,
                     "hspace": hspace, "wspace": wspace},
    )

    def _limits(index, axis):
        """(lo, hi) for one band, from whatever point it holds."""
        vals = [
            (bx if axis == "x" else by)
            for (r, c), items in placed.items()
            for _, bx, by in items
            if (c if axis == "x" else r) == index
        ]
        if not vals:
            return (0.0, 1.0)
        return (min(vals) - band_pad, max(vals) + band_pad)

    xlims = {0: _limits(0, "x"), 1: (xlo, xhi), 2: _limits(2, "x")}
    ylims = {0: _limits(0, "y"), 1: (ylo, yhi), 2: _limits(2, "y")}

    ax_mid = axes[1][1]
    plot_ubiquity_scatter(
        profile, meta=meta, blind_summary=None, ax=ax_mid,
        xlim=xlims[1], ylim=ylims[1], **kwargs,
    )

    attrs = getattr(blind_points, "attrs", {})
    for r in range(3):
        for c in range(3):
            ax = axes[r][c]
            if r not in used_rows or c not in used_cols:
                ax.axis("off")
                continue
            if (r, c) != (1, 1):
                _style(ax)
                ax.set_xlim(*xlims[c])
                ax.set_ylim(*ylims[r])
                if xlims[c][0] <= 0.0 <= xlims[c][1]:
                    ax.axvline(0.0, color=PALETTE["reference"], linewidth=1.4,
                               linestyle="--", zorder=2)
            for name, bx, by in placed.get((r, c), []):
                # Named in the legend, not beside the point: these panels are
                # only a few millimetres wide and any label overflows them.
                ax.scatter(
                    bx, by, s=70,
                    marker=BLIND_MARKERS.get(name, "o"),
                    color=PALETTE["reference"],
                    edgecolor="#fcfcfb", linewidth=1.0, zorder=4,
                )
            if r != 1:
                ax.set_yticks([round(np.mean(ylims[r]), 0)])
            if c != 1:
                ax.set_xticks([round(np.mean(xlims[c]), 0)])
            ax.set_xlabel("")
            ax.set_ylabel("")

    # Ticks only on the outer edges, and hide the spines the breaks replace.
    for r in range(3):
        for c in range(3):
            ax = axes[r][c]
            if not ax.axison:
                continue
            if r != max(used_rows):
                ax.tick_params(labelbottom=False, bottom=False)
                ax.spines["bottom"].set_visible(False)
            if r != min(used_rows):
                ax.spines["top"].set_visible(False)
            if c != min(used_cols):
                ax.tick_params(labelleft=False, left=False)
                ax.spines["left"].set_visible(False)
            if c != max(used_cols):
                ax.spines["right"].set_visible(False)

    kw = dict(
        marker=[(-1, -0.6), (1, 0.6)], markersize=6, linestyle="none",
        color=PALETTE["muted"], mec=PALETTE["muted"], mew=1, clip_on=False,
    )
    rows_used = sorted(used_rows)
    cols_used = sorted(used_cols)
    for upper, lower in zip(rows_used[:-1], rows_used[1:]):
        for c in cols_used:
            axes[upper][c].plot([0, 1], [0, 0],
                                transform=axes[upper][c].transAxes, **kw)
            axes[lower][c].plot([0, 1], [1, 1],
                                transform=axes[lower][c].transAxes, **kw)
    for left, right in zip(cols_used[:-1], cols_used[1:]):
        for r in rows_used:
            axes[r][left].plot([1, 1], [0, 1],
                               transform=axes[r][left].transAxes, **kw)
            axes[r][right].plot([0, 0], [0, 1],
                                transform=axes[r][right].transAxes, **kw)

    handles, texts = ax_mid.get_legend_handles_labels()
    legend = ax_mid.get_legend()
    if legend:
        legend.remove()
    for name in blind_points.index:
        handles.append(
            plt.Line2D(
                [], [], marker=BLIND_MARKERS.get(name, "o"), linestyle="none",
                markersize=6, color=PALETTE["reference"],
                label=blind_display_name(name, attrs),
            )
        )
    right_edge = max(
        axes[r][c].get_position().x1
        for r in rows_used for c in cols_used if axes[r][c].axison
    )
    fig.legend(
        handles=handles, frameon=False, fontsize=FONT["note"],
        loc="center left", bbox_to_anchor=(right_edge + 0.015, 0.5),
        bbox_transform=fig.transFigure, alignment="left",
    )

    fig.supxlabel(
        "Excess mass on ubiquitous cell types vs reference (pts)",
        fontsize=FONT["axis"], color=PALETTE["text"],
    )
    fig.supylabel(
        "Tissue Concordance Score (%)", fontsize=FONT["axis"], color=PALETTE["text"]
    )
    return fig, axes


def plot_ubiquity_arrows(
    ladder,
    excluded,
    ax=None,
    figsize=(9.0, 5.0),
    labels=None,
    n_tissues=None,
    wrap=62,
    show_gains=False,
    min_gain_span=6.0,
    show_legend=True,
):
    """Per method, TCS as progressively more ubiquitous cell types are excluded.

    A long span means the method's score rests on cell types expected almost
    everywhere. This is the attribution figure that `plot_ubiquity_scatter`
    cannot provide on its own.

    The legend names the excluded cell types outright rather than calling them
    "ubiquitous": the reader cannot otherwise tell what a 14/16 threshold
    actually removed, and the identity of those three or four types is the
    whole substance of the failure mode.

    Parameters
    ----------
    ladder, excluded : as returned by `ubiquity_ladder`.
    n_tissues : int or None
        Total tissue count, so the legend can read "≥8/16 tissues" rather than
        the bare threshold.
    wrap : int
        Character width at which the legend entries wrap.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _style(ax)
    labels = labels or {}

    steps = [c for c in ladder.columns if c.startswith("tcs_excl_")]
    steps = sorted(steps, key=lambda c: -int(c.rsplit("_", 1)[1]))
    colours = [PALETTE["series_1"], PALETTE["series_2"], PALETTE["series_3"]]

    # Ordered by full TCS: the figure is about how far each method falls from
    # its headline score, which is only legible if the headline is the sort key.
    ordered = ladder.sort_values("tcs")
    y = np.arange(len(ordered))
    for i, (_, row) in enumerate(ordered.iterrows()):
        xs = [row["tcs"]] + [row[c] for c in steps]
        # A plain connector, run all the way from the axis: anchored at zero,
        # the segment's length reads as the score itself rather than as a
        # floating interval. The dots are ordered left to right by
        # construction, so an arrowhead adds ink without adding information.
        ax.plot(
            [0.0, xs[0] * 100],
            [i, i],
            color=PALETTE["muted"],
            linewidth=1.2,
            alpha=0.7,
            zorder=2,
            solid_capstyle="butt",
        )
        for j, x in enumerate(xs):
            ax.scatter(
                x * 100,
                i,
                s=38,
                color=colours[j % len(colours)],
                zorder=3,
                edgecolor="#fcfcfb",
                linewidth=1.0,
            )

        if show_gains:
            # Each span is what one tier of cell types is worth: the axis to
            # the most restricted score, then each relaxation in turn.
            bounds = [0.0] + [x * 100 for x in reversed(xs)]
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if hi - lo < min_gain_span:
                    continue
                ax.annotate(
                    f"+{hi - lo:.1f}",
                    ((lo + hi) / 2, i),
                    textcoords="offset points",
                    xytext=(0, 4),
                    fontsize=FONT["small"],
                    color=PALETTE["muted"],
                    ha="center",
                    va="bottom",
                )

    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(m, m) for m in ordered.index], fontsize=FONT["legend"])
    ax.set_xlabel("Tissue Concordance Score (%)", fontsize=FONT["axis"])
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlim(left=0.0)

    def _describe(k):
        # Always name the excluded cell types: "ubiquitous" is not checkable by
        # the reader, and the identity of these types is the whole point.
        names = excluded.get(k, [])
        denom = f"/{n_tissues}" if n_tissues else ""
        head = f"excluding {len(names)} types expected in ≥{k}{denom} tissues:  "
        return "\n".join(textwrap.wrap(head + ", ".join(names), width=wrap))

    if not show_legend:
        return ax

    ax.scatter([], [], s=38, color=colours[0], label="all expected cell types")
    for j, col in enumerate(steps):
        k = int(col.rsplit("_", 1)[1])
        ax.scatter(
            [], [], s=38, color=colours[(j + 1) % len(colours)], label=_describe(k)
        )
    ax.legend(
        frameon=False,
        fontsize=FONT["note"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=1,
        handletextpad=0.6,
        labelspacing=0.7,
        alignment="left",
    )
    return ax


def plot_tcs_vs_divergence(
    pairs, ax=None, figsize=(5.6, 4.4), label_top=6, labels=None
):
    """Within-set divergence against the TCS gap, one point per method pair.

    Points at x ~ 0 with high y are pairs TCS calls equivalent while they
    disagree substantially about composition inside the expected set.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _style(ax)

    ax.scatter(
        pairs["abs_delta_tcs"] * 100,
        pairs["divergence"] * 100,
        s=34,
        color=PALETTE["series_1"],
        alpha=0.85,
        edgecolor="#fcfcfb",
        linewidth=0.8,
        zorder=3,
    )

    ax.margins(x=0.14, y=0.12)

    # Label the pairs that make the point: small TCS gap, large divergence.
    blind = pairs.assign(
        _score=pairs["divergence"] / (pairs["abs_delta_tcs"] + 1e-4)
    ).nlargest(label_top, "_score")
    labels = labels or {}
    midpoint = np.mean(ax.get_xlim())
    for _, row in blind.iterrows():
        x = row["abs_delta_tcs"] * 100
        right = x < midpoint  # label points on the left half rightwards
        name_a = labels.get(row["method_a"], row["method_a"])
        name_b = labels.get(row["method_b"], row["method_b"])
        ax.annotate(
            f"{name_a} / {name_b}",
            (x, row["divergence"] * 100),
            textcoords="offset points",
            xytext=(6 if right else -6, 5),
            fontsize=FONT["annot"],
            color=PALETTE["text"],
            ha="left" if right else "right",
        )

    ax.set_xlabel("|TCS difference| (pts)", fontsize=FONT["axis"])
    ax.set_ylabel("within-expected-set divergence (TVD, %)", fontsize=FONT["axis"])
    return ax


def common_label_cells(group_results, label_top=4, min_common=3):
    """(tissue, cell type) pairs worth labelling in *every* panel of a grid.

    Labelling each panel's own top-`label_top` makes the panels individually
    informative but mutually incomparable: a cell named in one panel and not
    another may be absent, or may simply have ranked fifth. Labelling one
    shared set instead lets the reader track the same cells across schemes.

    The set is the intersection of the per-panel top-`label_top`. That can be
    small or empty, so if fewer than `min_common` survive it falls back to
    ranking cells by how many panels list them, breaking ties on summed
    absolute deviation.

    Returns
    -------
    list of (tissue, cell_type)
    """
    from collections import Counter

    tops, weight = [], Counter()
    for res in group_results.values():
        frame = res["cells"]
        ordered = frame.reindex(
            frame["mean_across_methods"].abs().sort_values(ascending=False).index
        ).head(label_top)
        keys = set(zip(ordered["tissue"], ordered["cell_type"]))
        tops.append(keys)
        for key, value in zip(
            zip(ordered["tissue"], ordered["cell_type"]),
            ordered["mean_across_methods"].abs(),
        ):
            weight[key] += float(value)

    if not tops:
        return []

    shared = set.intersection(*tops)
    if len(shared) >= min_common:
        return sorted(shared, key=lambda k: -weight[k])

    counts = Counter(key for panel in tops for key in panel)
    ranked = sorted(counts, key=lambda k: (-counts[k], -weight[k]))
    return ranked[:min_common]


#: Text sizes, in points, for every figure this module draws. One place to
#: change so that figures land on the page with consistent text -- see
#: `set_font_scale`, and the width-matching in the runner: a figure saved wider
#: than the text block is scaled down by LaTeX, shrinking its text with it, so
#: uniform on-page text needs uniform saved widths.
_FONT_BASE = {
    "axis": 10.0,     # axis titles
    "title": 10.0,    # panel titles
    "tick": 9.0,      # tick labels
    "legend": 9.0,    # legend entries
    "note": 8.5,      # reference-line captions and figure notes
    "annot": 8.0,     # point labels
    "small": 7.5,     # dense in-bar / in-cell text
    "tiny": 6.5,      # heat-map tick labels
}

FONT = dict(_FONT_BASE)


def set_font_scale(scale=1.0):
    """Scale every text size at once, relative to the module defaults."""
    FONT.update({k: v * scale for k, v in _FONT_BASE.items()})
    return FONT


#: Target width, in inches, for every saved figure. Matching the text block of
#: the document means LaTeX includes each figure at scale 1, so a point in the
#: figure is a point on the page.
FIGURE_WIDTH = 6.5


#: Marker shapes for individually identified cells. All differ from the "o"
#: used by the ordinary points, so a highlighted cell reads as highlighted
#: before its shape is decoded.
HIGHLIGHT_MARKERS = ("s", "^", "D", "v", "P", "X", "*", "p")


def extreme_label_cells(group_results, per_side=2, always_include=()):
    """The union of each group's most over- and under-predicted cells.

    Taking `per_side` from *each* side of the reference, rather than the top
    few by absolute deviation, keeps the two directions represented: over- and
    under-prediction have different causes here, and a pure ranking by
    magnitude can silently return only one of them.

    The union rather than the intersection, so a cell that is extreme for one
    group and unremarkable for the others still appears -- that contrast is
    itself worth seeing.

    Parameters
    ----------
    always_include : iterable of (tissue, cell_type)
        Cells added regardless of rank, appended after the ranked ones. Size of
        deviation is not the only reason a cell matters: a moderate deviation
        that flips which cell type a tissue is called for is invisible to this
        ranking but not to the reader.

    Returns
    -------
    list of (tissue, cell_type)
        Ordered by how many groups list the cell, then by mean deviation, with
        `always_include` entries last.
    """
    from collections import Counter

    counts, weight = Counter(), {}
    for res in group_results.values():
        frame = res["cells"].sort_values("mean_across_methods")
        picks = pd.concat([frame.head(per_side), frame.tail(per_side)])
        for tissue, cell_type, mean in zip(
            picks["tissue"], picks["cell_type"], picks["mean_across_methods"]
        ):
            key = (tissue, cell_type)
            counts[key] += 1
            weight[key] = float(mean)

    ranked = sorted(counts, key=lambda k: (-counts[k], -abs(weight[k])))
    for key in always_include:
        key = tuple(key)
        if key not in counts:
            ranked.append(key)
    return ranked


def highlight_handles(cells, highlight_cells, markers=HIGHLIGHT_MARKERS):
    """Legend handles naming each highlighted cell, in its own marker shape.

    Identifying the cells in the legend rather than on the plot keeps the
    clouds unobscured; with only a handful of cells, a shape apiece is
    unambiguous and needs no leader lines or index numbers.
    """
    import matplotlib.pyplot as plt

    lookup = cells.set_index(["tissue", "cell_type"])
    handles = []
    for i, key in enumerate(highlight_cells):
        key = tuple(key)
        try:
            expected = bool(lookup.loc[key, "expected"])
        except KeyError:
            continue
        handles.append(
            plt.Line2D(
                [], [],
                marker=markers[i % len(markers)],
                linestyle="none",
                markersize=6,
                color=PALETTE["series_1"] if expected else PALETTE["series_2"],
                markeredgecolor=PALETTE["text"],
                markeredgewidth=0.8,
                label=f"{key[1]} / {key[0]}",
            )
        )
    return handles


def plot_shared_vs_specific(
    cells,
    ax=None,
    figsize=(5.6, 4.4),
    label_top=6,
    xlim=None,
    ylim=None,
    label_cells=None,
    highlight_cells=None,
):
    """Mean deviation across methods against its spread, per (tissue, cell type).

    Bottom-right (large |mean|, small spread) is shared by every method and so
    attributable to the reference, the atlas, or the data. Upward is
    method-specific.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _style(ax)

    expected = cells["expected"].fillna(False).astype(bool)
    highlight = [tuple(key) for key in (highlight_cells or [])]
    highlighted = pd.Series(
        [
            (tissue, cell_type) in set(highlight)
            for tissue, cell_type in zip(cells["tissue"], cells["cell_type"])
        ],
        index=cells.index,
    )

    for mask, colour, label in (
        (expected & ~highlighted, PALETTE["series_1"], "expected"),
        (~expected & ~highlighted, PALETTE["series_2"], "not expected"),
    ):
        ax.scatter(
            cells.loc[mask, "mean_across_methods"] * 100,
            cells.loc[mask, "sd_across_methods"] * 100,
            s=26,
            color=colour,
            alpha=0.8,
            # No surface-coloured ring here: the cloud is dense enough that
            # overlapping rings merge into a white blob and eat the marks.
            edgecolor="none",
            zorder=3,
            label=label,
        )

    # Highlighted cells sit on top in their own shapes, with a dark edge so
    # they separate from the cloud without needing a fourth colour.
    lookup = cells.set_index(["tissue", "cell_type"])
    for i, key in enumerate(highlight):
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        ax.scatter(
            float(row["mean_across_methods"]) * 100,
            float(row["sd_across_methods"]) * 100,
            s=72,
            marker=HIGHLIGHT_MARKERS[i % len(HIGHLIGHT_MARKERS)],
            color=PALETTE["series_1"] if row["expected"] else PALETTE["series_2"],
            edgecolor=PALETTE["text"],
            linewidth=0.8,
            zorder=5,
        )

    ax.margins(x=0.16)
    # Limits must be final before labels are placed: the de-collision works in
    # normalised axes coordinates, so rescaling afterwards would invalidate it.
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if label_cells is None:
        candidates = cells
        # Highlighted cells are named in the legend, so on-plot text would
        # only repeat it over the top of the cloud.
        cap = 0 if highlight else label_top
    else:
        wanted = {tuple(key) for key in label_cells}
        candidates = cells[
            [
                (tissue, cell_type) in wanted
                for tissue, cell_type in zip(cells["tissue"], cells["cell_type"])
            ]
        ]
        cap = len(candidates)
    candidates = candidates.reindex(
        candidates["mean_across_methods"].abs().sort_values(ascending=False).index
    )

    halo = [pe.withStroke(linewidth=2.2, foreground="#fcfcfb")]
    placed, n_placed = [], 0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # An explicit label set exists so the same cells can be compared across
    # panels, so every one of them has to appear: crowded labels are nudged
    # rather than dropped. Without one, dropping is fine -- the next-ranked
    # cell is no less valid a choice.
    enforce_all = label_cells is not None
    height_pts = (
        ax.get_figure().get_size_inches()[1] * ax.get_position().height * 72.0
    )

    for _, row in candidates.iterrows():
        if n_placed >= cap:
            break
        x = row["mean_across_methods"] * 100
        y = row["sd_across_methods"] * 100
        # Greedy de-collision in normalised axes space. Fixed offsets are not
        # enough here -- the extremes cluster along the low-spread edge.
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        collides = any(
            abs(fx - px) < 0.36 and abs(fy - py) < 0.075 for px, py in placed
        )
        if collides and not enforce_all:
            continue
        dy = 3.0
        if collides:
            dy += 12.0 if n_placed % 2 else -12.0
        placed.append((fx, fy + (dy - 3.0) / max(height_pts, 1.0)))
        n_placed += 1
        # Push away from the origin, where the cloud is densest -- but flip
        # back inward near either edge, or the label runs off the axes.
        right = x > 0
        if fx < 0.28:
            right = True
        elif fx > 0.72:
            right = False
        ax.annotate(
            f"{row['cell_type']} / {row['tissue']}",
            (x, y),
            textcoords="offset points",
            xytext=(7 if right else -7, dy),
            fontsize=FONT["annot"],
            color=PALETTE["text"],
            ha="left" if right else "right",
            va="center",
            path_effects=halo,
        )

    ax.axvline(0.0, color=PALETTE["reference"], linewidth=1.0, zorder=2)
    ax.set_xlabel("mean predicted - reference across methods (pts)", fontsize=FONT["axis"])
    ax.set_ylabel("spread across methods (SD, pts)", fontsize=FONT["axis"])
    ax.legend(frameon=False, fontsize=FONT["legend"])
    return ax


def plot_shared_vs_specific_grid(
    group_results,
    ncols=2,
    figsize=None,
    label_top=5,
    order=None,
    label_cells=None,
    highlight_cells=None,
):
    """One shared-vs-specific panel per group, on shared axes.

    Shared axes are the point: a group whose cloud is tall relative to the
    others disagrees internally more than the others do, and that is only
    visible if the panels are directly comparable.

    Parameters
    ----------
    group_results : dict
        Output of `group_error_tensors`.
    label_top : int
        Labels per panel when `label_cells` is None -- each panel names its own
        most extreme cells, which reads well panel by panel but leaves the
        panels mutually incomparable.
    label_cells : iterable of (tissue, cell_type), or None
        Label this one set in every panel instead, so the same cells can be
        tracked across groups. See `common_label_cells`.
    """
    import matplotlib.pyplot as plt

    keys = list(order) if order else order_groups(group_results)
    nrows = int(np.ceil(len(keys) / ncols))
    ncols = min(ncols, len(keys))
    figsize = figsize or (4.4 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    xlim = max(
        abs(group_results[k]["cells"]["mean_across_methods"]).max() for k in keys
    ) * 105
    ylim = max(
        group_results[k]["cells"]["sd_across_methods"].max() for k in keys
    ) * 105

    for i, key in enumerate(keys):
        ax = axes[i // ncols][i % ncols]
        res = group_results[key]
        plot_shared_vs_specific(
            res["cells"],
            ax=ax,
            label_top=label_top,
            xlim=(-xlim, xlim),
            ylim=(-0.02 * ylim, ylim),
            label_cells=label_cells,
            highlight_cells=highlight_cells,
        )
        shared = res["variance"].get("cell_effect", float("nan"))
        ax.set_title(
            f"{group_full_label(key)}  (n={res['n_methods']}, shared {shared:.0%})",
            fontsize=FONT["axis"],
            color=PALETTE["text"],
        )
        ax.get_legend().remove() if ax.get_legend() else None
        if i % ncols:
            ax.set_ylabel("")
        # One shared x title below the whole grid, not one per column.
        ax.set_xlabel("")

    for j in range(len(keys), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel(
        "mean predicted - reference across methods (pts)",
        fontsize=FONT["axis"],
        color=PALETTE["text"],
    )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=PALETTE["series_1"],
                   label="expected"),
        plt.Line2D([], [], marker="o", linestyle="", color=PALETTE["series_2"],
                   label="not expected"),
    ]
    ncol = 2
    if highlight_cells:
        handles += highlight_handles(
            group_results[keys[0]]["cells"], highlight_cells
        )
        ncol = 3
    fig.legend(
        handles=handles, frameon=False, fontsize=FONT["legend"], ncol=ncol,
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    return fig, axes


def plot_dominance(
    consensus, ax=None, figsize=(7.0, 4.6), min_fraction=0.0, title=None
):
    """Per tissue, which dominant cell type the methods call, and how often.

    Segments are coloured by `DOMINANCE_STATUS`, three states rather than two:
    a call can be right, wrong in a way TCS cannot see (the wrong cell type is
    still inside the expected set), or wrong in a way it can. The middle state
    is the one the figure exists to show, so it gets its own colour rather than
    a footnote marker.

    Parameters
    ----------
    title : str or None
        Panel title -- use it to say which methods are in the panel, since the
        baselines and the syto schemes are worth plotting separately.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _style(ax)

    sub = consensus[consensus["frac_methods"] >= min_fraction].copy()
    tissues = sorted(sub["tissue"].unique(), reverse=True)
    ypos = {t: i for i, t in enumerate(tissues)}
    status_colour = {
        DOMINANCE_STATUS[0]: PALETTE["series_3"],
        DOMINANCE_STATUS[1]: PALETTE["series_1"],
        DOMINANCE_STATUS[2]: PALETTE["series_2"],
    }

    # A label is drawn only when it fits inside its own segment: a name that
    # overruns spills past the neighbouring segment or off the axes, and then
    # reads as belonging to the wrong cell type.
    fig_width_in = ax.get_figure().get_size_inches()[0] * ax.get_position().width
    char_width = 6.5 / 72 * 0.55 * (100 / max(fig_width_in, 1e-6))

    # Stack segments per tissue in descending width, so the reference call (if
    # present) is easy to find and the labels have room.
    for tissue in tissues:
        grp = sub[sub["tissue"] == tissue].sort_values("frac_methods", ascending=False)
        left = 0.0
        for _, row in grp.iterrows():
            width = row["frac_methods"] * 100
            ax.barh(
                ypos[tissue], width, left=left, height=0.62,
                color=status_colour[dominance_status(row)],
                edgecolor="#fcfcfb", linewidth=1.4, zorder=3,
            )
            name = row["pred_dominant"]
            if width >= len(name) * char_width:
                ax.annotate(
                    name,
                    (left + width / 2, ypos[tissue]),
                    ha="center", va="center", fontsize=FONT["small"],
                    color="#ffffff", zorder=4,
                )
            left += width

    ax.set_yticks(range(len(tissues)))
    ax.set_yticklabels(tissues, fontsize=FONT["legend"])
    ax.set_xlabel("share of methods calling this cell type dominant (%)", fontsize=FONT["axis"])
    ax.set_xlim(0, 100)
    if title:
        ax.set_title(title, fontsize=FONT["axis"], color=PALETTE["text"])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=status_colour[s], label=s)
        for s in DOMINANCE_STATUS
    ]
    ax.legend(
        handles=handles, frameon=False, fontsize=FONT["note"], ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, 1.06 if title else 1.01),
    )
    return ax


def save_dominance_plots(
    dominance_groups, outdir, ext="pdf", figsize=(7.0, 4.6), dpi=200, **kwargs
):
    """One untitled dominance figure per group, written to `outdir`.

    Titles are omitted deliberately: these are meant to be arranged as subplots
    in the paper, where a per-panel title would duplicate the caption.

    Returns
    -------
    dict
        group -> written path.
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = {}
    for group, res in dominance_groups.items():
        slug = "".join(
            c if c.isalnum() else "_" for c in group_label(group).lower()
        ).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        path = outdir / f"dominance_{slug}.{ext}"

        fig, ax = plt.subplots(figsize=figsize)
        plot_dominance(res["consensus"], ax=ax, title=None, **kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written[group] = path
    return written


def dominance_breakdown(dominance, meta, tissue, by=("classifier", "deconvolver", "calibrator")):
    """What distinguishes the methods that disagree about a tissue's dominant type.

    When a group splits -- 40 of 60 Soft Labels configurations calling
    Adipocytes dominant in adipose tissue and 20 calling something else -- the
    question is whether the dissenters share a component. This crosstabs the
    dominant call against each configuration field so the answer is visible
    rather than guessed.

    Returns
    -------
    dict
        field -> DataFrame (rows: predicted dominant type, cols: field value).
    """
    sub = dominance[dominance["tissue"] == tissue].copy()
    for field in by:
        sub[field] = meta.loc[sub["method"], field].to_numpy()
    return {
        field: pd.crosstab(sub["pred_dominant"], sub[field]) for field in by
    }


def plot_logratio_grid(
    err, methods=None, vmax=3.0, figsize=None, cell_types=None, ncols=4
):
    """Small multiples of log2(predicted / reference), tissues x cell types.

    A column that is the same colour in every panel is a deviation no method
    escapes; panel-to-panel differences are method-specific. Reference-limited
    on the between-compartment axis -- read `compartment_profile` alongside.

    Cells where the reference is exactly zero are drawn in neutral gray, not at
    the top of the ramp. "This cell type is absent from the tissue's reference"
    is a statement about the mapping, not a measured over-prediction ratio, and
    colouring it maximal red would let the mapping's coverage masquerade as
    method error across the whole panel.
    """
    import matplotlib.pyplot as plt

    methods = list(methods or err["method"].unique())
    if cell_types is None:
        # Keep cell types that matter somewhere: expected, or given real mass.
        keep = err.groupby("cell_type").agg(
            any_expected=("expected", "any"), max_pred=("pred", "max")
        )
        cell_types = sorted(
            keep.index[keep["any_expected"] | (keep["max_pred"] >= 0.02)]
        )

    nrows = int(np.ceil(len(methods) / ncols))
    ncols = min(ncols, len(methods))
    figsize = figsize or (3.4 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    cmap = diverging_cmap()
    cmap.set_bad(PALETTE["grid"])

    tissues = sorted(err["tissue"].unique())
    absent = (
        err.pivot_table(index="tissue", columns="cell_type", values="ref", aggfunc="max")
        .reindex(index=tissues, columns=cell_types)
        .fillna(0.0)
        .to_numpy()
        <= 0
    )
    for i, method in enumerate(methods):
        ax = axes[i // ncols][i % ncols]
        mat = (
            err[err["method"] == method]
            .pivot_table(index="tissue", columns="cell_type", values="log2_ratio")
            .reindex(index=tissues, columns=cell_types)
            .to_numpy()
        )
        im = ax.imshow(
            np.where(absent, np.nan, mat),
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(method, fontsize=FONT["legend"], color=PALETTE["text"])
        # Only the lowest panel in each column carries x labels -- the grid is
        # ragged, so "lowest" is per column, not simply the last row.
        bottom_of_column = i + ncols >= len(methods)
        ax.set_xticks(range(len(cell_types)))
        ax.set_xticklabels(
            cell_types if bottom_of_column else [], rotation=90, fontsize=FONT["tiny"]
        )
        ax.set_yticks(range(len(tissues)))
        ax.set_yticklabels(tissues if i % ncols == 0 else [], fontsize=FONT["tiny"])
        ax.tick_params(length=0, colors=PALETTE["muted"])

    for j in range(len(methods), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.subplots_adjust(hspace=0.18)

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("log2(predicted / reference)", fontsize=FONT["legend"])
    cbar.ax.tick_params(labelsize=FONT["tick"], colors=PALETTE["muted"])
    cbar.ax.annotate(
        "gray = absent\nfrom reference",
        (0.5, -0.06),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=FONT["annot"],
        color=PALETTE["muted"],
    )
    return fig, axes


def plot_threshold_stability(
    sensitivity,
    ax=None,
    figsize=(6.0, 4.0),
    level=None,
    labels=None,
    max_methods=8,
    reference_threshold=0.01,
    show_dropped_note=False,
    xpad_left=0.4,
    xpad_right=0.15,
):
    """Method rank against the expected-set threshold -- a bump chart.

    Flat lines mean the 1% cutoff is not driving the ranking.

    The categorical palette has eight slots and hues are never cycled, so at
    most `max_methods` series are drawn: the best-ranked at
    `reference_threshold`. Facet the remainder rather than raising the cap.
    Set `show_dropped_note` to title the panel with how many were left out.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _style(ax)

    labels = labels or {}
    level = level or sensitivity["level"].iloc[0]
    sub = sensitivity[sensitivity["level"] == level]

    order = (
        sub[sub["threshold"] == reference_threshold]
        .sort_values("rank")["method"]
        .tolist()
    )
    order = order or sorted(sub["method"].unique())
    dropped = max(0, len(order) - max_methods)
    order = order[:max_methods]
    sub = sub[sub["method"].isin(order)]

    # Categorical slots in fixed order, assigned per entity and never cycled.
    slots = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]
    for i, method in enumerate(order):
        grp = sub[sub["method"] == method].sort_values("threshold")
        ax.plot(
            grp["threshold"] * 100,
            grp["rank"],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            color=slots[i],
            label=labels.get(method, method),
        )
        last = grp.iloc[-1]
        ax.annotate(
            labels.get(method, method),
            (last["threshold"] * 100, last["rank"]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=FONT["annot"],
            color=PALETTE["text"],
            va="center",
        )

    ax.invert_yaxis()
    ax.set_xlabel("Expected-set threshold (%)", fontsize=FONT["axis"])
    ax.set_ylabel("Rank by TCS (1 = best)", fontsize=FONT["axis"])
    # A negative threshold is not a thing, so do not let the axis imply one,
    # and stop just past the last marker: the direct labels sit outside the
    # axes and are kept by the tight bounding box, so the spine need not
    # stretch to hold them.
    thresholds = sorted(sub["threshold"].unique() * 100)
    ax.set_xlim(thresholds[0] - xpad_left, thresholds[-1] + xpad_right)
    if dropped and show_dropped_note:
        ax.set_title(
            f"top {len(order)} of {len(order) + dropped} methods",
            fontsize=FONT["legend"],
            color=PALETTE["muted"],
        )
    # Every line is directly labelled, so identity never rests on colour alone
    # and a legend would only repeat the labels.
    return ax
