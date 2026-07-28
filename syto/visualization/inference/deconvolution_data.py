"""Pure transforms turning the long-format deconvolution results DataFrame into
a ReportModel consumed by the PDF and CLI renderers. No plotting here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

BASELINE_DECONVOLVERS: tuple[str, ...] = (
    "uxm",
    "celfie",
    "celfieish",
    "epidish",
    "epidish_cbs",
    "epidish_cp",
)


def collect_baseline_labels(
    deconvolvers, defaults: Sequence[str] = BASELINE_DECONVOLVERS
):
    """Ordered baseline result labels: defaults first, then the actual labels
    produced by ``deconvolvers``.

    Handles custom result names (e.g. an EpiDISH entry labelled
    ``epidish_houseman``) and EM-checkpoint suffixes (``celfieish_10_steps``),
    which a static list cannot know in advance.  Each deconvolver contributes
    ``f"{name}_{n}_steps"`` per checkpoint when ``em_checkpoints`` is set, else
    its ``name``.
    """
    labels = list(defaults)
    seen = set(labels)
    for d in deconvolvers:
        checkpoints = getattr(d, "em_checkpoints", None)
        names = (
            [f"{d.name}_{n}_steps" for n in checkpoints] if checkpoints else [d.name]
        )
        for name in names:
            if name not in seen:
                labels.append(name)
                seen.add(name)
    return labels


REQUIRED_COLUMNS = (
    "CellType",
    "Classifier",
    "Deconvolver",
    "Calibrator",
    "PredictedProportion",
)


@dataclass(frozen=True)
class MethodInfo:
    key: str
    classifier: str
    deconvolver: str
    calibrator: str
    is_baseline: bool
    divergence: float


@dataclass
class ReportModel:
    cell_types: list[str]  # ordered by consensus desc
    methods: list[MethodInfo]  # ordered baseline | syto(hierarchy)
    matrix: np.ndarray  # (n_cell_types, n_methods), aligned to the orders above
    consensus: np.ndarray  # median per cell type
    mean: np.ndarray  # mean per cell type
    spread_min: np.ndarray
    spread_max: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    top_n: int
    known_truth: Optional[str]
    file_name: str

    @property
    def n_cell_types(self) -> int:
        return len(self.cell_types)

    @property
    def top_cell_types(self) -> list[str]:
        return self.cell_types[: self.top_n]

    @property
    def other_consensus(self) -> float:
        return float(self.consensus[self.top_n :].sum())


def _method_key(classifier: str, deconvolver: str, calibrator: str) -> str:
    return f"{classifier}·{deconvolver}·{calibrator}"


def _is_baseline(deconvolver: str, baseline_deconvolvers: Sequence[str]) -> bool:
    return deconvolver in set(baseline_deconvolvers)


def build_report_model(
    df: pd.DataFrame,
    top_n: int = 10,
    known_truth: Optional[str] = None,
    baseline_deconvolvers: Sequence[str] = BASELINE_DECONVOLVERS,
) -> ReportModel:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["__key"] = [
        _method_key(c, d, k)
        for c, d, k in zip(df["Classifier"], df["Deconvolver"], df["Calibrator"])
    ]

    # cell_type × method matrix (mean collapses accidental duplicate rows)
    pivot = df.pivot_table(
        index="CellType", columns="__key", values="PredictedProportion", aggfunc="mean"
    )
    vals = pivot.to_numpy(dtype=float)

    # per-cell statistics across methods (columns)
    consensus = np.nanmedian(vals, axis=1)
    mean = np.nanmean(vals, axis=1)
    spread_min = np.nanmin(vals, axis=1)
    spread_max = np.nanmax(vals, axis=1)
    q25 = np.nanpercentile(vals, 25, axis=1)
    q75 = np.nanpercentile(vals, 75, axis=1)

    # order cell types by consensus desc (stable)
    ct_order = np.argsort(-consensus, kind="stable")
    cell_types = [str(pivot.index[i]) for i in ct_order]

    # per-method metadata (one row per unique key)
    meta = (
        df[["__key", "Classifier", "Deconvolver", "Calibrator"]]
        .drop_duplicates("__key")
        .set_index("__key")
    )
    baseline_rank = {d: i for i, d in enumerate(baseline_deconvolvers)}

    def sort_key(key: str):
        row = meta.loc[key]
        if _is_baseline(row["Deconvolver"], baseline_deconvolvers):
            return (
                0,
                baseline_rank.get(row["Deconvolver"], 99),
                str(row["Calibrator"]),
                "",
            )
        return (
            1,
            str(row["Classifier"]),
            str(row["Deconvolver"]),
            str(row["Calibrator"]),
        )

    ordered_keys = sorted(pivot.columns, key=sort_key)

    # reindex matrix to (cell_type order, method order)
    matrix = pivot.reindex(
        index=[pivot.index[i] for i in ct_order], columns=ordered_keys
    ).to_numpy(dtype=float)
    consensus_ordered = consensus[ct_order]

    methods: list[MethodInfo] = []
    for j, key in enumerate(ordered_keys):
        col = matrix[:, j]
        divergence = float(np.nansum(np.abs(col - consensus_ordered)))
        row = meta.loc[key]
        methods.append(
            MethodInfo(
                key=key,
                classifier=str(row["Classifier"]),
                deconvolver=str(row["Deconvolver"]),
                calibrator=str(row["Calibrator"]),
                is_baseline=_is_baseline(row["Deconvolver"], baseline_deconvolvers),
                divergence=divergence,
            )
        )

    file_name = (
        str(df["FileName"].iloc[0])
        if "FileName" in df.columns and len(df)
        else "unknown"
    )

    return ReportModel(
        cell_types=cell_types,
        methods=methods,
        matrix=matrix,
        consensus=consensus_ordered,
        mean=mean[ct_order],
        spread_min=spread_min[ct_order],
        spread_max=spread_max[ct_order],
        q25=q25[ct_order],
        q75=q75[ct_order],
        top_n=min(top_n, len(cell_types)),
        known_truth=known_truth,
        file_name=file_name,
    )
