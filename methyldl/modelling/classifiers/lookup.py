"""
Soft-label lookup classifier with KNN fallback.

The classifier is essentially a lookup table keyed by (region_name, cpg_signature)
whose values are soft-label probability vectors.  When a key is missing at predict
time it falls back to 1-NN (or ties-averaged) using the Jaccard signature distance.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from methyldl.data.soft_labeling import (
    extract_cpg_signature,
    signature_distance,
    apply_normalized_knn_smoothing,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SoftLabelConfig:
    """All tuneable knobs live here."""

    num_classes: int = 39
    label_col: str = "original_label"

    # KNN smoothing parameters (used during *fit* to build the table)
    min_reads: int = 30
    max_distance: float = 0.41

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SoftLabelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────


class LookupClassifier:
    """Lookup-table classifier backed by soft labels.

    Parameters
    ----------
    config : SoftLabelConfig
        Smoothing / labelling hyper-parameters.

    Usage
    -----
    >>> clf = SoftLabelClassifier(SoftLabelConfig(min_reads=30, max_distance=0.41))
    >>> clf.fit(train_df)
    >>> preds = clf.predict(test_df)
    >>> clf.save("model.pkl")
    >>> clf = SoftLabelClassifier.load("model.pkl")
    """

    def __init__(self, config: Optional[SoftLabelConfig] = None):
        self.config = config or SoftLabelConfig()

        # Populated by fit() --------------------------------------------------
        # {(region, cpg_sig): {"soft_label": [...], "normalized_counts": [...]}}
        self._lookup: Dict[tuple, dict] = {}
        # Per-region index for fast NN search
        self._region_index: Dict[str, List[Tuple[tuple, dict]]] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "LookupClassifier":
        """Build the soft-label lookup table from a single DataFrame.

        Parameters
        ----------
        df : DataFrame
            Raw data frame containing at least the columns ``name``,
            ``trimmed_start``, ``pattern``, and ``<label_col>``.
            If you need to pool multiple splits, concatenate them
            before calling fit.

        Returns
        -------
        self
        """
        self._lookup = self._build_mapping(df)

        # Build per-region index for NN fallback
        self._region_index = {}
        for (region, sig), entry in self._lookup.items():
            self._region_index.setdefault(region, []).append((sig, entry))

        self._is_fitted = True
        logger.info(
            "Fitted with %d unique (region, signature) keys across %d regions.",
            len(self._lookup),
            len(self._region_index),
        )
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict soft labels for every row in *df*.

        The input frame must have the same schema used during fit (at minimum
        ``name``, ``trimmed_start``, ``pattern``).

        Returns a copy of *df* with extra columns:
        ``prediction_0``, ``prediction_1``, …, ``prediction_{num_classes-1}``,
        and ``prediction_source`` (``"exact"`` or ``"1nn"``).
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict().")

        num_classes = self.config.num_classes
        df = df.copy()

        # Ensure cpg_sig exists
        if "cpg_sig" not in df.columns:
            tqdm.pandas(desc="Extracting signatures")
            df["cpg_sig"] = df.progress_apply(extract_cpg_signature, axis=1)
        # Guarantee tuple type
        df["cpg_sig"] = df["cpg_sig"].apply(
            lambda x: tuple(x) if not isinstance(x, tuple) else x
        )

        pred_matrix = np.empty((len(df), num_classes))
        sources = []

        for i, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Predicting soft labels")
        ):
            key = (row["name"], row["cpg_sig"])
            entry = self._lookup.get(key)

            if entry is not None:
                pred_matrix[i] = entry["soft_label"]
                sources.append("exact")
            else:
                pred_matrix[i] = self._fallback_nn(row["name"], row["cpg_sig"])
                sources.append("1nn")

        pred_cols = [f"prediction_{j}" for j in range(num_classes)]
        df[pred_cols] = pred_matrix
        df["prediction_source"] = sources
        return df

    # ------------------------------------------------------------------ save / load
    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted classifier to disk."""
        path = Path(path)
        payload = {
            "config": self.config.to_dict(),
            "lookup": {self._key_to_str(k): v for k, v in self._lookup.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved classifier to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LookupClassifier":
        """Load a previously saved classifier."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        config = SoftLabelConfig.from_dict(payload["config"])
        clf = cls(config)

        clf._lookup = {cls._str_to_key(k): v for k, v in payload["lookup"].items()}

        # Rebuild per-region index
        clf._region_index = {}
        for (region, sig), entry in clf._lookup.items():
            clf._region_index.setdefault(region, []).append((sig, entry))

        clf._is_fitted = True
        logger.info("Loaded classifier with %d keys from %s", len(clf._lookup), path)
        return clf

    # ================================================================== private

    def _build_mapping(self, df: pd.DataFrame) -> Dict[tuple, dict]:
        """Raw DataFrame → {(region, cpg_sig): {soft_label, normalized_counts}}."""
        cfg = self.config
        df = df.copy()

        # Ensure cpg_sig column
        if "cpg_sig" not in df.columns:
            tqdm.pandas(desc="Extracting signatures")
            df["cpg_sig"] = df.progress_apply(extract_cpg_signature, axis=1)
        df["cpg_sig"] = df["cpg_sig"].apply(
            lambda x: tuple(x) if not isinstance(x, tuple) else x
        )

        # Ensure label column is int
        df[cfg.label_col] = df[cfg.label_col].astype(np.int32)

        # Build base counts
        base_counts = (
            df.groupby(["name", "cpg_sig", cfg.label_col]).size().unstack(fill_value=0)
        )
        base_counts = base_counts.reindex(columns=range(cfg.num_classes), fill_value=0)
        base_counts = base_counts.reset_index()
        class_cols = list(range(cfg.num_classes))
        base_counts["total_reads"] = base_counts[class_cols].sum(axis=1)

        # Smooth
        mapping_df = apply_normalized_knn_smoothing(
            base_counts,
            min_reads=cfg.min_reads,
            max_distance=cfg.max_distance,
            num_classes=cfg.num_classes,
        )

        # Convert to dict
        table: Dict[tuple, dict] = {}
        for _, row in mapping_df.iterrows():
            key = (row["name"], row["cpg_sig"])
            table[key] = {
                "soft_label": (
                    row["soft_label"]
                    if isinstance(row["soft_label"], list)
                    else list(row["soft_label"])
                ),
                "normalized_counts": (
                    row["normalized_counts"]
                    if isinstance(row["normalized_counts"], list)
                    else list(row["normalized_counts"])
                ),
            }
        return table

    def _fallback_nn(self, region: str, query_sig: tuple) -> list:
        """1-NN fallback within the same region.

        If multiple neighbours share the minimum distance, their
        *normalized counts* are summed and re-normalised to produce
        the predicted soft label.
        """
        num_classes = self.config.num_classes
        candidates = self._region_index.get(region)

        if not candidates:
            # Region completely unseen — return uniform
            logger.warning("Region '%s' not in lookup; returning uniform.", region)
            return [1.0 / num_classes] * num_classes

        # Find minimum distance
        best_dist = float("inf")
        best_entries: List[dict] = []

        for sig, entry in candidates:
            d = signature_distance(query_sig, sig)
            if d < best_dist:
                best_dist = d
                best_entries = [entry]
            elif d == best_dist:
                best_entries.append(entry)

        # Aggregate normalised counts of all tied NNs
        agg_counts = np.zeros(num_classes)
        for entry in best_entries:
            agg_counts += np.array(entry["normalized_counts"])

        total = agg_counts.sum()
        if total > 0:
            return (agg_counts / total).tolist()
        return [1.0 / num_classes] * num_classes

    # ---- serialisation helpers (tuples ↔ JSON-safe strings) ----

    @staticmethod
    def _key_to_str(key: tuple) -> str:
        """(region, cpg_sig_tuple) → JSON-safe string."""
        region, sig = key
        return json.dumps([region, sig])

    @staticmethod
    def _str_to_key(s: str) -> tuple:
        """Inverse of _key_to_str."""
        region, sig = json.loads(s)
        return (region, tuple(tuple(x) for x in sig))
