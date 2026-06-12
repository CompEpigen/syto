"""
Lookup classifier with KNN fallback, supporting soft and hard labels.

The classifier is essentially a lookup table keyed by (region_name, cpg_signature)
whose values are label vectors (soft probabilities or hard one-hot / averaged one-hot).
When a key is missing at predict time it falls back to 1-NN (or ties-averaged) using
the Jaccard signature distance.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.classification.evaluation import (
    compute_metrics,
    compute_metrics_soft_labels,
)

from syto.data.labelers.data_driven_soft_labeler import DataDrivenSoftLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)
from syto.classification.classifiers.abstract_read_classifier import (
    AbstractReadClassifier,
)
from syto.classification.mlflow_tracking import mlflow_tracked_fit

_module_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LabelConfig:
    """All tuneable knobs live here."""

    num_classes: int = 39
    label_col: str = "original_label"
    label_mode: str = "soft"  # "soft" or "hard"

    # KNN smoothing parameters (used during *fit* to build the table, soft mode only)
    min_reads: int = 30
    max_distance: float = 0.41

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelConfig":
        # pylint: disable=no-member
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────


class LookupClassifier(AbstractReadClassifier):
    """Lookup-table classifier backed by soft or hard labels.

    Parameters
    ----------
    config : LabelConfig
        Smoothing / labelling hyper-parameters.

    Usage
    -----
    >>> clf = LookupClassifier(LabelConfig(min_reads=30, max_distance=0.41))
    >>> clf.fit(train_df)
    >>> preds = clf.predict(test_df)
    >>> clf.save("model.pkl")
    >>> clf = LookupClassifier.load("model.pkl")
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        self.config = config or LabelConfig()

        # Handler used to extract binary CpG signatures and compute distances
        self.signature_handler = BinaryCpGSignatureHandler(
            start_column="trimmed_start", methylation_pattern_column="pattern"
        )

        # Populated by fit() --------------------------------------------------
        # {(region, cpg_sig): {"soft_label": [...], "normalized_counts": [...]}}
        self._lookup: Dict[tuple, dict] = {}
        # Per-region index for fast NN search
        self._region_index: Dict[str, List[Tuple[tuple, dict]]] = {}
        self._is_fitted: bool = False
        self.history: List[dict] = []

    @property
    def n_keys(self) -> int:
        """Number of unique (region, signature) keys in the lookup table."""
        return len(self._lookup)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        compute_train_metrics: bool = True,
        **kwargs,
    ) -> "LookupClassifier":
        """Build the soft-label lookup table from a single DataFrame.

        Parameters
        ----------
        df : DataFrame
            Raw data frame containing at least the columns ``name``,
            ``trimmed_start``, ``pattern``, and ``<label_col>``.
            If you need to pool multiple splits, concatenate them
            before calling fit.
        val_data : DataFrame, optional
            Validation DataFrame with the same schema as ``df``. When provided,
            validation metrics are computed and stored in :attr:`history`.
        compute_train_metrics : bool, default True
            Whether to run :meth:`predict` on the training data to compute
            train-side sklearn metrics. Set to ``False`` for large datasets
            where the row-by-row prediction pass is prohibitively slow.

        Returns
        -------
        self
        """
        t_start = time.time()
        self._lookup = self._build_mapping(df)

        # Build per-region index for NN fallback
        self._region_index = {}
        for (region, sig), entry in self._lookup.items():
            self._region_index.setdefault(region, []).append((sig, entry))

        self._is_fitted = True
        elapsed = time.time() - t_start

        fit_record: dict = {
            "n_keys": self.n_keys,
            "n_regions": len(self._region_index),
            "label_mode": self.config.label_mode,
            "elapsed_time": elapsed,
        }

        if compute_train_metrics:
            train_preds_df = self.predict(df)
            fit_record.update(self._compute_fit_metrics(df, train_preds_df, "train"))

        if val_data is not None:
            val_preds_df = self.predict(val_data)
            fit_record.update(self._compute_fit_metrics(val_data, val_preds_df, "val"))

        self.history.append(fit_record)

        _module_logger.info(
            "Fitted with %d unique (region, signature) keys across %d regions "
            "in %.2fs.%s%s",
            self.n_keys,
            len(self._region_index),
            elapsed,
            (
                " Train: acc=%.4f f1=%.4f"
                % (
                    fit_record.get("train_accuracy", float("nan")),
                    fit_record.get("train_f1", float("nan")),
                )
                if compute_train_metrics
                else ""
            ),
            (
                " Val: acc=%.4f f1=%.4f"
                % (
                    fit_record.get("val_accuracy", float("nan")),
                    fit_record.get("val_f1", float("nan")),
                )
                if val_data is not None
                else ""
            ),
        )
        return self

    # ------------------------------------------------------------------ metrics helper
    def _compute_fit_metrics(
        self, df: pd.DataFrame, predictions_df: pd.DataFrame, prefix: str
    ) -> dict:
        """Compute sklearn classification metrics from fit-time predictions.

        Parameters
        ----------
        df : DataFrame
            The ground-truth DataFrame (must contain ``config.label_col``).
        predictions_df : DataFrame
            Output of :meth:`predict` with ``prediction_0`` … columns.
        prefix : str
            Prefix prepended to each metric key (e.g. ``"train"`` or ``"val"``).

        Returns
        -------
        dict
            Keys like ``{prefix}_accuracy``, ``{prefix}_f1``, etc.
        """
        num_classes = self.config.num_classes
        pred_cols = [f"prediction_{j}" for j in range(num_classes)]
        predictions_proba = predictions_df[pred_cols].to_numpy()

        labels = df[self.config.label_col].to_numpy()

        if self.config.label_mode == "soft":
            # Build soft label matrix from integer labels for compute_metrics_soft_labels
            metrics = compute_metrics_soft_labels((predictions_proba, labels))
        else:
            metrics = compute_metrics((predictions_proba, labels))

        return {f"{prefix}_{k}": v for k, v in metrics.items()}

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
            df["cpg_sig"] = df.progress_apply(
                self.signature_handler.extract_signature, axis=1
            )
        # Guarantee tuple type
        df["cpg_sig"] = df["cpg_sig"].apply(
            lambda x: (
                tuple(tuple(int(e) for e in p) for p in x)
                if not isinstance(x, tuple)
                else x
            )
        )

        label_key = "hard_label" if self.config.label_mode == "hard" else "soft_label"
        pred_matrix = np.empty((len(df), num_classes))
        sources = []

        for i, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Predicting labels")
        ):
            key = (row["name"], row["cpg_sig"])
            entry = self._lookup.get(key)

            if entry is not None:
                pred_matrix[i] = entry[label_key]
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
        path.parent.mkdir(parents=True, exist_ok=True)
        file_extension = path.suffix.lower()

        if file_extension == ".pkl":
            _module_logger.warning(
                "Saving in .pkl format is deprecated; please switch to .joblib"
            )
            payload = {
                "config": self.config.to_dict(),
                "lookup": {self._key_to_str(k): v for k, v in self._lookup.items()},
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif file_extension == ".joblib":
            joblib.dump(value=self, filename=path)
        else:
            raise ValueError(
                f"Unsupported file extension '{file_extension}' for saving."
                " Use .joblib or .pkl."
            )

        _module_logger.info("Saved LookupClassifier to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path, None] = None, **kwargs) -> "LookupClassifier":
        """Load a previously saved classifier, or build a fresh instance if path is None.

        When ``path`` is None, ``kwargs`` are interpreted as :class:`LabelConfig`
        fields (``num_classes``, ``label_col``, ``label_mode``, ``min_reads``,
        ``max_distance``) via :meth:`LabelConfig.from_dict`. Unrecognized keys
        (e.g. shared classifier-factory kwargs such as ``num_labels`` or
        ``seq_length``) are silently ignored.
        """
        if path is None:
            return cls(LabelConfig.from_dict(kwargs))

        path = Path(path)
        file_extension = path.suffix.lower()

        if file_extension == ".joblib":
            clf = joblib.load(path)
            if not isinstance(clf, cls):
                raise ValueError(
                    f"Loaded object from {path} is not a LookupClassifier."
                )
        elif file_extension == ".pkl":
            _module_logger.warning(
                "Loading from .pkl format is deprecated; please switch to .joblib"
            )
            with open(path, "rb") as f:
                payload = pickle.load(f)

            config = LabelConfig.from_dict(payload["config"])
            clf = cls(config)

            clf._lookup = {cls._str_to_key(k): v for k, v in payload["lookup"].items()}

            # Rebuild per-region index
            clf._region_index = {}
            for (region, sig), entry in clf._lookup.items():
                clf._region_index.setdefault(region, []).append((sig, entry))

            clf._is_fitted = True
        else:
            raise ValueError(
                f"Unsupported file extension '{file_extension}' for loading."
                " Use .joblib or .pkl."
            )

        _module_logger.info(
            "Loaded LookupClassifier with %d keys from %s", clf.n_keys, path
        )
        return clf

    # ================================================================== private

    def _build_mapping(self, df: pd.DataFrame) -> Dict[tuple, dict]:
        """Raw DataFrame → {(region, cpg_sig): {label_vector, counts}}."""
        cfg = self.config
        df = df.copy()

        if cfg.label_mode != "hard":
            return self._build_soft_mapping(df)

        # Ensure cpg_sig column

        if "cpg_sig" not in df.columns:
            tqdm.pandas(desc="Extracting signatures")
            df["cpg_sig"] = df.progress_apply(
                self.signature_handler.extract_signature, axis=1
            )
        df["cpg_sig"] = df["cpg_sig"].apply(
            lambda x: (
                tuple(tuple(int(e) for e in p) for p in x)
                if not isinstance(x, tuple)
                else x
            )
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

        return self._build_hard_mapping(base_counts, class_cols)

    def _build_soft_mapping(self, df: pd.DataFrame) -> Dict[tuple, dict]:
        """Build the soft-label lookup table via data-driven KNN smoothing."""
        cfg = self.config

        reads_df = df.copy()
        # DataDrivenSoftLabeler expects an integer "original_label" column
        if cfg.label_col != "original_label":
            reads_df["original_label"] = reads_df[cfg.label_col]
        reads_df["original_label"] = reads_df["original_label"].astype(np.int32)

        # Drop columns that DataDrivenSoftLabeler.compute_labels itself produces,
        # so its merge back onto reads_df doesn't collide with same-named columns
        # already present in df (e.g. a "soft_label" from a previous labeling pass).
        labeler_output_cols = {
            "signature",
            "total_reads",
            "raw_counts_pooled",
            "num_signatures_pooled",
            "soft_label",
            *(f"raw_counts_{c}" for c in range(cfg.num_classes)),
            *(f"pooled_counts_{c}" for c in range(cfg.num_classes)),
            *(f"weighted_counts_{c}" for c in range(cfg.num_classes)),
        }
        reads_df = reads_df.drop(
            columns=[c for c in labeler_output_cols if c in reads_df.columns]
        )

        labeler = DataDrivenSoftLabeler(
            distance_name="jaccard",
            signature_handler=self.signature_handler,
        )
        labeled_df = labeler.compute_labels(
            reads_df,
            perform_pooling=True,
            min_reads=cfg.min_reads,
            max_distance=cfg.max_distance,
            num_classes=cfg.num_classes,
            keep_intermediate_values=True,
        )

        # The weighted (globally normalized) counts play the role of the
        # "normalized_counts" used by the lookup table and the NN fallback.
        weighted_cols = [f"weighted_counts_{c}" for c in range(cfg.num_classes)]
        mapping_df = labeled_df.drop_duplicates(subset=["name", "signature"])

        table: Dict[tuple, dict] = {}
        for _, row in mapping_df.iterrows():
            key = (row["name"], row["signature"])
            table[key] = {
                "soft_label": (
                    row["soft_label"]
                    if isinstance(row["soft_label"], list)
                    else list(row["soft_label"])
                ),
                "normalized_counts": [float(row[col]) for col in weighted_cols],
            }
        return table

    def _build_hard_mapping(
        self, base_counts: pd.DataFrame, class_cols: list
    ) -> Dict[tuple, dict]:
        """Build the hard-label lookup table (argmax with tie-averaging).

        Counts are first normalized using global sequencing-depth weights
        (same method as the data-driven soft labeler) before the
        argmax is computed.
        """
        cfg = self.config

        # Sanity check: the highest class (by global count) must dominate
        global_counts = base_counts[class_cols].sum(axis=0).sort_values(ascending=False)
        if len(global_counts) >= 2:
            top, runner_up = global_counts.iloc[0], global_counts.iloc[1]
            if top < 10 * runner_up:
                top_cls = int(global_counts.index[0])
                runner_cls = int(global_counts.index[1])
                raise ValueError(
                    f"Hard-label sanity check failed: class {top_cls} has "
                    f"{int(top)} reads but class {runner_cls} has {int(runner_up)} "
                    f"reads (ratio {top / runner_up:.1f}x < 10x required). "
                    f"This suggests the background class does not dominate "
                    f"as expected."
                )

        # Global normalization weights (median / per-class total)
        global_class_counts = base_counts[class_cols].sum(axis=0)
        nonzero = global_class_counts[global_class_counts > 0]
        median_count = nonzero.median()
        epsilon = 1e-9
        class_weights = (median_count / (global_class_counts + epsilon)).values

        raw_matrix = base_counts[class_cols].values.astype(float)
        normalized_matrix = raw_matrix * class_weights[np.newaxis, :]

        table: Dict[tuple, dict] = {}

        for idx, row in base_counts.iterrows():
            counts = normalized_matrix[idx]
            max_val = counts.max()
            max_mask = counts == max_val
            n_tied = max_mask.sum()

            # One-hot if strict argmax, averaged one-hot if tied
            label_vec = np.zeros(cfg.num_classes)
            label_vec[max_mask] = 1.0 / n_tied

            key = (row["name"], row["cpg_sig"])
            table[key] = {
                "hard_label": label_vec.tolist(),
                "normalized_counts": normalized_matrix[idx].tolist(),
            }
        return table

    def _fallback_nn(self, region: str, query_sig: tuple) -> list:
        """1-NN fallback within the same region.

        Soft mode: normalized counts of tied NNs are summed and re-normalised.
        Hard mode: hard-label vectors of tied NNs are averaged.
        """
        num_classes = self.config.num_classes
        candidates = self._region_index.get(region)

        if not candidates:
            if self.config.label_mode == "soft":
                # Region completely unseen — return uniform
                _module_logger.warning(
                    "Region '%s' not in lookup; returning uniform.", region
                )
                return [1.0 / num_classes] * num_classes
            elif self.config.label_mode == "hard":
                # Region completely unseen — return one hot of background class (last class)
                _module_logger.warning(
                    "Region '%s' not in lookup; returning background class.", region
                )
                background_class = num_classes - 1
                label_vec = np.zeros(num_classes)
                label_vec[background_class] = 1.0
                return label_vec.tolist()

        # Find minimum distance
        best_dist = float("inf")
        best_entries: List[dict] = []

        for sig, entry in candidates:
            d = self.signature_handler.compute_jaccard_distance(query_sig, sig)
            if d < best_dist:
                best_dist = d
                best_entries = [entry]
            elif d == best_dist:
                best_entries.append(entry)

        if self.config.label_mode == "hard":
            # Average the hard-label vectors of all tied NNs
            agg = np.zeros(num_classes)
            for entry in best_entries:
                agg += np.array(entry["hard_label"])
            return (agg / len(best_entries)).tolist()

        # Soft mode: aggregate normalised counts of all tied NNs
        agg_counts = np.zeros(num_classes)
        for entry in best_entries:
            agg_counts += np.array(entry["normalized_counts"])

        total = agg_counts.sum()
        if total > 0:
            return (agg_counts / total).tolist()
        return [1.0 / num_classes] * num_classes

    # ---- serialisation helpers (tuples <-> JSON-safe strings) ----

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

    def predict_split(self, split_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict method for compatibility with AbstractReadClassifier interface."""
        return self.predict(split_df)

    @mlflow_tracked_fit
    def fit_split(
        self,
        train_df: pd.DataFrame,
        val_df: Union[pd.DataFrame, None] = None,
        output_dir: Union[str, Path, None] = None,
        **kwargs,
    ) -> "LookupClassifier":
        """Fit the classifier on training data for compatibility with AbstractReadClassifier."""
        self.fit(df=train_df, val_data=val_df, **kwargs)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.save(Path(output_dir) / "lookup_classifier.joblib")
        return self
