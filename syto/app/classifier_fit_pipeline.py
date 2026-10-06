"""
Classifier Fitting Pipeline
============================

Fits a read-level classifier (``EpigenDnabert2``, ``MethylBert``, ``Dismir``,
``CancerDetectorClassifier`` or ``LookupClassifier``) on one or more datasets
via ``AbstractReadClassifier.fit_classificaton``.

If an ``mlflow`` section is present in the configuration, the tracking URI
and experiment are configured once up front; each dataset's ``fit_classificaton``
call is then recorded as its own MLflow run by the
``mlflow_tracked_fit`` decorator on the classifier.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from syto.classification.classifiers.lazy_classifier_factory import (
    read_classifier_factory,
)
from syto.classification.fit_data import (
    detect_format,
    load_legacy_split,
    load_columnar_split,
    apply_label_rename,
    apply_pattern_length_filter,
    deduplicate_reads,
    DEFAULT_DEDUP_KEYS,
    READ_COUNT_COLUMN,
)


class ClassifierFittingPipeline:
    """Orchestrate read-level classifier fitting end-to-end.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    logger : logging.Logger
        Logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Optional per-epoch hook, set programmatically by an external driver
        # (e.g. the Optuna tuner in Scripts/). Not a config key: it is a
        # callable, and the effective config is dumped as YAML.
        self.epoch_callback = None

        self.model_cfg = config.get("model", {})
        self.training_cfg = config.get("training", {})
        output_cfg = config.get("output", {})
        self.mlflow_cfg = config.get("mlflow", {})

        self.model_arch = self.model_cfg["architecture"].lower()
        self.data_path = Path(config["data_path"])
        # ``output_dir`` may be given at the top level (most configs and the
        # ``--output_dir`` CLI override) or nested under an ``output:`` section.
        self.base_output_dir = Path(
            output_cfg.get("output_dir") or config.get("output_dir", "output")
        )

        self.data_format = config.get("data_format", "auto")
        self.split_column = config.get("split_column", "split")
        self.label_column = config.get("label_column")
        self.min_pattern_length = config.get("min_pattern_length")
        # Evaluate on a random subset of the validation split. A full pass over
        # a large valid split can cost more than the training it interrupts;
        # subsampling keeps a frequent eval cadence affordable. Reads are
        # sampled, so models that chunk reads (MethylBert) see proportionally
        # more examples than this number.
        self.valid_subsample = config.get("valid_subsample")
        # Collapse identical reads into one row carrying a ``read_count``
        # multiplicity. On train the classifier turns counts into sampling or
        # loss weights; on valid into count-weighted loss and metrics, which
        # equal the read-for-read values (the model is deterministic in eval
        # mode) at a fraction of the prediction cost.
        self.deduplicate_train = bool(config.get("deduplicate_train", False))
        self.deduplicate_valid = bool(config.get("deduplicate_valid", False))
        self.dedup_keys = list(config.get("dedup_keys") or DEFAULT_DEDUP_KEYS)
        self.seed = config.get("seed", 42)

        datasets = config.get("datasets", ["all"])
        if isinstance(datasets, str):
            datasets = ["all"] if datasets == "all" else [datasets]
        self.datasets: List[str] = datasets

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Fit the configured classifier on each requested dataset.

        Returns ``{dataset_name: best_epoch_metrics}``. The selected epoch is the
        best one seen during the fit, not the last: with early stopping the saved
        checkpoint is the best epoch, so reporting the last would describe a
        model that was never written. Datasets whose classifier exposes no
        ``history`` are omitted.
        """
        self.logger.info("Seeding RNGs with seed %d", self.seed_everything())
        self._configure_mlflow()

        self.logger.info(
            f"Starting classifier fit workflow for architecture: {self.model_arch}"
        )

        summary: Dict[str, Dict[str, Any]] = {}
        for dataset_name in self.datasets:
            best = self._fit_dataset(dataset_name)
            if best is not None:
                summary[dataset_name] = best
        return summary

    def _best_epoch_record(self, history) -> Optional[Dict[str, Any]]:
        """Pick the best record from a fit ``history`` by the configured metric."""
        metric = self.config.get("best_metric", "val_loss")
        mode = self.config.get("best_mode", "min" if metric.endswith("loss") else "max")
        usable = [h for h in (history or []) if h.get(metric) is not None]
        if not usable:
            return None
        pick = min if mode == "min" else max
        best = dict(pick(usable, key=lambda h: h[metric]))
        best["best_epoch"] = best.get("epoch")
        best["best_metric"] = metric
        return best

    # ═══════════════════════════════════════════════════════════════
    #  Per-dataset fitting
    # ═══════════════════════════════════════════════════════════════

    def _fit_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Detect layout, load only fit-relevant columns, and fit the classifier."""
        if dataset_name == "all":
            dataset_path = self.data_path
            out_dir = self.base_output_dir / "all"
        else:
            dataset_path = self.data_path / dataset_name
            out_dir = self.base_output_dir / dataset_name

        self.logger.info(f"Processing dataset: {dataset_name} at {dataset_path}")

        fmt = detect_format(
            dataset_path, override=self.data_format, split_column=self.split_column
        )
        self.logger.info(f"Detected data format: {fmt}")

        self.logger.info("Initializing classifier...")
        classifier = read_classifier_factory(
            name=self.model_arch,
            path=None,  # Start fresh
            **self._classifier_kwargs(),
        )

        declared_columns = classifier.required_fit_columns(self.config)
        if self.label_column:
            declared_columns = declared_columns + [self.label_column]

        if (self.deduplicate_train or self.deduplicate_valid) and not getattr(
            classifier, "supports_read_counts", False
        ):
            raise ValueError(
                "deduplicate_train/deduplicate_valid is set but the "
                f"{self.model_arch!r} classifier does not consume read counts; "
                "de-duplicated reads without them would silently change the "
                "objective and the metrics."
            )

        try:
            train_df = self._load_split(dataset_path, "train", fmt, declared_columns)
            self.logger.info(f"Loaded train split: {len(train_df)} rows")
        except FileNotFoundError as e:
            self.logger.error(f"Error loading train data: {e}")
            return None

        try:
            val_df = self._load_split(dataset_path, "valid", fmt, declared_columns)
            self.logger.info(f"Loaded valid split: {len(val_df)} rows")
            if self.valid_subsample:
                val_df = self._subsample_valid(val_df, int(self.valid_subsample))
        except FileNotFoundError:
            self.logger.warning(
                "No valid split found, continuing without validation data"
            )
            val_df = None

        # Persist the effective config (post CLI-override) next to the fit
        # outputs so it is captured by the run's ``log_artifacts(out_dir)``.
        self._dump_effective_config(out_dir)

        self.logger.info("Fitting classifier...")

        # The entire ``training:`` section is forwarded as kwargs, so an optional
        # ``training.early_stopping`` block reaches HF classifiers as the
        # ``early_stopping`` kwarg (see MethylBert/EpigenDnabert2.fit_classificaton).
        classifier.fit_classificaton(
            train_df=train_df,
            val_df=val_df,
            output_dir=out_dir,
            mlflow_run_name=f"{self.model_arch}_{dataset_name}_{self.label_column}",
            mlflow_tags={"architecture": self.model_arch, "dataset": dataset_name},
            mlflow_extra_params=self._mlflow_extra_params(classifier),
            mlflow_extra_artifacts=self._mlflow_extra_artifacts(),
            track_with_mlflow=self.mlflow_cfg.get("enabled", True),
            epoch_callback=self.epoch_callback,
            **self.training_cfg,
        )

        self.logger.info(f"Finished processing {dataset_name}")
        return self._best_epoch_record(getattr(classifier, "history", None))

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    def _subsample_valid(self, val_df, n_reads):
        """Keep a random ``n_reads`` reads of the valid split (seeded).

        On a de-duplicated split the draw is over reads, not unique rows: a
        multivariate hypergeometric draw thins each row's ``read_count`` exactly
        as sampling ``n_reads`` reads without replacement from the duplicated
        split would.
        """
        if READ_COUNT_COLUMN not in val_df.columns:
            if len(val_df) <= n_reads:
                return val_df
            out = val_df.sample(n=n_reads, random_state=self.seed)
        else:
            counts = val_df[READ_COUNT_COLUMN].to_numpy()
            if counts.sum() <= n_reads:
                return val_df
            drawn = np.random.default_rng(self.seed).multivariate_hypergeometric(
                counts, n_reads
            )
            out = val_df.assign(**{READ_COUNT_COLUMN: drawn})[drawn > 0]
        self.logger.info(
            "Subsampled valid split to %d reads in %d rows (seed %d)",
            n_reads,
            len(out),
            self.seed,
        )
        return out.reset_index(drop=True)

    def _dump_effective_config(self, out_dir: Path) -> None:
        """Write the effective (post-override) config into the fit output dir.

        Captured by the run's ``log_artifacts(out_dir)``, so the exact config
        that produced the run is logged as an MLflow artifact.
        """
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "run_config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning(f"Could not write run_config.yaml: {exc}")

    def _mlflow_extra_artifacts(self) -> List[str]:
        """Standalone files to log as run artifacts (the run log file, if any).

        The log file path is resolved from the active logging handlers rather
        than passed in explicitly.
        """
        log_file = self._resolve_log_file()
        return [log_file] if log_file else []

    def _resolve_log_file(self) -> Optional[str]:
        """Return the path of the active ``FileHandler`` log file, if configured.

        Walks the pipeline logger's handler chain up to the root (records
        propagate there), so it finds the file handler whether it was attached
        to this logger or configured on the root via ``logging.basicConfig``.
        """
        logger = self.logger
        while logger:
            for handler in logger.handlers:
                base = getattr(handler, "baseFilename", None)
                if base:
                    return base
            if not getattr(logger, "propagate", True):
                break
            logger = logger.parent
        return None

    def _configure_mlflow(self) -> None:
        """Configure the MLflow tracking URI and experiment, if requested."""
        if not self.mlflow_cfg.get("enabled", True):
            return

        import mlflow  # pylint: disable=import-outside-toplevel

        tracking_uri = self.mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = self.mlflow_cfg.get("experiment_name")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    def seed_everything(self) -> int:
        """Seed Python, NumPy and torch RNGs from ``seed`` for reproducible fits.

        Without this the model initialisation and batch order differ run to run,
        so two fits of the same config can land far apart -- which makes a
        hyperparameter sweep unreadable. Returns the seed actually applied.
        """
        import random

        import numpy as np
        import torch

        seed = int(self.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return seed

    def _classifier_kwargs(self) -> Dict[str, Any]:
        """Build the kwargs passed to ``read_classifier_factory`` / the classifier's ``__init__``."""
        model_cfg = self.model_cfg
        return {
            "num_labels": model_cfg.get("num_labels", 2),
            "num_grg_labels": model_cfg.get("num_grg_labels", 39),
            "seq_length": self.config.get("max_sequence_length", 150),
            "foundation_model_path": model_cfg.get("foundation_model"),
            "classifier_head_implementation": model_cfg.get(
                "classifier_head_implementation", "grg_attention_based"
            ),
            "grg_label_column": model_cfg.get("grg_label_column", "dmr_ctype_label"),
            "dismir_flavor": model_cfg.get("flavor", "lstm"),
            "cancer_detector_prior_type": model_cfg.get(
                "cancer_detector_prior_type", "uniform"
            ),
            "soft_labels": model_cfg.get("soft_labels", False),
            # Architecture/loss knobs for hyperparameter tuning. Defaults match
            # the values that were hardcoded before they became configurable, so
            # omitting them reproduces earlier runs.
            "dropout_prob": model_cfg.get("dropout_prob", 0.2),
            "penalty_scale": model_cfg.get("penalty_scale", 1.0),
            "batch_size": self.config.get("prediction_batch_size", 2200),
            # LookupClassifier.LabelConfig fields, forwarded as-is. Unused by
            # other architectures and filtered by LabelConfig.from_dict.
            **model_cfg.get("lookup_config", {}),
            # MethylBert cwce/focal-loss tuning fields, forwarded as-is. Unused
            # by other architectures and ignored by their load()/__init__.
            **model_cfg.get("methylbert_config", {}),
        }

    def _mlflow_extra_params(self, classifier) -> Dict[str, Any]:
        """Run metadata to log as MLflow params beyond the training kwargs.

        Covers the pipeline-level knobs that shape the fit but are not part of
        ``training_cfg`` (label/split selection, read filtering, sequence
        length), plus each architecture's own relevant init params exposed via
        ``classifier.mlflow_fit_params`` under a ``model.*`` namespace. Nested
        dicts are flattened by the ``mlflow_tracked_fit`` decorator.
        """
        return {
            "data_format": self.data_format,
            "split_column": self.split_column,
            "label_column": self.label_column,
            "min_pattern_length": self.min_pattern_length,
            "deduplicate_train": self.deduplicate_train,
            "deduplicate_valid": self.deduplicate_valid,
            "dedup_keys": (
                self.dedup_keys
                if self.deduplicate_train or self.deduplicate_valid
                else None
            ),
            "max_sequence_length": self.config.get("max_sequence_length"),
            "model": classifier.mlflow_fit_params(),
        }

    def _load_split(self, dataset_path, split, fmt, declared_columns):
        """Load one split, projecting/filtering for columnar and renaming the label.

        Filtering (and, for a de-duplicated train split, collapsing) runs per
        part file, so peak RAM is bounded by one part rather than the split.
        """
        deduplicate = {"train": self.deduplicate_train, "valid": self.deduplicate_valid}.get(
            split, False
        )
        reduce_part = self._part_reducer(declared_columns, deduplicate)
        if fmt == "legacy":
            df = reduce_part(load_legacy_split(dataset_path, split))
        else:
            load_columns = declared_columns + (
                [k for k in self.dedup_keys if k not in declared_columns]
                if deduplicate
                else []
            )
            df = load_columnar_split(
                dataset_path,
                split,
                declared_columns=load_columns,
                split_column=self.split_column,
                per_fragment=reduce_part,
            )
            if df.empty:
                raise FileNotFoundError(
                    f"No rows for split {split!r} in {dataset_path}"
                )

        if deduplicate:
            self.logger.info(
                "De-duplicated %s split: %d reads -> %d unique rows",
                split,
                int(df["read_count"].sum()),
                len(df),
            )

        if self.label_column:
            df = apply_label_rename(
                df, self.label_column, self.model_cfg.get("soft_labels", False)
            )
        return df

    def _part_reducer(self, declared_columns, deduplicate):
        """Per-part transform: pattern-length filter, then optional de-duplication.

        Key columns loaded only to de-duplicate on are dropped afterwards. A
        group split across part files is merely collapsed less; its counts stay
        exact either way.
        """
        extra_keys = [k for k in self.dedup_keys if k not in declared_columns]

        def reduce_part(part):
            if self.min_pattern_length:
                part = apply_pattern_length_filter(part, self.min_pattern_length)
            if deduplicate:
                part = deduplicate_reads(part, self.dedup_keys).drop(
                    columns=extra_keys
                )
            return part

        return reduce_part
