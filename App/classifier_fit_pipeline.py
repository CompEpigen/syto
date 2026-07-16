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

        datasets = config.get("datasets", ["all"])
        if isinstance(datasets, str):
            datasets = ["all"] if datasets == "all" else [datasets]
        self.datasets: List[str] = datasets

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Fit the configured classifier on each requested dataset."""
        self._configure_mlflow()

        self.logger.info(
            f"Starting classifier fit workflow for architecture: {self.model_arch}"
        )

        for dataset_name in self.datasets:
            self._fit_dataset(dataset_name)

    # ═══════════════════════════════════════════════════════════════
    #  Per-dataset fitting
    # ═══════════════════════════════════════════════════════════════

    def _fit_dataset(self, dataset_name: str) -> None:
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

        try:
            train_df = self._load_split(dataset_path, "train", fmt, declared_columns)
            self.logger.info(f"Loaded train split: {len(train_df)} rows")
        except FileNotFoundError as e:
            self.logger.error(f"Error loading train data: {e}")
            return

        try:
            val_df = self._load_split(dataset_path, "valid", fmt, declared_columns)
            self.logger.info(f"Loaded valid split: {len(val_df)} rows")
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
            **self.training_cfg,
        )

        self.logger.info(f"Finished processing {dataset_name}")

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

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
            "max_sequence_length": self.config.get("max_sequence_length"),
            "model": classifier.mlflow_fit_params(),
        }

    def _load_split(self, dataset_path, split, fmt, declared_columns):
        """Load one split, projecting/filtering for columnar and renaming the label."""
        if fmt == "legacy":
            df = load_legacy_split(dataset_path, split)
        else:
            df = load_columnar_split(
                dataset_path,
                split,
                declared_columns=declared_columns,
                split_column=self.split_column,
            )
            if df.empty:
                raise FileNotFoundError(
                    f"No rows for split {split!r} in {dataset_path}"
                )

        if self.min_pattern_length:
            df = apply_pattern_length_filter(df, self.min_pattern_length)

        if self.label_column:
            df = apply_label_rename(
                df, self.label_column, self.model_cfg.get("soft_labels", False)
            )
        return df
