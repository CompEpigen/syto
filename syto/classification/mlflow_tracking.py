"""
MLflow tracking decorator for ``AbstractReadClassifier.fit_classificaton`` implementations.

Rather than re-introducing a monolithic "experiment wrapper" per model
architecture, ``mlflow_tracked_fit`` wraps any ``fit_classificaton`` method that
follows the ``AbstractReadClassifier`` signature
(``fit_classificaton(self, train_df, val_df=None, output_dir=None, **kwargs)``) and
records the call as an MLflow run:

- hyperparameters from ``kwargs`` (and dataset sizes) are logged as params
- everything written to ``output_dir`` by ``fit_classificaton`` is logged as run
  artifacts
- train/validation metrics that ``fit_classificaton`` already computed (exposed via
  ``classifier.history``, a list of per-fit metric dicts with ``train_*``/
  ``val_*`` keys) are logged as metrics
"""

import dataclasses
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Dict

import mlflow

_module_logger = logging.getLogger(__name__)

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _flatten_for_mlflow(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    """Recursively flatten dicts/dataclasses into MLflow-loggable scalar params."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)

    if isinstance(value, dict):
        for key, sub_value in value.items():
            _flatten_for_mlflow(
                f"{prefix}.{key}" if prefix else str(key), sub_value, out
            )
    elif isinstance(value, _SCALAR_TYPES):
        out[prefix] = value
    # Non-scalar, non-dict values (callables, tensors, ...) are not loggable as params.


def _fit_metrics(classifier) -> Dict[str, float]:
    """Pull the ``train_*``/``val_*`` metrics from the most recent ``fit_classificaton`` call.

    Classifiers record per-fit metrics (computed via ``compute_metrics`` /
    ``compute_metrics_soft_labels``) in ``self.history``, a list of dicts
    appended to on each ``fit_classificaton`` call. Returns the scalar ``train_*``/
    ``val_*`` entries of the last record, or ``{}`` if unavailable.
    """
    history = getattr(classifier, "history", None)
    if not history:
        return {}

    last_record = history[-1]
    return {
        key: value
        for key, value in last_record.items()
        if isinstance(value, (int, float))
        and (key.startswith("train_") or key.startswith("val_"))
    }


def mlflow_tracked_fit(fit_classificaton: Callable) -> Callable:
    """Decorate ``fit_classificaton`` so each call is recorded as an MLflow run.

    Recognized (and consumed) ``kwargs``:

    - ``track_with_mlflow`` (bool, default ``True``): set to ``False`` to run
      ``fit_classificaton`` without any MLflow involvement (e.g. in unit tests).
    - ``mlflow_run_name`` (str): name for the MLflow run.
    - ``mlflow_tags`` (dict): extra tags to set on the run.
    - ``mlflow_extra_params`` (dict): additional params to log, flattened like
      the training kwargs. Log-only: consumed here and never forwarded to the
      wrapped ``fit_classificaton``. Used by the caller to record run metadata
      that is not a fit hyperparameter (dataset/split/label config, and each
      architecture's relevant init params via ``mlflow_fit_params``).

    The tracking URI and experiment are expected to already be configured
    (e.g. via ``mlflow.set_tracking_uri`` / ``mlflow.set_experiment``) before
    ``fit_classificaton`` is called.
    """

    @functools.wraps(fit_classificaton)
    def wrapper(self, train_df, val_df=None, output_dir=None, **kwargs):
        if not kwargs.pop("track_with_mlflow", True):
            return fit_classificaton(
                self, train_df, val_df=val_df, output_dir=output_dir, **kwargs
            )

        run_name = kwargs.pop("mlflow_run_name", None)
        tags = kwargs.pop("mlflow_tags", None) or {}
        extra_params = kwargs.pop("mlflow_extra_params", None) or {}

        with mlflow.start_run(
            run_name=run_name, tags=tags, nested=mlflow.active_run() is not None
        ):
            params: Dict[str, Any] = {
                "classifier": type(self).__name__,
                "train_rows": len(train_df),
            }
            if val_df is not None:
                params["val_rows"] = len(val_df)
            if output_dir is not None:
                params["output_dir"] = str(output_dir)
            for key, value in kwargs.items():
                _flatten_for_mlflow(key, value, params)
            _flatten_for_mlflow("", extra_params, params)
            mlflow.log_params(params)

            try:
                result = fit_classificaton(
                    self, train_df, val_df=val_df, output_dir=output_dir, **kwargs
                )
            except Exception as exc:
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error", str(exc))
                raise

            if output_dir is not None and Path(output_dir).exists():
                mlflow.log_artifacts(str(output_dir))

            metrics = _fit_metrics(result)
            if metrics:
                mlflow.log_metrics(metrics)

            return result

    return wrapper
