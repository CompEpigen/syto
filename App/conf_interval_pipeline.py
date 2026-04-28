"""
In this pipeline, we use the stored predictions from the output
of the calibration pipeline to recompute metrics with confidence intervals using the bootstrap method.
The results are stored in a new .csv file in the output directory.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from methyldl.deconvolution.evaluation import compute_deconvolution_metrics

CALIBRATION_METHODS = [
    "uncalibrated",
    "linear_clip0_normalize",
    "linear_simplex_projection",
    "vector_scaling",
]


class ConfidenceIntervalPipeline:
    """Recompute deconvolution metrics with bootstrap confidence intervals.

    Reads the saved prediction ``.npz`` files produced by
    :class:`CalibratorFittingPipeline` and evaluates each
    deconvolver / calibration-method pair using
    :func:`compute_deconvolution_metrics` with ``compute_ci=True``.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.  Expected keys:

        * ``calibration_results_dir`` - directory written by the
          calibration pipeline (contains ``<name>_calibrators_and_predictions/``
          sub-directories).
        * ``output_dir`` - where the summary CSV will be saved.
        * ``labels_dict_path`` - path to a JSON label dictionary.
        * ``deconvolvers`` - list of dicts, each with a ``name`` key.
        * ``calibration_methods`` (optional) - list of method names to
          evaluate.  Defaults to :data:`CALIBRATION_METHODS`.
        * ``n_resamples`` (optional) - bootstrap resamples (default 10 000).
        * ``confidence_level`` (optional) - CI level (default 0.95).
        * ``random_state`` (optional) - RNG seed (default 42).
    logger : logging.Logger
        Logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Labels
        labels_dict_path = config["labels_dict_path"]
        with open(labels_dict_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}

        # Directories
        self.calibration_results_dir = config["calibration_results_dir"]
        self.output_dir = config.get("output_dir", self.calibration_results_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Bootstrap parameters
        self.n_resamples = config.get("n_resamples", 10_000)
        self.confidence_level = config.get("confidence_level", 0.95)
        self.random_state = config.get("random_state", 42)

        # Which calibration methods to evaluate
        self.calibration_methods: List[str] = config.get(
            "calibration_methods", CALIBRATION_METHODS
        )

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """Execute the full confidence-interval pipeline.

        Returns
        -------
        dict
            ``{deconv_name: {method: metrics_dict}}``
        """
        deconvolvers_cfg = self.config.get("deconvolvers", [])
        deconv_names = [cfg["name"] for cfg in deconvolvers_cfg]

        # ── Stage 1: Load predictions ─────────────────────────────
        predictions, target_proportions = self._load_predictions(deconv_names)

        # ── Stage 2: Evaluate with CIs ────────────────────────────
        results: Dict[str, Any] = {}
        for deconv_name, methods_preds in predictions.items():
            self.logger.info(f"Computing CI metrics for deconvolver: {deconv_name}")
            results[deconv_name] = {}

            for method, test_pred in methods_preds.items():
                self.logger.info(f"  [{deconv_name}] {method} ...")
                try:
                    metrics = compute_deconvolution_metrics(
                        test_pred,
                        target_proportions,
                        class_names=list(self.labels_dict.values()),
                        n_resamples=self.n_resamples,
                        confidence_level=self.confidence_level,
                        random_state=self.random_state,
                        compute_ci=True,
                        return_per_sample=True,
                    )
                    results[deconv_name][method] = metrics
                    self.logger.info(
                        f"    MSE={metrics['mse']:.6f}  "
                        f"CI=[{metrics.get('mse_ci_lower', float('nan')):.6f}, "
                        f"{metrics.get('mse_ci_upper', float('nan')):.6f}]"
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    self.logger.error(
                        f"    Failed for {deconv_name}/{method}: {e}",
                        exc_info=True,
                    )

        # ── Stage 3: Summary ──────────────────────────────────────
        self._log_summary(results)
        self._save_summary_csv(results)
        self._save_per_sample_csv(results)

        return results

    # ═══════════════════════════════════════════════════════════════
    #  Data loading
    # ═══════════════════════════════════════════════════════════════

    def _load_predictions(
        self, deconv_names: List[str]
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
        """Load saved predictions from calibration pipeline output.

        Returns
        -------
        predictions : dict
            ``{deconv_name: {method: test_pred_array}}``
        target_proportions : np.ndarray
            Ground-truth proportions (loaded once from the first
            ``uncalibrated_predictions.npz``).
        """
        self.logger.info(
            f"Loading predictions from {self.calibration_results_dir}"
        )

        predictions: Dict[str, Dict[str, np.ndarray]] = {}
        target_proportions: np.ndarray | None = None

        for deconv_name in deconv_names:
            deconv_dir = os.path.join(
                self.calibration_results_dir,
                f"{deconv_name}_calibrators_and_predictions",
            )
            if not os.path.isdir(deconv_dir):
                self.logger.warning(f"  {deconv_dir} not found, skipping")
                continue

            predictions[deconv_name] = {}

            for method in self.calibration_methods:
                if method == "uncalibrated":
                    pred_path = os.path.join(
                        deconv_dir, "uncalibrated_predictions.npz"
                    )
                elif method == "vector_scaling":
                    pred_path = os.path.join(
                        deconv_dir, "vector_scaling_predictions.npz"
                    )
                else:
                    # e.g. "linear_clip0_normalize" -> "linear_clip0_normalize_predictions.npz"
                    pred_path = os.path.join(
                        deconv_dir, f"{method}_predictions.npz"
                    )

                if not os.path.exists(pred_path):
                    self.logger.warning(
                        f"  [{deconv_name}] {method}: {pred_path} not found, skipping"
                    )
                    continue

                data = np.load(pred_path)
                predictions[deconv_name][method] = data["test_pred"]

                # Load target proportions once
                if target_proportions is None and "test_target" in data:
                    target_proportions = data["test_target"]

            self.logger.info(
                f"  {deconv_name}: loaded methods {list(predictions[deconv_name].keys())}"
            )

        if target_proportions is None:
            raise ValueError(
                "Could not load target proportions from any "
                "uncalibrated_predictions.npz file. Ensure at least one "
                "deconvolver has 'uncalibrated' predictions saved."
            )

        self.logger.info(
            f"Loaded predictions for {len(predictions)} deconvolver(s), "
            f"target shape: {target_proportions.shape}"
        )
        return predictions, target_proportions

    # ═══════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════

    def _log_summary(self, results: Dict[str, Dict[str, dict]]) -> None:
        """Log a summary table of all results."""
        self.logger.info("=" * 70)
        self.logger.info("CONFIDENCE INTERVAL SUMMARY")
        self.logger.info("=" * 70)

        for deconv_name, methods in results.items():
            self.logger.info(f"  Deconvolver: {deconv_name}")
            for method, m in methods.items():
                self.logger.info(
                    f"    {method:30s}  "
                    f"R2={m.get('overall_r2', m.get('r2', float('nan'))):.6f}  "
                    f"MAE={m['mae']:.6f} [{m.get('mae_ci_lower', float('nan')):.6f}, {m.get('mae_ci_upper', float('nan')):.6f}]  "
                    f"MSE={m['mse']:.6f} [{m.get('mse_ci_lower', float('nan')):.6f}, {m.get('mse_ci_upper', float('nan')):.6f}]  "
                    f"KL={m['kl']:.6f} [{m.get('kl_ci_lower', float('nan')):.6f}, {m.get('kl_ci_upper', float('nan')):.6f}]"
                )

        self.logger.info("=" * 70)

    def _save_summary_csv(self, results: Dict[str, Dict[str, dict]]) -> None:
        """Save a CSV with one row per (deconvolver, calibration_method) pair."""
        rows: list = []

        # Collect all metric keys from the first available result
        metric_keys: List[str] = []
        for methods in results.values():
            for m in methods.values():
                metric_keys = sorted(m.keys())
                break
            if metric_keys:
                break

        for deconv_name, methods in results.items():
            for method, m in methods.items():
                row = {
                    "deconvolver": deconv_name,
                    "calibration_method": method,
                }
                for key in metric_keys:
                    row[key] = m.get(key)
                rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "confidence_intervals_summary.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CI summary CSV to {csv_path}")

    def _save_per_sample_csv(self, results: Dict[str, Dict[str, dict]]) -> None:
        """Save a CSV with per-sample MSE, MAE and KL for each (deconvolver, method) pair."""
        rows: list = []

        for deconv_name, methods in results.items():
            for method, m in methods.items():
                mse_arr = m.get("mse_per_sample")
                mae_arr = m.get("mae_per_sample")
                kl_arr = m.get("kl_per_sample")
                if mse_arr is None:
                    continue
                for i in range(len(mse_arr)):
                    rows.append(
                        {
                            "deconvolver": deconv_name,
                            "calibration_method": method,
                            "sample_index": i,
                            "mse": mse_arr[i],
                            "mae": mae_arr[i],
                            "kl": kl_arr[i],
                        }
                    )

        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.output_dir, "per_sample_metrics.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved per-sample metrics CSV to {csv_path}")
