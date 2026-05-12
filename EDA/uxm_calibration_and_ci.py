"""
UXM Calibration & Confidence Interval Script
=============================================

This script:
1. Loads the UXM pseudobulk "multinomial" predictions and target proportions
   from the ExtendedProportions experiment directory.
2. Fits calibrators (LinearCalibrator, VectorScalingCalibratorCV) on the
   validation set predictions — saving calibrator weights and calibrated
   predictions in the same fashion as App/calibration_pipeline.py.
3. Computes confidence intervals on the test set predictions (uncalibrated
   and calibrated) and saves the results in the same fashion as
   App/conf_interval_pipeline.py.
"""

import json
import logging
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────
from methyldl.deconvolution.evaluation import compute_deconvolution_metrics
from methyldl.deconvolution.linear_calibrator import LinearCalibrator
from methyldl.deconvolution.vector_scaling_calibrator import VectorScalingCalibratorCV

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════

DATA_DIR = (
    "/staging/leuven/stg_00118/methylDL/experiments/"
    "ExtendedProportions/pseudobulk/uxm_results/pseudobulk"
)
LABELS_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "App",
    "labels_dict.json",
)
OUTPUT_DIR = os.path.join(DATA_DIR, "calibration_results")

LINEAR_NORM_METHODS = [
    "clip0-normalize",
    "simplex-projection",
]
CALIBRATION_METHODS = [
    "uncalibrated",
    "linear_clip0_normalize",
    "linear_simplex_projection",
    "vector_scaling",
]

# Bootstrap CI parameters
N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

DECONV_NAME = "uxm"

# ══════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════


def load_data():
    """Load UXM multinomial predictions and target proportions."""
    logger.info("Loading data from %s", DATA_DIR)

    val_pred = np.load(
        os.path.join(DATA_DIR, "uxm_results_valid_uniform_multinomial.npz")
    )["arr_0"]
    test_pred = np.load(
        os.path.join(DATA_DIR, "uxm_results_test_uniform_multinomial.npz")
    )["arr_0"]
    y_valid = np.load(os.path.join(DATA_DIR, "target_proportions_valid.npz"))["arr_0"]
    y_test = np.load(os.path.join(DATA_DIR, "target_proportions_test.npz"))["arr_0"]

    logger.info(
        "  val_pred=%s  test_pred=%s  y_valid=%s  y_test=%s",
        val_pred.shape,
        test_pred.shape,
        y_valid.shape,
        y_test.shape,
    )
    return val_pred, test_pred, y_valid, y_test


def load_labels_dict():
    """Load the labels dictionary."""
    with open(LABELS_DICT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels_dict = {int(k): v for k, v in raw.items()}
    logger.info("Loaded %d labels from %s", len(labels_dict), LABELS_DICT_PATH)
    return labels_dict


# ══════════════════════════════════════════════════════════════════
#  Part 1 — Calibration (mirrors App/calibration_pipeline.py)
# ══════════════════════════════════════════════════════════════════


def run_calibration(val_pred, test_pred, y_valid, y_test, deconv_out):
    """Fit calibrators and save predictions, mirroring CalibratorFittingPipeline."""

    results = {}

    # ── 1a: Uncalibrated ──────────────────────────────────────────
    logger.info("[%s] Saving uncalibrated predictions ...", DECONV_NAME)
    uncal_path = os.path.join(deconv_out, "uncalibrated_predictions.npz")
    if os.path.exists(uncal_path):
        logger.info("  Found existing file, loading: %s", uncal_path)
        data = np.load(uncal_path)
        val_pred_loaded = data["val_pred"]
        test_pred_loaded = data["test_pred"]
    else:
        val_pred_loaded = val_pred
        test_pred_loaded = test_pred
        np.savez_compressed(
            uncal_path,
            val_pred=val_pred,
            test_pred=test_pred,
            val_target=y_valid,
            test_target=y_test,
        )

    val_metrics = compute_deconvolution_metrics(val_pred_loaded, y_valid)
    test_metrics = compute_deconvolution_metrics(test_pred_loaded, y_test)
    logger.info(
        "  Uncalibrated  val MSE=%.6f  test MSE=%.6f",
        val_metrics["mse"],
        test_metrics["mse"],
    )
    results["uncalibrated"] = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # ── 1b: Linear calibration ────────────────────────────────────
    logger.info("[%s] Fitting linear calibrator ...", DECONV_NAME)
    linear_cal_path = os.path.join(deconv_out, "linear_calibrator.npz")
    if os.path.exists(linear_cal_path):
        logger.info("  Found existing calibrator, loading: %s", linear_cal_path)
        linear_calibrator = LinearCalibrator()
        linear_calibrator.load_calibration_parameters(linear_cal_path)
    else:
        linear_calibrator = LinearCalibrator()
        linear_calibrator.fit(val_pred_loaded, y_valid)
        linear_calibrator.save_calibration_parameters(linear_cal_path)
        logger.info("  Saved linear calibrator to %s", linear_cal_path)

    for norm_method in LINEAR_NORM_METHODS:
        short_name = norm_method.replace("-", "_")
        pred_path = os.path.join(deconv_out, f"linear_{short_name}_predictions.npz")
        if os.path.exists(pred_path):
            logger.info(
                "  Found existing predictions for '%s', loading: %s",
                norm_method,
                pred_path,
            )
            data = np.load(pred_path)
            val_calib = data["val_pred"]
            test_calib = data["test_pred"]
        else:
            val_calib, _ = linear_calibrator.predict(
                val_pred_loaded, norm_method=norm_method
            )
            test_calib, _ = linear_calibrator.predict(
                test_pred_loaded, norm_method=norm_method
            )
            np.savez_compressed(pred_path, val_pred=val_calib, test_pred=test_calib)

        val_m = compute_deconvolution_metrics(val_calib, y_valid)
        test_m = compute_deconvolution_metrics(test_calib, y_test)
        logger.info(
            "  Linear (%s)  val MSE=%.6f  test MSE=%.6f",
            norm_method,
            val_m["mse"],
            test_m["mse"],
        )
        results[f"linear_{short_name}"] = {
            "val_metrics": val_m,
            "test_metrics": test_m,
        }

    # ── 1c: VectorScaling calibration ─────────────────────────────
    logger.info("[%s] Fitting VectorScalingCalibratorCV ...", DECONV_NAME)
    vs_path = os.path.join(deconv_out, "vector_scaling_calibrator.npz")
    if os.path.exists(vs_path):
        logger.info("  Found existing calibrator, loading: %s", vs_path)
        vs_calibrator = VectorScalingCalibratorCV()
        vs_calibrator.load(vs_path)
    else:
        vs_calibrator = VectorScalingCalibratorCV(
            reg_lambda_list=[0.0, 1e-4, 1e-3],
            lr_list=[1e-3, 1e-2],
            max_iter_list=[1000],
            optimizer="adam",
            scheduler="plateau",
            patience=50,
            n_folds=5,
            batch_size=None,
            verbose=True,
            plateau_factor=0.5,
            plateau_patience=10,
        )
        vs_calibrator.fit(val_pred_loaded, y_valid)
        vs_calibrator.save(vs_path)
        logger.info("  Saved VectorScaling calibrator to %s", vs_path)

    logger.info(
        "  Best CV loss: %.6f, params: %s",
        vs_calibrator.best_cv_val_loss_,
        vs_calibrator.best_params_,
    )

    vs_pred_path = os.path.join(deconv_out, "vector_scaling_predictions.npz")
    if os.path.exists(vs_pred_path):
        logger.info("  Found existing VS predictions, loading: %s", vs_pred_path)
        data = np.load(vs_pred_path)
        val_vs = data["val_pred"]
        test_vs = data["test_pred"]
    else:
        val_vs = vs_calibrator.predict_proba(val_pred_loaded)
        test_vs = vs_calibrator.predict_proba(test_pred_loaded)
        np.savez_compressed(vs_pred_path, val_pred=val_vs, test_pred=test_vs)

    val_vs_m = compute_deconvolution_metrics(val_vs, y_valid)
    test_vs_m = compute_deconvolution_metrics(test_vs, y_test)
    logger.info(
        "  VectorScaling  val MSE=%.6f  test MSE=%.6f",
        val_vs_m["mse"],
        test_vs_m["mse"],
    )
    results["vector_scaling"] = {
        "val_metrics": val_vs_m,
        "test_metrics": test_vs_m,
        "best_params": vs_calibrator.best_params_,
        "best_cv_val_loss": vs_calibrator.best_cv_val_loss_,
    }

    return results


def save_calibration_summary_csv(results, output_dir):
    """Save a CSV with one row per calibration method (mirrors CalibratorFittingPipeline)."""
    rows = []
    for calib_name, data in results.items():
        if "test_metrics" not in data:
            continue
        m = data["test_metrics"]
        rows.append(
            {
                "deconvolver": DECONV_NAME,
                "calibration_method": calib_name,
                "overall_r2": m["r2"],
                "loa_lower": m["loa_lower"],
                "loa_upper": m["loa_upper"],
                "worst_class_loa_lower": m["worst_class_loa_lower"],
                "worst_class_loa_upper": m["worst_class_loa_upper"],
                "mae": m["mae"],
                "mse": m["mse"],
                "kl": m["kl"],
            }
        )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "calibration_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved calibration summary CSV to %s", csv_path)


# ══════════════════════════════════════════════════════════════════
#  Part 2 — Confidence Intervals (mirrors App/conf_interval_pipeline.py)
# ══════════════════════════════════════════════════════════════════


def run_confidence_intervals(deconv_out, labels_dict, output_dir):
    """Load saved predictions, compute CI metrics, and save summaries."""

    class_names = list(labels_dict.values())

    # ── Load predictions ──────────────────────────────────────────
    predictions = {}
    target_proportions = None

    for method in CALIBRATION_METHODS:
        if method == "uncalibrated":
            pred_path = os.path.join(deconv_out, "uncalibrated_predictions.npz")
        elif method == "vector_scaling":
            pred_path = os.path.join(deconv_out, "vector_scaling_predictions.npz")
        else:
            pred_path = os.path.join(deconv_out, f"{method}_predictions.npz")

        if not os.path.exists(pred_path):
            logger.warning("  %s: %s not found, skipping", method, pred_path)
            continue

        data = np.load(pred_path)
        predictions[method] = data["test_pred"]

        if target_proportions is None and "test_target" in data:
            target_proportions = data["test_target"]

    if target_proportions is None:
        raise ValueError(
            "Could not load target proportions from uncalibrated_predictions.npz"
        )

    logger.info(
        "Loaded predictions for %d methods, target shape: %s",
        len(predictions),
        target_proportions.shape,
    )

    # ── Compute metrics with CIs ──────────────────────────────────
    results = {}
    for method, test_pred in predictions.items():
        logger.info("  [%s] %s ...", DECONV_NAME, method)
        try:
            metrics = compute_deconvolution_metrics(
                test_pred,
                target_proportions,
                class_names=class_names,
                n_resamples=N_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                random_state=RANDOM_STATE,
                compute_ci=True,
                return_per_sample=True,
            )
            results[method] = metrics
            logger.info(
                "    MSE=%.6f  CI=[%.6f, %.6f]",
                metrics["mse"],
                metrics.get("mse_ci_lower", float("nan")),
                metrics.get("mse_ci_upper", float("nan")),
            )
        except Exception:
            logger.error("    Failed for %s/%s", DECONV_NAME, method, exc_info=True)

    # ── Save CI summary CSV ───────────────────────────────────────
    save_ci_summary_csv(results, output_dir)
    save_per_sample_csv(results, output_dir)

    # ── Log summary ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CONFIDENCE INTERVAL SUMMARY")
    logger.info("=" * 70)
    for method, m in results.items():
        logger.info(
            "  %30s  R2=%.6f  MAE=%.6f [%.6f, %.6f]  MSE=%.6f [%.6f, %.6f]  "
            "KL=%.6f [%.6f, %.6f]",
            method,
            m.get("overall_r2", m.get("r2", float("nan"))),
            m["mae"],
            m.get("mae_ci_lower", float("nan")),
            m.get("mae_ci_upper", float("nan")),
            m["mse"],
            m.get("mse_ci_lower", float("nan")),
            m.get("mse_ci_upper", float("nan")),
            m["kl"],
            m.get("kl_ci_lower", float("nan")),
            m.get("kl_ci_upper", float("nan")),
        )
    logger.info("=" * 70)

    return results


def save_ci_summary_csv(results, output_dir):
    """Save a CSV with one row per calibration method (mirrors ConfidenceIntervalPipeline)."""
    rows = []

    metric_keys = []
    for m in results.values():
        metric_keys = sorted(m.keys())
        break

    for method, m in results.items():
        row = {
            "deconvolver": DECONV_NAME,
            "calibration_method": method,
        }
        for key in metric_keys:
            row[key] = m.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "confidence_intervals_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved CI summary CSV to %s", csv_path)


def save_per_sample_csv(results, output_dir):
    """Save per-sample MSE, MAE, KL (mirrors ConfidenceIntervalPipeline)."""
    rows = []
    for method, m in results.items():
        mse_arr = m.get("mse_per_sample")
        mae_arr = m.get("mae_per_sample")
        kl_arr = m.get("kl_per_sample")
        if mse_arr is None:
            continue
        for i in range(len(mse_arr)):
            rows.append(
                {
                    "deconvolver": DECONV_NAME,
                    "calibration_method": method,
                    "sample_index": i,
                    "mse": mse_arr[i],
                    "mae": mae_arr[i],
                    "kl": kl_arr[i],
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "per_sample_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved per-sample metrics CSV to %s", csv_path)


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    deconv_out = os.path.join(OUTPUT_DIR, f"{DECONV_NAME}_calibrators_and_predictions")
    os.makedirs(deconv_out, exist_ok=True)

    labels_dict = load_labels_dict()
    val_pred, test_pred, y_valid, y_test = load_data()

    # ── Part 1: Calibration ───────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PART 1: CALIBRATION")
    logger.info("=" * 70)
    calib_results = run_calibration(val_pred, test_pred, y_valid, y_test, deconv_out)
    save_calibration_summary_csv(calib_results, OUTPUT_DIR)

    # ── Part 2: Confidence Intervals ──────────────────────────────
    logger.info("=" * 70)
    logger.info("PART 2: CONFIDENCE INTERVALS")
    logger.info("=" * 70)
    run_confidence_intervals(deconv_out, labels_dict, OUTPUT_DIR)

    logger.info("All done. Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
