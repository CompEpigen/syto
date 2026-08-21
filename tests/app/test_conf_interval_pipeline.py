"""Confidence-interval pipeline: config contract and calibration-method names.

Two things about this pipeline are easy to break silently, and both did break
once: the key it reads the deconvolver list from, and the spelling of the
linear clip-normalize method.  A wrong key yields zero deconvolvers and an
error about target proportions that points nowhere near the cause; a wrong
method name is not an error at all, it just drops one of the four calibration
methods from the summary.  Both are pinned here against the names the
calibration pipeline actually writes.
"""

import json
import logging
import os
import tempfile
import unittest

import numpy as np

from syto.app.calibration_pipeline import LINEAR_NORM_METHODS
from syto.app.cli import validate_config
from syto.app.conf_interval_pipeline import (
    CALIBRATION_METHODS,
    ConfidenceIntervalPipeline,
)

LABELS = {0: "ct_a", 1: "ct_b", 2: "ct_c"}
DECONVOLVERS = ["mlp", "nnls"]


def _written_method_names():
    """The method names fit-calibration turns into ``<name>_predictions.npz``.

    Mirrors CalibratorFittingPipeline: ``uncalibrated``, one entry per
    LINEAR_NORM_METHODS with hyphens swapped for underscores and a ``linear_``
    prefix, plus ``vector_scaling``.
    """
    linear = [f"linear_{m.replace('-', '_')}" for m in LINEAR_NORM_METHODS]
    return ["uncalibrated", *linear, "vector_scaling"]


def _make_calibration_dir(root, methods):
    rng = np.random.default_rng(0)
    target = rng.dirichlet(np.ones(len(LABELS)), size=8)
    for name in DECONVOLVERS:
        sub = os.path.join(root, f"{name}_calibrators_and_predictions")
        os.makedirs(sub, exist_ok=True)
        for method in methods:
            pred = np.clip(target + rng.normal(0, 0.01, target.shape), 0, None)
            pred /= pred.sum(axis=1, keepdims=True)
            np.savez(
                os.path.join(sub, f"{method}_predictions.npz"),
                test_pred=pred,
                test_target=target,
            )
    return target


class TestCalibrationMethodNames(unittest.TestCase):
    """The names this pipeline looks for must be the ones calibration writes."""

    def test_every_written_method_is_looked_for(self):
        missing = set(_written_method_names()) - set(CALIBRATION_METHODS)
        self.assertEqual(
            missing,
            set(),
            f"fit-calibration writes {sorted(missing)} but confidence-intervals "
            "never looks for them; they would be dropped from the summary "
            "without an error",
        )


class TestDeconvolverListKey(unittest.TestCase):
    """The deconvolver list is read from ``deconvolver_names``."""

    def test_config_requires_deconvolver_names(self):
        config = {"calibration_results_dir": "x", "labels_dict_path": "y"}
        with self.assertRaises(ValueError) as ctx:
            validate_config(config, "confidence_intervals")
        self.assertIn("deconvolver_names", str(ctx.exception))

    def test_deconvolver_names_drives_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = os.path.join(tmp, "calibration_results")
            _make_calibration_dir(calib, _written_method_names())
            labels_path = os.path.join(tmp, "labels.json")
            with open(labels_path, "w", encoding="utf-8") as fh:
                json.dump({str(k): v for k, v in LABELS.items()}, fh)

            config = {
                "calibration_results_dir": calib,
                "labels_dict_path": labels_path,
                "output_dir": os.path.join(tmp, "out"),
                "deconvolver_names": DECONVOLVERS,
                "n_resamples": 20,
            }
            validate_config(config, "confidence_intervals")
            logger = logging.getLogger("test_conf_interval")
            logger.setLevel(logging.CRITICAL)
            results = ConfidenceIntervalPipeline(config=config, logger=logger).run()

            self.assertEqual(sorted(results), sorted(DECONVOLVERS))
            for name in DECONVOLVERS:
                self.assertEqual(
                    sorted(results[name]),
                    sorted(_written_method_names()),
                    f"{name}: not every method fit-calibration wrote was evaluated",
                )


if __name__ == "__main__":
    unittest.main()
