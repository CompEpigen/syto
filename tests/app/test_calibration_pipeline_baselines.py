"""Calibration of baselines whose predictions are already stored as parquet.

The pseudobulk deconvolution pipeline writes one long-format parquet per split
(``pb_index``, ``cell_type``, ``target_proportion``, ``<model_key>``).  These
tests cover reading those back into the ``(n_samples, n_classes)`` matrices the
calibrators expect, and driving the calibration pipeline from them without any
HDF5 file, feature mask, or saved deconvolver.
"""

import json
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from syto.app.calibration_pipeline import (
    CalibratorFittingPipeline,
    load_predictions_from_parquet,
)
from syto.app.cli import validate_config

LABELS = {0: "ct_a", 1: "ct_b", 2: "ct_c"}


def _long_frame(pred, target, column, labels=LABELS, shuffle=True):
    """Build a long-format parquet frame from (n_pb, n_ct) matrices."""
    rows = []
    for pb_index in range(pred.shape[0]):
        for i, name in labels.items():
            rows.append(
                {
                    "pb_index": pb_index,
                    "cell_type": name,
                    "target_proportion": float(target[pb_index, i]),
                    column: float(pred[pb_index, i]),
                }
            )
    df = pd.DataFrame(rows)
    if shuffle:
        # The producing pipeline collects rows via imap_unordered, so on-disk
        # row order carries no meaning.
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return df


def _write_baseline_dir(directory, column="epidish", n_pb=8, splits=("valid", "test")):
    """Write baseline parquets and return {split: (pred, target)}."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(7)
    expected = {}
    for offset, split in enumerate(splits):
        pred = rng.random((n_pb, len(LABELS))) + offset
        target = rng.random((n_pb, len(LABELS)))
        target = target / target.sum(axis=1, keepdims=True)
        _long_frame(pred, target, column).to_parquet(
            os.path.join(directory, f"{split}_deconvolution_results.parquet"),
            index=False,
        )
        expected[split] = (pred, target)
    return expected


class TestLoadPredictionsFromParquet(unittest.TestCase):
    def test_pivots_shuffled_rows_into_labels_dict_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _write_baseline_dir(tmp)

            loaded = load_predictions_from_parquet(tmp, LABELS)

            for split, (pred, target) in expected.items():
                got_pred, got_target = loaded[split]
                np.testing.assert_allclose(got_pred, pred)
                np.testing.assert_allclose(got_target, target)

    def test_infers_the_prediction_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _write_baseline_dir(tmp, column="celfie_300_steps")

            loaded = load_predictions_from_parquet(tmp, LABELS)

            np.testing.assert_allclose(loaded["valid"][0], expected["valid"][0])

    def test_explicit_prediction_column_selects_among_several(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _write_baseline_dir(tmp, column="epidish")
            path = os.path.join(tmp, "valid_deconvolution_results.parquet")
            df = pd.read_parquet(path)
            df["other_model"] = 0.0
            df.to_parquet(path, index=False)

            loaded = load_predictions_from_parquet(
                tmp, LABELS, splits=("valid",), prediction_column="epidish"
            )

            np.testing.assert_allclose(loaded["valid"][0], expected["valid"][0])

    def test_ambiguous_prediction_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_baseline_dir(tmp)
            path = os.path.join(tmp, "valid_deconvolution_results.parquet")
            df = pd.read_parquet(path)
            df["other_model"] = 0.0
            df.to_parquet(path, index=False)

            with self.assertRaises(ValueError) as ctx:
                load_predictions_from_parquet(tmp, LABELS, splits=("valid",))
            self.assertIn("prediction_column", str(ctx.exception))

    def test_missing_split_parquet_names_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_baseline_dir(tmp, splits=("test",))

            with self.assertRaises(FileNotFoundError) as ctx:
                load_predictions_from_parquet(tmp, LABELS, splits=("valid",))
            self.assertIn("valid_deconvolution_results.parquet", str(ctx.exception))

    def test_cell_type_missing_from_parquet_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_baseline_dir(tmp)
            path = os.path.join(tmp, "valid_deconvolution_results.parquet")
            df = pd.read_parquet(path)
            df = df[df["cell_type"] != "ct_c"]
            df.to_parquet(path, index=False)

            with self.assertRaises(ValueError) as ctx:
                load_predictions_from_parquet(tmp, LABELS, splits=("valid",))
            self.assertIn("ct_c", str(ctx.exception))

    def test_duplicate_pb_index_cell_type_pair_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_baseline_dir(tmp)
            path = os.path.join(tmp, "valid_deconvolution_results.parquet")
            df = pd.read_parquet(path)
            pd.concat([df, df.head(1)]).to_parquet(path, index=False)

            with self.assertRaises(ValueError) as ctx:
                load_predictions_from_parquet(tmp, LABELS, splits=("valid",))
            self.assertIn("duplicate", str(ctx.exception).lower())

    def test_num_output_labels_truncates_to_leading_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _write_baseline_dir(tmp)

            loaded = load_predictions_from_parquet(
                tmp, LABELS, splits=("valid",), num_output_labels=2
            )

            pred, target = loaded["valid"]
            self.assertEqual(pred.shape, (8, 2))
            np.testing.assert_allclose(pred, expected["valid"][0][:, :2])
            np.testing.assert_allclose(target, expected["valid"][1][:, :2])


class TestCalibrationPipelineFromStoredPredictions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.labels_path = os.path.join(self.tmp.name, "labels.json")
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in LABELS.items()}, f)
        self.logger = logging.getLogger("test")

    def _config(self, baseline_dir, **overrides):
        config = {
            "labels_dict_path": self.labels_path,
            "output_dir": os.path.join(self.tmp.name, "out"),
            "deconvolvers": [
                {"name": "epidish_trainonly", "predictions_dir": baseline_dir}
            ],
        }
        config.update(overrides)
        return config

    def test_constructs_without_hdf5_mask_or_deconvolvers_dir(self):
        baseline_dir = os.path.join(self.tmp.name, "epidish")
        _write_baseline_dir(baseline_dir)

        pipeline = CalibratorFittingPipeline(self._config(baseline_dir), self.logger)

        self.assertIsNone(pipeline.pseudobulk_path)
        self.assertIsNone(pipeline.features_mask_path)
        self.assertIsNone(pipeline.deconvolvers_dir)

    def test_uncalibrated_stage_reads_parquet_and_caches_npz(self):
        baseline_dir = os.path.join(self.tmp.name, "epidish")
        # The pipeline reads every configured split (train included, as in the
        # production configs) but calibrates on valid/test only.
        expected = _write_baseline_dir(
            baseline_dir, splits=("train", "valid", "test")
        )
        pipeline = CalibratorFittingPipeline(self._config(baseline_dir), self.logger)
        deconv_out = os.path.join(self.tmp.name, "out", "deconv")
        os.makedirs(deconv_out, exist_ok=True)

        val_pred, y_valid, test_pred, y_test = pipeline._load_uncalibrated(
            "epidish_trainonly",
            {"name": "epidish_trainonly", "predictions_dir": baseline_dir},
            deconv_out,
            {},
            {},
        )

        np.testing.assert_allclose(val_pred, expected["valid"][0])
        np.testing.assert_allclose(y_valid, expected["valid"][1])
        np.testing.assert_allclose(test_pred, expected["test"][0])
        np.testing.assert_allclose(y_test, expected["test"][1])

        cached = np.load(os.path.join(deconv_out, "uncalibrated_predictions.npz"))
        np.testing.assert_allclose(cached["val_pred"], expected["valid"][0])
        np.testing.assert_allclose(cached["test_target"], expected["test"][1])

    def test_cached_npz_short_circuits_the_parquet_read(self):
        baseline_dir = os.path.join(self.tmp.name, "epidish")
        _write_baseline_dir(baseline_dir)
        pipeline = CalibratorFittingPipeline(self._config(baseline_dir), self.logger)
        deconv_out = os.path.join(self.tmp.name, "out", "deconv")
        os.makedirs(deconv_out, exist_ok=True)
        sentinel = np.full((4, 3), 0.25)
        np.savez_compressed(
            os.path.join(deconv_out, "uncalibrated_predictions.npz"),
            val_pred=sentinel,
            test_pred=sentinel,
            val_target=sentinel,
            test_target=sentinel,
        )

        val_pred, _, _, _ = pipeline._load_uncalibrated(
            "epidish_trainonly",
            {"name": "epidish_trainonly", "predictions_dir": baseline_dir},
            deconv_out,
            {},
            {},
        )

        np.testing.assert_allclose(val_pred, sentinel)

    def test_targets_disagreeing_with_the_hdf5_are_rejected(self):
        baseline_dir = os.path.join(self.tmp.name, "epidish")
        expected = _write_baseline_dir(
            baseline_dir, splits=("train", "valid", "test")
        )
        pipeline = CalibratorFittingPipeline(self._config(baseline_dir), self.logger)
        deconv_out = os.path.join(self.tmp.name, "out", "deconv")
        os.makedirs(deconv_out, exist_ok=True)
        wrong = expected["valid"][1] + 0.5

        with self.assertRaises(ValueError) as ctx:
            pipeline._load_uncalibrated(
                "epidish_trainonly",
                {"name": "epidish_trainonly", "predictions_dir": baseline_dir},
                deconv_out,
                {},
                {"valid": wrong, "test": expected["test"][1]},
            )
        self.assertIn("target", str(ctx.exception).lower())

    def test_duplicate_deconvolver_names_are_rejected(self):
        baseline_dir = os.path.join(self.tmp.name, "epidish")
        _write_baseline_dir(baseline_dir)
        config = self._config(baseline_dir)
        config["deconvolvers"] = [
            {"name": "uxm", "predictions_dir": baseline_dir},
            {"name": "uxm", "predictions_dir": baseline_dir},
        ]

        with self.assertRaises(ValueError) as ctx:
            CalibratorFittingPipeline(config, self.logger)
        self.assertIn("uxm", str(ctx.exception))


class TestFitCalibrationConfigValidation(unittest.TestCase):
    def _base(self):
        return {
            "labels_dict_path": "syto/app/labels_dict.json",
            "output_dir": "/tmp/out",
        }

    def test_prediction_backed_config_needs_no_model_inputs(self):
        config = self._base()
        config["deconvolvers"] = [{"name": "epidish", "predictions_dir": "/tmp/e"}]

        validate_config(config, "fit_calibration")  # must not raise

    def test_model_backed_config_still_requires_model_inputs(self):
        config = self._base()
        config["deconvolvers"] = [{"name": "xgb"}]

        with self.assertRaises(ValueError) as ctx:
            validate_config(config, "fit_calibration")
        self.assertIn("pseudobulk_path", str(ctx.exception))

    def test_mixed_config_requires_model_inputs(self):
        config = self._base()
        config["deconvolvers"] = [
            {"name": "epidish", "predictions_dir": "/tmp/e"},
            {"name": "xgb"},
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_config(config, "fit_calibration")
        self.assertIn("pseudobulk_path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
