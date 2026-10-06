"""Validation subsampling in the classifier fitting pipeline."""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from syto.app.classifier_fit_pipeline import ClassifierFittingPipeline


def _dataset(directory, n_valid=500):
    """A minimal columnar dataset with train and valid splits."""
    d = Path(directory) / "final" / "region_bucket=0"
    d.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "input_ids": ["ACGTACGTAC"] * (100 + n_valid),
            "methylation_ids": ["0101010101"] * (100 + n_valid),
            "original_label": [0] * (100 + n_valid),
            "dmr_ctype_label": [0] * (100 + n_valid),
            "soft_label": [[0.5, 0.5]] * (100 + n_valid),
            "split": ["train"] * 100 + ["valid"] * n_valid,
        }
    )
    df.to_parquet(d / "part-0.parquet", index=False)


def _config(directory, **overrides):
    cfg = {
        "model": {"architecture": "dismir", "num_labels": 2},
        "data_path": str(directory),
        "datasets": "final",
        "label_column": "soft_label",
        "output_dir": str(Path(directory) / "out"),
        "mlflow": {"enabled": False},
        "training": {},
    }
    cfg.update(overrides)
    return cfg


class TestValidSubsample(unittest.TestCase):
    def _run(self, **overrides):
        with tempfile.TemporaryDirectory() as d:
            _dataset(d)
            pipeline = ClassifierFittingPipeline(
                _config(d, **overrides), logging.getLogger("t")
            )
            classifier = MagicMock()
            classifier.required_fit_columns.return_value = [
                "input_ids",
                "methylation_ids",
            ]
            with patch(
                "syto.app.classifier_fit_pipeline.read_classifier_factory",
                return_value=classifier,
            ):
                pipeline.run()
            return classifier.fit_classificaton.call_args.kwargs

    def test_valid_split_is_capped_and_train_is_untouched(self):
        kwargs = self._run(valid_subsample=120, seed=7)
        self.assertEqual(len(kwargs["val_df"]), 120)
        self.assertEqual(len(kwargs["train_df"]), 100)

    def test_subsample_is_deterministic_for_a_seed(self):
        first = self._run(valid_subsample=50, seed=3)["val_df"]
        second = self._run(valid_subsample=50, seed=3)["val_df"]
        pd.testing.assert_frame_equal(first, second)

    def test_no_subsample_keeps_every_validation_row(self):
        self.assertEqual(len(self._run()["val_df"]), 500)

    def test_a_cap_above_the_split_size_is_a_no_op(self):
        self.assertEqual(len(self._run(valid_subsample=10_000)["val_df"]), 500)


if __name__ == "__main__":
    unittest.main()
