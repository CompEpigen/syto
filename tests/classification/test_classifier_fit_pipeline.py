import logging
import unittest
import tempfile
from pathlib import Path

import pandas as pd

import App.classifier_fit_pipeline as pipeline_mod
from App.classifier_fit_pipeline import ClassifierFittingPipeline


class _StubClassifier:
    last = None

    def __init__(self):
        _StubClassifier.last = self
        self.train_df = None
        self.val_df = None

    def required_fit_columns(self, config):
        return ["input_ids", "methylation_ids", "dmr_ctype_label"]

    def fit_classificaton(self, train_df, val_df=None, output_dir=None, **kwargs):
        self.train_df = train_df
        self.val_df = val_df
        return self


class TestClassifierFitPipelineColumnar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "ds"
        for bucket in (0, 1):
            part = self.data_dir / f"region_bucket={bucket}"
            part.mkdir(parents=True)
            pd.DataFrame(
                {
                    "input_ids": ["A", "B"],
                    "methylation_ids": ["0101", "1100"],
                    "dmr_ctype_label": [bucket, bucket],
                    "soft_label_pooled": [[0.2, 0.8], [0.7, 0.3]],
                    "soft_label_other": [[0.9, 0.1], [0.4, 0.6]],
                    "split": ["train", "valid"],
                    "region_bucket": [bucket, bucket],
                }
            ).to_parquet(part / "part-0.parquet")

        self._orig_factory = pipeline_mod.read_classifier_factory
        pipeline_mod.read_classifier_factory = lambda **kw: _StubClassifier()

    def tearDown(self):
        pipeline_mod.read_classifier_factory = self._orig_factory
        self.tmp.cleanup()

    def _config(self):
        return {
            "model": {"architecture": "dismir", "soft_labels": True,
                      "grg_label_column": "dmr_ctype_label"},
            "training": {},
            "output": {"output_dir": str(Path(self.tmp.name) / "out")},
            "mlflow": {"enabled": False},
            "data_path": str(self.data_dir),
            "datasets": "all",
            "label_column": "soft_label_pooled",
        }

    def test_columnar_projects_renames_and_filters(self):
        pipe = ClassifierFittingPipeline(self._config(), logging.getLogger("t"))
        pipe.run()
        clf = _StubClassifier.last
        # label renamed to canonical soft_label; other label column pruned.
        self.assertEqual(
            set(clf.train_df.columns),
            {"input_ids", "methylation_ids", "dmr_ctype_label", "soft_label"},
        )
        self.assertNotIn("soft_label_other", clf.train_df.columns)
        # train filter: one row per bucket.
        self.assertEqual(len(clf.train_df), 2)
        # valid split loaded too.
        self.assertEqual(len(clf.val_df), 2)
