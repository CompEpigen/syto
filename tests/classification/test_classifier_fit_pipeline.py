import logging
import unittest
import tempfile
from pathlib import Path

import pandas as pd

import syto.app.classifier_fit_pipeline as pipeline_mod
from syto.app.classifier_fit_pipeline import ClassifierFittingPipeline


class _StubClassifier:
    last = None

    def __init__(self):
        _StubClassifier.last = self
        self.train_df = None
        self.val_df = None

    def required_fit_columns(self, config):
        return ["input_ids", "methylation_ids", "dmr_ctype_label"]

    def mlflow_fit_params(self):
        return {"flavour": "lstm"}

    def fit_classificaton(self, train_df, val_df=None, output_dir=None, **kwargs):
        self.train_df = train_df
        self.val_df = val_df
        self.fit_kwargs = kwargs
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
            "model": {
                "architecture": "dismir",
                "soft_labels": True,
                "grg_label_column": "dmr_ctype_label",
            },
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

    def test_mlflow_extra_params_carry_pipeline_and_model_metadata(self):
        pipe = ClassifierFittingPipeline(self._config(), logging.getLogger("t"))
        pipe.run()
        clf = _StubClassifier.last
        extra = clf.fit_kwargs["mlflow_extra_params"]
        # Pipeline-level knobs that are not part of training_cfg.
        self.assertEqual(extra["label_column"], "soft_label_pooled")
        self.assertEqual(extra["split_column"], "split")
        self.assertIsNone(extra["min_pattern_length"])
        self.assertIsNone(extra["max_sequence_length"])
        # Architecture-relevant params, namespaced under "model".
        self.assertEqual(extra["model"], {"flavour": "lstm"})

    def test_effective_config_dumped_next_to_outputs(self):
        pipe = ClassifierFittingPipeline(self._config(), logging.getLogger("t"))
        pipe.run()
        run_config = Path(self.tmp.name) / "out" / "all" / "run_config.yaml"
        self.assertTrue(run_config.exists())
        import yaml

        dumped = yaml.safe_load(run_config.read_text())
        self.assertEqual(dumped["model"]["architecture"], "dismir")
        self.assertEqual(dumped["label_column"], "soft_label_pooled")

    def test_log_file_forwarded_as_extra_artifact(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as lf:
            log_path = lf.name
        logger = logging.getLogger("pipeline_artifact_test")
        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)
        try:
            pipe = ClassifierFittingPipeline(self._config(), logger)
            pipe.run()
            clf = _StubClassifier.last
            self.assertIn(log_path, clf.fit_kwargs["mlflow_extra_artifacts"])
        finally:
            logger.removeHandler(handler)
            handler.close()
            Path(log_path).unlink(missing_ok=True)


class TestClassifierFitPipelinePatternFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "ds"
        part = self.data_dir / "region_bucket=0"
        part.mkdir(parents=True)
        # marked-CpG counts within the train split: 4, 2, 1
        pd.DataFrame(
            {
                "input_ids": ["A", "B", "C", "D"],
                "methylation_ids": ["0101", "10", "1", "0011"],
                "dmr_ctype_label": [0, 0, 0, 0],
                "soft_label_pooled": [[0.2, 0.8]] * 4,
                "split": ["train", "train", "train", "valid"],
                "region_bucket": [0, 0, 0, 0],
            }
        ).to_parquet(part / "part-0.parquet")

        self._orig_factory = pipeline_mod.read_classifier_factory
        pipeline_mod.read_classifier_factory = lambda **kw: _StubClassifier()

    def tearDown(self):
        pipeline_mod.read_classifier_factory = self._orig_factory
        self.tmp.cleanup()

    def _config(self):
        return {
            "model": {
                "architecture": "dismir",
                "soft_labels": True,
                "grg_label_column": "dmr_ctype_label",
            },
            "training": {},
            "output": {"output_dir": str(Path(self.tmp.name) / "out")},
            "mlflow": {"enabled": False},
            "data_path": str(self.data_dir),
            "datasets": "all",
            "label_column": "soft_label_pooled",
            "min_pattern_length": 2,
        }

    def test_min_pattern_length_drops_short_reads(self):
        pipe = ClassifierFittingPipeline(self._config(), logging.getLogger("t"))
        pipe.run()
        clf = _StubClassifier.last
        # train had counts 4, 2, 1 -> drops the length-1 read, keeps two.
        self.assertEqual(len(clf.train_df), 2)
        self.assertEqual(set(clf.train_df["methylation_ids"]), {"0101", "10"})
        # valid read has count 4 -> retained.
        self.assertEqual(len(clf.val_df), 1)


class TestClassifierFitPipelineTuningKnobs(unittest.TestCase):
    """Tuning hyperparameters must reach the classifier, and runs must be seedable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _config(self, model_extra=None, **top):
        cfg = {
            "model": {
                "architecture": "dismir",
                "soft_labels": True,
                "grg_label_column": "cluster_label",
                **(model_extra or {}),
            },
            "training": {},
            "output": {"output_dir": str(Path(self.tmp.name) / "out")},
            "mlflow": {"enabled": False},
            "data_path": str(Path(self.tmp.name) / "ds"),
            "datasets": "all",
            "label_column": "soft_label_pooled",
        }
        cfg.update(top)
        return cfg

    def test_penalty_scale_forwarded_to_classifier(self):
        pipe = ClassifierFittingPipeline(
            self._config({"penalty_scale": 0.25}), logging.getLogger("t")
        )
        self.assertEqual(pipe._classifier_kwargs()["penalty_scale"], 0.25)

    def test_dropout_prob_forwarded_to_classifier(self):
        pipe = ClassifierFittingPipeline(
            self._config({"dropout_prob": 0.35}), logging.getLogger("t")
        )
        self.assertEqual(pipe._classifier_kwargs()["dropout_prob"], 0.35)

    def test_tuning_knobs_default_to_current_behaviour(self):
        pipe = ClassifierFittingPipeline(self._config(), logging.getLogger("t"))
        kwargs = pipe._classifier_kwargs()
        self.assertEqual(kwargs["penalty_scale"], 1.0)
        self.assertEqual(kwargs["dropout_prob"], 0.2)

    def test_seed_makes_torch_initialisation_reproducible(self):
        import torch

        def first_weight(seed):
            ClassifierFittingPipeline(
                self._config(seed=seed), logging.getLogger("t")
            ).seed_everything()
            return torch.nn.Linear(8, 8).weight.detach().clone()

        self.assertTrue(torch.equal(first_weight(1234), first_weight(1234)))
        self.assertFalse(torch.equal(first_weight(1234), first_weight(99)))

    def test_run_seeds_before_fitting(self):
        pipe = ClassifierFittingPipeline(self._config(seed=7), logging.getLogger("t"))
        calls = []
        pipe.seed_everything = lambda: calls.append("seeded") or 7
        pipe.datasets = []  # no datasets: isolate the seeding step
        pipe.run()
        self.assertEqual(calls, ["seeded"])


class _HistoryStubClassifier(_StubClassifier):
    """Stub exposing a ``history`` like Dismir's: one record per epoch."""

    def fit_classificaton(self, train_df, val_df=None, output_dir=None, **kwargs):
        super().fit_classificaton(train_df, val_df, output_dir, **kwargs)
        self.history = [
            {"epoch": 1, "val_loss": 0.50, "val_f1": 0.10},
            {"epoch": 2, "val_loss": 0.20, "val_f1": 0.80},  # best
            {"epoch": 3, "val_loss": 0.30, "val_f1": 0.40},  # last
        ]
        return self


class TestClassifierFitPipelineRunSummary(unittest.TestCase):
    """run() must hand back the BEST epoch per dataset, not the last."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "ds"
        part = self.data_dir / "region_bucket=0"
        part.mkdir(parents=True)
        pd.DataFrame(
            {
                "input_ids": ["A", "B"],
                "methylation_ids": ["0101", "1100"],
                "dmr_ctype_label": [0, 0],
                "soft_label_pooled": [[0.2, 0.8], [0.7, 0.3]],
                "split": ["train", "valid"],
                "region_bucket": [0, 0],
            }
        ).to_parquet(part / "part-0.parquet")
        self._orig = pipeline_mod.read_classifier_factory
        pipeline_mod.read_classifier_factory = lambda **kw: _HistoryStubClassifier()

    def tearDown(self):
        pipeline_mod.read_classifier_factory = self._orig
        self.tmp.cleanup()

    def _pipe(self):
        return ClassifierFittingPipeline(
            {
                "model": {"architecture": "dismir", "soft_labels": True,
                          "grg_label_column": "dmr_ctype_label"},
                "training": {},
                "output": {"output_dir": str(Path(self.tmp.name) / "out")},
                "mlflow": {"enabled": False},
                "data_path": str(self.data_dir),
                "datasets": "all",
                "label_column": "soft_label_pooled",
            },
            logging.getLogger("t"),
        )

    def test_run_returns_best_epoch_metrics_per_dataset(self):
        summary = self._pipe().run()
        self.assertIn("all", summary)
        self.assertEqual(summary["all"]["best_epoch"], 2)
        self.assertAlmostEqual(summary["all"]["val_loss"], 0.20)
        self.assertAlmostEqual(summary["all"]["val_f1"], 0.80)

    def test_run_summary_selects_on_a_configurable_metric(self):
        pipe = self._pipe()
        pipe.config["best_metric"] = "val_f1"
        pipe.config["best_mode"] = "max"
        summary = pipe.run()
        self.assertEqual(summary["all"]["best_epoch"], 2)

    def test_epoch_callback_attribute_is_forwarded_to_the_classifier(self):
        pipe = self._pipe()

        def hook(_record):
            return True

        pipe.epoch_callback = hook
        pipe.run()
        self.assertIs(
            _HistoryStubClassifier.last.fit_kwargs.get("epoch_callback"), hook
        )

    def test_epoch_callback_absent_by_default(self):
        self._pipe().run()
        self.assertIsNone(
            _HistoryStubClassifier.last.fit_kwargs.get("epoch_callback")
        )
