import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import syto.classification.mlflow_tracking as mlflow_tracking
from syto.classification.mlflow_tracking import mlflow_tracked_fit


class _Classifier:
    history: list = []

    @mlflow_tracked_fit
    def fit_classificaton(self, train_df, val_df=None, output_dir=None, **kwargs):
        # Records exactly what the wrapper forwarded to the real fit.
        self.forwarded_kwargs = kwargs
        return self


class TestMlflowExtraParams(unittest.TestCase):
    def setUp(self):
        # mlflow is bound at import time in the module under test. Patch it there
        # directly so the stub is used regardless of whether the real mlflow was
        # already imported by another test (import order is not deterministic).
        patcher = mock.patch.object(mlflow_tracking, "mlflow")
        self.mlflow_stub = patcher.start()
        self.addCleanup(patcher.stop)
        self.mlflow_stub.active_run.return_value = None

    def test_extra_params_logged_flattened_and_not_forwarded(self):
        train_df = pd.DataFrame({"x": [1, 2, 3]})
        clf = _Classifier().fit_classificaton(
            train_df,
            epochs=5,
            mlflow_extra_params={
                "label_column": "soft_label_pooled",
                "min_pattern_length": None,
                "model": {"flavour": "lstm", "num_labels": 2},
            },
        )

        # Log-only: the extra params never reach the wrapped fit.
        self.assertEqual(clf.forwarded_kwargs, {"epochs": 5})

        logged = self.mlflow_stub.log_params.call_args[0][0]
        # Training kwargs and dataset sizes are still logged.
        self.assertEqual(logged["epochs"], 5)
        self.assertEqual(logged["train_rows"], 3)
        # Extra params flatten: top-level scalars keep their key, nested dicts dot.
        self.assertEqual(logged["label_column"], "soft_label_pooled")
        self.assertIsNone(logged["min_pattern_length"])
        self.assertEqual(logged["model.flavour"], "lstm")
        self.assertEqual(logged["model.num_labels"], 2)

    def test_extra_artifacts_logged_and_not_forwarded(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "methylbert.yaml")
            log = os.path.join(d, "classifier_fit.log")
            for p in (cfg, log):
                with open(p, "w") as f:
                    f.write("x")
            missing = os.path.join(d, "does_not_exist.log")

            clf = _Classifier().fit_classificaton(
                pd.DataFrame({"x": [1]}),
                epochs=1,
                mlflow_extra_artifacts=[cfg, log, missing],
            )

            # Log-only: never forwarded to the wrapped fit.
            self.assertEqual(clf.forwarded_kwargs, {"epochs": 1})

            logged = {c.args[0] for c in self.mlflow_stub.log_artifact.call_args_list}
            # Existing artifacts logged; the missing path is skipped, not raised.
            self.assertIn(cfg, logged)
            self.assertIn(log, logged)
            self.assertNotIn(missing, logged)


if __name__ == "__main__":
    unittest.main()
