import unittest

from App.wizard.tasks.classifier_fit import ClassifierFitWizard
from App.wizard.tasks import TASK_REGISTRY


class TestClassifierFitSchema(unittest.TestCase):
    def setUp(self):
        self.wiz = ClassifierFitWizard()
        self.specs = self.wiz.field_specs()
        self.by_key = {s.key: s for s in self.specs}

    def test_registered(self):
        self.assertIn("classifier_fit", TASK_REGISTRY)
        self.assertIs(TASK_REGISTRY["classifier_fit"], ClassifierFitWizard)

    def test_architecture_is_first(self):
        first = self.specs[0]
        self.assertEqual(first.key, "model.architecture")
        self.assertEqual(
            set(first.choices),
            {"dismir", "methylbert", "cancer_detector", "lookup", "epigenbert2"},
        )

    def _arch(self, name):
        return {"model.architecture": name}

    def test_flavor_dismir_only(self):
        w = self.by_key["model.flavor"].when
        self.assertTrue(w(self._arch("dismir")))
        self.assertFalse(w(self._arch("methylbert")))

    def test_foundation_model_transformers_only(self):
        w = self.by_key["model.foundation_model"].when
        self.assertTrue(w(self._arch("methylbert")))
        self.assertTrue(w(self._arch("epigenbert2")))
        self.assertFalse(w(self._arch("dismir")))
        self.assertFalse(w(self._arch("lookup")))

    def test_num_grg_labels_dismir_and_methylbert_only(self):
        w = self.by_key["model.num_grg_labels"].when
        self.assertTrue(w(self._arch("dismir")))
        self.assertTrue(w(self._arch("methylbert")))
        self.assertFalse(
            w(self._arch("epigenbert2"))
        )  # epigenbert2 has no num_grg_labels

    def test_grg_label_column_in_model_for_dismir_only(self):
        w = self.by_key["model.grg_label_column"].when
        self.assertTrue(w(self._arch("dismir")))
        self.assertFalse(w(self._arch("methylbert")))
        self.assertFalse(w(self._arch("epigenbert2")))

    def test_grg_label_column_in_training_for_hf_only(self):
        w = self.by_key["training.grg_label_column"].when
        self.assertTrue(w(self._arch("methylbert")))
        self.assertTrue(w(self._arch("epigenbert2")))
        self.assertFalse(w(self._arch("dismir")))

    def test_lookup_config_lookup_only(self):
        w = self.by_key["model.lookup_config.label_mode"].when
        self.assertTrue(w(self._arch("lookup")))
        self.assertFalse(w(self._arch("dismir")))

    def test_epigenbert2_methylation_flags_only_epigenbert2(self):
        w = self.by_key["model.use_cpg_methylation"].when
        self.assertTrue(w(self._arch("epigenbert2")))
        self.assertFalse(w(self._arch("methylbert")))

    def test_cancer_detector_training_fields_gated(self):
        w = self.by_key["training.eps_beta_fit"].when
        self.assertTrue(w(self._arch("cancer_detector")))
        self.assertFalse(w(self._arch("dismir")))

    def test_training_args_hf_only_and_has_expert_fields(self):
        lr = self.by_key["training.training_args.learning_rate"]
        self.assertTrue(lr.when(self._arch("methylbert")))
        self.assertTrue(lr.when(self._arch("epigenbert2")))
        self.assertFalse(lr.when(self._arch("dismir")))
        # some training_args are expert-tier
        self.assertEqual(
            self.by_key["training.training_args.adam_beta1"].tier, "expert"
        )

    def test_label_column_mandatory_for_neural(self):
        spec = self.by_key["label_column"]
        self.assertTrue(spec.when(self._arch("dismir")))
        self.assertTrue(spec.when(self._arch("methylbert")))
        self.assertTrue(spec.when(self._arch("epigenbert2")))
        self.assertFalse(spec.when(self._arch("lookup")))
        self.assertFalse(spec.when(self._arch("cancer_detector")))
        # mandatory -> rejects blank
        self.assertIsNotNone(spec.validate(""))

    def test_data_format_choices(self):
        spec = self.by_key["data_format"]
        self.assertEqual(set(spec.choices), {"auto", "legacy", "columnar"})
        self.assertEqual(spec.default, "auto")

    def test_split_column_default(self):
        spec = self.by_key["split_column"]
        self.assertEqual(spec.default, "split")


class TestClassifierFitBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = ClassifierFitWizard()

    def _shared(self, arch):
        return {
            "model.architecture": arch,
            "data_path": "/data/loyfer",
            "datasets": "all",
            "max_sequence_length": 150,
            "prediction_batch_size": 128,
            "output.output_dir": "tmp/out",
            "mlflow.enabled": True,
            "mlflow.experiment_name": "exp",
            "mlflow.tracking_uri": "",  # blank -> omitted
        }

    def test_cancer_detector_minimal_model(self):
        ans = self._shared("cancer_detector")
        ans.update(
            {
                "training.col_n_meth_cpgs": "M",
                "training.col_n_unmeth_cpgs": "U",
                "training.col_label": "label",
                "training.col_marker_label": "dmr_ctype_label",
                "training.eps_beta_fit": 0.01,
                "training.class_prior_type": "train_freq",
            }
        )
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["model"], {"architecture": "cancer_detector"})
        self.assertEqual(cfg["training"]["class_prior_type"], "train_freq")
        self.assertNotIn("tracking_uri", cfg["mlflow"])  # blank omitted
        self.assertEqual(cfg["datasets"], "all")

    def test_dismir_model_and_training(self):
        ans = self._shared("dismir")
        ans.update(
            {
                "model.flavor": "lstm",
                "model.num_labels": 39,
                "model.classifier_head_implementation": "grg_attention_based",
                "model.num_grg_labels": 39,
                "model.grg_label_column": "dmr_ctype_label",
                "model.soft_labels": True,
                "training.epochs": 100,
                "training.batch_size": 128,
                "training.patience": 30,
                "training.optimizer_type": "ADAM",
                "training.lr": 0.001,
                "training.weight_decay": 1.0e-6,
                "training.variable_length": False,
                "training.verbose": 1,
            }
        )
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["model"]["flavor"], "lstm")
        self.assertEqual(cfg["model"]["grg_label_column"], "dmr_ctype_label")
        self.assertNotIn("training_args", cfg["training"])
        self.assertNotIn("foundation_model", cfg["model"])
        self.assertEqual(cfg["training"]["epochs"], 100)

    def test_lookup_nested_config(self):
        ans = self._shared("lookup")
        ans.update(
            {
                "model.lookup_config.num_classes": 39,
                "model.lookup_config.label_col": "original_label",
                "model.lookup_config.label_mode": "soft",
                "model.lookup_config.min_reads": 0,
                "model.lookup_config.max_distance": 0,
                "training.compute_train_metrics": True,
            }
        )
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["model"]["lookup_config"]["label_mode"], "soft")
        self.assertEqual(cfg["model"]["lookup_config"]["min_reads"], 0)
        self.assertNotIn("num_labels", cfg["model"])
        self.assertEqual(cfg["training"], {"compute_train_metrics": True})

    def test_methylbert_training_args_nested_and_grg_in_training(self):
        ans = self._shared("methylbert")
        ans.update(
            {
                "model.foundation_model": "foundationalModels/methylbert_hg19_12l",
                "model.num_labels": 40,
                "model.num_grg_labels": 39,
                "model.classifier_head_implementation": "grg_attention_based",
                "model.soft_labels": False,
                "training.grg_label_column": "dmr_ctype_label",
                "training.training_args.learning_rate": 0.0004,
                "training.training_args.per_device_train_batch_size": 512,
                "training.training_args.num_train_epochs": 1,
            }
        )
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["model"]["num_labels"], 40)
        self.assertNotIn("flavor", cfg["model"])
        self.assertNotIn("grg_label_column", cfg["model"])  # lives in training
        self.assertEqual(cfg["training"]["grg_label_column"], "dmr_ctype_label")
        self.assertEqual(cfg["training"]["training_args"]["learning_rate"], 0.0004)

    def test_epigenbert2_no_num_grg_labels(self):
        ans = self._shared("epigenbert2")
        ans.update(
            {
                "model.foundation_model": "foundationalModels/DNABERT-2-117M",
                "model.num_labels": 39,
                "model.use_cpg_methylation": True,
                "model.use_m6a_methylation": False,
                "model.classifier_head_implementation": "grg_attention_based",
                "model.soft_labels": True,
                "training.grg_label_column": "dmr_ctype_label",
                "training.training_args.learning_rate": 0.00001,
            }
        )
        cfg = self.wiz.build_config(ans)
        self.assertNotIn("num_grg_labels", cfg["model"])
        self.assertTrue(cfg["model"]["use_cpg_methylation"])
        self.assertEqual(cfg["training"]["training_args"]["learning_rate"], 0.00001)


if __name__ == "__main__":
    unittest.main()
