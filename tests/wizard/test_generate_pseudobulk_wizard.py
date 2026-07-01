import unittest

from App.wizard.tasks.generate_pseudobulk import GeneratePseudobulkWizard
from App.wizard.tasks import TASK_REGISTRY


class TestGeneratePseudobulkSchema(unittest.TestCase):
    def setUp(self):
        self.wiz = GeneratePseudobulkWizard()
        self.specs = self.wiz.field_specs()
        self.by_key = {s.key: s for s in self.specs if s.kind != "list_section"}

    def test_registered(self):
        self.assertIn("generate_pseudobulk", TASK_REGISTRY)
        self.assertIs(TASK_REGISTRY["generate_pseudobulk"], GeneratePseudobulkWizard)

    def test_input_type_first(self):
        self.assertEqual(self.specs[0].key, "input_type")
        self.assertEqual(
            set(self.specs[0].choices), {"raw_splits", "pre_predicted"}
        )

    def test_classifier_type_always_limited_choices(self):
        spec = self.by_key["classifier_type"]
        self.assertIsNone(spec.when)  # always asked (metadata)
        self.assertEqual(
            set(spec.choices), {"dismir", "methylbert", "cancer_detector"}
        )

    def test_classifier_checkpoint_raw_splits_only(self):
        w = self.by_key["classifier_checkpoint"].when
        self.assertTrue(w({"input_type": "raw_splits"}))
        self.assertFalse(w({"input_type": "pre_predicted"}))

    def test_atlas_raw_splits_only(self):
        for key in ("atlas_path", "atlas_name"):
            w = self.by_key[key].when
            self.assertTrue(w({"input_type": "raw_splits"}))
            self.assertFalse(w({"input_type": "pre_predicted"}))

    def test_classifier_config_seq_length_raw_only(self):
        w = self.by_key["classifier_config.seq_length"].when
        self.assertTrue(w({"input_type": "raw_splits"}))
        self.assertFalse(w({"input_type": "pre_predicted"}))

    def test_dismir_flavor_requires_raw_and_dismir(self):
        w = self.by_key["classifier_config.dismir_flavor"].when
        self.assertTrue(
            w({"input_type": "raw_splits", "classifier_type": "dismir"})
        )
        self.assertFalse(
            w({"input_type": "raw_splits", "classifier_type": "methylbert"})
        )
        self.assertFalse(
            w({"input_type": "pre_predicted", "classifier_type": "dismir"})
        )

    def test_foundation_model_requires_raw_and_methylbert(self):
        w = self.by_key["classifier_config.foundation_model"].when
        self.assertTrue(
            w({"input_type": "raw_splits", "classifier_type": "methylbert"})
        )
        self.assertFalse(
            w({"input_type": "raw_splits", "classifier_type": "dismir"})
        )

    def test_grg_fields_only_for_grg_archs(self):
        w = self.by_key["classifier_config.num_grg_labels"].when
        self.assertTrue(
            w({"input_type": "raw_splits", "classifier_type": "dismir"})
        )
        self.assertFalse(
            w({"input_type": "raw_splits", "classifier_type": "cancer_detector"})
        )

    def test_split_information_is_list_section(self):
        section = next(s for s in self.specs if s.key == "split_information")
        self.assertEqual(section.kind, "list_section")

    def test_n_pseudobulks_gated_on_limit(self):
        w = self.by_key["n_pseudobulks_to_sample"].when
        self.assertTrue(w({"limit_pseudobulks": True}))
        self.assertFalse(w({"limit_pseudobulks": False}))


class TestGeneratePseudobulkBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = GeneratePseudobulkWizard()

    def _common(self):
        return {
            "classifier_type": "dismir",
            "labels_dict_path": "App/labels_dict.json",
            "num_labels": 39,
            "labeling_scheme": "soft_labels",
            "split_information": {
                "train": {"data_path": "/tmp/train.pkl",
                          "target_proportions_path": "/tmp/tp_train.npz"},
                "valid": {"data_path": "/tmp/valid.pkl",
                          "target_proportions_path": "/tmp/tp_valid.npz"},
            },
            "n_reads_to_sample": 475000,
            "class_label_column": "original_label",
            "grg_label_column": "dmr_ctype_label",
            "grg_sampling_method": "uniform_multinomial",
            "substitution_method": "uniform_number",
            "n_workers": 5,
            "batch_size": 100,
            "output_dir": "tmp/",
            "limit_pseudobulks": False,
        }

    def test_pre_predicted_omits_classifier_and_atlas(self):
        ans = self._common()
        ans["input_type"] = "pre_predicted"
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["input_type"], "pre_predicted")
        self.assertNotIn("classifier_checkpoint", cfg)
        self.assertNotIn("classifier_config", cfg)
        self.assertNotIn("atlas_path", cfg)
        self.assertNotIn("atlas_name", cfg)
        # metadata still present
        self.assertEqual(cfg["classifier_type"], "dismir")
        self.assertEqual(cfg["labeling_scheme"], "soft_labels")
        # split information nested
        self.assertEqual(
            cfg["split_information"]["train"]["data_path"], "/tmp/train.pkl"
        )
        # the gate helper key never leaks
        self.assertNotIn("limit_pseudobulks", cfg)
        self.assertNotIn("n_pseudobulks_to_sample", cfg)

    def test_raw_splits_includes_classifier_config_and_atlas(self):
        ans = self._common()
        ans.update({
            "input_type": "raw_splits",
            "classifier_checkpoint": "/tmp/weight.pt",
            "atlas_path": "/tmp/atlas.tsv",
            "atlas_name": "atlas",
            "classifier_config.seq_length": 150,
            "classifier_config.soft_labels": True,
            "classifier_config.num_labels": 39,
            "classifier_config.batch_size": 2200,
            "classifier_config.classifier_head_implementation": "grg_attention_based",
            "classifier_config.num_grg_labels": 39,
            "classifier_config.grg_label_column": "dmr_ctype_label",
            "classifier_config.dismir_flavor": "lstm",
        })
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["classifier_checkpoint"], "/tmp/weight.pt")
        self.assertEqual(cfg["atlas_name"], "atlas")
        self.assertEqual(cfg["classifier_config"]["seq_length"], 150)
        self.assertEqual(cfg["classifier_config"]["dismir_flavor"], "lstm")
        self.assertNotIn("foundation_model", cfg["classifier_config"])

    def test_limit_pseudobulks_written_when_enabled(self):
        ans = self._common()
        ans["input_type"] = "pre_predicted"
        ans["limit_pseudobulks"] = True
        ans["n_pseudobulks_to_sample"] = 2000
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["n_pseudobulks_to_sample"], 2000)
        self.assertNotIn("limit_pseudobulks", cfg)


if __name__ == "__main__":
    unittest.main()
