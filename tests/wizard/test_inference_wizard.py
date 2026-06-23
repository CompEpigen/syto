import unittest

from App.wizard.tasks.inference import InferenceWizard
from App.wizard.tasks import TASK_REGISTRY


class TestInferenceWizardSchema(unittest.TestCase):
    def setUp(self):
        self.wiz = InferenceWizard()
        self.specs = self.wiz.field_specs()
        self.by_key = {s.key: s for s in self.specs if s.kind != "list_section"}

    def test_registered(self):
        self.assertIn("inference", TASK_REGISTRY)
        self.assertIs(TASK_REGISTRY["inference"], InferenceWizard)

    def test_classifier_type_is_first_and_has_five_choices(self):
        first = self.specs[0]
        self.assertEqual(first.key, "classifier.classifier_type")
        self.assertEqual(
            set(first.choices),
            {"dismir", "methylbert", "cancer_detector", "lookup", "epigenbert2"},
        )

    def test_dismir_flavor_only_when_dismir(self):
        spec = self.by_key["classifier.dismir_flavor"]
        self.assertTrue(spec.when({"classifier.classifier_type": "dismir"}))
        self.assertFalse(spec.when({"classifier.classifier_type": "lookup"}))

    def test_foundation_model_only_for_transformer_archs(self):
        spec = self.by_key["classifier.foundation_model"]
        self.assertTrue(spec.when({"classifier.classifier_type": "methylbert"}))
        self.assertTrue(spec.when({"classifier.classifier_type": "epigenbert2"}))
        self.assertFalse(spec.when({"classifier.classifier_type": "dismir"}))

    def test_pseudobulk_only_for_prior_strategies(self):
        spec = self.by_key["pseudobulk_h5_path"]
        self.assertTrue(
            spec.when({"fill_in_missing_labels": True,
                       "missing_label_strategy": "prior_blending"})
        )
        self.assertFalse(
            spec.when({"fill_in_missing_labels": True,
                       "missing_label_strategy": "zeroes"})
        )

    def test_bam_processing_fields_are_expert_tier(self):
        self.assertEqual(self.by_key["bam_processing.n_jobs"].tier, "expert")
        self.assertEqual(self.by_key["max_sequence_length"].tier, "expert")


class TestInferenceBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = InferenceWizard()

    def _minimal_answers(self):
        return {
            "classifier.classifier_type": "dismir",
            "classifier.dismir_flavor": "lstm",
            "classifier.foundation_model": None,
            "classifier.classifier_head_implementation": "grg_attention_based",
            "classifier.soft_labels": True,
            "checkpoint_path": "/tmp/weight.pt",
            "labels_dict_path": "App/labels_dict.json",
            "num_labels": 39,
            "atlas_path": "/tmp/atlas.tsv",
            "atlas_name": "atlas",
            "input.type": "bam",
            "input.bam_path": "/tmp/x.bam",
            "input.reference_path": "/tmp/hg38.fa.gz",
            "input.data_type": "wgbs",
            "input.parsed_reads_path": None,
            "input.chromosomes": "all",
            "fill_in_missing_labels": True,
            "missing_label_strategy": "prior_blending",
            "pseudobulk_h5_path": "/tmp/pb.h5",
            "deconvolution.methods": [],
            "deconvolution.baselines": [],
            "output_dir": "/tmp/out",
            "output.save_processed_reads": False,
            "output.save_predictions": True,
            "output.save_deconvolution": True,
            "bam_processing.n_jobs": 2,
            "bam_processing.min_mapq": 10,
            "bam_processing.require_flags": 3,
            "bam_processing.exclude_flags": 1796,
            "bam_processing.min_cpgs": 4,
            "bam_processing.merge_pairs": True,
            "bam_processing.ont_methyl_tr": 180,
            "max_sequence_length": 150,
            "prediction_batch_size": 2200,
            "features_mask_path": "",
            "cell_type_match_dict_path": "",
        }

    def test_nests_dot_keys(self):
        cfg = self.wiz.build_config(self._minimal_answers())
        self.assertEqual(cfg["classifier"]["classifier_type"], "dismir")
        self.assertEqual(cfg["input"]["bam_path"], "/tmp/x.bam")
        self.assertEqual(cfg["output"]["save_predictions"], True)

    def test_chromosomes_all_stays_string(self):
        cfg = self.wiz.build_config(self._minimal_answers())
        self.assertEqual(cfg["input"]["chromosomes"], "all")

    def test_chromosomes_list_is_split(self):
        ans = self._minimal_answers()
        ans["input.chromosomes"] = "chr1, chr2,chr3"
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["input"]["chromosomes"], ["chr1", "chr2", "chr3"])

    def test_blank_optional_paths_become_none(self):
        cfg = self.wiz.build_config(self._minimal_answers())
        self.assertIsNone(cfg["features_mask_path"])
        self.assertIsNone(cfg["cell_type_match_dict_path"])

    def test_deconvolution_block_carries_lists(self):
        ans = self._minimal_answers()
        ans["deconvolution.methods"] = [
            {"name": "xgboost", "enabled": True, "use_callibration": True,
             "checkpoint_path": "/tmp/xgb.joblib",
             "calibrators_dir": "/tmp/cal"}
        ]
        ans["deconvolution.baselines"] = [
            {"model": "uxm", "enabled": True, "atlas_path": "/tmp/atlas.tsv",
             "ignore_cells": ["Megakaryocytes"]}
        ]
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["deconvolution"]["methods"][0]["name"], "xgboost")
        self.assertEqual(cfg["deconvolution"]["baselines"][0]["model"], "uxm")


if __name__ == "__main__":
    unittest.main()
