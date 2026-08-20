import unittest

from syto.app.wizard.tasks.inference import InferenceWizard
from syto.app.wizard.tasks import TASK_REGISTRY


class TestInferenceWizardSchema(unittest.TestCase):
    def setUp(self):
        self.wiz = InferenceWizard()
        self.specs = self.wiz.field_specs()
        self.by_key = {s.key: s for s in self.specs if s.kind != "list_section"}

    def test_registered(self):
        self.assertIn("inference", TASK_REGISTRY)
        self.assertIs(TASK_REGISTRY["inference"], InferenceWizard)

    def test_run_syto_gate_is_first(self):
        self.assertEqual(self.specs[0].key, "run_syto")

    def test_classifier_block_gated_on_run_syto(self):
        spec = self.by_key["classifier.classifier_type"]
        self.assertTrue(spec.when({"run_syto": True}))
        self.assertFalse(spec.when({"run_syto": False}))

    def test_dismir_flavor_requires_syto_and_dismir(self):
        spec = self.by_key["classifier.dismir_flavor"]
        self.assertTrue(
            spec.when({"run_syto": True, "classifier.classifier_type": "dismir"})
        )
        self.assertFalse(
            spec.when({"run_syto": False, "classifier.classifier_type": "dismir"})
        )
        self.assertFalse(
            spec.when({"run_syto": True, "classifier.classifier_type": "lookup"})
        )

    def test_labeling_scheme_select_replaces_soft_labels(self):
        self.assertIn("classifier.labeling_scheme", self.by_key)
        self.assertNotIn("classifier.soft_labels", self.by_key)
        spec = self.by_key["classifier.labeling_scheme"]
        self.assertEqual(spec.kind, "select")
        self.assertEqual(spec.choices, ["Soft Labels", "Hard Labels"])

    def test_labels_dict_default(self):
        self.assertEqual(
            self.by_key["labels_dict_path"].default, "syto/app/labels_dict.json"
        )

    def test_input_type_includes_predicted_reads_and_single_data_path(self):
        self.assertEqual(
            set(self.by_key["input.type"].choices),
            {"bam", "parsed_reads", "predicted_reads"},
        )
        self.assertIn("input.data_path", self.by_key)
        self.assertNotIn("input.bam_path", self.by_key)
        self.assertNotIn("input.parsed_reads_path", self.by_key)
        self.assertNotIn("input.predicted_reads_path", self.by_key)

    def test_syto_atlas_under_deconvolution_scope_and_gated(self):
        spec = self.by_key["deconvolution.syto.atlas_path"]
        self.assertTrue(spec.when({"run_syto": True}))
        self.assertFalse(spec.when({"run_syto": False}))
        self.assertNotIn("atlas_path", self.by_key)  # no top-level atlas

    def test_syto_methods_list_section_gated(self):
        section = next(s for s in self.specs if s.key == "deconvolution.syto.methods")
        self.assertEqual(section.kind, "list_section")
        self.assertTrue(section.when({"run_syto": True}))
        self.assertFalse(section.when({"run_syto": False}))

    def test_pseudobulk_only_for_prior_strategies(self):
        spec = self.by_key["pseudobulk_path"]
        self.assertTrue(
            spec.when(
                {
                    "fill_in_missing_labels": True,
                    "missing_label_strategy": "prior_blending",
                }
            )
        )
        self.assertFalse(
            spec.when(
                {"fill_in_missing_labels": True, "missing_label_strategy": "zeroes"}
            )
        )


class TestInferenceBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = InferenceWizard()

    def _syto_answers(self):
        return {
            "run_syto": True,
            "classifier.classifier_type": "dismir",
            "classifier.dismir_flavor": "lstm",
            "classifier.foundation_model": None,
            "classifier.classifier_head_implementation": "grg_attention_based",
            "classifier.labeling_scheme": "Soft Labels",
            "checkpoint_path": "/tmp/weight.pt",
            "features_mask_path": "/tmp/mask.npz",
            "labels_dict_path": "syto/app/labels_dict.json",
            "num_labels": 39,
            "deconvolution.syto.atlas_path": "/tmp/atlas.tsv",
            "deconvolution.syto.atlas_name": "atlas",
            "input.type": "bam",
            "input.data_path": "/tmp/x.bam",
            "input.reference_path": "/tmp/hg38.fa.gz",
            "input.data_type": "wgbs",
            "input.chromosomes": "all",
            "fill_in_missing_labels": True,
            "missing_label_strategy": "prior_blending",
            "pseudobulk_path": "/tmp/pb.h5",
            "deconvolution.syto.methods": [],
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
            "bam_processing.restrict_to_atlas": True,
            "bam_processing.ont_methyl_tr": 180,
            "bam_processing.ont_unmethyl_tr": 75,
            "max_sequence_length": 150,
            "prediction_batch_size": 2200,
            "cell_type_match_dict_path": "",
        }

    def test_single_data_path_nested(self):
        cfg = self.wiz.build_config(self._syto_answers())
        self.assertEqual(cfg["input"]["data_path"], "/tmp/x.bam")
        self.assertNotIn("bam_path", cfg["input"])

    def test_labeling_scheme_maps_to_soft_labels(self):
        ans = self._syto_answers()
        cfg = self.wiz.build_config(ans)
        self.assertIs(cfg["classifier"]["soft_labels"], True)
        self.assertNotIn("labeling_scheme", cfg["classifier"])

        ans["classifier.labeling_scheme"] = "Hard Labels"
        cfg = self.wiz.build_config(ans)
        self.assertIs(cfg["classifier"]["soft_labels"], False)

    def test_syto_atlas_nested_under_deconvolution(self):
        cfg = self.wiz.build_config(self._syto_answers())
        self.assertEqual(cfg["deconvolution"]["syto"]["atlas_path"], "/tmp/atlas.tsv")
        self.assertEqual(cfg["deconvolution"]["syto"]["atlas_name"], "atlas")
        self.assertNotIn("atlas_path", cfg)  # not top-level

    def test_syto_methods_canonical_dispatch_names(self):
        ans = self._syto_answers()
        ans["deconvolution.syto.methods"] = [
            {
                "name": "3Layer_MLP",
                "enabled": True,
                "use_callibration": True,
                "checkpoint_path": "/tmp/mlp.pt",
                "calibrators_dir": "/tmp/cal",
            },
            {
                "name": "ls",
                "flavor": "nnls",
                "enabled": True,
                "use_callibration": True,
                "checkpoint_path": "/tmp/nnls.joblib",
                "calibrators_dir": "/tmp/cal",
            },
        ]
        cfg = self.wiz.build_config(ans)
        methods = cfg["deconvolution"]["syto"]["methods"]
        self.assertEqual(methods[0]["name"], "3Layer_MLP")
        self.assertEqual(methods[1]["name"], "ls")
        self.assertEqual(methods[1]["flavor"], "nnls")

    def test_no_syto_omits_classifier_and_syto_block(self):
        ans = {
            "run_syto": False,
            "classifier.classifier_type": None,
            "checkpoint_path": None,
            "features_mask_path": None,
            "deconvolution.syto.atlas_path": None,
            "deconvolution.syto.atlas_name": None,
            "labels_dict_path": "syto/app/labels_dict.json",
            "num_labels": 39,
            "input.type": "predicted_reads",
            "input.data_path": "/tmp/preds.pkl",
            "input.chromosomes": "all",
            "fill_in_missing_labels": False,
            "deconvolution.baselines": [
                {
                    "model": "uxm",
                    "enabled": True,
                    "atlas_path": "/tmp/atlas.tsv",
                    "ignore_cells": ["Megakaryocytes"],
                }
            ],
            "output_dir": "/tmp/out",
            "output.save_predictions": True,
            "cell_type_match_dict_path": "",
        }
        cfg = self.wiz.build_config(ans)
        self.assertNotIn("classifier", cfg)
        self.assertNotIn("checkpoint_path", cfg)
        self.assertNotIn("features_mask_path", cfg)
        self.assertNotIn("syto", cfg["deconvolution"])
        self.assertEqual(cfg["deconvolution"]["baselines"][0]["model"], "uxm")
        self.assertEqual(cfg["input"]["data_path"], "/tmp/preds.pkl")

    def test_chromosomes_list_is_split(self):
        ans = self._syto_answers()
        ans["input.chromosomes"] = "chr1, chr2,chr3"
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["input"]["chromosomes"], ["chr1", "chr2", "chr3"])

    def test_irrelevant_classifier_fields_omitted_for_cancer_detector(self):
        # Engine omits when-False keys; build_config must not resurrect them.
        ans = {
            "run_syto": True,
            "classifier.classifier_type": "cancer_detector",
            "labels_dict_path": "syto/app/labels_dict.json",
            "num_labels": 39,
            "deconvolution.syto.atlas_path": "/tmp/atlas.tsv",
            "deconvolution.syto.atlas_name": "atlas",
            "input.type": "bam",
            "input.data_path": "/tmp/x.bam",
            "input.reference_path": "/tmp/r.fa",
            "input.data_type": "wgbs",
            "input.chromosomes": "all",
            "fill_in_missing_labels": False,
            "deconvolution.syto.methods": [],
            "deconvolution.baselines": [],
            "output_dir": "/tmp/out",
        }
        cfg = self.wiz.build_config(ans)
        self.assertEqual(cfg["classifier"]["classifier_type"], "cancer_detector")
        self.assertNotIn("dismir_flavor", cfg["classifier"])
        self.assertNotIn("foundation_model", cfg["classifier"])
        self.assertNotIn("classifier_head_implementation", cfg["classifier"])
        self.assertNotIn("soft_labels", cfg["classifier"])

    def test_chunked_inference_block_is_nested_when_enabled(self):
        ans = self._syto_answers()
        ans["chunked_inference.enabled"] = True
        ans["chunked_inference.chunk_by"] = "grg"
        ans["chunked_inference.progress_bar"] = True
        cfg = self.wiz.build_config(ans)
        self.assertEqual(
            cfg["chunked_inference"],
            {"enabled": True, "chunk_by": "grg", "progress_bar": True},
        )

    def test_chunked_inference_omitted_when_not_answered(self):
        cfg = self.wiz.build_config(self._syto_answers())
        self.assertNotIn("chunked_inference", cfg)

    def test_bam_processing_omitted_for_non_bam_input(self):
        ans = self._syto_answers()
        ans["input.type"] = "parsed_reads"
        del ans["input.reference_path"]
        del ans["input.data_type"]
        for k in list(ans):
            if k.startswith("bam_processing."):
                del ans[k]
        cfg = self.wiz.build_config(ans)
        self.assertNotIn("bam_processing", cfg)


class _StubEngine:
    """Engine stub for section handlers: scripted ask_checkbox + ask_one."""

    def __init__(self, checkbox_return, answers_by_key):
        self._checkbox = checkbox_return
        self._by_key = answers_by_key

    def ask_checkbox(self, label, choices):
        return self._checkbox

    def ask_one(self, spec, answers):
        return self._by_key.get(spec.key, spec.default)


class TestEpidishBaselineSection(unittest.TestCase):
    def test_epidish_in_baselines_list(self):
        from syto.app.wizard.tasks.inference import BASELINES

        self.assertIn("epidish", BASELINES)

    def test_rpc_entry_auto_named(self):
        from syto.app.wizard.tasks.inference import _deconv_baselines_section

        eng = _StubEngine(
            ["epidish"],
            {
                "_epidish_atlas": "/tmp/atlas.csv",
                "_epidish_ref": "hg38",
                "_epidish_method": "RPC",
                "_epidish_maxit": 50,
                "_epidish_name": "",
            },
        )
        answers = {}
        _deconv_baselines_section(eng, answers)
        entry = answers["deconvolution.baselines"][0]
        self.assertEqual(entry["model"], "epidish")
        self.assertEqual(entry["method"], "RPC")
        self.assertEqual(entry["maxit"], 50)
        self.assertEqual(entry["reference_genome"], "hg38")
        self.assertEqual(entry["atlas_path"], "/tmp/atlas.csv")
        self.assertNotIn("name", entry)  # blank override -> auto-named downstream
        self.assertNotIn("constraint", entry)

    def test_cp_entry_with_name_override(self):
        from syto.app.wizard.tasks.inference import _deconv_baselines_section

        eng = _StubEngine(
            ["epidish"],
            {
                "_epidish_atlas": "/tmp/atlas.csv",
                "_epidish_ref": "hg19",
                "_epidish_method": "CP",
                "_epidish_constraint": "inequality",
                "_epidish_name": "epidish_houseman",
            },
        )
        answers = {}
        _deconv_baselines_section(eng, answers)
        entry = answers["deconvolution.baselines"][0]
        self.assertEqual(entry["method"], "CP")
        self.assertEqual(entry["constraint"], "inequality")
        self.assertEqual(entry["name"], "epidish_houseman")
        self.assertNotIn("maxit", entry)


if __name__ == "__main__":
    unittest.main()
