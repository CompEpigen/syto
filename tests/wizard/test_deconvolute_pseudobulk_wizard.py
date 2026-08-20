import unittest
from unittest import mock

from syto.app.wizard.tasks import TASK_REGISTRY
from syto.app.wizard.tasks import deconvolute_pseudobulk as dp
from syto.app.wizard.tasks.deconvolute_pseudobulk import DeconvolutePseudobulkWizard


class _StubEngine:
    """Minimal engine with scripted, sequential ask_checkbox / ask_one returns."""

    def __init__(self, checkbox=None, one=None):
        self._checkbox = list(checkbox or [])
        self._one = list(one or [])
        self.calls = []  # choices offered on each ask_checkbox call
        self.one_specs = []  # FieldSpecs passed to each ask_one call

    def ask_checkbox(self, label, choices):
        self.calls.append(list(choices))
        return self._checkbox.pop(0)

    def ask_one(self, spec, answers):
        self.one_specs.append(spec)
        return self._one.pop(0)


class TestSplitsSection(unittest.TestCase):
    def test_resolved_from_file(self):
        with mock.patch.object(
            dp, "_list_splits", return_value=["train", "valid", "test"]
        ):
            eng = _StubEngine(checkbox=[["train", "test"]])
            answers = {"pseudobulk_path": "x.h5"}
            dp._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "test"])
        self.assertEqual(eng.calls[0], ["train", "valid", "test"])

    def test_key_omitted_when_file_unreadable(self):
        with mock.patch.object(dp, "_list_splits", return_value=[]):
            eng = _StubEngine(checkbox=[])  # ask_checkbox must NOT be called
            answers = {"pseudobulk_path": "x.h5"}
            dp._splits_section(eng, answers)
        self.assertNotIn("splits", answers)
        self.assertEqual(eng.calls, [])

    def test_key_omitted_when_nothing_selected(self):
        with mock.patch.object(dp, "_list_splits", return_value=["train", "valid"]):
            eng = _StubEngine(checkbox=[[]])
            answers = {"pseudobulk_path": "x.h5"}
            dp._splits_section(eng, answers)
        self.assertNotIn("splits", answers)


class TestBaselinesSection(unittest.TestCase):
    def test_four_baselines_offered(self):
        eng = _StubEngine(checkbox=[["uxm"]], one=["/tmp/atlas.tsv", "Megakaryocytes"])
        answers = {}
        dp._baselines_section(eng, answers)
        self.assertEqual(eng.calls[0], ["uxm", "celfieish", "celfie", "epidish"])

    def test_uxm_entry_shape(self):
        eng = _StubEngine(checkbox=[["uxm"]], one=["/tmp/atlas.tsv", "Megakaryocytes"])
        answers = {}
        dp._baselines_section(eng, answers)
        self.assertEqual(
            answers["baselines"],
            [
                {
                    "model": "uxm",
                    "enabled": True,
                    "atlas_path": "/tmp/atlas.tsv",
                    "ignore_cells": ["Megakaryocytes"],
                }
            ],
        )

    def test_celfieish_em_checkpoints(self):
        eng = _StubEngine(
            checkbox=[["celfieish"]],
            one=["/tmp/atlas.csv", "hg38", "em_checkpoints", "100, 300"],
        )
        answers = {}
        dp._baselines_section(eng, answers)
        entry = answers["baselines"][0]
        self.assertEqual(entry["model"], "celfieish")
        self.assertEqual(entry["reference_genome"], "hg38")
        self.assertEqual(entry["em_checkpoints"], [100, 300])

    def test_reprompts_until_at_least_one_selected(self):
        eng = _StubEngine(
            checkbox=[[], ["uxm"]], one=["/tmp/atlas.tsv", "Megakaryocytes"]
        )
        answers = {}
        dp._baselines_section(eng, answers)
        self.assertEqual(len(eng.calls), 2)
        self.assertEqual(len(answers["baselines"]), 1)


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.specs = DeconvolutePseudobulkWizard().field_specs()
        self.keys = [s.key for s in self.specs]
        self.by_key = {s.key: s for s in self.specs}

    def test_registered(self):
        self.assertIn("deconvolute_pseudobulk", TASK_REGISTRY)
        self.assertIs(
            TASK_REGISTRY["deconvolute_pseudobulk"], DeconvolutePseudobulkWizard
        )

    def test_labels_dict_first(self):
        self.assertEqual(self.keys[0], "labels_dict_path")

    def test_pseudobulk_before_splits(self):
        self.assertLess(self.keys.index("pseudobulk_path"), self.keys.index("splits"))

    def test_sections_are_list_sections(self):
        for key in ("splits", "baselines"):
            self.assertEqual(self.by_key[key].kind, "list_section")

    def test_no_syto_deconvolvers_asked(self):
        # This task runs read-based baselines only.
        self.assertNotIn("deconvolvers", self.keys)

    def test_expert_tiers(self):
        self.assertEqual(self.by_key["n_workers"].tier, "core")
        self.assertEqual(self.by_key["batch_size"].tier, "expert")
        self.assertEqual(self.by_key["class_label_column"].tier, "expert")

    def test_class_label_column_default_matches_pipeline(self):
        self.assertEqual(self.by_key["class_label_column"].default, "original_label")


class TestBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = DeconvolutePseudobulkWizard()

    def test_lists_passthrough(self):
        answers = {
            "labels_dict_path": "syto/app/labels_dict.json",
            "pseudobulk_path": "/tmp/pb.h5",
            "splits": ["train", "test"],
            "baselines": [
                {"model": "uxm", "enabled": True, "atlas_path": "/tmp/atlas.tsv"}
            ],
            "output_dir": "/tmp/out",
            "n_workers": 12,
            "batch_size": 1,
            "class_label_column": "original_label",
        }
        cfg = self.wiz.build_config(answers)
        self.assertEqual(cfg["splits"], ["train", "test"])
        self.assertEqual(cfg["baselines"][0]["model"], "uxm")
        self.assertEqual(cfg["n_workers"], 12)

    def test_drops_blank_strings(self):
        answers = {"output_dir": "", "labels_dict_path": "syto/app/labels_dict.json"}
        cfg = self.wiz.build_config(answers)
        self.assertNotIn("output_dir", cfg)
        self.assertIn("labels_dict_path", cfg)

    def test_splits_absent_when_not_provided(self):
        answers = {"labels_dict_path": "syto/app/labels_dict.json"}
        cfg = self.wiz.build_config(answers)
        self.assertNotIn("splits", cfg)


class TestGeneratedConfigSatisfiesValidator(unittest.TestCase):
    def test_build_config_has_pipeline_required_fields(self):
        # Mirrors syto/app/cli.py validate_config
        # required_fields["deconvolute_pseudobulk"].
        answers = {
            "labels_dict_path": "syto/app/labels_dict.json",
            "pseudobulk_path": "/tmp/pb.h5",
            "baselines": [{"model": "celfie", "enabled": True, "atlas_path": "/a.csv"}],
            "output_dir": "/tmp/out",
            "n_workers": 1,
            "batch_size": 1,
            "class_label_column": "original_label",
        }
        cfg = DeconvolutePseudobulkWizard().build_config(answers)
        for field in ("pseudobulk_path", "output_dir", "labels_dict_path"):
            self.assertIn(field, cfg)


if __name__ == "__main__":
    unittest.main()
