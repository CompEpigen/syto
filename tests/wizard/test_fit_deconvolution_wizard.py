import unittest
from unittest import mock

from App.wizard.tasks import fit_deconvolution as fd


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


class TestDeconvolversSection(unittest.TestCase):
    def test_all_five_offered_in_canonical_order(self):
        eng = _StubEngine(checkbox=[["nnls"]])
        answers = {}
        fd._deconvolvers_section(eng, answers)
        self.assertEqual(eng.calls[0], ["xgb", "swn", "mlp", "nnls", "psls"])

    def test_default_params_attached_in_selection_order(self):
        eng = _StubEngine(checkbox=[["swn", "xgb", "psls", "nnls"]])
        answers = {}
        fd._deconvolvers_section(eng, answers)
        self.assertEqual(
            answers["deconvolvers"],
            [
                {"name": "swn", "params": {"device": "cuda"}},
                {"name": "xgb"},
                {"name": "psls", "params": {"n_workers": 2}},
                {"name": "nnls"},
            ],
        )


class TestSplitsSection(unittest.TestCase):
    def test_resolved_from_file(self):
        with mock.patch.object(
            fd, "_list_splits", return_value=["train", "valid", "test"]
        ):
            eng = _StubEngine(checkbox=[["train", "valid"]])
            answers = {"pseudobulk_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(eng.calls[0], ["train", "valid", "test"])

    def test_fallback_when_unreadable(self):
        with mock.patch.object(fd, "_list_splits", return_value=[]):
            eng = _StubEngine(checkbox=[])  # ask_checkbox must NOT be called
            answers = {"pseudobulk_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid", "test"])
        self.assertEqual(eng.calls, [])

    def test_reprompts_until_train_and_valid_included(self):
        with mock.patch.object(
            fd, "_list_splits", return_value=["train", "valid", "test"]
        ):
            eng = _StubEngine(
                checkbox=[["valid"], ["train", "valid"]]
            )  # 1st omits train
            answers = {"pseudobulk_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(len(eng.calls), 2)

    def test_accepts_selection_when_file_lacks_required(self):
        with mock.patch.object(fd, "_list_splits", return_value=["train", "test"]):
            eng = _StubEngine(checkbox=[["train", "test"]])
            answers = {"pseudobulk_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "test"])
        self.assertEqual(len(eng.calls), 1)  # no reprompt for absent 'valid'


class TestGuaranteeColumnsSection(unittest.TestCase):
    def test_declined_leaves_key_absent(self):
        eng = _StubEngine(one=[False])
        answers = {}
        fd._guarantee_columns_section(eng, answers)
        self.assertNotIn("guarantee_columns_selection", answers)

    def test_accepted_parses_comma_separated_ints(self):
        eng = _StubEngine(one=[True, "38, 5"])
        answers = {}
        fd._guarantee_columns_section(eng, answers)
        self.assertEqual(answers["guarantee_columns_selection"], [38, 5])

    def test_accepted_reprompts_on_non_integer(self):
        eng = _StubEngine(one=[True, "x, 5", "38, 5"])
        answers = {}
        fd._guarantee_columns_section(eng, answers)
        self.assertEqual(answers["guarantee_columns_selection"], [38, 5])
        # 1 bool prompt + 2 text prompts (first invalid, second valid)
        self.assertEqual(len(eng.one_specs), 3)


from App.wizard.tasks import TASK_REGISTRY
from App.wizard.tasks.fit_deconvolution import FitDeconvolutionWizard


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.specs = FitDeconvolutionWizard().field_specs()
        self.keys = [s.key for s in self.specs]
        self.by_key = {s.key: s for s in self.specs}

    def test_registered(self):
        self.assertIn("fit_deconvolution", TASK_REGISTRY)
        self.assertIs(TASK_REGISTRY["fit_deconvolution"], FitDeconvolutionWizard)

    def test_labels_dict_first(self):
        self.assertEqual(self.keys[0], "labels_dict_path")

    def test_input_labels_before_output_labels(self):
        self.assertLess(
            self.keys.index("num_input_labels"), self.keys.index("num_output_labels")
        )

    def test_both_label_counts_are_core(self):
        self.assertEqual(self.by_key["num_input_labels"].tier, "core")
        self.assertEqual(self.by_key["num_output_labels"].tier, "core")

    def test_pseudobulk_before_splits(self):
        self.assertLess(
            self.keys.index("pseudobulk_path"), self.keys.index("splits")
        )

    def test_feature_selection_mode_is_select(self):
        spec = self.by_key["feature_selection_mode"]
        self.assertEqual(spec.kind, "select")
        self.assertEqual(set(spec.choices), {"cutoff", "top_features"})

    def test_feature_cutoff_gated_on_cutoff_mode(self):
        w = self.by_key["feature_cutoff"].when
        self.assertTrue(w({"feature_selection_mode": "cutoff"}))
        self.assertFalse(w({"feature_selection_mode": "top_features"}))

    def test_top_features_gated_on_top_features_mode(self):
        w = self.by_key["top_features"].when
        self.assertTrue(w({"feature_selection_mode": "top_features"}))
        self.assertFalse(w({"feature_selection_mode": "cutoff"}))

    def test_plot_is_expert(self):
        self.assertEqual(self.by_key["generate_feature_selection_plot"].tier, "expert")

    def test_sections_are_list_sections(self):
        for key in ("splits", "guarantee_columns_selection", "deconvolvers"):
            self.assertEqual(self.by_key[key].kind, "list_section")


class TestBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = FitDeconvolutionWizard()

    def test_nested_and_lists_passthrough(self):
        answers = {
            "labels_dict_path": "App/labels_dict.json",
            "num_input_labels": 39,
            "num_output_labels": 39,
            "pseudobulk_path": "/tmp/pb.h5",
            "splits": ["train", "valid", "test"],
            "feature_selection_mode": "top_features",
            "top_features": 156,
            "guarantee_diagonal_selection": True,
            "guarantee_columns_selection": [38],
            "generate_feature_selection_plot": False,
            "deconvolvers": [
                {"name": "swn", "params": {"device": "cuda"}},
                {"name": "nnls"},
            ],
            "output_dir": "/tmp/out",
        }
        cfg = self.wiz.build_config(answers)
        self.assertEqual(cfg["pseudobulk_path"], "/tmp/pb.h5")
        self.assertEqual(cfg["splits"], ["train", "valid", "test"])
        self.assertEqual(cfg["top_features"], 156)
        self.assertEqual(cfg["guarantee_columns_selection"], [38])
        self.assertEqual(cfg["deconvolvers"][0]["params"]["device"], "cuda")
        self.assertNotIn("params", cfg["deconvolvers"][1])
        # wizard-only gate key never leaks
        self.assertNotIn("feature_selection_mode", cfg)

    def test_drops_blank_strings(self):
        answers = {"output_dir": "", "labels_dict_path": "App/labels_dict.json"}
        cfg = self.wiz.build_config(answers)
        self.assertNotIn("output_dir", cfg)
        self.assertIn("labels_dict_path", cfg)

    def test_guarantee_columns_absent_when_not_provided(self):
        answers = {"labels_dict_path": "App/labels_dict.json"}
        cfg = self.wiz.build_config(answers)
        self.assertNotIn("guarantee_columns_selection", cfg)


class TestGeneratedConfigSatisfiesValidator(unittest.TestCase):
    def test_build_config_has_pipeline_required_fields(self):
        # Mirrors App/main.py validate_config required_fields["fit_deconvolution"].
        answers = {
            "labels_dict_path": "App/labels_dict.json",
            "num_input_labels": 39,
            "num_output_labels": 39,
            "pseudobulk_path": "/tmp/pb.h5",
            "splits": ["train", "valid"],
            "feature_selection_mode": "cutoff",
            "feature_cutoff": 1.1,
            "guarantee_diagonal_selection": True,
            "generate_feature_selection_plot": False,
            "deconvolvers": [{"name": "nnls"}],
            "output_dir": "/tmp/out",
        }
        cfg = FitDeconvolutionWizard().build_config(answers)
        for field in ("pseudobulk_path", "output_dir", "labels_dict_path"):
            self.assertIn(field, cfg)


if __name__ == "__main__":
    unittest.main()
