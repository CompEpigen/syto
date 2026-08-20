import os
import tempfile
import unittest
from unittest import mock

from syto.app.wizard.tasks import fit_calibration as fc


class TestDiscoveryHelpers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_available_deconvolvers_scans_files_in_canonical_order(self):
        for fname in ("swn_best_deconvolver.pt", "xgb_deconvolver.joblib"):
            open(os.path.join(self.tmp, fname), "w").close()
        # xgb precedes swn in canonical order regardless of touch order
        self.assertEqual(fc._available_deconvolvers(self.tmp), ["xgb", "swn"])

    def test_available_deconvolvers_empty_when_none_present(self):
        self.assertEqual(fc._available_deconvolvers(self.tmp), [])


class _StubEngine:
    """Minimal engine exposing ask_checkbox with scripted, sequential returns."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # choices offered on each ask_checkbox call

    def ask_checkbox(self, label, choices):
        self.calls.append(list(choices))
        return self._responses.pop(0)


class TestSplitsSection(unittest.TestCase):
    def test_splits_resolved_from_file(self):
        with mock.patch.object(
            fc, "_list_splits", return_value=["train", "valid", "test"]
        ):
            eng = _StubEngine([["train", "valid"]])
            answers = {"pseudobulk_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(eng.calls[0], ["train", "valid", "test"])

    def test_splits_fallback_when_unreadable(self):
        with mock.patch.object(fc, "_list_splits", return_value=[]):
            eng = _StubEngine([])  # ask_checkbox must NOT be called
            answers = {"pseudobulk_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid", "test"])
        self.assertEqual(eng.calls, [])

    def test_splits_reprompts_until_valid_included(self):
        with mock.patch.object(
            fc, "_list_splits", return_value=["train", "valid", "test"]
        ):
            eng = _StubEngine([["train"], ["train", "valid"]])  # 1st omits valid
            answers = {"pseudobulk_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(len(eng.calls), 2)

    def test_splits_no_valid_in_file_accepts_single_selection(self):
        with mock.patch.object(fc, "_list_splits", return_value=["train", "test"]):
            eng = _StubEngine([["train", "test"]])
            answers = {"pseudobulk_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "test"])
        self.assertEqual(len(eng.calls), 1)  # no reprompt when file has no 'valid'


class TestDeconvolversSection(unittest.TestCase):
    def test_default_params_attached_in_selection_order(self):
        with mock.patch.object(
            fc,
            "_available_deconvolvers",
            return_value=["xgb", "swn", "mlp", "nnls", "psls"],
        ):
            eng = _StubEngine([["swn", "xgb", "psls"]])
            answers = {"deconvolvers_dir": "d"}
            fc._deconvolvers_section(eng, answers)
        self.assertEqual(
            answers["deconvolvers"],
            [
                {"name": "swn", "params": {"device": "cuda"}},
                {"name": "xgb"},
                {"name": "psls", "params": {"n_workers": 2}},
            ],
        )

    def test_discovery_offers_only_present(self):
        with mock.patch.object(
            fc, "_available_deconvolvers", return_value=["swn", "mlp"]
        ):
            eng = _StubEngine([["swn"]])
            answers = {"deconvolvers_dir": "d"}
            fc._deconvolvers_section(eng, answers)
        self.assertEqual(eng.calls[0], ["swn", "mlp"])
        self.assertEqual(
            answers["deconvolvers"], [{"name": "swn", "params": {"device": "cuda"}}]
        )

    def test_fallback_to_all_known_when_none_found(self):
        with mock.patch.object(fc, "_available_deconvolvers", return_value=[]):
            eng = _StubEngine([["nnls"]])
            answers = {"deconvolvers_dir": "d"}
            fc._deconvolvers_section(eng, answers)
        self.assertEqual(eng.calls[0], ["xgb", "swn", "mlp", "nnls", "psls"])
        self.assertEqual(answers["deconvolvers"], [{"name": "nnls"}])


from syto.app.wizard.tasks import TASK_REGISTRY
from syto.app.wizard.tasks.fit_calibration import FitCalibrationWizard


class TestFitCalibrationSchema(unittest.TestCase):
    def setUp(self):
        self.specs = FitCalibrationWizard().field_specs()
        self.keys = [s.key for s in self.specs]
        self.by_key = {s.key: s for s in self.specs}

    def test_registered(self):
        self.assertIn("fit_calibration", TASK_REGISTRY)
        self.assertIs(TASK_REGISTRY["fit_calibration"], FitCalibrationWizard)

    def test_labels_dict_first(self):
        self.assertEqual(self.keys[0], "labels_dict_path")

    def test_pseudobulk_asked_before_splits(self):
        self.assertLess(self.keys.index("pseudobulk_path"), self.keys.index("splits"))

    def test_deconvolvers_dir_before_deconvolvers(self):
        self.assertLess(
            self.keys.index("deconvolvers_dir"), self.keys.index("deconvolvers")
        )

    def test_num_input_labels_is_expert(self):
        self.assertEqual(self.by_key["num_input_labels"].tier, "expert")

    def test_sections_are_list_sections(self):
        self.assertEqual(self.by_key["splits"].kind, "list_section")
        self.assertEqual(self.by_key["deconvolvers"].kind, "list_section")


class TestFitCalibrationBuildConfig(unittest.TestCase):
    def setUp(self):
        self.wiz = FitCalibrationWizard()

    def test_nested_and_lists_passthrough(self):
        answers = {
            "labels_dict_path": "syto/app/labels_dict.json",
            "num_output_labels": 39,
            "num_input_labels": 39,
            "pseudobulk_path": "/tmp/pb.h5",
            "splits": ["train", "valid", "test"],
            "features_mask_path": "/tmp/mask.npz",
            "deconvolvers_dir": "/tmp/deconv/",
            "deconvolvers": [
                {"name": "swn", "params": {"device": "cuda"}},
                {"name": "xgb"},
            ],
            "output_dir": "/tmp/out",
        }
        cfg = self.wiz.build_config(answers)
        self.assertEqual(cfg["pseudobulk_path"], "/tmp/pb.h5")
        self.assertEqual(cfg["splits"], ["train", "valid", "test"])
        self.assertEqual(cfg["deconvolvers"][0]["params"]["device"], "cuda")
        self.assertNotIn("params", cfg["deconvolvers"][1])

    def test_drops_blank_strings(self):
        answers = {"output_dir": "", "labels_dict_path": "syto/app/labels_dict.json"}
        cfg = self.wiz.build_config(answers)
        self.assertNotIn("output_dir", cfg)
        self.assertIn("labels_dict_path", cfg)


if __name__ == "__main__":
    unittest.main()
