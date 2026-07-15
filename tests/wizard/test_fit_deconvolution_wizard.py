import unittest
from unittest import mock

from App.wizard.tasks import fit_deconvolution as fd


class _StubEngine:
    """Minimal engine with scripted, sequential ask_checkbox / ask_one returns."""

    def __init__(self, checkbox=None, one=None):
        self._checkbox = list(checkbox or [])
        self._one = list(one or [])
        self.calls = []       # choices offered on each ask_checkbox call
        self.one_specs = []    # FieldSpecs passed to each ask_one call

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
            answers = {"pseudobulk_h5_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(eng.calls[0], ["train", "valid", "test"])

    def test_fallback_when_unreadable(self):
        with mock.patch.object(fd, "_list_splits", return_value=[]):
            eng = _StubEngine(checkbox=[])  # ask_checkbox must NOT be called
            answers = {"pseudobulk_h5_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid", "test"])
        self.assertEqual(eng.calls, [])

    def test_reprompts_until_train_and_valid_included(self):
        with mock.patch.object(
            fd, "_list_splits", return_value=["train", "valid", "test"]
        ):
            eng = _StubEngine(checkbox=[["valid"], ["train", "valid"]])  # 1st omits train
            answers = {"pseudobulk_h5_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(len(eng.calls), 2)

    def test_accepts_selection_when_file_lacks_required(self):
        with mock.patch.object(fd, "_list_splits", return_value=["train", "test"]):
            eng = _StubEngine(checkbox=[["train", "test"]])
            answers = {"pseudobulk_h5_path": "x.h5"}
            fd._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "test"])
        self.assertEqual(len(eng.calls), 1)  # no reprompt for absent 'valid'


if __name__ == "__main__":
    unittest.main()
