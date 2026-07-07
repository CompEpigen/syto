import os
import tempfile
import unittest
from unittest import mock

import h5py

from App.wizard.tasks import fit_calibration as fc


class TestDiscoveryHelpers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_list_splits_reads_outputs_groups(self):
        path = os.path.join(self.tmp, "pb.h5")
        with h5py.File(path, "w") as f:
            outputs = f.create_group("outputs")
            outputs.create_group("train")
            outputs.create_group("valid")
        self.assertEqual(sorted(fc._list_splits(path)), ["train", "valid"])

    def test_list_splits_missing_file_returns_empty(self):
        self.assertEqual(fc._list_splits(os.path.join(self.tmp, "nope.h5")), [])

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
        with mock.patch.object(fc, "_list_splits", return_value=["train", "valid", "test"]):
            eng = _StubEngine([["train", "valid"]])
            answers = {"pseudobulk_h5_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(eng.calls[0], ["train", "valid", "test"])

    def test_splits_fallback_when_unreadable(self):
        with mock.patch.object(fc, "_list_splits", return_value=[]):
            eng = _StubEngine([])  # ask_checkbox must NOT be called
            answers = {"pseudobulk_h5_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid", "test"])
        self.assertEqual(eng.calls, [])

    def test_splits_reprompts_until_valid_included(self):
        with mock.patch.object(fc, "_list_splits", return_value=["train", "valid", "test"]):
            eng = _StubEngine([["train"], ["train", "valid"]])  # 1st omits valid
            answers = {"pseudobulk_h5_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "valid"])
        self.assertEqual(len(eng.calls), 2)

    def test_splits_no_valid_in_file_accepts_single_selection(self):
        with mock.patch.object(fc, "_list_splits", return_value=["train", "test"]):
            eng = _StubEngine([["train", "test"]])
            answers = {"pseudobulk_h5_path": "x.h5"}
            fc._splits_section(eng, answers)
        self.assertEqual(answers["splits"], ["train", "test"])
        self.assertEqual(len(eng.calls), 1)  # no reprompt when file has no 'valid'


if __name__ == "__main__":
    unittest.main()
