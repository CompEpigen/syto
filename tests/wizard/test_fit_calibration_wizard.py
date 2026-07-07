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


if __name__ == "__main__":
    unittest.main()
