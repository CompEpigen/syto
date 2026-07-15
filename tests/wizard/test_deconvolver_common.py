import os
import tempfile
import unittest

import h5py

from App.wizard.tasks import deconvolver_common as dc


class TestListSplits(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_reads_outputs_groups(self):
        path = os.path.join(self.tmp, "pb.h5")
        with h5py.File(path, "w") as f:
            outputs = f.create_group("outputs")
            outputs.create_group("train")
            outputs.create_group("valid")
        self.assertEqual(sorted(dc.list_splits(path)), ["train", "valid"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(dc.list_splits(os.path.join(self.tmp, "nope.h5")), [])


if __name__ == "__main__":
    unittest.main()
