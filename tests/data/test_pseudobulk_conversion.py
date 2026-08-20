"""Tests for the HDF5 -> columnar conversion script.

The conversion reads the *consolidated* HDF5 layout, which differs from the
per-batch layout in ways that are easy to miss: consolidated pseudobulk groups
carry no ``index`` attribute, for instance, because the index lives in the
group name.  These tests exercise the script against real consolidated stores
built by the fixture so that such differences fail here rather than part-way
through a multi-hour batch conversion.
"""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from syto.data.pseudobulk_store import open_pseudobulk_store
from tests.data.pseudobulk_fixtures import PseudobulkFixture

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "Scripts"
    / "convert_pseudobulk_to_columnar.py"
)


def _load_converter():
    """Import the conversion script as a module."""
    spec = importlib.util.spec_from_file_location("convert_pseudobulk", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_pseudobulk"] = module
    spec.loader.exec_module(module)
    return module


convert_module = _load_converter()


class ConversionCaseMixin:
    """Convert a fixture and assert the result is equivalent to its source."""

    fixture_kwargs: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(cls.tmp, **cls.fixture_kwargs)
        cls.h5_path = cls.fixture.build_hdf5()
        cls.dest = cls.tmp / "converted"
        convert_module.convert(cls.h5_path, cls.dest, dtype="float64", pb_chunk=3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_verifies_as_equivalent(self):
        """The bundled verifier reports no problems."""
        problems = convert_module.verify_equivalence(
            self.h5_path, self.dest, n_read_subsets=5
        )
        self.assertEqual(problems, [])

    def test_counts_match(self):
        """Every split keeps its pseudobulk count."""
        h5 = open_pseudobulk_store(self.h5_path)
        col = open_pseudobulk_store(self.dest)
        self.assertEqual(sorted(h5.list_splits()), sorted(col.list_splits()))
        for split in h5.list_splits():
            self.assertEqual(h5.count_pseudobulks(split), col.count_pseudobulks(split))

    def test_indices_recovered_from_group_names(self):
        """Seeds land in index order despite no stored index attribute.

        The consolidated layout names groups ``i_<index>`` and stores no
        ``index`` attribute, so a converter that trusts the attribute fails and
        one that trusts HDF5 key order mis-orders past ``i_9``.
        """
        col = open_pseudobulk_store(self.dest)
        split = col.list_splits()[0]
        seeds = [seed for seed, _, _ in col.iter_pseudobulk_params(split)]
        expected = [1000 + i for i in range(self.fixture.n_pseudobulks)]
        self.assertEqual(seeds, expected)


class TestConversionSquare(ConversionCaseMixin, unittest.TestCase):
    """Predictions equal to classes."""

    fixture_kwargs = {"n_classes": 3, "n_grg": 2, "seed": 21}


class TestConversionBackgroundClass(ConversionCaseMixin, unittest.TestCase):
    """An extra prediction column, as in the background-class runs."""

    fixture_kwargs = {"n_classes": 3, "n_grg": 2, "n_predictions": 4, "seed": 22}


class TestConversionThreeSplitsPastTen(ConversionCaseMixin, unittest.TestCase):
    """More than ten pseudobulks, so lexicographic key order would be wrong."""

    fixture_kwargs = {
        "splits": ("train", "valid", "test"),
        "n_pseudobulks": 23,
        "batch_size": 7,
        "seed": 23,
    }


class TestConversionDetectsCorruption(unittest.TestCase):
    """The verifier must actually fail when the stores disagree."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fixture = PseudobulkFixture(self.tmp, n_pseudobulks=6, seed=24)
        self.h5_path = self.fixture.build_hdf5()
        self.dest = self.tmp / "converted"
        convert_module.convert(self.h5_path, self.dest, dtype="float64", pb_chunk=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_perturbed_features_are_detected(self):
        """A single altered feature value is reported."""
        import pyarrow.parquet as pq
        import pyarrow as pa

        path = self.dest / "outputs" / "train" / "features.parquet"
        table = pq.read_table(path)
        column = "prediction_0_wavg"
        values = table.column(column).to_numpy(zero_copy_only=False).copy()
        values[0] += 1.0
        index = table.schema.get_field_index(column)
        table = table.set_column(index, column, pa.array(values))
        pq.write_table(table, path, compression="zstd", version="2.6")

        problems = convert_module.verify_equivalence(
            self.h5_path, self.dest, n_read_subsets=2
        )
        self.assertTrue(any("features differ" in p for p in problems), problems)

    def test_truncated_inputs_are_detected(self):
        """Losing input rows breaks read-subset reconstruction."""
        import pyarrow.parquet as pq

        path = self.dest / "inputs" / "train.parquet"
        table = pq.read_table(path)
        pq.write_table(table.slice(0, table.num_rows - 1), path, version="2.6")

        problems = convert_module.verify_equivalence(
            self.h5_path, self.dest, n_read_subsets=3
        )
        self.assertTrue(problems)


class TestFloat32Tolerance(unittest.TestCase):
    """Narrowed features need a tolerance; exact comparison must reject them."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fixture = PseudobulkFixture(self.tmp, n_pseudobulks=6, seed=25)
        self.h5_path = self.fixture.build_hdf5()
        self.dest = self.tmp / "converted32"
        convert_module.convert(self.h5_path, self.dest, dtype="float32", pb_chunk=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exact_comparison_rejects_float32(self):
        """Without a tolerance, float32 is correctly reported as different."""
        problems = convert_module.verify_equivalence(
            self.h5_path, self.dest, n_read_subsets=2
        )
        self.assertTrue(any("features differ" in p for p in problems), problems)

    def test_tolerant_comparison_accepts_float32(self):
        """With the documented tolerance, float32 verifies."""
        problems = convert_module.verify_equivalence(
            self.h5_path, self.dest, n_read_subsets=2, rtol=1e-6, atol=1e-8
        )
        self.assertEqual(problems, [])

    def test_integer_payloads_stay_exact(self):
        """Seeds and count matrices are unaffected by the feature dtype."""
        h5 = open_pseudobulk_store(self.h5_path)
        col = open_pseudobulk_store(self.dest)
        for (sh, ch, _), (sc, cc, _) in zip(
            h5.iter_pseudobulk_params("train"), col.iter_pseudobulk_params("train")
        ):
            self.assertEqual(sh, sc)
            np.testing.assert_array_equal(ch, cc)


if __name__ == "__main__":
    unittest.main()
