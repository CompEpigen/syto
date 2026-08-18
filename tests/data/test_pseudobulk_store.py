"""Tests for the format-independent pseudobulk store contract."""

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader
from syto.data.pseudobulk_store import (
    BasePseudobulkReader,
    PseudobulkStore,
    detect_store_format,
    open_pseudobulk_store,
)
from tests.data.pseudobulk_fixtures import PseudobulkFixture

N_CLASSES = 3
N_GRG = 2
N_PSEUDOBULKS = 4
FIRST_SEED = 1000


def _build_small_h5(directory: Path) -> Path:
    """Write a miniature but structurally faithful consolidated HDF5 file."""
    fixture = PseudobulkFixture(
        directory,
        n_classes=N_CLASSES,
        n_grg=N_GRG,
        n_pseudobulks=N_PSEUDOBULKS,
        splits=("train",),
        batch_size=N_PSEUDOBULKS,
    )
    return fixture.build_hdf5()


class TestDetectStoreFormat(unittest.TestCase):
    """Tests for path-based backend selection."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_h5_file_is_hdf5(self):
        """A .h5 file is recognised as an HDF5 store."""
        path = self.tmp / "pseudobulk.h5"
        path.touch()
        self.assertEqual(detect_store_format(path), "hdf5")

    def test_directory_with_h5_is_hdf5(self):
        """A directory holding pseudobulk.h5 resolves to the HDF5 backend."""
        (self.tmp / "pseudobulk.h5").touch()
        self.assertEqual(detect_store_format(self.tmp), "hdf5")

    def test_directory_with_manifest_is_columnar(self):
        """A directory holding manifest.json is a columnar store."""
        (self.tmp / "manifest.json").write_text("{}")
        self.assertEqual(detect_store_format(self.tmp), "columnar")

    def test_manifest_wins_over_stray_h5(self):
        """A manifest identifies the store even if a .h5 sits beside it."""
        (self.tmp / "manifest.json").write_text("{}")
        (self.tmp / "pseudobulk.h5").touch()
        self.assertEqual(detect_store_format(self.tmp), "columnar")

    def test_missing_path_raises(self):
        """A path that does not exist is reported as missing."""
        with self.assertRaises(FileNotFoundError):
            detect_store_format(self.tmp / "nope.h5")

    def test_unrecognised_directory_raises(self):
        """An empty directory is not a store."""
        with self.assertRaises(ValueError):
            detect_store_format(self.tmp)

    def test_unrecognised_suffix_raises(self):
        """A file with an unrelated suffix is not a store."""
        path = self.tmp / "pseudobulk.parquet"
        path.touch()
        with self.assertRaises(ValueError):
            detect_store_format(path)


class TestOpenPseudobulkStore(unittest.TestCase):
    """Tests for the factory and the contract it returns."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.h5_path = _build_small_h5(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_returns_hdf5_reader_for_file(self):
        """Opening a .h5 file yields the HDF5 backend."""
        store = open_pseudobulk_store(self.h5_path)
        self.assertIsInstance(store, PseudobulkHDF5Reader)

    def test_directory_resolves_to_inner_h5(self):
        """Opening the containing directory resolves to pseudobulk.h5."""
        store = open_pseudobulk_store(self.h5_path.parent)
        self.assertEqual(store.path, self.h5_path)
        self.assertEqual(store.list_splits(), ["train"])

    def test_satisfies_the_protocol(self):
        """The returned reader structurally satisfies PseudobulkStore."""
        self.assertIsInstance(open_pseudobulk_store(self.h5_path), PseudobulkStore)

    def test_hdf5_reader_derives_from_base(self):
        """The HDF5 reader inherits the shared reconstruction logic."""
        self.assertTrue(issubclass(PseudobulkHDF5Reader, BasePseudobulkReader))


class TestCountPseudobulks(unittest.TestCase):
    """Tests for the method that replaces the pipeline's direct h5py access."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.h5_path = _build_small_h5(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_counts_match_generated(self):
        """The count equals the number of pseudobulks written."""
        store = open_pseudobulk_store(self.h5_path)
        self.assertEqual(store.count_pseudobulks("train"), N_PSEUDOBULKS)

    def test_agrees_with_reading_the_matrices(self):
        """The count agrees with the first axis of the feature matrix."""
        store = open_pseudobulk_store(self.h5_path)
        features, _ = store.read_pseudobulk_matrices("train", N_CLASSES)
        self.assertEqual(store.count_pseudobulks("train"), features.shape[0])

    def test_unknown_split_raises(self):
        """An absent split raises rather than returning a misleading zero."""
        store = open_pseudobulk_store(self.h5_path)
        with self.assertRaises(KeyError):
            store.count_pseudobulks("nonexistent")


class TestSharedReconstruction(unittest.TestCase):
    """The reconstruction path derived in the base class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.h5_path = _build_small_h5(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_read_subsets_agree_with_params_and_state(self):
        """iter_pseudobulk_read_subsets composes the two primitives faithfully."""
        from syto.data.pseudobulk_generator import (
            _sample_read_ids_from_grouped_dataframe,
        )

        store = open_pseudobulk_store(self.h5_path)
        input_df, indices = store.build_reconstruction_state("train")

        expected = []
        for seed, n_per_class_per_grg, target in store.iter_pseudobulk_params("train"):
            read_ids = _sample_read_ids_from_grouped_dataframe(
                n_per_class_per_grg, indices, seed=seed
            )
            expected.append((input_df.iloc[read_ids].reset_index(drop=True), target))

        actual = list(store.iter_pseudobulk_read_subsets("train"))
        self.assertEqual(len(actual), len(expected))
        for (reads_a, target_a), (reads_e, target_e) in zip(actual, expected):
            pd.testing.assert_frame_equal(reads_a, reads_e)
            np.testing.assert_array_equal(target_a, target_e)

    def test_reconstruction_is_deterministic(self):
        """Two iterations of the same store yield identical reads."""
        store = open_pseudobulk_store(self.h5_path)
        first = [r for r, _ in store.iter_pseudobulk_read_subsets("train")]
        second = [r for r, _ in store.iter_pseudobulk_read_subsets("train")]
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)

    def test_params_are_yielded_in_index_order(self):
        """Seeds come back in pseudobulk index order, not HDF5 key order."""
        store = open_pseudobulk_store(self.h5_path)
        seeds = [seed for seed, _, _ in store.iter_pseudobulk_params("train")]
        self.assertEqual(seeds, [FIRST_SEED + i for i in range(N_PSEUDOBULKS)])

    def test_prediction_indices_reject_missing_column(self):
        """Asking for more prediction classes than exist is an error."""
        store = open_pseudobulk_store(self.h5_path)
        with self.assertRaises(KeyError):
            store.read_pseudobulk_matrices("train", N_CLASSES + 5)


if __name__ == "__main__":
    unittest.main()
