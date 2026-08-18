"""Tests for the columnar pseudobulk backend.

A columnar store and an HDF5 store built
from the *same* generation batches must be indistinguishable through the
:class:`~syto.data.pseudobulk_store.PseudobulkStore` contract.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from syto.data.pseudobulk_columnar import (
    FORMAT_VERSION,
    PseudobulkColumnarReader,
    PseudobulkColumnarWriter,
)
from syto.data.pseudobulk_store import detect_store_format, open_pseudobulk_store
from tests.data.pseudobulk_fixtures import CLASS_LABEL_COLUMN, PseudobulkFixture


class ColumnarEquivalenceMixin:
    """Assertions comparing a columnar store against an HDF5 store."""

    fixture: PseudobulkFixture
    h5_path: Path
    columnar_path: Path

    def assert_stores_equivalent(self, num_pred_classes=None):
        """Both backends agree across the whole read contract."""
        h5 = open_pseudobulk_store(self.h5_path)
        col = open_pseudobulk_store(self.columnar_path)
        n_pred = num_pred_classes or self.fixture.n_predictions

        self.assertEqual(sorted(h5.list_splits()), sorted(col.list_splits()))

        for split in sorted(h5.list_splits()):
            self.assertEqual(
                h5.count_pseudobulks(split), col.count_pseudobulks(split), split
            )

            fh, ph = h5.read_pseudobulk_matrices(split, n_pred)
            fc, pc = col.read_pseudobulk_matrices(split, n_pred)
            np.testing.assert_array_equal(fh, fc)
            np.testing.assert_array_equal(ph, pc)

            np.testing.assert_array_equal(
                h5.read_pure_feature_matrix(split, n_pred),
                col.read_pure_feature_matrix(split, n_pred),
            )
            pd.testing.assert_frame_equal(
                h5.read_uniform_prior(split), col.read_uniform_prior(split)
            )

            params_h5 = list(h5.iter_pseudobulk_params(split))
            params_col = list(col.iter_pseudobulk_params(split))
            self.assertEqual(len(params_h5), len(params_col))
            for (sh, ch, th), (sc, cc, tc) in zip(params_h5, params_col):
                self.assertEqual(sh, sc)
                np.testing.assert_array_equal(ch, cc)
                np.testing.assert_array_equal(th, tc)

            subsets_h5 = list(h5.iter_pseudobulk_read_subsets(split, CLASS_LABEL_COLUMN))
            subsets_col = list(col.iter_pseudobulk_read_subsets(split, CLASS_LABEL_COLUMN))
            self.assertEqual(len(subsets_h5), len(subsets_col))
            for (rh, th), (rc, tc) in zip(subsets_h5, subsets_col):
                pd.testing.assert_frame_equal(
                    rh.reset_index(drop=True),
                    rc.reset_index(drop=True),
                    check_dtype=False,
                )
                np.testing.assert_array_equal(th, tc)


class TestColumnarEquivalenceSquare(unittest.TestCase, ColumnarEquivalenceMixin):
    """Predictions == classes, two splits."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(cls.tmp, n_classes=3, n_grg=2, seed=1)
        cls.h5_path, cls.columnar_path = cls.fixture.build_both()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_equivalent(self):
        """The two backends are indistinguishable through the contract."""
        self.assert_stores_equivalent()

    def test_fewer_prediction_classes_requested(self):
        """Requesting a prefix of the prediction columns agrees too."""
        self.assert_stores_equivalent(num_pred_classes=2)


class TestColumnarEquivalenceBackgroundClass(unittest.TestCase, ColumnarEquivalenceMixin):
    """Predictions exceed classes, as in the background-class runs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(
            cls.tmp, n_classes=3, n_grg=2, n_predictions=4, seed=2
        )
        cls.h5_path, cls.columnar_path = cls.fixture.build_both()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_equivalent(self):
        """An extra prediction column does not disturb the contract."""
        self.assert_stores_equivalent()

    def test_target_proportions_stay_at_class_count(self):
        """Targets keep the class width even with more prediction columns."""
        col = open_pseudobulk_store(self.columnar_path)
        _, targets = col.read_pseudobulk_matrices("train", 4)
        self.assertEqual(targets.shape[1], self.fixture.n_classes)


class TestColumnarEquivalenceThreeSplits(unittest.TestCase, ColumnarEquivalenceMixin):
    """Three splits, matching the runs that carry a test split."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(
            cls.tmp, splits=("train", "valid", "test"), n_pseudobulks=5, seed=3
        )
        cls.h5_path, cls.columnar_path = cls.fixture.build_both()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_equivalent(self):
        """All three splits agree."""
        self.assert_stores_equivalent()

    def test_all_splits_present(self):
        """The manifest enumerates every split."""
        col = open_pseudobulk_store(self.columnar_path)
        self.assertEqual(sorted(col.list_splits()), ["test", "train", "valid"])


class TestColumnarEquivalenceProductionShape(unittest.TestCase, ColumnarEquivalenceMixin):
    """The widths the published runs actually use: 39 GR groups, 40 predictions.

    Small fixtures cannot surface reshape or integer-width mistakes that only
    appear once the GR-group and class axes are the same size, which is exactly
    the case in every real run.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(
            cls.tmp,
            n_classes=39,
            n_grg=39,
            n_predictions=40,
            n_pseudobulks=25,
            splits=("train", "valid"),
            batch_size=10,
            rows_per_group=2,
            seed=11,
        )
        cls.h5_path, cls.columnar_path = cls.fixture.build_both()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_equivalent(self):
        """Square class/GR axes do not get transposed anywhere."""
        self.assert_stores_equivalent()

    def test_counts_orientation_preserved(self):
        """n_reads_per_gr keeps (n_classes, n_grg) orientation, not its transpose."""
        h5 = open_pseudobulk_store(self.h5_path)
        col = open_pseudobulk_store(self.columnar_path)
        _, counts_h5, _ = next(iter(h5.iter_pseudobulk_params("train")))
        _, counts_col, _ = next(iter(col.iter_pseudobulk_params("train")))
        self.assertEqual(counts_h5.shape, (39, 39))
        np.testing.assert_array_equal(counts_h5, counts_col)


class TestColumnarLayout(unittest.TestCase):
    """The on-disk shape of the store."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(cls.tmp, n_classes=3, n_grg=2, seed=4)
        cls.path = cls.fixture.build_columnar()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_detected_as_columnar(self):
        """The store directory is recognised by the factory."""
        self.assertEqual(detect_store_format(self.path), "columnar")
        self.assertIsInstance(
            open_pseudobulk_store(self.path), PseudobulkColumnarReader
        )

    def test_expected_files_exist(self):
        """Every documented file is written."""
        for rel in [
            "manifest.json",
            "metadata.json",
            "parameters.json",
            "inputs/train.parquet",
            "outputs/train/features.parquet",
            "outputs/train/proportions.parquet",
            "outputs/train/n_reads_per_gr.parquet",
            "outputs/train/pure_profiles.npz",
        ]:
            self.assertTrue((self.path / rel).exists(), rel)

    def test_manifest_records_shape_and_dtype(self):
        """The manifest is self-describing."""
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.assertEqual(manifest["format_version"], FORMAT_VERSION)
        self.assertEqual(manifest["feature_dtype"], "float64")
        train = manifest["splits"]["train"]
        self.assertEqual(train["n_pseudobulks"], self.fixture.n_pseudobulks)
        self.assertEqual(train["n_grg"], self.fixture.n_grg)
        self.assertEqual(train["n_classes"], self.fixture.n_classes)
        self.assertEqual(train["feature_columns"], self.fixture.columns)
        self.assertTrue(train["has_pure_profiles"])
        self.assertEqual(train["n_input_rows"], len(self.fixture.input_dfs["train"]))

    def test_features_row_count_and_index(self):
        """features.parquet is one row per (pseudobulk, GR group), in order."""
        table = pq.read_table(self.path / "outputs/train/features.parquet")
        n_pb, n_grg = self.fixture.n_pseudobulks, self.fixture.n_grg
        self.assertEqual(table.num_rows, n_pb * n_grg)
        pb_index = table.column("pb_index").to_numpy()
        np.testing.assert_array_equal(pb_index, np.repeat(np.arange(n_pb), n_grg))
        grg_row = table.column("grg_row").to_numpy()
        np.testing.assert_array_equal(grg_row, np.tile(np.arange(n_grg), n_pb))

    def test_counts_row_count(self):
        """n_reads_per_gr.parquet is one row per (pseudobulk, class)."""
        table = pq.read_table(self.path / "outputs/train/n_reads_per_gr.parquet")
        self.assertEqual(
            table.num_rows, self.fixture.n_pseudobulks * self.fixture.n_classes
        )

    def test_inputs_preserve_row_order(self):
        """The input frame round-trips in exactly the generation order."""
        original = self.fixture.input_dfs["train"]
        stored = pq.read_table(self.path / "inputs/train.parquet").to_pandas()
        pd.testing.assert_frame_equal(stored, original)


class TestColumnarWriterGuards(unittest.TestCase):
    """The writer refuses inputs it cannot represent faithfully."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ragged_features_rejected(self):
        """A pseudobulk with the wrong number of GR-group rows is an error."""
        from syto.data.pseudobulk_hdf5_utils import PseudobulkResult

        writer = PseudobulkColumnarWriter(self.tmp / "store")
        columns = ["dmr_ctype_label", "prediction_0_wavg", "total_weight", "n_reads"]
        split = writer.open_split("train", columns)
        split.open()
        good = PseudobulkResult(
            index=0,
            target_proportions=np.array([1.0]),
            actual_proportions=np.array([1.0]),
            n_reads_sampled=3,
            n_samples_per_class_per_grg=np.array([[1, 2]]),
            seed=7,
            aggregated_features=pd.DataFrame(np.zeros((2, 4)), columns=columns),
        )
        split.write_results([good])

        ragged = PseudobulkResult(
            index=1,
            target_proportions=np.array([1.0]),
            actual_proportions=np.array([1.0]),
            n_reads_sampled=3,
            n_samples_per_class_per_grg=np.array([[1, 2]]),
            seed=8,
            aggregated_features=pd.DataFrame(np.zeros((3, 4)), columns=columns),
        )
        with self.assertRaises(ValueError):
            split.write_results([ragged])
        split.close()

    def test_missing_manifest_is_reported(self):
        """Opening a directory without a manifest fails clearly."""
        (self.tmp / "empty").mkdir()
        with self.assertRaises(FileNotFoundError):
            PseudobulkColumnarReader(self.tmp / "empty")

    def test_unknown_split_raises(self):
        """Reading an absent split raises KeyError."""
        fixture = PseudobulkFixture(self.tmp, n_pseudobulks=2, seed=5)
        store = open_pseudobulk_store(fixture.build_columnar())
        with self.assertRaises(KeyError):
            store.count_pseudobulks("nope")
        with self.assertRaises(KeyError):
            store.read_pseudobulk_matrices("nope", 3)


class TestFeatureDtype(unittest.TestCase):
    """Storing narrowed features is a deliberate, recorded choice."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fixture = PseudobulkFixture(self.tmp, n_classes=3, n_grg=2, seed=6)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_is_lossless(self):
        """float64 by default, matching the HDF5 backend exactly."""
        h5 = open_pseudobulk_store(self.fixture.build_hdf5())
        col = open_pseudobulk_store(self.fixture.build_columnar())
        fh, _ = h5.read_pseudobulk_matrices("train", 3)
        fc, _ = col.read_pseudobulk_matrices("train", 3)
        self.assertEqual(fc.dtype, np.float64)
        np.testing.assert_array_equal(fh, fc)

    def test_float32_is_close_and_recorded(self):
        """float32 stays within tolerance and is stamped in the manifest."""
        h5 = open_pseudobulk_store(self.fixture.build_hdf5())
        path = self.fixture.build_columnar(
            self.tmp / "columnar32", feature_dtype="float32"
        )
        col = open_pseudobulk_store(path)
        fh, _ = h5.read_pseudobulk_matrices("train", 3)
        fc, _ = col.read_pseudobulk_matrices("train", 3)
        self.assertEqual(fc.dtype, np.float32)
        np.testing.assert_allclose(fh, fc, rtol=1e-6, atol=1e-7)
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertEqual(manifest["feature_dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
