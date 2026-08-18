"""Every consumer of a pseudobulk store works against both on-disk layouts.
"""

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from App.wizard import validators as v
from App.wizard.tasks.deconvolver_common import list_splits
from syto.data.pseudobulk_store import open_pseudobulk_store
from tests.data.pseudobulk_fixtures import PseudobulkFixture

N_CLASSES = 3
N_GRG = 2
N_PSEUDOBULKS = 6


class DualFormatBase(unittest.TestCase):
    """Builds one fixture consolidated into both layouts."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.fixture = PseudobulkFixture(
            cls.tmp,
            n_classes=N_CLASSES,
            n_grg=N_GRG,
            n_pseudobulks=N_PSEUDOBULKS,
            splits=("train", "valid"),
            seed=31,
        )
        cls.h5_path = cls.fixture.build_hdf5()
        cls.columnar_path = cls.fixture.build_columnar()

        cls.labels_path = cls.tmp / "labels_dict.json"
        cls.labels_path.write_text(
            json.dumps({str(i): f"ct_{i}" for i in range(N_CLASSES)})
        )
        cls.logger = logging.getLogger("dual-format-test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @property
    def store_paths(self):
        """The same data in both layouts."""
        return {"hdf5": self.h5_path, "columnar": self.columnar_path}


class TestDeconvolutionFittingPipeline(DualFormatBase):
    """App/deconvolution_pipeline.py"""

    def _pipeline(self, store_path):
        from App.deconvolution_pipeline import DeconvolutionFittingPipeline

        return DeconvolutionFittingPipeline(
            {
                "labels_dict_path": str(self.labels_path),
                "pseudobulk_path": str(store_path),
                "output_dir": str(self.tmp / f"out_{Path(store_path).name}"),
                "num_input_labels": N_CLASSES,
                "num_output_labels": N_CLASSES,
            },
            self.logger,
        )

    def test_constructs_against_both_layouts(self):
        """The pipeline opens either store without knowing which it got."""
        for kind, path in self.store_paths.items():
            with self.subTest(kind=kind):
                self.assertEqual(
                    sorted(self._pipeline(path).reader.list_splits()),
                    ["train", "valid"],
                )

    def test_reads_identical_matrices(self):
        """Feature and proportion matrices agree across layouts."""
        readers = {k: self._pipeline(p).reader for k, p in self.store_paths.items()}
        for split in ("train", "valid"):
            fh, ph = readers["hdf5"].read_pseudobulk_matrices(split, N_CLASSES)
            fc, pc = readers["columnar"].read_pseudobulk_matrices(split, N_CLASSES)
            np.testing.assert_array_equal(fh, fc)
            np.testing.assert_array_equal(ph, pc)

    def test_reads_identical_pure_profiles(self):
        """Pure profiles, which drive feature selection, agree across layouts."""
        readers = {k: self._pipeline(p).reader for k, p in self.store_paths.items()}
        np.testing.assert_array_equal(
            readers["hdf5"].read_pure_feature_matrix("train", N_CLASSES),
            readers["columnar"].read_pure_feature_matrix("train", N_CLASSES),
        )


class TestCalibrationPipeline(DualFormatBase):
    """App/calibration_pipeline.py"""

    def _pipeline(self, store_path):
        from App.calibration_pipeline import CalibratorFittingPipeline

        return CalibratorFittingPipeline(
            {
                "labels_dict_path": str(self.labels_path),
                "pseudobulk_path": str(store_path),
                "output_dir": str(self.tmp / f"cal_{Path(store_path).name}"),
                "deconvolvers": [],
                "num_input_labels": N_CLASSES,
                "num_output_labels": N_CLASSES,
            },
            self.logger,
        )

    def test_reads_identical_matrices(self):
        """Calibration sees the same pseudobulks from either layout."""
        readers = {k: self._pipeline(p).reader for k, p in self.store_paths.items()}
        for split in ("train", "valid"):
            fh, ph = readers["hdf5"].read_pseudobulk_matrices(split, N_CLASSES)
            fc, pc = readers["columnar"].read_pseudobulk_matrices(split, N_CLASSES)
            np.testing.assert_array_equal(fh, fc)
            np.testing.assert_array_equal(ph, pc)

    def test_reader_is_none_without_a_store(self):
        """A config with no store still constructs, as before."""
        from App.calibration_pipeline import CalibratorFittingPipeline

        pipeline = CalibratorFittingPipeline(
            {
                "labels_dict_path": str(self.labels_path),
                "output_dir": str(self.tmp / "cal_none"),
                "deconvolvers": [],
            },
            self.logger,
        )
        self.assertIsNone(pipeline.reader)


class TestBaselineDeconvolutionPipeline(DualFormatBase):
    """App/pseudobulk_deconvolution_pipeline.py, including the removed h5py leak."""

    def test_count_split_pseudobulks_agrees(self):
        """The count that replaced the direct h5py access works on both."""
        from App.pseudobulk_deconvolution_pipeline import (
            PseudobulkDeconvolutionPipeline as Pipeline,
        )

        counts = {}
        for kind, path in self.store_paths.items():
            reader = open_pseudobulk_store(path)
            counts[kind] = Pipeline._count_split_pseudobulks(reader, "train")
        self.assertEqual(counts["hdf5"], N_PSEUDOBULKS)
        self.assertEqual(counts["columnar"], N_PSEUDOBULKS)

    def test_count_returns_none_for_unknown_split(self):
        """An unreadable count degrades to None rather than raising.

        It only sizes a progress bar, so the pipeline must survive it.
        """
        from App.pseudobulk_deconvolution_pipeline import (
            PseudobulkDeconvolutionPipeline as Pipeline,
        )

        for kind, path in self.store_paths.items():
            with self.subTest(kind=kind):
                reader = open_pseudobulk_store(path)
                self.assertIsNone(Pipeline._count_split_pseudobulks(reader, "nope"))

    def test_reconstruction_state_agrees(self):
        """Baseline deconvolution reconstructs the same reads from either layout."""
        h5 = open_pseudobulk_store(self.h5_path)
        col = open_pseudobulk_store(self.columnar_path)
        df_h5, idx_h5 = h5.build_reconstruction_state("train")
        df_col, idx_col = col.build_reconstruction_state("train")
        pd.testing.assert_frame_equal(df_h5, df_col, check_dtype=False)
        self.assertEqual(set(idx_h5), set(idx_col))
        for key in idx_h5:
            np.testing.assert_array_equal(idx_h5[key], idx_col[key])


class TestInferenceUniformPrior(DualFormatBase):
    """App/inference.py loads its uniform prior from either layout."""

    def test_uniform_prior_agrees(self):
        """The prior used for missing-label filling is layout-independent."""
        h5 = open_pseudobulk_store(self.h5_path)
        col = open_pseudobulk_store(self.columnar_path)
        pd.testing.assert_frame_equal(
            h5.read_uniform_prior("train"), col.read_uniform_prior("train")
        )

    def test_inference_module_uses_the_factory(self):
        """The call site was actually migrated, not just the import."""
        source = Path("App/inference.py").read_text()
        self.assertIn("open_pseudobulk_store", source)
        self.assertNotIn("PseudobulkHDF5Reader", source)


class TestWizardDualFormat(DualFormatBase):
    """The wizard enumerates splits and validates paths for both layouts."""

    def test_list_splits_agrees(self):
        """Split discovery works for a file and for a directory."""
        for kind, path in self.store_paths.items():
            with self.subTest(kind=kind):
                self.assertEqual(sorted(list_splits(str(path))), ["train", "valid"])

    def test_list_splits_still_swallows_errors(self):
        """A bad path yields an empty list rather than raising."""
        self.assertEqual(list_splits("/nonexistent/store.h5"), [])

    def test_validator_accepts_both_layouts(self):
        """Validation passes for a .h5 file and a columnar directory."""
        for kind, path in self.store_paths.items():
            with self.subTest(kind=kind):
                self.assertIsNone(v.pseudobulk_store(str(path)))

    def test_validator_accepts_directory_holding_h5(self):
        """A run directory containing pseudobulk.h5 is accepted."""
        self.assertIsNone(v.pseudobulk_store(str(self.h5_path.parent)))

    def test_validator_rejects_missing_and_unrecognised(self):
        """Bad paths produce a message instead of passing silently."""
        self.assertIsNotNone(v.pseudobulk_store(str(self.tmp / "nope.h5")))
        stray = self.tmp / "stray"
        stray.mkdir(exist_ok=True)
        self.assertIsNotNone(v.pseudobulk_store(str(stray)))


if __name__ == "__main__":
    unittest.main()
