"""Tests for the hdf5_utils module."""

import tempfile
import shutil
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from syto.data.hdf5_utils import (
    PseudobulkResult,
    GenerationParameters,
    CheckpointManager,
    HDF5BatchWriter,
)


class TestPseudobulkResult(unittest.TestCase):
    """Tests for PseudobulkResult dataclass."""

    def test_to_dict_returns_all_fields(self):
        """Verify to_dict() includes all required fields."""
        result = PseudobulkResult(
            index=0,
            target_proportions=np.array([0.5, 0.5]),
            actual_proportions=np.array([0.48, 0.52]),
            n_reads_really_sampled=100,
            n_samples_per_class_per_grg=np.array([[50, 50], [48, 52]]),
            seed=42,
            aggregated_features=pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
        )
        d = result.to_dict()
        self.assertIn("index", d)
        self.assertIn("target_proportions", d)
        self.assertIn("seed", d)


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager."""

    def setUp(self):
        """Set up temporary directory for checkpoint files."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.manager = CheckpointManager(self.output_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_create_and_load_config(self):
        """Verify config can be created, saved, and loaded correctly."""
        config = self.manager.create_config(
            parameters_hash="abc123",
            splits_order=["train", "valid", "test"],
            n_pseudobulks_per_split={"train": 100, "valid": 20, "test": 20},
        )
        loaded = self.manager.load_config()
        self.assertEqual(loaded.parameters_hash, "abc123")
        self.assertEqual(loaded.splits_order, ["train", "valid", "test"])

    def test_create_and_load_split_checkpoint(self):
        """Verify split checkpoint can be created with correct batch count."""
        checkpoint = self.manager.create_split_checkpoint("train", 100, 10)
        self.assertEqual(checkpoint.total_batches, 10)
        loaded = self.manager.load_split_checkpoint("train")
        self.assertEqual(loaded.split, "train")
        self.assertEqual(loaded.total_batches, 10)

    def test_mark_batch_completed(self):
        """Verify completed batches are tracked correctly."""
        self.manager.create_split_checkpoint("train", 100, 10)
        self.manager.mark_batch_completed("train", 0)
        self.manager.mark_batch_completed("train", 1)
        loaded = self.manager.load_split_checkpoint("train")
        self.assertEqual(loaded.completed_batches, [0, 1])

    def test_get_missing_batches(self):
        """Verify get_missing_batches returns only incomplete batch indices."""
        self.manager.create_split_checkpoint("train", 50, 10)  # 5 batches
        self.manager.mark_batch_completed("train", 0)
        self.manager.mark_batch_completed("train", 2)
        missing = self.manager.get_missing_batches("train")
        self.assertEqual(missing, [1, 3, 4])


class TestHDF5BatchWriter(unittest.TestCase):
    """Tests for HDF5BatchWriter."""

    def setUp(self):
        """Set up temporary directory for batch files."""
        self.temp_dir = tempfile.mkdtemp()
        self.batches_dir = Path(self.temp_dir) / "batches"
        self.writer = HDF5BatchWriter(self.batches_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_write_and_read_batch(self):
        """Verify batch data survives a write-read round trip."""
        results = [
            PseudobulkResult(
                index=0,
                target_proportions=np.array([0.5, 0.5]),
                actual_proportions=np.array([0.48, 0.52]),
                n_reads_really_sampled=100,
                n_samples_per_class_per_grg=np.array([[25, 25], [24, 26]]),
                seed=42,
                aggregated_features=pd.DataFrame(
                    {"pred_0_wavg": [0.3, 0.7], "pred_1_wavg": [0.6, 0.4]}
                ),
            ),
            PseudobulkResult(
                index=1,
                target_proportions=np.array([0.3, 0.7]),
                actual_proportions=np.array([0.29, 0.71]),
                n_reads_really_sampled=100,
                n_samples_per_class_per_grg=np.array([[15, 15], [35, 35]]),
                seed=43,
                aggregated_features=pd.DataFrame(
                    {"pred_0_wavg": [0.2, 0.8], "pred_1_wavg": [0.5, 0.5]}
                ),
            ),
        ]
        self.writer.write_batch(0, results)

        # Verify file exists
        self.assertTrue(self.writer.batch_exists(0))

        # Read back and verify
        loaded = self.writer.read_batch(0)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].index, 0)
        self.assertEqual(loaded[0].seed, 42)
        np.testing.assert_array_almost_equal(
            loaded[0].target_proportions, np.array([0.5, 0.5])
        )

    def test_atomic_write_no_temp_file_left(self):
        """Verify no temporary files remain after a successful write."""
        results = [
            PseudobulkResult(
                index=0,
                target_proportions=np.array([1.0]),
                actual_proportions=np.array([1.0]),
                n_reads_really_sampled=10,
                n_samples_per_class_per_grg=np.array([[10]]),
                seed=1,
                aggregated_features=pd.DataFrame({"col": [1]}),
            )
        ]
        self.writer.write_batch(5, results)

        # Check no .tmp files exist
        tmp_files = list(self.batches_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)


class TestGenerationParameters(unittest.TestCase):
    """Tests for GenerationParameters."""

    def test_to_hash_is_deterministic(self):
        """Verify identical parameters produce the same hash."""
        params1 = GenerationParameters(
            cell_types_mapping={"A": 0, "B": 1},
            gr_groups_mapping={"gr1": 0, "gr2": 1},
            substitution_method="uniform_number",
            grg_sampling_method="uniform_multinomial",
        )
        params2 = GenerationParameters(
            cell_types_mapping={"A": 0, "B": 1},
            gr_groups_mapping={"gr1": 0, "gr2": 1},
            substitution_method="uniform_number",
            grg_sampling_method="uniform_multinomial",
        )
        self.assertEqual(params1.to_hash(), params2.to_hash())

    def test_different_params_different_hash(self):
        """Verify different parameters produce different hashes."""
        params1 = GenerationParameters(
            cell_types_mapping={"A": 0, "B": 1},
            gr_groups_mapping={"gr1": 0, "gr2": 1},
            substitution_method="uniform_number",
            grg_sampling_method="uniform_multinomial",
        )
        params2 = GenerationParameters(
            cell_types_mapping={"A": 0, "C": 2},  # Different
            gr_groups_mapping={"gr1": 0, "gr2": 1},
            substitution_method="uniform_number",
            grg_sampling_method="uniform_multinomial",
        )
        self.assertNotEqual(params1.to_hash(), params2.to_hash())


if __name__ == "__main__":
    unittest.main()
