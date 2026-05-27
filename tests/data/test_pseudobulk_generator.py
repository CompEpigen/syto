"""Tests for the pseudobulk_generator module."""

import tempfile
import shutil
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from syto.data.pseudobulk_generator import (
    PseudobulkGenerator,
    _sample_reads_per_grg_uniform_multinomial,
    _sample_read_ids_from_grouped_dataframe,
    build_target_columns,
)
from syto.data.hdf5_utils import (
    PseudobulkResult,
    PureProfileResult,
    GenerationMetadata,
    GenerationParameters,
    HDF5BatchWriter,
)
from syto.modelling.prediction_aggregation import aggregate_predictions_by_grg_optimized

# pylint: disable=protected-access


def build_prediction_columns(n_classes: int = 39):
    """Return the prediction columns expected by the pseudo-bulk helpers."""
    return [f"prediction_{index}" for index in range(n_classes)]


def build_minimal_split_dataframe(labels: list, n_gr_groups: int = 39):
    """Create a minimal split dataframe for testing."""
    rows = []
    prediction_columns = build_prediction_columns()

    for label in labels:
        for grg_label in range(n_gr_groups):
            row = {
                "original_label": label,
                "dmr_ctype_label": grg_label,
                "dmr_ctype": f"ctype_{grg_label}",
                "name": f"region_{label}_{grg_label}",
                "chr": "chr1",
                "NCPGS": 2,
                "M_rate": 0.5,
                "label": label,
            }
            for pred_idx, pred_col in enumerate(prediction_columns):
                row[pred_col] = (pred_idx + 1) / 100.0
            rows.append(row)

    return pd.DataFrame(rows)


def build_realistic_split_dataframe(
    labels: list, n_gr_groups: int = 5, n_reads_per_group: int = 100
):
    """Create a more realistic split dataframe with multiple reads per group."""
    rows = []
    n_classes = len(labels)

    for label in labels:
        for grg_label in range(n_gr_groups):
            for read_idx in range(n_reads_per_group):
                row = {
                    "original_label": label,
                    "dmr_ctype_label": grg_label,
                    "dmr_ctype": f"ctype_{grg_label}",
                    "name": f"region_{label}_{grg_label}_{read_idx}",
                    "chr": "chr1",
                    "NCPGS": np.random.randint(1, 10),
                    "M_rate": np.random.random(),
                    "methylation_level": np.random.random(),  # Required for methylation_level_wavg
                    "label": label,
                    "chromosome": "chr1",
                }
                # Create prediction values that vary by read
                for pred_idx in range(n_classes):
                    if pred_idx == label:
                        row[f"prediction_{pred_idx}"] = 0.8 + 0.2 * np.random.random()
                    else:
                        row[f"prediction_{pred_idx}"] = 0.1 * np.random.random()
                rows.append(row)

    return pd.DataFrame(rows)


class TestSampleReadsPerGRGUniformMultinomial(unittest.TestCase):
    """Tests for _sample_reads_per_grg_uniform_multinomial function."""

    def test_returns_correct_shape(self):
        """Verify the output array has shape (n_classes, n_grg)."""
        n_samples_per_class = np.array([100, 200, 50])
        n_grg = 10
        result = _sample_reads_per_grg_uniform_multinomial(n_samples_per_class, n_grg)
        self.assertEqual(result.shape, (3, 10))

    def test_row_sums_match_input(self):
        """Verify that row sums equal the requested samples per class."""
        n_samples_per_class = np.array([100, 200, 50])
        n_grg = 10
        result = _sample_reads_per_grg_uniform_multinomial(n_samples_per_class, n_grg)
        np.testing.assert_array_equal(result.sum(axis=1), n_samples_per_class)

    def test_handles_zero_samples(self):
        """Verify that zero samples for a class results in all zeros for that row."""
        n_samples_per_class = np.array([0, 100, 0])
        n_grg = 5
        result = _sample_reads_per_grg_uniform_multinomial(n_samples_per_class, n_grg)
        self.assertEqual(result[0].sum(), 0)
        self.assertEqual(result[1].sum(), 100)
        self.assertEqual(result[2].sum(), 0)


class TestSampleReadIdsFromGroupedDataframe(unittest.TestCase):
    """Tests for _sample_read_ids_from_grouped_dataframe function."""

    def test_returns_correct_number_of_samples(self):
        """Verify the output array length matches the total requested samples."""
        n_samples = np.array([[5, 3], [2, 4]])  # 2 classes, 2 GRGs
        indices = {
            (0, 0): np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (0, 1): np.array([10, 11, 12, 13, 14]),
            (1, 0): np.array([20, 21, 22, 23, 24]),
            (1, 1): np.array([30, 31, 32, 33, 34, 35]),
        }
        result = _sample_read_ids_from_grouped_dataframe(n_samples, indices, seed=42)
        self.assertEqual(len(result), n_samples.sum())

    def test_reproducible_with_same_seed(self):
        """Verify that using the same seed produces identical results."""
        n_samples = np.array([[5, 3], [2, 4]])
        indices = {
            (0, 0): np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (0, 1): np.array([10, 11, 12, 13, 14]),
            (1, 0): np.array([20, 21, 22, 23, 24]),
            (1, 1): np.array([30, 31, 32, 33, 34, 35]),
        }
        result1 = _sample_read_ids_from_grouped_dataframe(n_samples, indices, seed=42)
        result2 = _sample_read_ids_from_grouped_dataframe(n_samples, indices, seed=42)
        np.testing.assert_array_equal(result1, result2)


class TestGenerateSinglePseudobulk(unittest.TestCase):
    """Tests for PseudobulkGenerator.generate_single_pseudobulk."""

    def test_returns_pseudobulk_result(self):
        """Verify the method returns a valid PseudobulkResult with expected fields."""
        df = build_minimal_split_dataframe(labels=[0, 1], n_gr_groups=5)
        indices_dict = {}
        for (label, grg), group in df.groupby(["original_label", "dmr_ctype_label"]):
            indices_dict[(label, grg)] = group.index.to_numpy()

        result = PseudobulkGenerator.generate_single_pseudobulk(
            n_reads_to_sample=100,
            n_gr_groups=5,
            indices_per_class_and_grg=indices_dict,
            read_df=df,
            target_proportions=np.array([0.5, 0.5]),
            grg_grouping_columns=["dmr_ctype_label", "dmr_ctype"],
            seed=42,
            index=0,
        )

        self.assertIsInstance(result, PseudobulkResult)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.seed, 42)
        np.testing.assert_almost_equal(result.target_proportions.sum(), 1.0)

    def test_seed_reproducibility(self):
        """Test that the same seed produces the same read sampling.

        Note: The seed controls read ID selection within each (class, GRG) group,
        but the multinomial distribution of samples across GRGs uses global random
        state. So we verify the seed is stored and target proportions match.
        """
        df = build_minimal_split_dataframe(labels=[0, 1], n_gr_groups=5)
        indices_dict = {}
        for (label, grg), group in df.groupby(["original_label", "dmr_ctype_label"]):
            indices_dict[(label, grg)] = group.index.to_numpy()

        result1 = PseudobulkGenerator.generate_single_pseudobulk(
            n_reads_to_sample=100,
            n_gr_groups=5,
            indices_per_class_and_grg=indices_dict,
            read_df=df,
            target_proportions=np.array([0.5, 0.5]),
            grg_grouping_columns=["dmr_ctype_label", "dmr_ctype"],
            seed=123,
            index=0,
        )
        result2 = PseudobulkGenerator.generate_single_pseudobulk(
            n_reads_to_sample=100,
            n_gr_groups=5,
            indices_per_class_and_grg=indices_dict,
            read_df=df,
            target_proportions=np.array([0.5, 0.5]),
            grg_grouping_columns=["dmr_ctype_label", "dmr_ctype"],
            seed=123,
            index=0,
        )

        # Same seed should be stored
        self.assertEqual(result1.seed, result2.seed)
        self.assertEqual(result1.seed, 123)

        # Target proportions should match input
        np.testing.assert_array_almost_equal(
            result1.target_proportions, np.array([0.5, 0.5])
        )
        np.testing.assert_array_almost_equal(
            result2.target_proportions, np.array([0.5, 0.5])
        )


class TestPseudobulkGeneratorColumnArguments(unittest.TestCase):
    """Tests for PseudobulkGenerator class_label_column and grg_label_column arguments."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_dataframe_with_custom_columns(
        self,
        class_col: str = "original_label",
        grg_col: str = "dmr_ctype_label",
    ) -> pd.DataFrame:
        """Create a minimal dataframe with custom column names."""
        rows = []
        for label in [0, 1]:
            for grg_label in range(3):
                row = {
                    class_col: label,
                    grg_col: grg_label,
                    "dmr_ctype": f"ctype_{grg_label}",
                    "NCPGS": 2,
                    "prediction_0": 0.5,
                    "prediction_1": 0.5,
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def _create_generator(
        self,
        splits_df: dict,
        class_label_column: str = "original_label",
        grg_label_column: str = "dmr_ctype_label",
        grg_grouping_columns: list = None,
    ) -> PseudobulkGenerator:
        """Create a PseudobulkGenerator instance for testing."""
        if grg_grouping_columns is None:
            grg_grouping_columns = [grg_label_column, "dmr_ctype"]

        metadata = GenerationMetadata(
            gr_id_column="name",
            labeling_scheme="test",
            classifier="test",
            data_watermark="test",
        )
        parameters = GenerationParameters(
            cell_types_mapping={"class_0": 0, "class_1": 1},
            gr_groups_mapping={"0": 0, "1": 1, "2": 2},
            substitution_method="uniform_number",
            gr_sampling_method="uniform_multinomial",
        )
        target_proportions = np.array([[0.5, 0.5], [0.3, 0.7]])

        return PseudobulkGenerator(
            splits_df=splits_df,
            output_directory=self.output_dir,
            target_proportions_per_split={"train": target_proportions},
            batch_size=10,
            n_workers=1,
            metadata=metadata,
            parameters=parameters,
            class_label_column=class_label_column,
            grg_label_column=grg_label_column,
            grg_grouping_columns=grg_grouping_columns,
        )

    def test_default_column_arguments(self):
        """Test that default column arguments work correctly."""
        df = self._create_dataframe_with_custom_columns()
        splits_df = {"train": df}

        generator = self._create_generator(splits_df)

        self.assertEqual(generator.class_label_column, "original_label")
        self.assertEqual(generator.grg_label_column, "dmr_ctype_label")
        # Check that indices were precomputed correctly
        self.assertIn("train", generator._indices_per_split)

    def test_custom_class_label_column(self):
        """Test that a custom class_label_column is used correctly."""
        df = self._create_dataframe_with_custom_columns(class_col="my_class_label")
        splits_df = {"train": df}

        generator = self._create_generator(
            splits_df,
            class_label_column="my_class_label",
        )

        self.assertEqual(generator.class_label_column, "my_class_label")
        # Check that indices were precomputed using the custom column
        self.assertIn("train", generator._indices_per_split)
        indices = generator._indices_per_split["train"]
        # Should have 2 classes * 3 GRGs = 6 groups
        self.assertEqual(len(indices), 6)

    def test_custom_grg_label_column(self):
        """Test that a custom grg_label_column is used correctly."""
        df = self._create_dataframe_with_custom_columns(grg_col="my_grg_label")
        splits_df = {"train": df}

        generator = self._create_generator(
            splits_df,
            grg_label_column="my_grg_label",
            grg_grouping_columns=["my_grg_label", "dmr_ctype"],
        )

        self.assertEqual(generator.grg_label_column, "my_grg_label")

    def test_custom_both_columns(self):
        """Test using both custom class_label_column and grg_label_column."""
        df = self._create_dataframe_with_custom_columns(
            class_col="cell_type",
            grg_col="region_group",
        )
        splits_df = {"train": df}

        generator = self._create_generator(
            splits_df,
            class_label_column="cell_type",
            grg_label_column="region_group",
            grg_grouping_columns=["region_group", "dmr_ctype"],
        )

        self.assertEqual(generator.class_label_column, "cell_type")
        self.assertEqual(generator.grg_label_column, "region_group")
        indices = generator._indices_per_split["train"]
        self.assertEqual(len(indices), 6)

    def test_grg_label_not_in_grouping_columns_raises(self):
        """Test that AssertionError is raised when grg_label_column is not in grg_grouping_columns."""
        df = self._create_dataframe_with_custom_columns()
        splits_df = {"train": df}

        with self.assertRaises(AssertionError) as context:
            self._create_generator(
                splits_df,
                grg_label_column="dmr_ctype_label",
                grg_grouping_columns=["other_column", "dmr_ctype"],  # Missing grg_label
            )

        self.assertIn("dmr_ctype_label", str(context.exception))
        self.assertIn("must be in", str(context.exception))


class TestBuildTargetColumns(unittest.TestCase):
    """Tests for build_target_columns function."""

    def test_default_columns(self):
        """Verify default columns match legacy structure."""
        cols = build_target_columns(num_prediction_classes=39)
        self.assertIn("dmr_ctype_label", cols)
        self.assertIn("dmr_ctype", cols)
        self.assertIn("prediction_0_wavg", cols)
        self.assertIn("prediction_38_wavg", cols)
        self.assertIn("methylation_level_wavg", cols)
        self.assertIn("total_weight", cols)
        self.assertIn("n_reads", cols)
        self.assertIn("chromosome", cols)
        self.assertIn("label", cols)

    def test_custom_num_classes(self):
        """Verify prediction columns match num_prediction_classes."""
        cols = build_target_columns(num_prediction_classes=5)
        pred_cols = [c for c in cols if c.startswith("prediction_")]
        self.assertEqual(len(pred_cols), 5)
        self.assertIn("prediction_4_wavg", cols)
        self.assertNotIn("prediction_5_wavg", cols)


class TestReproducibilityFromStoredData(unittest.TestCase):
    """Tests verifying that stored data can reproduce aggregated features.

    This is critical for reproducibility: given the seed, n_samples_per_class_per_grg,
    and input DataFrame, we should be able to exactly reproduce the sampled reads
    and aggregated features.
    """

    def test_seed_and_n_samples_reproduce_read_ids(self):
        """Test that seed + n_samples_per_class_per_grg reproduce exact same read_ids."""
        df = build_realistic_split_dataframe(
            labels=[0, 1], n_gr_groups=5, n_reads_per_group=50
        )
        indices_dict = {}
        for (label, grg), group in df.groupby(["original_label", "dmr_ctype_label"]):
            indices_dict[(label, grg)] = group.index.to_numpy()

        # Define a fixed n_samples_per_class_per_grg
        n_samples = np.array(
            [
                [10, 8, 12, 9, 11],  # Class 0
                [15, 12, 18, 13, 17],  # Class 1
            ]
        )
        seed = 12345

        # Sample twice with same parameters
        read_ids_1 = _sample_read_ids_from_grouped_dataframe(
            n_samples, indices_dict, seed
        )
        read_ids_2 = _sample_read_ids_from_grouped_dataframe(
            n_samples, indices_dict, seed
        )

        np.testing.assert_array_equal(read_ids_1, read_ids_2)

    def test_stored_data_reproduces_aggregated_features(self):
        """Test that stored seed + n_samples_per_class_per_grg + df reproduces aggregated_features."""
        np.random.seed(42)  # Fix seed for reproducible test data
        df = build_realistic_split_dataframe(
            labels=[0, 1], n_gr_groups=5, n_reads_per_group=50
        )
        indices_dict = {}
        for (label, grg), group in df.groupby(["original_label", "dmr_ctype_label"]):
            indices_dict[(label, grg)] = group.index.to_numpy()

        grg_grouping_columns = ["dmr_ctype_label", "dmr_ctype"]
        columns_to_keep = build_target_columns(num_prediction_classes=2)

        # Generate a pseudobulk and store its metadata
        result = PseudobulkGenerator.generate_single_pseudobulk(
            n_reads_to_sample=200,
            n_gr_groups=5,
            indices_per_class_and_grg=indices_dict,
            read_df=df,
            target_proportions=np.array([0.4, 0.6]),
            grg_grouping_columns=grg_grouping_columns,
            columns_to_keep=columns_to_keep,
            seed=54321,
            index=0,
        )

        # Now reproduce using only the stored data
        stored_seed = result.seed
        stored_n_samples = result.n_samples_per_class_per_grg

        # Resample read IDs using stored data
        reproduced_read_ids = _sample_read_ids_from_grouped_dataframe(
            stored_n_samples, indices_dict, stored_seed
        )

        # Reaggregate
        reproduced_features = aggregate_predictions_by_grg_optimized(
            df.iloc[reproduced_read_ids],
            grg_grouping_columns,
            weight_col="NCPGS",
        )
        if columns_to_keep is not None:
            reproduced_features = reproduced_features[columns_to_keep]

        # Compare with original
        pd.testing.assert_frame_equal(
            result.aggregated_features.reset_index(drop=True),
            reproduced_features.reset_index(drop=True),
        )

    def test_hdf5_roundtrip_preserves_reproducibility_data(self):
        """Test that HDF5 storage preserves all data needed for reproducibility."""
        temp_dir = tempfile.mkdtemp()
        try:
            np.random.seed(42)
            df = build_realistic_split_dataframe(
                labels=[0, 1], n_gr_groups=3, n_reads_per_group=20
            )
            indices_dict = {}
            for (label, grg), group in df.groupby(
                ["original_label", "dmr_ctype_label"]
            ):
                indices_dict[(label, grg)] = group.index.to_numpy()

            grg_grouping_columns = ["dmr_ctype_label", "dmr_ctype"]

            # Generate result
            original_result = PseudobulkGenerator.generate_single_pseudobulk(
                n_reads_to_sample=50,
                n_gr_groups=3,
                indices_per_class_and_grg=indices_dict,
                read_df=df,
                target_proportions=np.array([0.5, 0.5]),
                grg_grouping_columns=grg_grouping_columns,
                seed=99999,
                index=0,
            )

            # Write to HDF5
            batches_dir = Path(temp_dir) / "batches"
            writer = HDF5BatchWriter(batches_dir)
            writer.write_batch(0, [original_result])

            # Read back
            loaded_results = writer.read_batch(0)
            loaded_result = loaded_results[0]

            # Verify stored data matches
            self.assertEqual(loaded_result.seed, original_result.seed)
            np.testing.assert_array_equal(
                loaded_result.n_samples_per_class_per_grg,
                original_result.n_samples_per_class_per_grg,
            )
            np.testing.assert_array_almost_equal(
                loaded_result.target_proportions,
                original_result.target_proportions,
            )
            np.testing.assert_array_almost_equal(
                loaded_result.actual_proportions,
                original_result.actual_proportions,
            )

            # Reproduce from loaded data
            reproduced_read_ids = _sample_read_ids_from_grouped_dataframe(
                loaded_result.n_samples_per_class_per_grg,
                indices_dict,
                loaded_result.seed,
            )
            reproduced_features = aggregate_predictions_by_grg_optimized(
                df.iloc[reproduced_read_ids],
                grg_grouping_columns,
                weight_col="NCPGS",
            )

            # Get numeric columns from loaded result for comparison
            numeric_cols = loaded_result.aggregated_features.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            # Compare numeric columns
            pd.testing.assert_frame_equal(
                loaded_result.aggregated_features[numeric_cols].reset_index(drop=True),
                reproduced_features[numeric_cols].reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            shutil.rmtree(temp_dir)


class TestPureProfileGeneration(unittest.TestCase):
    """Tests for pure profile generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_generator(self, n_classes: int = 2, n_gr_groups: int = 3):
        """Create a PseudobulkGenerator for testing."""
        np.random.seed(42)
        df = build_realistic_split_dataframe(
            labels=list(range(n_classes)),
            n_gr_groups=n_gr_groups,
            n_reads_per_group=50,
        )
        splits_df = {"train": df}

        cell_types = {f"class_{i}": i for i in range(n_classes)}
        metadata = GenerationMetadata(
            gr_id_column="name",
            labeling_scheme="test",
            classifier="test",
            data_watermark="test",
        )
        parameters = GenerationParameters(
            cell_types_mapping=cell_types,
            gr_groups_mapping={str(i): i for i in range(n_gr_groups)},
            substitution_method="uniform_number",
            gr_sampling_method="uniform_multinomial",
        )
        target_proportions = np.array([[1.0 / n_classes] * n_classes])

        return PseudobulkGenerator(
            splits_df=splits_df,
            output_directory=self.output_dir,
            target_proportions_per_split={"train": target_proportions},
            batch_size=10,
            n_workers=1,
            metadata=metadata,
            parameters=parameters,
            n_reads_to_sample=100,
        )

    def test_pure_profile_returns_correct_structure(self):
        """Test that _generate_pure_profiles returns correct PureProfileResult structure."""
        generator = self._create_generator(n_classes=2, n_gr_groups=3)

        # Create checkpoint first (required by the method)
        generator.checkpoint_manager.create_split_checkpoint("train", 1, 10)

        pure_profile = generator._generate_pure_profiles("train")

        self.assertIsInstance(pure_profile, PureProfileResult)
        self.assertEqual(pure_profile.split_name, "train")
        # feature_matrices should have shape (n_classes, n_gr_groups, n_numeric_features)
        self.assertEqual(pure_profile.feature_matrices.shape[0], 2)  # n_classes
        self.assertEqual(pure_profile.feature_matrices.shape[1], 3)  # n_gr_groups
        # uniform_prior should have shape (n_gr_groups, n_numeric_features)
        self.assertEqual(pure_profile.uniform_prior.shape[0], 3)
        # Check column names are stored
        self.assertIsNotNone(pure_profile.numeric_columns)
        self.assertGreater(len(pure_profile.numeric_columns), 0)

    def test_pure_profile_uniform_prior_is_average(self):
        """Test that uniform_prior is the average of feature_matrices across classes."""
        generator = self._create_generator(n_classes=3, n_gr_groups=4)
        generator.checkpoint_manager.create_split_checkpoint("train", 1, 10)

        pure_profile = generator._generate_pure_profiles("train")

        expected_prior = pure_profile.feature_matrices.mean(axis=0)
        np.testing.assert_array_almost_equal(
            pure_profile.uniform_prior,
            expected_prior,
        )

    def test_pure_profile_string_columns_stored(self):
        """Test that string columns are properly stored in PureProfileResult."""
        generator = self._create_generator(n_classes=2, n_gr_groups=3)
        generator.checkpoint_manager.create_split_checkpoint("train", 1, 10)

        pure_profile = generator._generate_pure_profiles("train")

        # String columns should be stored if present in target columns
        if pure_profile.string_columns:
            self.assertIsNotNone(pure_profile.string_matrices)
            self.assertEqual(pure_profile.string_matrices.shape[0], 2)  # n_classes


class TestGenerateSingleSplit(unittest.TestCase):
    """Tests for _generate_single_split method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_generates_correct_number_of_batches(self):
        """Test that correct number of batch files are created."""
        np.random.seed(42)
        df = build_realistic_split_dataframe(
            labels=[0, 1], n_gr_groups=3, n_reads_per_group=30
        )
        splits_df = {"train": df}

        metadata = GenerationMetadata(
            gr_id_column="name",
            labeling_scheme="test",
            classifier="test",
            data_watermark="test",
        )
        parameters = GenerationParameters(
            cell_types_mapping={"class_0": 0, "class_1": 1},
            gr_groups_mapping={"0": 0, "1": 1, "2": 2},
            substitution_method="uniform_number",
            gr_sampling_method="uniform_multinomial",
        )
        # 5 pseudobulks, batch_size=2 -> 3 batches
        target_proportions = np.array(
            [
                [0.5, 0.5],
                [0.3, 0.7],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )

        generator = PseudobulkGenerator(
            splits_df=splits_df,
            output_directory=self.output_dir,
            target_proportions_per_split={"train": target_proportions},
            batch_size=2,
            n_workers=1,
            metadata=metadata,
            parameters=parameters,
            n_reads_to_sample=50,
        )

        # Run generation for train split
        generator.checkpoint_manager.create_config(
            parameters_hash=parameters.to_hash(),
            splits_order=["train"],
            n_pseudobulks_per_split={"train": 5},
        )
        generator._generate_single_split("train")

        # Check batches were created
        batches_dir = generator.checkpoint_manager.get_split_batches_dir("train")
        batch_files = list(batches_dir.glob("batch_*.h5"))
        self.assertEqual(len(batch_files), 3)  # ceil(5/2) = 3

        # Check checkpoint was updated
        checkpoint = generator.checkpoint_manager.load_split_checkpoint("train")
        self.assertEqual(sorted(checkpoint.completed_batches), [0, 1, 2])

    def test_resume_skips_completed_batches(self):
        """Test that generation resumes from where it left off."""
        np.random.seed(42)
        df = build_realistic_split_dataframe(
            labels=[0, 1], n_gr_groups=3, n_reads_per_group=30
        )
        splits_df = {"train": df}

        metadata = GenerationMetadata(
            gr_id_column="name",
            labeling_scheme="test",
            classifier="test",
            data_watermark="test",
        )
        parameters = GenerationParameters(
            cell_types_mapping={"class_0": 0, "class_1": 1},
            gr_groups_mapping={"0": 0, "1": 1, "2": 2},
            substitution_method="uniform_number",
            gr_sampling_method="uniform_multinomial",
        )
        target_proportions = np.array(
            [
                [0.5, 0.5],
                [0.3, 0.7],
                [0.7, 0.3],
            ]
        )

        generator = PseudobulkGenerator(
            splits_df=splits_df,
            output_directory=self.output_dir,
            target_proportions_per_split={"train": target_proportions},
            batch_size=1,
            n_workers=1,
            metadata=metadata,
            parameters=parameters,
            n_reads_to_sample=50,
        )

        # Pre-create checkpoint with batch 0 already completed
        generator.checkpoint_manager.create_split_checkpoint("train", 3, 1)
        generator.checkpoint_manager.mark_batch_completed("train", 0)

        # Manually write batch 0 to simulate partial completion
        batch_writer = HDF5BatchWriter(
            generator.checkpoint_manager.get_split_batches_dir("train")
        )
        dummy_result = PseudobulkResult(
            index=0,
            target_proportions=np.array([0.5, 0.5]),
            actual_proportions=np.array([0.5, 0.5]),
            actual_n_reads_sampled=50,
            n_samples_per_class_per_grg=np.array([[25], [25]]),
            seed=1,
            aggregated_features=pd.DataFrame({"col": [1, 2, 3]}),
        )
        batch_writer.write_batch(0, [dummy_result])

        # Run generation - should skip batch 0
        generator._generate_single_split("train")

        checkpoint = generator.checkpoint_manager.load_split_checkpoint("train")
        self.assertEqual(sorted(checkpoint.completed_batches), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
