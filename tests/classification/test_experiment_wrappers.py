import unittest
from unittest.mock import patch, MagicMock
import os
import warnings
from uuid import uuid4
import tempfile
import shutil
import subprocess
import sys
import time
import socket
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import torch

warnings.filterwarnings("ignore")

# Import the experiment wrapper classes - adjust imports as needed
from syto.classification.experiment_wrappers import (
    AbstractMLFlowExperiment,
    DismirMLflowExperiment,
    EpigenBERT2MLflowExperiment,
    MethylBertMLflowExperiment,
    RestartOnPoorPerformanceCallback,
)
from syto.classification.classifiers.dnabert2 import (
    TrainingArguments as EpigenTrainingArguments,
)
from syto.data.dataset import generate_example_data
from syto.classification.classifiers.methylbert import default_methylbert_config


def is_singularity_container():
    """Check if running inside a Singularity container."""
    return os.path.exists("/.singularity.d") or os.environ.get("SINGULARITY_CONTAINER")


def get_mlflow_storage_path():
    """
    Get appropriate storage path for MLflow artifacts.
    In Singularity containers, use mounted external storage.
    """
    if is_singularity_container():
        # Check common mount points for external storage in Singularity
        potential_paths = [
            "/external_storage/mlflow_tests",
            "/mnt/external/mlflow_tests",
            "/scratch/mlflow_tests",
            "/tmp/mlflow_tests",
        ]
        for path in potential_paths:
            parent = Path(path).parent
            if parent.exists() and os.access(parent, os.W_OK):
                return path

    # Default to system temp directory
    return tempfile.mkdtemp(prefix="mlflow_test_")


def find_free_port():
    """Find a free port for MLflow server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class MLflowServerManager:
    """Manages MLflow server lifecycle for tests."""

    def __init__(self, backend_store_uri=None, artifact_root=None):
        self.backend_store_uri = backend_store_uri or get_mlflow_storage_path()
        self.artifact_root = artifact_root or os.path.join(
            self.backend_store_uri, "artifacts"
        )
        self.port = find_free_port()
        self.server_process = None
        self.tracking_uri = f"http://127.0.0.1:{self.port}"

        # Ensure directories exist
        os.makedirs(self.backend_store_uri, exist_ok=True)
        os.makedirs(self.artifact_root, exist_ok=True)

    def start(self, timeout=30):
        """Start MLflow server."""
        # Prefer the CLI when available, otherwise run via current interpreter.
        mlflow_executable = shutil.which("mlflow")
        if mlflow_executable:
            cmd = [
                mlflow_executable,
                "server",
                "--backend-store-uri",
                f"file://{self.backend_store_uri}",
                "--default-artifact-root",
                f"file://{self.artifact_root}",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                "1",
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--backend-store-uri",
                f"file://{self.backend_store_uri}",
                "--default-artifact-root",
                f"file://{self.artifact_root}",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                "1",
            ]

        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "MLFLOW_TRACKING_URI": self.tracking_uri},
        )

        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_server_ready():
                print(f"MLflow server started on {self.tracking_uri}")
                return True
            time.sleep(0.5)

        raise RuntimeError(f"MLflow server failed to start within {timeout} seconds")

    def _is_server_ready(self):
        """Check if MLflow server is ready."""
        try:
            import requests

            response = requests.get(f"{self.tracking_uri}/health", timeout=1)
            return response.status_code == 200
        except:
            return False

    def stop(self):
        """Stop MLflow server and cleanup."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

        # Cleanup directories
        if (
            os.path.exists(self.backend_store_uri)
            and "/mlflow_test_" in self.backend_store_uri
        ):
            shutil.rmtree(self.backend_store_uri, ignore_errors=True)


class ExperimentTestBase(unittest.TestCase):
    """Base test class with common setup for experiment tests."""

    @classmethod
    def setUpClass(cls):
        """Set up MLflow server for all tests in the class."""
        cls.mlflow_manager = MLflowServerManager()
        cls.mlflow_manager.start()
        mlflow.set_tracking_uri(cls.mlflow_manager.tracking_uri)

    @classmethod
    def tearDownClass(cls):
        """Tear down MLflow server after all tests."""
        cls.mlflow_manager.stop()

    def setUp(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="exp_test_")
        self.data_path = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_path)

        # Create sample data structure
        self._create_sample_data_structure()

    def verify_experiment_was_logged(self, experiment, min_runs=1):
        """Verify that experiment has logged runs in MLflow.

        Args:
            experiment: The experiment object with experiment_name attribute
            min_runs: Minimum number of runs expected (default: 1)

        Returns:
            list: List of runs for further assertions if needed
        """
        # Get experiment details
        current_experiment = mlflow.get_experiment_by_name(experiment.experiment_name)
        self.assertIsNotNone(
            current_experiment, f"Experiment '{experiment.experiment_name}' not found"
        )

        experiment_id = current_experiment.experiment_id

        # Search for runs
        client = mlflow.tracking.MlflowClient(
            tracking_uri=self.mlflow_manager.tracking_uri
        )
        runs = client.search_runs(experiment_ids=[experiment_id])

        # Verify runs exist
        self.assertGreaterEqual(
            len(runs), min_runs, f"Expected at least {min_runs} runs, found {len(runs)}"
        )
        return runs

    def tearDown(self):
        """Clean up after each test."""
        # Clean up MLflow runs
        try:
            mlflow.end_run()
        except:
            pass

        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_sample_data_structure(self):
        """Create sample data directory structure with parquet files."""
        # Create multiple dataset directories
        datasets = ["sample_dataset_1", "sample_dataset_2", "sample_dataset_3"]

        for dataset in datasets:
            dataset_path = os.path.join(self.data_path, dataset)
            os.makedirs(dataset_path)

            # Create sample parquet files for each split
            for split in ["train", "valid", "test", "rest"]:
                df = self._create_sample_dataframe(split)
                df.to_parquet(os.path.join(dataset_path, f"{split}.parquet"))

    def _create_sample_dataframe(self, split, num_samples=10):
        """Create a sample dataframe for testing."""
        np.random.seed(42)  # For reproducibility

        data = []
        for i in range(num_samples):
            seq_len = np.random.randint(100, 500)
            dna_seq = "".join(np.random.choice(["A", "T", "C", "G"], seq_len))
            meth_seq = "".join(np.random.choice(["0", "1", "2"], seq_len))

            # Add some structure for MethylBert (DMRs)
            dmr_status = "".join(np.random.choice(["0", "1"], seq_len))
            name = uuid4()
            data.append(
                {
                    "read_name": str(name),
                    "input_ids": dna_seq,
                    "methylation_ids": meth_seq,
                    "dmr_status": dmr_status,
                    "label": np.random.randint(0, 2),
                    "sum_abs_areastat": np.random.uniform(3000, 6000),
                    "chromosome": "chr1",
                    "original_file": "some_sam_file",
                }
            )

        return pd.DataFrame(data)


class TestRestartOnPoorPerformanceCallback(unittest.TestCase):
    """Test the RestartOnPoorPerformanceCallback class."""

    def test_callback_initialization(self):
        """Test callback initialization with different parameters."""
        callback = RestartOnPoorPerformanceCallback(
            eval_loss_threshold=0.7, check_at_step=100, max_retries=3
        )

        self.assertEqual(callback.eval_loss_threshold, 0.7)
        self.assertEqual(callback.check_at_step, 100)
        self.assertEqual(callback.max_retries, 3)
        self.assertEqual(callback.retry_count, 0)
        self.assertFalse(callback.should_restart)
        self.assertFalse(callback.has_checked)

    def test_callback_triggers_restart_on_poor_performance(self):
        """Test that callback triggers restart when performance is poor."""
        callback = RestartOnPoorPerformanceCallback(
            eval_loss_threshold=0.5, check_at_step=10, max_retries=3
        )

        # Mock trainer state with poor performance
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.log_history = [{"eval_loss": 0.8, "step": 10}]

        mock_control = MagicMock()
        mock_control.should_training_stop = False

        # Call callback
        result_control = callback.on_log(None, mock_state, mock_control)

        self.assertTrue(callback.has_checked)
        self.assertTrue(callback.should_restart)
        self.assertTrue(mock_control.should_training_stop)

    def test_callback_continues_on_good_performance(self):
        """Test that callback continues when performance is good."""
        callback = RestartOnPoorPerformanceCallback(
            eval_loss_threshold=0.5, check_at_step=10, max_retries=3
        )

        # Mock trainer state with good performance
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.log_history = [{"eval_loss": 0.3, "step": 10}]

        mock_control = MagicMock()
        mock_control.should_training_stop = False

        # Call callback
        result_control = callback.on_log(None, mock_state, mock_control)

        self.assertTrue(callback.has_checked)
        self.assertFalse(callback.should_restart)
        self.assertFalse(mock_control.should_training_stop)

    def test_callback_respects_max_retries(self):
        """Test that callback respects maximum retry limit."""
        callback = RestartOnPoorPerformanceCallback(
            eval_loss_threshold=0.5, check_at_step=10, max_retries=2
        )

        # Simulate multiple retries
        callback.retry_count = 2  # Already at max

        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.log_history = [{"eval_loss": 0.8, "step": 10}]

        mock_control = MagicMock()
        mock_control.should_training_stop = False

        # Call callback
        callback.on_log(None, mock_state, mock_control)

        # Should not trigger restart when max retries reached
        self.assertFalse(callback.should_restart)
        self.assertFalse(mock_control.should_training_stop)

    def test_callback_reset_for_retry(self):
        """Test callback reset functionality."""
        callback = RestartOnPoorPerformanceCallback(
            eval_loss_threshold=0.5, check_at_step=10, max_retries=3
        )

        # Set some state
        callback.should_restart = True
        callback.has_checked = True

        # Reset for retry
        callback.reset_for_retry()

        self.assertEqual(callback.retry_count, 1)
        self.assertFalse(callback.should_restart)
        self.assertFalse(callback.has_checked)


class TestAbstractMLFlowExperiment(ExperimentTestBase):
    """Test the AbstractMLFlowExperiment base class."""

    def test_initialization_with_valid_data(self):
        """Test initialization with valid data directory."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=1000,
        )

        self.assertEqual(experiment.max_sequence_length, 1000)
        self.assertEqual(
            len(experiment.data_dirs), 3
        )  # sample_dataset_1, sample_dataset_2, sample_dataset_3
        self.assertIn("sample_dataset_1", experiment.data_dirs)
        self.assertIn("sample_dataset_2", experiment.data_dirs)

    def test_initialization_without_experiment_name_raises_error(self):
        """Test that initialization without experiment name raises error."""
        with self.assertRaises(ValueError) as context:
            AbstractMLFlowExperiment(
                data_path=self.data_path,
                experiment_name=None,
                tracking_uri=self.mlflow_manager.tracking_uri,
            )

        self.assertIn("Experiment name must be provided", str(context.exception))

    def test_initialization_with_invalid_path_raises_error(self):
        """Test initialization with non-existent path raises error."""
        with self.assertRaises(ValueError) as context:
            AbstractMLFlowExperiment(
                data_path="/non/existent/path",
                experiment_name="test_experiment",
                tracking_uri=self.mlflow_manager.tracking_uri,
            )

        self.assertIn("does not exist", str(context.exception))

    def test_find_data_directories(self):
        """Test finding data directories with required files."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        # Check that all directories were found
        self.assertEqual(len(experiment.data_dirs), 3)

        # Check that paths are correct
        for name, path in experiment.data_dirs.items():
            self.assertTrue(path.exists())
            self.assertTrue((path / "train.parquet").exists())
            self.assertTrue((path / "valid.parquet").exists())
            self.assertTrue((path / "test.parquet").exists())

    def test_calculate_data_stats(self):
        """Test calculation of dataset statistics."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        # Calculate stats for a dataset
        train_path = os.path.join(self.data_path, "sample_dataset_1", "train.parquet")
        stats = experiment._calculate_data_stats(train_path)

        self.assertIn("num_samples", stats)
        self.assertIn("num_positive", stats)
        self.assertIn("num_negative", stats)
        self.assertIn("positive_ratio", stats)
        self.assertIn("avg_sequence_length", stats)

        self.assertEqual(stats["num_samples"], 10)
        self.assertTrue(0 <= stats["positive_ratio"] <= 1)

    def test_calculate_data_stats_with_incorrect_file(self):
        """Test that calculating data stats with incorrect file returns empty dictionary."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        empty_res = experiment._calculate_data_stats(
            os.path.join(self.data_path, "sample_dataset_1", "nonexistent.parquet")
        )

        self.assertEqual(empty_res, {})

    def test_calculate_metrics_binary_classification(self):
        """Test metric calculation with various scenarios."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        # Test with normal values
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        y_pred_binary = np.array([0, 1, 0, 1, 1])

        metrics = experiment._calculate_metrics(y_true, y_pred_proba, y_pred_binary)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("mcc", metrics)

        # Test with NaN values
        y_true_nan = np.array([0, 1, np.nan, 1, 1])
        y_pred_proba_nan = np.array([0.1, 0.9, 0.5, np.nan, 0.7])

        metrics_nan = experiment._calculate_metrics(
            y_true_nan, y_pred_proba_nan, y_pred_binary
        )
        self.assertIn("nan_percentage", metrics_nan)
        self.assertIn("num_nans_predictions_proba", metrics_nan)

    def test_calculate_metrics_binary_classification_preds_shaped_N2(self):
        """Test metric calculation when binary predictions are shaped (N, 2) instead of (N,)."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred_proba = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]]
        )
        y_pred_binary = None

        metrics = experiment._calculate_metrics(y_true, y_pred_proba, y_pred_binary)

        expected_accuracy = 1.0
        self.assertIn("accuracy", metrics)
        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy, places=10)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("mcc", metrics)

    def test_calculate_metrics_multiclass_classification(self):
        """Test metric calculation for multi-class classification branch."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred_proba = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.05, 0.90, 0.05],
                [0.10, 0.10, 0.80],
                [0.20, 0.60, 0.20],
                [0.70, 0.20, 0.10],
                [0.05, 0.15, 0.80],
            ]
        )

        metrics = experiment._calculate_metrics(
            y_true,
            y_pred_proba,
            y_pred_binary=None,
            num_classes=3,
        )

        self.assertTrue(metrics["is_multiclass"])
        self.assertEqual(metrics["num_classes"], 3)
        self.assertNotIn("threshold", metrics)

        # Perfect predictions should yield perfect scores.
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=10)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=10)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=10)
        self.assertAlmostEqual(metrics["f1_score"], 1.0, places=10)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0, places=10)
        self.assertAlmostEqual(metrics["mcc"], 1.0, places=10)

        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

        self.assertAlmostEqual(metrics["precision_class_0"], 1.0, places=10)
        self.assertAlmostEqual(metrics["precision_class_1"], 1.0, places=10)
        self.assertAlmostEqual(metrics["precision_class_2"], 1.0, places=10)
        self.assertAlmostEqual(metrics["recall_class_0"], 1.0, places=10)
        self.assertAlmostEqual(metrics["recall_class_1"], 1.0, places=10)
        self.assertAlmostEqual(metrics["recall_class_2"], 1.0, places=10)
        self.assertAlmostEqual(metrics["f1_class_0"], 1.0, places=10)
        self.assertAlmostEqual(metrics["f1_class_1"], 1.0, places=10)
        self.assertAlmostEqual(metrics["f1_class_2"], 1.0, places=10)

        self.assertEqual(metrics["num_nans_predictions_proba"], 0)
        self.assertEqual(metrics["nan_percentage"], 0.0)

    def test_calculate_metrics_error(self):
        """Test that metric calculation correctly terminate when error is raised"""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        y_true = np.array([0, 1, 0])
        y_pred_proba = np.array([0.1, 0.9])  # Incorrect shape
        y_pred_binary = np.array([0, 1, 0])

        result = experiment._calculate_metrics(y_true, y_pred_proba, y_pred_binary)

        self.assertIn("error", result)
        self.assertIn("num_total_samples", result)

    def test_aggregate_predictions_multiclass(self):
        """Test multiclass chunk-to-read aggregation with weighted probabilities."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri,
        )

        data_chunked = pd.DataFrame(
            {
                "read_name": ["read_a", "read_a", "read_b", "read_b"],
                "num_cpgs": [2, 1, 1, 3],
                "label": [1, 1, 0, 0],
            }
        )

        predictions = np.array(
            [
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
                [0.6, 0.2, 0.2],
                [0.2, 0.2, 0.6],
            ]
        )

        (
            data_chunked_out,
            labels,
            predictions_binary,
            predictions_proba,
            best_threshold,
        ) = experiment._aggregate_predictions(
            data_chunked=data_chunked,
            predictions=predictions,
            num_classes=3,
        )

        self.assertIsNone(best_threshold)

        self.assertIn("predictions_proba_class_0", data_chunked_out.columns)
        self.assertIn("predictions_proba_class_1", data_chunked_out.columns)
        self.assertIn("predictions_proba_class_2", data_chunked_out.columns)

        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(predictions_binary.tolist(), [1, 2])

        expected_predictions_proba = [
            [0.1, (0.7 * 2 + 0.2 * 1) / 3, (0.2 * 2 + 0.7 * 1) / 3],
            [0.3, 0.2, 0.5],
        ]
        self.assertEqual(predictions_proba.shape, (2, 3))
        for i, row in enumerate(expected_predictions_proba):
            for j, expected_value in enumerate(row):
                self.assertAlmostEqual(
                    predictions_proba[i, j], expected_value, places=12
                )


class TestDismirMLFlowExperiment(ExperimentTestBase):
    """Test the DismirMLFlowExperiment class."""

    @patch("syto.classification.experiment_wrappers")
    def test_dismir_initialization(self, mock_dismir_class):
        """Test Dismir experiment initialization."""
        experiment = DismirMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dismir",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=500,
            model_flavor="lstm",
        )

        self.assertEqual(experiment.model_flavor, "lstm")
        self.assertEqual(experiment.max_sequence_length, 500)

    def test_dismir_training_single_dataset(self):
        """Test Dismir training on a single dataset."""

        experiment = DismirMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dismir_training",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=500,
            model_flavor="lstm",
        )

        # Run training
        results = experiment.train_dataset(
            "sample_dataset_1",
            epochs=2,
            batch_size=32,
            output_dir=self.temp_dir,  # Use temp directory for output
        )

        # Verify results
        self.assertIn("dataset", results)
        self.assertIn("metrics", results)
        self.assertIn("data_stats", results)
        self.assertEqual(results["dataset"], "sample_dataset_1")


class TestEpigenBERT2MLflowExperiment(ExperimentTestBase):
    """Test the EpigenBERT2MLflowExperiment class."""

    def test_dnabert2_initialization(self):
        """Test DNABert2 experiment initialization - integration test."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=512,
            foundation_model_huggingface="zhihan1996/DNABERT-2-117M",
        )

        self.assertEqual(experiment.max_sequence_length, 512)
        self.assertEqual(
            experiment.foundation_model_huggingface, "zhihan1996/DNABERT-2-117M"
        )

    def test_dnabert2_with_restart_callback(self):
        """Test DNABert2 training with restart callback - integration test."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2_restart",
            tracking_uri=self.mlflow_manager.tracking_uri,
            foundation_model_huggingface="foundationalModels/DNABERT-2-117M",
            max_sequence_length=128,  # Smaller for faster testing
        )

        # Run with restart callback - use minimal epochs for speed
        results = experiment.train_dataset(
            "sample_dataset_1",  # Use your test dataset name
            num_train_epochs=1,  # Minimal epochs for testing
            per_device_train_batch_size=4,  # Small batch size for testing
            per_device_eval_batch_size=4,
            eval_loss_threshold=0.7,
            check_at_step=10,  # Lower step count for testing
            output_dir=self.temp_dir,  # Use temp directory for output
        )

        # Verify results structure
        self.assertIsNotNone(results)
        self.assertIn("dataset", results)
        self.assertIn("metrics", results)
        _ = self.verify_experiment_was_logged(experiment)

    @patch("syto.classification.experiment_wrappers.SupervisedDataset")
    @patch("syto.classification.experiment_wrappers.EpigenDnabert2")
    def test_progressive_predict_train_valid_standard_branch(
        self, mock_epigen_class, mock_supervised_dataset
    ):
        """Test _progressive_predict uses standard prediction path for train/valid short sequences."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2_progressive_standard_branch",
            tracking_uri=self.mlflow_manager.tracking_uri,
            foundation_model_huggingface="foundationalModels/DNABERT-2-117M",
            max_sequence_length=128,
        )

        data_df = pd.DataFrame(
            {
                "input_ids": ["ATCGAT", "GGCCTA", "TTAACC"],
                "methylation_ids": ["010101", "001122", "112200"],
                "label": [0, 1, 0],
            }
        )

        mock_model_instance = MagicMock()
        mock_model_instance.tokenizer = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_result = MagicMock()
        mock_result.predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        mock_result.label_ids = np.array([0, 1, 0])
        mock_model_instance.predict.return_value = mock_result
        mock_epigen_class.return_value = mock_model_instance

        mock_dataset = MagicMock()
        mock_supervised_dataset.return_value = mock_dataset

        predictions, labels, used_lengths = experiment._progressive_predict(
            data_df=data_df,
            checkpoint_path="/tmp/fake_checkpoint/model.safetensors",
            split_name="train",
        )

        self.assertEqual(predictions.shape, (3, 2))
        self.assertEqual(labels.tolist(), [0, 1, 0])
        self.assertEqual(used_lengths, [experiment.max_sequence_length] * 3)

        mock_epigen_class.assert_called_once_with(
            fine_tuned_model_path="/tmp/fake_checkpoint/model.safetensors",
            max_sequence_length=experiment.max_sequence_length,
            use_cpg_methylation=experiment.use_cpg_methylation,
            use_m6a_methylation=experiment.use_m6a_methylation,
            foundation_model_huggingface=experiment.foundation_model_huggingface,
            use_triton=experiment.use_triton,
        )
        mock_supervised_dataset.assert_called_once()
        mock_model_instance.model.eval.assert_called_once()
        mock_model_instance.predict.assert_called_once_with(
            mock_dataset, batch_size=None
        )

    @patch("syto.classification.experiment_wrappers.mlflow.log_metric")
    @patch.object(EpigenBERT2MLflowExperiment, "_progressive_predict")
    @patch.object(EpigenBERT2MLflowExperiment, "_get_best_checkpoint")
    @patch("syto.classification.experiment_wrappers.EpigenDnabert2")
    def test_dnabert2_train_dataset_multiclass(
        self,
        mock_epigen_class,
        mock_get_best_checkpoint,
        mock_progressive_predict,
        mock_log_metric,
    ):
        """Test EpigenBERT2 train_dataset in a multi-class setting."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2_multiclass",
            tracking_uri=self.mlflow_manager.tracking_uri,
            foundation_model_huggingface="foundationalModels/DNABERT-2-117M",
            max_sequence_length=128,
        )

        # Mock model instance used during training loop.
        mock_model_instance = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_model_instance.model.parameters.return_value = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        ]
        mock_model_instance.trainer = MagicMock()
        mock_epigen_class.return_value = mock_model_instance

        # Provide a valid fake checkpoint path for artifact logging.
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-1")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
        with open(checkpoint_path, "wb") as f:
            f.write(b"fake")
        mock_get_best_checkpoint.return_value = checkpoint_path

        # Force multiclass predictions for all splits.
        mock_progressive_predict.return_value = (
            np.array(
                [
                    [0.90, 0.05, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                    [0.10, 0.80, 0.10],
                ]
            ),
            np.array([0, 1, 2, 1]),
            [128, 128, 128, 128],
        )

        results = experiment.train_dataset(
            "sample_dataset_1",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_steps=1,
            save_steps=1,
            check_at_step=1,
            output_dir=self.temp_dir,
        )

        self.assertIsNotNone(results)
        self.assertEqual(results["dataset"], "sample_dataset_1")
        self.assertIn("metrics", results)

        for split in experiment.splits:
            self.assertIn(split, results["metrics"])
            self.assertEqual(results["metrics"][split]["num_classes"], 3)
            self.assertTrue(results["metrics"][split]["is_multiclass"])

        self.assertTrue(mock_model_instance.fine_tune.called)
        self.assertEqual(mock_progressive_predict.call_count, len(experiment.splits))

    @patch("syto.classification.experiment_wrappers.mlflow.log_metric")
    @patch.object(EpigenBERT2MLflowExperiment, "_progressive_predict")
    @patch.object(EpigenBERT2MLflowExperiment, "_get_best_checkpoint")
    @patch("syto.classification.experiment_wrappers.EpigenDnabert2")
    def test_dnabert2_train_dataset_multiclass_with_training_args(
        self,
        mock_epigen_class,
        mock_get_best_checkpoint,
        mock_progressive_predict,
        mock_log_metric,
    ):
        """Test EpigenBERT2 train_dataset in multiclass mode with explicit TrainingArguments."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2_multiclass_training_args",
            tracking_uri=self.mlflow_manager.tracking_uri,
            foundation_model_huggingface="foundationalModels/DNABERT-2-117M",
            max_sequence_length=128,
        )

        mock_model_instance = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_model_instance.model.parameters.return_value = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        ]
        mock_model_instance.trainer = MagicMock()
        mock_epigen_class.return_value = mock_model_instance

        checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-2")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
        with open(checkpoint_path, "wb") as f:
            f.write(b"fake")
        mock_get_best_checkpoint.return_value = checkpoint_path

        mock_progressive_predict.return_value = (
            np.array(
                [
                    [0.90, 0.05, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                    [0.10, 0.80, 0.10],
                ]
            ),
            np.array([0, 1, 2, 1]),
            [128, 128, 128, 128],
        )

        training_args = EpigenTrainingArguments(
            output_dir=self.temp_dir,
            run_name="epigenbert2_multiclass_args",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=3e-5,
            eval_strategy="steps",
            eval_steps=1,
            save_steps=1,
            logging_steps=1,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            overwrite_output_dir=True,
        )

        results = experiment.train_dataset(
            "sample_dataset_1",
            training_args=training_args,
            check_at_step=1,
            output_dir=self.temp_dir,
        )

        self.assertIsNotNone(results)
        self.assertEqual(results["dataset"], "sample_dataset_1")
        self.assertIn("metrics", results)

        for split in experiment.splits:
            self.assertIn(split, results["metrics"])
            self.assertEqual(results["metrics"][split]["num_classes"], 3)
            self.assertTrue(results["metrics"][split]["is_multiclass"])

        self.assertTrue(mock_model_instance.fine_tune.called)
        self.assertEqual(mock_progressive_predict.call_count, len(experiment.splits))


class TestMethylBertMlFlowExperiment(ExperimentTestBase):
    """Test the MethylBertMLflowExperiment class."""

    def test_methylbert_initialization(self):
        """Test MethylBert experiment initialization - integration test."""
        experiment = MethylBertMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_methylbert",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=150,
            foundation_model_huggingface="hanyangii/methylbert_hg19_12l",
        )

        self.assertEqual(experiment.max_sequence_length, 150)
        self.assertEqual(
            experiment.foundation_model_huggingface, "hanyangii/methylbert_hg19_12l"
        )

    def test_prepare_methylbert_list_runs_correctly(self):
        """Test _prepare_methylbert_list builds valid rows from parquet input."""
        experiment = MethylBertMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_methylbert_prepare_list",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=100,
        )

        # No dmr_name columns exist in generated sample data, so this should
        # use the fallback branch where dmr_id is set to 0.
        dmrs = pd.DataFrame({"dmr_id": [0], "dmr_name": ["dmr_0"]})
        data_list = experiment._prepare_methylbert_list(
            experiment.data_dirs["sample_dataset_1"],
            "train",
            dmrs,
            split_to_chunks=False,
        )

        self.assertIsInstance(data_list, list)
        self.assertGreater(len(data_list), 1)
        self.assertEqual(
            data_list[0], ["dna_seq", "methyl_seq", "dmr_ctype", "dmr_label", "ctype"]
        )

        # Validate first data row format.
        first_row = data_list[1]
        self.assertEqual(len(first_row), 5)
        self.assertEqual(first_row[2], 1)  # dmr_ctype is hard-coded to 1
        self.assertEqual(first_row[3], 0)  # fallback dmr_id when no dmr columns
        self.assertIn(first_row[4], [0, 1])
        self.assertGreater(len(first_row[0]), 0)  # dna_seq
        self.assertGreater(len(first_row[1]), 0)  # methyl_seq

    def test_prepare_methylbert_list_uses_dmr_names_branch(self):
        """Test _prepare_methylbert_list correctly maps dmr_id when dmr_names exists."""
        experiment = MethylBertMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_methylbert_prepare_list_dmr_names",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=100,
        )

        dataset_dir = experiment.data_dirs["sample_dataset_1"]
        train_path = os.path.join(dataset_dir, "train.parquet")

        # Overwrite train split with controlled rows that contain dmr_names.
        df = pd.DataFrame(
            {
                "read_name": ["r1", "r2"],
                "input_ids": ["ATCGATCGAT", "GCGTATATGC"],
                "methylation_ids": ["0120120120", "1201201201"],
                "dmr_names": ["['dmr_a', 'dmr_b']", "['dmr_b']"],
                "label": [1, 0],
            }
        )
        df.to_parquet(train_path)

        dmrs = pd.DataFrame(
            {
                "dmr_id": [5, 9],
                "dmr_name": ["dmr_a", "dmr_b"],
            }
        )

        data_list = experiment._prepare_methylbert_list(
            dataset_dir,
            "train",
            dmrs,
            split_to_chunks=False,
        )

        self.assertIsInstance(data_list, list)
        self.assertEqual(
            data_list[0], ["dna_seq", "methyl_seq", "dmr_ctype", "dmr_label", "ctype"]
        )
        self.assertEqual(len(data_list), 3)

        # Uses first dmr from dmr_names list-like string for mapping.
        self.assertEqual(data_list[1][3], 5)
        self.assertEqual(data_list[2][3], 9)

    def test_methylbert_with_dmr_labels(self):
        """Test MethylBert with DMR labels - integration test."""
        experiment = MethylBertMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_methylbert_dmr",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=100,  # Smaller for faster testing
        )

        # Run with DMR labels - minimal training for testing
        results = experiment.train_dataset(
            "sample_dataset_1",
            epochs=1,  # Minimal epochs
            batch_size=4,
            output_dir=self.temp_dir,  # Use temp directory for output
        )

        # Verify results
        self.assertIsNotNone(results)
        self.assertIn("dataset_name", results)
        self.assertIn("metrics", results)

        run = self.verify_experiment_was_logged(experiment)[0]

        params = run.data.params
        self.assertIn("num_grg_labels", params)
        self.assertEqual(
            params["num_grg_labels"], "1"
        )  # By default if data doesn't posses dmr indexes, assigns 0 to all


if __name__ == "__main__":
    unittest.main()
