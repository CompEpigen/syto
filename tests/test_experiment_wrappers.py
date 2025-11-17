import unittest
import os
import tempfile
import shutil
import subprocess
import time
import socket
import signal
import atexit
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
from parameterized import parameterized
import pandas as pd
import numpy as np
import mlflow
import torch
import json
import pickle
from contextlib import contextmanager
from threading import Thread
import warnings
from uuid import uuid4
warnings.filterwarnings('ignore')

# Import the experiment wrapper classes - adjust imports as needed
from methyldl.modelling.experiment_wrappers import (
    AbstractMLFlowExperiment,
    DismirMLflowExperiment, 
    EpigenBERT2MLflowExperiment,
    MethylBertMLflowExperiment,
    RestartOnPoorPerformanceCallback
)
from methyldl.data.dataset import generate_example_data
from methyldl.modelling.methylbert import default_methylbert_config

def is_singularity_container():
    """Check if running inside a Singularity container."""
    return os.path.exists('/.singularity.d') or os.environ.get('SINGULARITY_CONTAINER')


def get_mlflow_storage_path():
    """
    Get appropriate storage path for MLflow artifacts.
    In Singularity containers, use mounted external storage.
    """
    if is_singularity_container():
        # Check common mount points for external storage in Singularity
        potential_paths = [
            '/external_storage/mlflow_tests',
            '/mnt/external/mlflow_tests',
            '/scratch/mlflow_tests',
            '/tmp/mlflow_tests'
        ]
        for path in potential_paths:
            parent = Path(path).parent
            if parent.exists() and os.access(parent, os.W_OK):
                return path
    
    # Default to system temp directory
    return tempfile.mkdtemp(prefix='mlflow_test_')


def find_free_port():
    """Find a free port for MLflow server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class MLflowServerManager:
    """Manages MLflow server lifecycle for tests."""
    
    def __init__(self, backend_store_uri=None, artifact_root=None):
        self.backend_store_uri = backend_store_uri or get_mlflow_storage_path()
        self.artifact_root = artifact_root or os.path.join(self.backend_store_uri, 'artifacts')
        self.port = find_free_port()
        self.server_process = None
        self.tracking_uri = f"http://127.0.0.1:{self.port}"
        
        # Ensure directories exist
        os.makedirs(self.backend_store_uri, exist_ok=True)
        os.makedirs(self.artifact_root, exist_ok=True)
        
    def start(self, timeout=30):
        """Start MLflow server."""
        cmd = [
            'mlflow', 'server',
            '--backend-store-uri', f'file://{self.backend_store_uri}',
            '--default-artifact-root', f'file://{self.artifact_root}',
            '--host', '127.0.0.1',
            '--port', str(self.port),
            '--workers', '1'
        ]
        
        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, 'MLFLOW_TRACKING_URI': self.tracking_uri}
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
            response = requests.get(f'{self.tracking_uri}/health', timeout=1)
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
        if os.path.exists(self.backend_store_uri) and '/mlflow_test_' in self.backend_store_uri:
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
        self.temp_dir = tempfile.mkdtemp(prefix='exp_test_')
        self.data_path = os.path.join(self.temp_dir, 'data')
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
        self.assertIsNotNone(current_experiment, 
                            f"Experiment '{experiment.experiment_name}' not found")
        
        experiment_id = current_experiment.experiment_id
        
        # Search for runs
        client = mlflow.tracking.MlflowClient(tracking_uri=self.mlflow_manager.tracking_uri)
        runs = client.search_runs(experiment_ids=[experiment_id])
        
        # Verify runs exist
        self.assertGreaterEqual(len(runs), min_runs,
                               f"Expected at least {min_runs} runs, found {len(runs)}")
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
        datasets = ['sample_dataset_1', 'sample_dataset_2', 'sample_dataset_3']
        
        for dataset in datasets:
            dataset_path = os.path.join(self.data_path, dataset)
            os.makedirs(dataset_path)
            
            # Create sample parquet files for each split
            for split in ['train', 'valid', 'test', 'rest']:
                df = self._create_sample_dataframe(split)
                df.to_parquet(os.path.join(dataset_path, f'{split}.parquet'))
    
    def _create_sample_dataframe(self, split, num_samples=10):
        """Create a sample dataframe for testing."""
        np.random.seed(42)  # For reproducibility
        
        data = []
        for i in range(num_samples):
            seq_len = np.random.randint(100, 500)
            dna_seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], seq_len))
            meth_seq = ''.join(np.random.choice(['0', '1', '2'], seq_len))
            
            # Add some structure for MethylBert (DMRs)
            dmr_status = ''.join(np.random.choice(['0', '1'], seq_len))
            name = uuid4()
            data.append({
                'read_name': str(name),
                'input_ids': dna_seq,
                'methylation_ids': meth_seq,
                'dmr_status': dmr_status,
                'label': np.random.randint(0, 2),
                'sum_abs_areastat': np.random.uniform(3000, 6000),
                'chromosome': "chr1",
                'original_file':"some_sam_file"
            })
        
        return pd.DataFrame(data)


class TestRestartOnPoorPerformanceCallback(unittest.TestCase):
    """Test the RestartOnPoorPerformanceCallback class."""
    
    def test_callback_initialization(self):
        """Test callback initialization with different parameters."""
        callback = RestartOnPoorPerformanceCallback(
            eval_loss_threshold=0.7,
            check_at_step=100,
            max_retries=3
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
            eval_loss_threshold=0.5,
            check_at_step=10,
            max_retries=3
        )
        
        # Mock trainer state with poor performance
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.log_history = [{'eval_loss': 0.8, 'step': 10}]
        
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
            eval_loss_threshold=0.5,
            check_at_step=10,
            max_retries=3
        )
        
        # Mock trainer state with good performance
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.log_history = [{'eval_loss': 0.3, 'step': 10}]
        
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
            eval_loss_threshold=0.5,
            check_at_step=10,
            max_retries=2
        )
        
        # Simulate multiple retries
        callback.retry_count = 2  # Already at max
        
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.log_history = [{'eval_loss': 0.8, 'step': 10}]
        
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
            eval_loss_threshold=0.5,
            check_at_step=10,
            max_retries=3
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
            max_sequence_length=1000
        )
        
        self.assertEqual(experiment.max_sequence_length, 1000)
        self.assertEqual(len(experiment.data_dirs), 3)  # sample_dataset_1, sample_dataset_2, sample_dataset_3
        self.assertIn('sample_dataset_1', experiment.data_dirs)
        self.assertIn('sample_dataset_2', experiment.data_dirs)
    
    def test_initialization_without_experiment_name_raises_error(self):
        """Test that initialization without experiment name raises error."""
        with self.assertRaises(ValueError) as context:
            AbstractMLFlowExperiment(
                data_path=self.data_path,
                experiment_name=None,
                tracking_uri=self.mlflow_manager.tracking_uri
            )
        
        self.assertIn("Experiment name must be provided", str(context.exception))
    
    def test_initialization_with_invalid_path_raises_error(self):
        """Test initialization with non-existent path raises error."""
        with self.assertRaises(ValueError) as context:
            AbstractMLFlowExperiment(
                data_path="/non/existent/path",
                experiment_name="test_experiment",
                tracking_uri=self.mlflow_manager.tracking_uri
            )
        
        self.assertIn("does not exist", str(context.exception))
    
    def test_find_data_directories(self):
        """Test finding data directories with required files."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri
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
            tracking_uri=self.mlflow_manager.tracking_uri
        )
        
        # Calculate stats for a dataset
        train_path = os.path.join(self.data_path, 'sample_dataset_1', 'train.parquet')
        stats = experiment._calculate_data_stats(train_path)
        
        self.assertIn('num_samples', stats)
        self.assertIn('num_positive', stats)
        self.assertIn('num_negative', stats)
        self.assertIn('positive_ratio', stats)
        self.assertIn('avg_sequence_length', stats)
        
        self.assertEqual(stats['num_samples'], 10)
        self.assertTrue(0 <= stats['positive_ratio'] <= 1)
    
    def test_calculate_metrics(self):
        """Test metric calculation with various scenarios."""
        experiment = AbstractMLFlowExperiment(
            data_path=self.data_path,
            experiment_name="test_experiment",
            tracking_uri=self.mlflow_manager.tracking_uri
        )
        
        # Test with normal values
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        y_pred_binary = np.array([0, 1, 0, 1, 1])
        
        metrics = experiment._calculate_metrics(y_true, y_pred_proba, y_pred_binary)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('mcc', metrics)
        
        # Test with NaN values
        y_true_nan = np.array([0, 1, np.nan, 1, 1])
        y_pred_proba_nan = np.array([0.1, 0.9, 0.5, np.nan, 0.7])
        
        metrics_nan = experiment._calculate_metrics(y_true_nan, y_pred_proba_nan, y_pred_binary)
        self.assertIn('nan_percentage', metrics_nan)
        self.assertIn('num_nans_predictions_proba', metrics_nan)


class TestDismirMLFlowExperiment(ExperimentTestBase):
    """Test the DismirMLFlowExperiment class."""
    
    @patch('methyldl.modelling.experiment_wrappers')
    def test_dismir_initialization(self, mock_dismir_class):
        """Test Dismir experiment initialization."""
        experiment = DismirMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dismir",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=500,
            model_flavor="lstm"
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
            model_flavor="lstm"
        )
        
        # Run training
        results = experiment.train_dataset('sample_dataset_1', epochs=2, batch_size=32,
                                           output_dir="test_container_tmp/dismir_output")
        
        # Verify results
        self.assertIn('dataset', results)
        self.assertIn('metrics', results)
        self.assertIn('data_stats', results)
        self.assertEqual(results['dataset'], 'sample_dataset_1')




class TestEpigenBERT2MLflowExperiment(ExperimentTestBase):
    """Test the EpigenBERT2MLflowExperiment class."""
    
    def test_dnabert2_initialization(self):
        """Test DNABert2 experiment initialization - integration test."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=512,
            foundation_model_huggingface="zhihan1996/DNABERT-2-117M"
        )
        
        self.assertEqual(experiment.max_sequence_length, 512)
        self.assertEqual(experiment.foundation_model_huggingface, "zhihan1996/DNABERT-2-117M")
    
    def test_dnabert2_with_restart_callback(self):
        """Test DNABert2 training with restart callback - integration test."""
        experiment = EpigenBERT2MLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_dnabert2_restart",
            tracking_uri=self.mlflow_manager.tracking_uri,
            foundation_model_huggingface = "foundationalModels/DNABERT-2-117M",
            max_sequence_length=128  # Smaller for faster testing
        )
        
        # Run with restart callback - use minimal epochs for speed
        results = experiment.train_dataset(
            'sample_dataset_1',  # Use your test dataset name
            num_train_epochs=1,  # Minimal epochs for testing
            per_device_train_batch_size=4,  # Small batch size for testing
            per_device_eval_batch_size=4, 
            eval_loss_threshold=0.7,
            check_at_step=10,  # Lower step count for testing
            output_dir="test_container_tmp/dnabert2_output"
        )
        
        # Verify results structure
        self.assertIsNotNone(results)
        self.assertIn('dataset', results)
        self.assertIn('metrics', results)
        _ = self.verify_experiment_was_logged(experiment)
        

class TestMethylBertMlFlowExperiment(ExperimentTestBase):
    """Test the MethylBertMLflowExperiment class."""
    
    def test_methylbert_initialization(self):
        """Test MethylBert experiment initialization - integration test."""
        experiment = MethylBertMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_methylbert",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=150,
            foundation_model_huggingface="hanyangii/methylbert_hg19_12l"  
        )
        
        self.assertEqual(experiment.max_sequence_length, 150)
        self.assertEqual(experiment.foundation_model_huggingface, "hanyangii/methylbert_hg19_12l")
    
    def test_methylbert_with_dmr_labels(self):
        """Test MethylBert with DMR labels - integration test."""
        experiment = MethylBertMLflowExperiment(
            data_path=self.data_path,
            experiment_name="test_methylbert_dmr",
            tracking_uri=self.mlflow_manager.tracking_uri,
            max_sequence_length=100  # Smaller for faster testing
        )
        
        # Run with DMR labels - minimal training for testing
        results = experiment.train_dataset(
            'sample_dataset_1',
            epochs=1,  # Minimal epochs
            batch_size=4,
            output_dir="test_container_tmp/methylbert_output"
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('dataset_name', results)
        self.assertIn('metrics', results)
        
        run = self.verify_experiment_was_logged(experiment)[0]
        
        params = run.data.params
        self.assertIn('num_dmr_labels', params)
        self.assertEqual(params['num_dmr_labels'], '1') # By default if data doesn't posses dmr indexes, assigns 0 to all


if __name__ == '__main__':
    unittest.main()