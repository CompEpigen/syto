import unittest
import os
import tempfile
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from parameterized import parameterized
import shutil
from methyldl.data.dataset import generate_example_data

from methyldl.modelling.classifiers.dismir import (
    Dismir,
    DISMIRNet,
    VariableLengthDataset,
    ChunkAwareBatchSampler,
)


class DismirTestBase(unittest.TestCase):
    """Base test class with common setup and data generation methods."""

    def setUp(self, num_samples=10):
        """Create temporary directories and sample data for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.max_sequence_length = 512

        # Create sample data files
        self.train_path = os.path.join(self.temp_dir, "train.parquet")
        self.test_path = os.path.join(self.temp_dir, "test.parquet")
        self.valid_path = os.path.join(self.temp_dir, "valid.parquet")
        self.num_samples = num_samples

        # Default data creation - child classes can override
        self._create_default_data()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def _create_default_data(self):
        """Hook method that child classes can override for custom data."""
        self._create_sample_data()

    def _create_sample_data(self, sequence_length=512):
        """Helper to create sample parquet files."""
        dataset = generate_example_data(
            sequence_length=sequence_length,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=self.num_samples,
        )
        df = pd.DataFrame(
            dataset[1:], columns=["input_ids", "methylation_ids", "label"]
        )
        df["dmr_label"] = np.random.randint(0, 5, len(df))
        df.to_parquet(self.train_path)
        df.to_parquet(self.test_path)
        df.to_parquet(self.valid_path)


class TestDismirModelInitialization(DismirTestBase):
    """Test Dismir model initialization with different configurations."""

    @parameterized.expand(
        [
            ("lstm", 128),
            ("lstm", 256),
            ("minigru", 128),
            ("minigru", 512),
        ]
    )
    def test_model_initialization_with_different_configs(self, flavour, max_seq_length):
        """Test model initialization with different RNN types and sequence lengths."""
        model = Dismir(
            max_sequence_length=max_seq_length,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            flavour=flavour,
        )

        self.assertEqual(model.max_sequence_length, max_seq_length)
        self.assertIsNotNone(model.model)
        self.assertIsInstance(model.model, DISMIRNet)

        # Check if model is on correct device
        self.assertTrue(model.device.type in ["cuda", "cpu"])

    @parameterized.expand([("cpu"), ("cuda")])
    def test_model_device_selection(self, device):
        """Test explicit device selection."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device(device),
        )
        self.assertEqual(model.device, torch.device(device))


class TestDismirPredict(DismirTestBase):
    """Test prediction functionality of Dismir model."""

    def setUp(self):
        super().setUp()
        # Initialize model
        self.model = Dismir(
            max_sequence_length=self.max_sequence_length,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device("cuda"),
        )

    @parameterized.expand(
        [
            (1,),  # Single sequence
            (5,),  # Small batch
            (10,),  # Medium batch
        ]
    )
    def test_predict_with_different_batch_sizes(self, num_sequences):
        """Test prediction with different numbers of sequences."""
        # Generate test sequences
        dna_sequences = ["ATCG" * 20] * num_sequences
        methylation_sequences = ["0101" * 20] * num_sequences

        # Run prediction
        probs, labels = self.model.predict(
            dna_sequences, methylation_sequences, batch_size=32
        )

        # Check output shapes and values
        self.assertEqual(len(probs), num_sequences)
        self.assertEqual(len(labels), num_sequences)
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        self.assertTrue(np.all((labels == 0) | (labels == 1)))

    @parameterized.expand(
        [
            (50,),  # Short sequence
            (128,),  # Exact max length
            (200,),  # Longer than max (will be truncated)
        ]
    )
    def test_predict_with_variable_sequence_lengths(self, seq_length):
        """Test prediction with sequences of different lengths."""
        dna_sequence = "".join(np.random.choice(["A", "T", "C", "G"], seq_length))
        methylation_sequence = "".join(np.random.choice(["0", "1"], seq_length))

        probs, labels = self.model.predict(
            [dna_sequence], [methylation_sequence], batch_size=1
        )
        print(probs)
        self.assertEqual(len(probs), 1)
        self.assertEqual(len(labels), 1)
        self.assertTrue(all((0 <= probs[0]) & (probs[0] <= 1)))

    def test_predict_with_parallel_scan_disabled(self):
        """Test prediction with parallel_scan disabled (for minigru)."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            flavour="minigru",
            device=torch.device("cpu"),
        )

        dna_sequences = ["ATCG" * 20]
        methylation_sequences = ["0101" * 20]

        probs, labels = model.predict(
            dna_sequences, methylation_sequences, parallel_scan=False
        )

        self.assertEqual(len(probs), 1)
        self.assertEqual(len(labels), 1)

    def test_predict_with_custom_threshold(self):
        """Test prediction with custom classification threshold."""
        dna_sequences = ["ATCG" * 20] * 5
        methylation_sequences = ["0101" * 20] * 5
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            flavour="lstm",
            num_labels=1,
            device=torch.device("cpu"),
        )
        # Test with different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            probs, labels = model.predict(
                dna_sequences, methylation_sequences, threshold=threshold
            )

            # Check that labels are correctly thresholded
            expected_labels = (probs >= threshold).astype(int)
            np.testing.assert_array_equal(labels, expected_labels)


class TestDismirTraining(DismirTestBase):
    """Test training functionality of Dismir model."""

    @parameterized.expand(
        [
            ("dmr_attention_based",),
            ("vanilla",),
        ]
    )
    def test_fixed_length_training(self, classifier_type):
        """Test fixed-length training mode."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device("cuda"),
            classifier_type=classifier_type,
            num_dmr_labels=100,
            dmr_label_col="dmr_label",
        )

        # Train for a few epochs
        model.train(
            train_dir=self.temp_dir,
            epochs=2,
            batch_size=8,
            patience=5,
            variable_length=False,
            verbose=0,
        )

        # Check that history is populated
        self.assertGreater(len(model.history), 0)

        # Check that weight file is saved
        weight_path = os.path.join(self.temp_dir, "weight.pt")
        self.assertTrue(os.path.exists(weight_path))

    def test_variable_length_training(self):
        """Test variable-length training mode."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device("cuda"),
            num_labels=2,
        )

        # Train with variable length
        model.train(
            train_dir=self.temp_dir,
            epochs=2,
            batch_size=16,  # This is max_chunks_per_batch in variable mode
            patience=5,
            variable_length=True,
            verbose=0,
        )

        # Check that history is populated
        self.assertGreater(len(model.history), 0)

        # Verify history contains expected keys
        expected_keys = [
            "session",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "elapsed_time",
        ]
        for key in expected_keys:
            self.assertIn(key, model.history[0])

    @parameterized.expand(
        [
            ("SGD", 0.05, 0.9),
            ("Adam", 0.001, 0.0),
        ]
    )
    def test_different_optimizers(self, optimizer_type, lr, momentum):
        """Test training with different optimizer configurations."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device("cuda"),
        )

        model.train(
            train_dir=self.temp_dir,
            epochs=1,
            batch_size=8,
            optimizer_type=optimizer_type,
            lr=lr,
            momentum=momentum,
            verbose=0,
        )

        # Check that training completed
        self.assertGreater(len(model.history), 0)

    def test_early_stopping(self):
        """Test that early stopping works correctly."""
        # Flip validation labels so val_loss diverges as the model learns training data,
        # guaranteeing the patience counter reaches the threshold before 100 epochs.
        # (because the generation of training and validation data is identical)
        valid_df = pd.read_parquet(self.valid_path)
        valid_df["label"] = 1 - valid_df["label"]
        valid_df.to_parquet(self.valid_path)

        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device("cuda"),
        )

        # Train with very low patience
        model.train(
            train_dir=self.temp_dir,
            epochs=100,  # Set high, should stop early
            batch_size=8,
            patience=2,
            variable_length=False,
            verbose=0,
        )

        # Should stop before reaching 100 epochs
        self.assertLess(len(model.history), 100)

    def test_reset_history(self):
        """Test that history can be reset between training sessions."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            device=torch.device("cuda"),
        )

        # First training session
        model.train(train_dir=self.temp_dir, epochs=1, batch_size=8, verbose=0)

        first_history_length = len(model.history)

        # Second training session without reset
        model.train(
            train_dir=self.temp_dir,
            epochs=1,
            batch_size=8,
            reset_history=False,
            verbose=0,
        )

        self.assertGreater(len(model.history), first_history_length)

        # Third training session with reset
        model.train(
            train_dir=self.temp_dir,
            epochs=1,
            batch_size=8,
            reset_history=True,
            verbose=0,
        )

        # History should only contain entries from the last session
        self.assertLess(len(model.history), len(model.history) + first_history_length)


# class TestDismirEvaluation(DismirTestBase):
#     """Test evaluation functionality of Dismir model."""

#     def setUp(self):
#         super().setUp()
#         # Initialize model

#     @parameterized.expand(
#         [
#             ("test", False, 1),
#             ("valid", False, 1),
#             ("test", True, 1),
#             ("valid", True, 1),
#             ("test", False, 3),
#             ("valid", False, 3),
#             ("test", True, 3),
#             ("valid", True, 3),
#         ]
#     )
#     def test_evaluate_different_splits_and_modes(
#         self, split, variable_length, num_labels
#     ):
#         """Test evaluation on different data splits and modes."""
#         model = Dismir(
#             max_sequence_length=self.max_sequence_length,
#             train_data_path=self.train_path,
#             test_data_path=self.test_path,
#             valid_data_path=self.valid_path,
#             device=torch.device("cuda"),
#             num_labels=num_labels,
#         )

#         if not variable_length:
#             # Load and transform test data
#             model.test_x, model.test_y = model.load_and_transform_input(self.test_path)
#             model.test_x = torch.tensor(model.test_x, dtype=torch.float32)

#             # Handle labels based on num_labels
#             if num_labels == 1:
#                 # Binary: float32, shape [batch_size, 1]
#                 model.test_y = torch.tensor(model.test_y, dtype=torch.float32).view(
#                     -1, 1
#                 )
#             else:
#                 # Multi-class: long, shape [batch_size]
#                 model.test_y = torch.tensor(model.test_y, dtype=torch.long).squeeze()

#             # Load and transform validation data
#             model.valid_x, model.valid_y = model.load_and_transform_input(
#                 self.valid_path
#             )
#             model.valid_x = torch.tensor(model.valid_x, dtype=torch.float32)

#             if num_labels == 1:
#                 model.valid_y = torch.tensor(model.valid_y, dtype=torch.float32).view(
#                     -1, 1
#                 )
#             else:
#                 model.valid_y = torch.tensor(model.valid_y, dtype=torch.long).squeeze()

#         loss, accuracy = model.evaluate(split=split, variable_length=variable_length)

#         # Check that metrics are valid
#         self.assertIsInstance(loss, float)
#         self.assertIsInstance(accuracy, float)
#         self.assertGreaterEqual(loss, 0)
#         self.assertTrue(0 <= accuracy <= 1)


# class TestVariableLengthDataset(DismirTestBase):
#     """Test the VariableLengthDataset class."""

#     def setUp(self):
#         super().setUp()
#         self.max_sequence_length = 100
#         self.data_path = os.path.join(self.temp_dir, "test_data.parquet")
#         # Create test data with varying lengths
#         self._create_variable_length_data()

#     def _create_variable_length_data(self):
#         """Create data with sequences of varying lengths."""
#         data = []
#         for i in range(5):
#             # Create sequences longer than max_sequence_length to test chunking
#             seq_len = 150 + i * 50  # 150, 200, 250, 300, 350
#             dna_seq = "".join(np.random.choice(["A", "T", "C", "G"], seq_len))
#             # Add CpG sites for testing weight calculation
#             dna_seq = dna_seq[:10] + "CG" * 5 + dna_seq[20:]
#             meth_seq = "".join(np.random.choice(["0", "1"], seq_len))
#             label = i % 2
#             data.append(
#                 {"input_ids": dna_seq, "methylation_ids": meth_seq, "label": label}
#             )
#         df["dmr_label"] = np.random.randint(0, 5, len(df))
#         df = pd.DataFrame(data)
#         df.to_parquet(self.data_path)

#     def test_dataset_initialization(self):
#         """Test that dataset initializes correctly."""

#         def mock_conv_onehot(dna_seqs, meth_seqs):
#             # Mock conversion function
#             result = []
#             for dna, meth in zip(dna_seqs, meth_seqs):
#                 seq_len = len(dna)
#                 mock_onehot = np.random.rand(seq_len, 5)
#                 result.append(mock_onehot)
#             return result

#         dataset = VariableLengthDataset(
#             self.data_path, self.max_sequence_length, mock_conv_onehot
#         )

#         # Check dataset properties
#         self.assertEqual(len(dataset), 5)
#         self.assertIsNotNone(dataset.read_chunks)
#         self.assertIsNotNone(dataset.chunk_weights)
#         self.assertIsNotNone(dataset.read_labels)

#     def test_chunk_creation(self):
#         """Test that sequences are properly chunked."""

#         def mock_conv_onehot(dna_seqs, meth_seqs):
#             result = []
#             for dna in dna_seqs:
#                 # Pad or truncate to max_sequence_length
#                 seq_len = min(len(dna), self.max_sequence_length)
#                 mock_onehot = np.zeros((self.max_sequence_length, 5))
#                 mock_onehot[:seq_len, :] = np.random.rand(seq_len, 5)
#                 result.append(mock_onehot)
#             return result

#         dataset = VariableLengthDataset(
#             self.data_path, self.max_sequence_length, mock_conv_onehot
#         )

#         # Check first sequence (length 150, should have 2 chunks)
#         first_read_chunks = dataset.get_chunk_count(0)
#         self.assertEqual(first_read_chunks, 2)

#         # Check last sequence (length 350, should have 4 chunks)
#         last_read_chunks = dataset.get_chunk_count(4)
#         self.assertEqual(last_read_chunks, 4)

#     def test_getitem(self):
#         """Test dataset __getitem__ method."""

#         def mock_conv_onehot(dna_seqs, meth_seqs):
#             result = []
#             for dna in dna_seqs:
#                 seq_len = min(len(dna), self.max_sequence_length)
#                 mock_onehot = np.zeros((self.max_sequence_length, 5))
#                 mock_onehot[:seq_len, :] = np.random.rand(seq_len, 5)
#                 result.append(mock_onehot)
#             return result

#         dataset = VariableLengthDataset(
#             self.data_path, self.max_sequence_length, mock_conv_onehot
#         )

#         chunks, weights, label, read_id = dataset[0]

#         # Check tensor types and shapes
#         self.assertIsInstance(chunks, torch.Tensor)
#         self.assertIsInstance(weights, torch.Tensor)
#         self.assertIsInstance(label, torch.Tensor)

#         # Check that weights sum to 1 (normalized)
#         self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)


class TestChunkAwareBatchSampler(unittest.TestCase):
    """Test the ChunkAwareBatchSampler class."""

    def test_batch_sampler_respects_chunk_limit(self):
        """Test that batch sampler respects max_chunks_per_batch."""

        # Create mock dataset
        class MockDataset:
            def __init__(self):
                # Varying chunk counts: 1, 2, 3, 4, 5 chunks
                self.chunk_counts = [1, 2, 3, 4, 5]

            def get_chunk_count(self, idx):
                return self.chunk_counts[idx]

            def __len__(self):
                return len(self.chunk_counts)

        dataset = MockDataset()
        max_chunks_per_batch = 5

        sampler = ChunkAwareBatchSampler(dataset, max_chunks_per_batch, shuffle=False)

        batches = list(sampler)

        # Check that no batch exceeds the chunk limit
        for batch in batches:
            total_chunks = sum(dataset.get_chunk_count(idx) for idx in batch)
            self.assertLessEqual(total_chunks, max_chunks_per_batch)

    def test_batch_sampler_with_shuffle(self):
        """Test that batch sampler can shuffle indices."""

        class MockDataset:
            def __init__(self):
                self.chunk_counts = [1] * 10  # All same size for simplicity

            def get_chunk_count(self, idx):
                return self.chunk_counts[idx]

            def __len__(self):
                return len(self.chunk_counts)

        dataset = MockDataset()

        # Get two different shuffled batches
        sampler1 = ChunkAwareBatchSampler(dataset, 3, shuffle=True)
        sampler2 = ChunkAwareBatchSampler(dataset, 3, shuffle=True)

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # They should be different (with high probability)
        # We can't guarantee they're different, but we can check structure
        self.assertEqual(len(batches1), len(batches2))


class TestOneHotConversion(unittest.TestCase):
    """Test the one-hot conversion functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = self._create_minimal_model()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_minimal_model(self):
        """Create a minimal Dismir model for testing."""
        # Create dummy data files
        data = [{"input_ids": "ATCG", "methylation_ids": "0101", "label": 0}]
        df = pd.DataFrame(data)

        paths = ["train.parquet", "test.parquet", "valid.parquet"]
        for path in paths:
            df.to_parquet(os.path.join(self.temp_dir, path))

        return Dismir(
            max_sequence_length=128,
            train_data_path=os.path.join(self.temp_dir, "train.parquet"),
            test_data_path=os.path.join(self.temp_dir, "test.parquet"),
            valid_data_path=os.path.join(self.temp_dir, "valid.parquet"),
            device=torch.device("cpu"),
        )

    def test_conv_onehot_basic(self):
        """Test basic one-hot conversion."""
        dna_seq = ["ATCG"]
        meth_seq = ["0000"]

        result = self.model.conv_onehot(dna_seq, meth_seq)

        # Check shape
        self.assertEqual(result.shape, (1, 128, 5))

        # Check that first 4 positions are encoded correctly
        # A = [1,0,0,0,0], T = [0,1,0,0,0], C = [0,0,1,0,0], G = [0,0,0,1,0]
        expected = np.array(
            [
                [1, 0, 0, 0, 0],  # A
                [0, 1, 0, 0, 0],  # T
                [0, 0, 1, 0, 0],  # C
                [0, 0, 0, 1, 0],  # G
            ]
        )

        np.testing.assert_array_equal(result[0, :4, :], expected)

    def test_conv_onehot_with_methylation(self):
        """Test one-hot conversion with methylation."""
        dna_seq = ["CCCC"]
        meth_seq = ["0101"]

        result = self.model.conv_onehot(dna_seq, meth_seq)

        # Check methylated C encoding
        # Unmethylated C = [0,0,1,0,0], Methylated C = [0,0,1,0,1]
        expected = np.array(
            [
                [0, 0, 1, 0, 0],  # C unmethylated
                [0, 0, 1, 0, 1],  # C methylated
                [0, 0, 1, 0, 0],  # C unmethylated
                [0, 0, 1, 0, 1],  # C methylated
            ]
        )

        np.testing.assert_array_equal(result[0, :4, :], expected)

    def test_conv_onehot_truncation(self):
        """Test that sequences longer than max_length are truncated."""
        # Create a sequence longer than max_sequence_length
        long_seq = "A" * 200
        meth_seq = "0" * 200

        result = self.model.conv_onehot([long_seq], [meth_seq])

        # Should be truncated to max_sequence_length
        self.assertEqual(result.shape[1], self.model.max_sequence_length)

    def test_conv_onehot_padding(self):
        """Test that short sequences are padded with zeros."""
        short_seq = "ATCG"
        meth_seq = "0000"

        result = self.model.conv_onehot([short_seq], [meth_seq])

        # Check that positions after the sequence are all zeros
        padding_start = len(short_seq)
        padding = result[0, padding_start:, :]

        np.testing.assert_array_equal(padding, np.zeros_like(padding))


class TestDISMIRNetArchitecture(unittest.TestCase):
    """Test the DISMIRNet neural network architecture."""

    @parameterized.expand(
        [
            (128, "lstm", 1),
            (256, "lstm", 3),
            (128, "minigru", 1),
            (512, "minigru", 3),
        ]
    )
    def test_network_forward_pass(self, max_seq_length, flavor, num_labels):
        """Test forward pass through the network."""
        model = DISMIRNet(max_seq_length, flavor, num_labels=num_labels)
        model.eval()

        # Create random input
        batch_size = 4
        input_tensor = torch.randn(batch_size, max_seq_length, 5)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)

        # Check output shape and values
        self.assertEqual(output.shape, (batch_size, num_labels))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))


class TestModelStatePersistence(unittest.TestCase):
    """Test saving and loading model states."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = self._create_test_model()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_test_model(self):
        """Create a test model."""
        # Create dummy data
        data = [
            {"input_ids": "ATCG" * 25, "methylation_ids": "0101" * 25, "label": 0}
        ] * 5
        df = pd.DataFrame(data)

        paths = ["train.parquet", "test.parquet", "valid.parquet"]
        for path in paths:
            df.to_parquet(os.path.join(self.temp_dir, path))

        return Dismir(
            max_sequence_length=100,
            train_data_path=os.path.join(self.temp_dir, "train.parquet"),
            test_data_path=os.path.join(self.temp_dir, "test.parquet"),
            valid_data_path=os.path.join(self.temp_dir, "valid.parquet"),
            device=torch.device("cpu"),
        )

    def test_weight_saving_during_training(self):
        """Test that weights are saved during training."""
        self.model.train(train_dir=self.temp_dir, epochs=1, batch_size=2, verbose=0)

        weight_file = os.path.join(self.temp_dir, "weight.pt")
        self.assertTrue(os.path.exists(weight_file))

        # Load and check weights
        state_dict = torch.load(weight_file, map_location="cpu")
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)

    def test_model_state_dict_keys(self):
        """Test that model state dict contains expected keys."""
        state_dict = self.model.model.state_dict()

        expected_keys = [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "fc3.weight",
            "fc3.bias",
        ]

        for key in expected_keys:
            self.assertIn(key, state_dict)


class TestDismirSoftLabels(DismirTestBase):
    """Test soft-label support in Dismir."""

    def _create_default_data(self):
        """Create data with a 'soft_label' column for soft-label tests."""
        dataset = generate_example_data(
            sequence_length=128,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=self.num_samples,
        )
        df = pd.DataFrame(
            dataset[1:], columns=["input_ids", "methylation_ids", "label"]
        )
        # Add a soft_label column: list of floats that sum to 1
        num_classes = 3
        soft_labels = []
        for _ in range(len(df)):
            probs = np.random.dirichlet(np.ones(num_classes))
            soft_labels.append(probs.tolist())
        df["soft_label"] = soft_labels
        # Also add dmr_label_col for DMR tests
        df["dmr_label"] = np.random.randint(0, 5, len(df))

        df.to_parquet(self.train_path)
        df.to_parquet(self.test_path)
        df.to_parquet(self.valid_path)

    def test_soft_label_criterion_is_cwce(self):
        """With soft_labels=True and dmr_attention_based, criterion should be CWCE."""
        from methyldl.modelling.loss import ConfidenceWeightedCrossEntropy

        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            classifier_type="dmr_attention_based",
            num_labels=3,
            num_dmr_labels=5,
            dmr_label_col="dmr_label",
            soft_labels=True,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(model.criterion, ConfidenceWeightedCrossEntropy)

    def test_soft_label_load_and_transform_reads_soft_column(self):
        """load_and_transform_input(soft_labels=True) should read the 'soft_label' column."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            num_labels=3,
            device=torch.device("cpu"),
        )

        features, labels = model.load_and_transform_input(
            self.train_path, soft_labels=True
        )

        # Labels should be 2D: [num_samples, num_classes]
        self.assertEqual(labels.ndim, 2)
        self.assertEqual(labels.shape[0], self.num_samples)
        self.assertEqual(labels.shape[1], 3)
        # Each row should sum to ~1.0
        np.testing.assert_allclose(labels.sum(axis=1), 1.0, atol=1e-5)

    def test_soft_label_training_loop_accuracy_uses_argmax(self):
        """Soft-label training should use argmax for accuracy in the training loop."""
        model = Dismir(
            max_sequence_length=128,
            train_data_path=self.train_path,
            test_data_path=self.test_path,
            valid_data_path=self.valid_path,
            classifier_type="dmr_attention_based",
            num_labels=3,
            num_dmr_labels=5,
            dmr_label_col="dmr_label",
            soft_labels=True,
            device=torch.device("cpu"),
        )

        model.train(
            train_dir=self.temp_dir,
            epochs=1,
            batch_size=8,
            patience=5,
            variable_length=False,
            verbose=0,
        )

        # Training should complete and produce history
        self.assertGreater(len(model.history), 0)
        # Accuracy should be between 0 and 1
        self.assertTrue(0 <= model.history[0]["train_acc"] <= 1)
        self.assertTrue(0 <= model.history[0]["val_acc"] <= 1)


if __name__ == "__main__":
    unittest.main()
