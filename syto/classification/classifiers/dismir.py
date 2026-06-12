import os
import os.path
from datetime import datetime
import time
from collections import defaultdict
from typing import Union
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from syto.classification.training_progress import (
    TrainingProgressTracker,
    is_in_notebook,
)
from syto.classification.evaluation import compute_metrics, compute_metrics_soft_labels
from syto.classification.classification_heads import (
    GRGAttentionClassificationHead,
)
from syto.classification.loss import ConfidenceWeightedCrossEntropy
from syto.classification.classifiers.minirnns.minRNNs import BiMinGRU
from syto.classification.classifiers.abstract_read_classifier import (
    AbstractReadClassifier,
)
from syto.classification.mlflow_tracking import mlflow_tracked_fit

from syto.data.dataset import resolve_column

_module_logger = logging.getLogger(__name__)


class DISMIRConfig:
    """Simple config class for GRGAttentionClassificationHead compatibility."""

    def __init__(
        self,
        hidden_size,
        num_labels,
        num_grg_labels,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
        layer_norm_eps=1e-12,
    ):
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_grg_labels = num_grg_labels
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps


class DISMIRNet(nn.Module):
    """
    PyTorch model mirroring the structure of the original Keras model from the paper.

    SHARED ENCODER:
    1. Conv1D -> ReLU -> MaxPool
    2. Dropout
    3. Bidirectional LSTM
    4. Conv1D -> ReLU -> MaxPool
    5. Dropout

    VANILLA CLASSIFIER (after flatten):
    6. Dense -> ReLU
    7. Dropout
    8. Dense -> ReLU
    9. Dense -> Sigmoid

    grg_attention_based CLASSIFIER:
    Uses GRGAttentionClassificationHead on the sequence output from encoder (before flatten).
    """

    def __init__(
        self,
        max_sequence_length,
        flavor="lstm",
        num_labels=1,
        classifier_type="vanilla",
        num_grg_labels=None,
        dropout_prob=0.2,
        soft_labels=False,
    ):
        super(DISMIRNet, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.num_labels = num_labels
        self.classifier_type = classifier_type
        self.num_grg_labels = num_grg_labels
        self.soft_labels = soft_labels

        # ============== SHARED ENCODER ==============
        # 1) First convolution block
        self.conv1 = nn.Conv1d(
            in_channels=5, out_channels=100, kernel_size=10, padding=5
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=dropout_prob)

        # 2) Bidirectional LSTM/MinGRU
        if flavor == "lstm":
            rnn = nn.LSTM(
                input_size=100,
                hidden_size=max_sequence_length // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif flavor == "minigru":
            rnn = BiMinGRU(
                input_dim=100,
                hidden_dim=max_sequence_length // 2,
                batch_first=True,
                use_init_hidden_state=False,
                num_layers=1,
            )
        else:
            raise ValueError(f"Unknown flavor: {flavor}. Choose 'lstm' or 'minigru'")

        self.rnn = rnn

        # 3) Second convolution block (part of shared encoder)
        self.conv2 = nn.Conv1d(
            in_channels=max_sequence_length, out_channels=100, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=dropout_prob)

        # Encoder output shape: (batch, 100, seq_len/4)
        # Flattened size for vanilla classifier
        self.encoder_output_size = 100 * (max_sequence_length // 4)
        # Sequence length after encoder (for attention)
        self.encoder_seq_len = max_sequence_length // 4
        # Hidden size for attention (conv2 output channels)
        self.encoder_hidden_size = 100

        # ============== CLASSIFIER HEAD ==============
        if classifier_type == "vanilla":
            _module_logger.info("Using vanilla classifier with FC layers")
            self._init_vanilla_classifier(dropout_prob)
            self.grg_classifier = None

        elif classifier_type == "grg_attention_based":
            _module_logger.info("Using attention-based classifier with GRG context")
            if num_grg_labels is None:
                raise ValueError(
                    "num_grg_labels must be provided for grg_attention_based classifier"
                )

            self._init_grg_attention_classifier(
                num_labels, num_grg_labels, dropout_prob
            )
            # Set vanilla FC components to None
            self.fc1 = None
            self.drop3 = None
            self.fc2 = None
            self.fc3 = None
            self.sigmoid = None
        else:
            raise ValueError(
                f"Unknown classifier_type: {classifier_type}. "
                "Choose 'vanilla' or 'grg_attention_based'"
            )

    def _init_vanilla_classifier(self, dropout_prob):
        """Initialize vanilla classifier components (FC layers only)."""
        self.fc1 = nn.Linear(self.encoder_output_size, 750)
        self.drop3 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(750, 300)
        self.fc3 = nn.Linear(300, self.num_labels)
        self.sigmoid = nn.Sigmoid()

    def _init_grg_attention_classifier(self, num_labels, num_grg_labels, dropout_prob):
        """Initialize GRG attention-based classifier."""
        # After encoder: shape is (batch, 100, seq_len/4)
        # Permuted for attention: (batch, seq_len/4, 100)
        # So hidden_size = 100 (conv2 output channels)
        config = DISMIRConfig(
            hidden_size=self.encoder_hidden_size,  # 100 (conv2 output channels)
            num_labels=num_labels,
            num_grg_labels=num_grg_labels,
            attention_probs_dropout_prob=dropout_prob,
            hidden_dropout_prob=dropout_prob,
            layer_norm_eps=1e-12,
        )
        self.grg_classifier = GRGAttentionClassificationHead(config)

    def _forward_encoder(self, x, parallel_scan=True):
        """
        Shared encoder forward pass.

        Args:
            x: Input tensor of shape (batch_size, max_sequence_length, 5)
            parallel_scan: Whether to use parallel scan for MinGRU

        Returns:
            Encoder output of shape (batch, 100, seq_len/4)
        """
        # (batch, seq_len, channels=5) -> (batch, channels=5, seq_len)
        x = x.permute(0, 2, 1)

        # First conv block
        x = self.conv1(x)  # (batch, 100, seq_len)
        x = self.relu(x)
        x = self.pool1(x)  # (batch, 100, seq_len/2)
        x = self.drop1(x)

        # RNN: expecting shape (batch, seq_len, features=100)
        x = x.permute(0, 2, 1)  # (batch, seq_len/2, 100)
        if isinstance(self.rnn, BiMinGRU) and not parallel_scan:
            x, _ = self.rnn(x, parallel_scan=False)
        else:
            x, _ = self.rnn(x)
        # x shape: (batch, seq_len/2, 2*hidden_size) = (batch, seq_len/2, max_sequence_length)

        # Second conv block
        x = x.permute(0, 2, 1)  # (batch, max_sequence_length, seq_len/2)
        x = self.conv2(x)  # (batch, 100, seq_len/2)
        x = self.relu(x)
        x = self.pool2(x)  # (batch, 100, seq_len/4)
        x = self.drop2(x)

        return x  # (batch, 100, seq_len/4)

    def forward(self, x, parallel_scan=True, grg_ids=None, attention_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, max_sequence_length, 5)
            parallel_scan: Whether to use parallel scan for MinGRU (only relevant for minigru flavor)
            grg_ids: [batch_size] - GRG labels for each sequence (required for grg_attention_based)
            attention_mask: [batch_size, seq_len] - attention mask for padding (optional, for grg_attention_based)

        Returns:
            For vanilla: output tensor of shape (batch_size, num_labels)
            For grg_attention_based: tuple of (logits, attention_weights)
        """
        # Shared encoder
        x = self._forward_encoder(x, parallel_scan)  # (batch, 100, seq_len/4)

        if self.classifier_type == "vanilla":
            return self._forward_vanilla(x)
        elif self.classifier_type == "grg_attention_based":
            return self._forward_grg_attention(x, grg_ids, attention_mask)

    def _forward_vanilla(self, x):
        """Forward pass for vanilla classifier (FC layers only)."""
        # Flatten encoder output
        x = x.reshape(x.size(0), -1)  # (batch, 100 * seq_len/4)

        # Fully connected layers
        x = self.fc1(x)  # (batch, 750)
        x = self.relu(x)
        x = self.drop3(x)
        x = self.fc2(x)  # (batch, 300)
        x = self.relu(x)
        x = self.fc3(x)  # (batch, num_labels)
        x = self.sigmoid(x)
        return x

    def _forward_grg_attention(self, x, grg_ids, attention_mask):
        """Forward pass for GRG attention-based classifier."""
        if grg_ids is None:
            raise ValueError(
                "grg_ids must be provided when using grg_attention_based classifier"
            )

        # x shape: (batch, 100, seq_len/4)
        # Permute for attention: (batch, seq_len/4, 100) = (batch, seq_len, hidden_size)
        x = x.permute(0, 2, 1)

        logits, attention_weights = self.grg_classifier(x, grg_ids, attention_mask)

        return logits, attention_weights


class VariableLengthDataset(Dataset):
    """
    Custom dataset for variable-length sequences that handles chunking.
    TODO: Initialization via providing dataset from RAM instead of from disk
    """

    def __init__(self, data_path_or_df, max_sequence_length, conv_onehot_func):
        self.max_sequence_length = max_sequence_length
        self.conv_onehot = conv_onehot_func
        if isinstance(data_path_or_df, (str, Path)):
            self.data = pd.read_parquet(data_path_or_df)
        else:
            self.data = data_path_or_df

        # Create mapping from read_id to chunks
        self.read_chunks = defaultdict(list)
        self.read_labels = {}
        self.chunk_weights = defaultdict(list)
        self.read_chunk_counts = {}

        self._prepare_chunks()

    def _prepare_chunks(self):
        """
        Prepare chunks for each read and calculate CpG-based weights.
        """
        for idx, row in self.data.iterrows():
            dna_seq = row["input_ids"]
            methylation_seq = row["methylation_ids"]
            label = row["label"]
            read_id = (
                idx  # Using index as read_id, you might have a specific read_id column
            )

            # Store label for this read
            self.read_labels[read_id] = label

            # Create chunks
            seq_len = len(dna_seq)
            chunks = []
            weights = []

            for start in range(0, seq_len, self.max_sequence_length):
                end = min(start + self.max_sequence_length, seq_len)

                # Extract chunk
                chunk_dna = dna_seq[start:end]
                chunk_methylation = methylation_seq[start:end]

                # Convert to one-hot
                chunk_onehot = self.conv_onehot([chunk_dna], [chunk_methylation])[0]
                chunks.append(chunk_onehot)

                # Calculate CpG count for weighting
                cpg_count = self._count_cpg(
                    chunk_dna[: end - start]
                )  # Only count real sequence, not padding
                weights.append(max(cpg_count, 1))  # Ensure minimum weight of 1

            # Normalize weights for this read
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            self.read_chunks[read_id] = chunks
            self.chunk_weights[read_id] = normalized_weights
            self.read_chunk_counts[read_id] = len(chunks)

    def _count_cpg(self, sequence):
        """Count CpG dinucleotides in a sequence."""
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i : i + 2] == "CG":
                count += 1
        return count

    def get_chunk_count(self, idx):
        """Get the number of chunks for a specific read."""
        read_id = list(self.read_labels.keys())[idx]
        return self.read_chunk_counts[read_id]

    def __len__(self):
        return len(self.read_labels)

    def __getitem__(self, idx):
        """
        Return all chunks for a read along with their weights and label.
        """
        read_id = list(self.read_labels.keys())[idx]
        chunks = torch.tensor(np.array(self.read_chunks[read_id]), dtype=torch.float32)
        weights = torch.tensor(self.chunk_weights[read_id], dtype=torch.float32)
        label = torch.tensor(self.read_labels[read_id], dtype=torch.float32)

        return chunks, weights, label, read_id


class ChunkAwareBatchSampler:
    """
    Custom batch sampler that ensures total chunks per batch doesn't exceed max_chunks_per_batch.
    """

    def __init__(self, dataset, max_chunks_per_batch, shuffle=True):
        self.dataset = dataset
        self.max_chunks_per_batch = max_chunks_per_batch
        self.shuffle = shuffle

        # Pre-compute chunk counts for all reads
        self.chunk_counts = [dataset.get_chunk_count(i) for i in range(len(dataset))]

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        batch = []
        current_chunk_count = 0

        for idx in indices:
            read_chunk_count = self.chunk_counts[idx]

            # If adding this read would exceed the limit, yield current batch and start new one
            if (
                current_chunk_count + read_chunk_count > self.max_chunks_per_batch
                and batch
            ):
                yield batch
                batch = []
                current_chunk_count = 0

            # Add the read to current batch
            batch.append(idx)
            current_chunk_count += read_chunk_count

            # If we've reached the limit exactly, yield the batch
            if current_chunk_count == self.max_chunks_per_batch:
                yield batch
                batch = []
                current_chunk_count = 0

        # Yield any remaining reads in the last batch
        if batch:
            yield batch

    def __len__(self):
        # Estimate number of batches (this is approximate)
        total_chunks = sum(self.chunk_counts)
        return (
            total_chunks + self.max_chunks_per_batch - 1
        ) // self.max_chunks_per_batch


def variable_length_collate_fn(batch):
    """
    Custom collate function for variable-length training.
    Groups all chunks from all reads in the batch.
    """
    all_chunks = []
    all_weights = []
    all_labels = []
    chunk_to_read_mapping = []

    for read_idx, (chunks, weights, label, read_id) in enumerate(batch):
        all_chunks.append(chunks)
        all_weights.append(weights)
        all_labels.append(label)

        # Map each chunk to its read index in the batch
        chunk_to_read_mapping.extend([read_idx] * len(chunks))

    # Concatenate all chunks
    all_chunks = torch.cat(all_chunks, dim=0)
    all_weights = torch.cat(all_weights, dim=0)
    all_labels = torch.stack(all_labels)
    chunk_to_read_mapping = torch.tensor(chunk_to_read_mapping, dtype=torch.long)

    return all_chunks, all_weights, all_labels, chunk_to_read_mapping


class Dismir(AbstractReadClassifier):
    """
    Equivalent class to the Keras-based Dismir, but using PyTorch internally.
    Supports both vanilla and GRG attention-based classifiers.
    """

    def __init__(
        self,
        max_sequence_length,
        train_data_path,
        test_data_path,
        valid_data_path,
        device=None,
        flavour="lstm",
        num_labels=2,
        classifier_type="vanilla",
        num_grg_labels=None,
        grg_label_column=None,
        soft_labels=False,
    ):
        self.max_sequence_length = max_sequence_length
        self.num_labels = num_labels
        self.classifier_type = classifier_type
        self.num_grg_labels = num_grg_labels
        self.grg_label_column = grg_label_column
        self.soft_labels = soft_labels

        # Validate GRG parameters
        if classifier_type == "grg_attention_based":
            if num_grg_labels is None:
                raise ValueError(
                    "num_grg_labels must be provided for grg_attention_based classifier"
                )
            if grg_label_column is None:
                raise ValueError(
                    "grg_label_column must be provided for grg_attention_based classifier"
                )

        # Set up loss function
        # Note: GRGAttentionClassificationHead returns raw logits (no sigmoid), so we need different loss
        if classifier_type == "vanilla":
            if num_labels == 1:
                self.criterion = nn.BCELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:  # grg_attention_based
            if num_labels == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                if soft_labels:
                    _module_logger.info(
                        "Soft Labels were selected as labeling scheme. The model will be trained with ConfidenceWeightedCrossEntropy"
                    )
                    self.criterion = ConfidenceWeightedCrossEntropy(num_labels)
                else:
                    self.criterion = nn.CrossEntropyLoss()

        # Use CUDA if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize the PyTorch model
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.valid_data_path = valid_data_path
        self.history = []

        self.model = DISMIRNet(
            max_sequence_length,
            flavour,
            num_labels,
            classifier_type=classifier_type,
            num_grg_labels=num_grg_labels,
        ).to(self.device)

    def conv_onehot(self, dna_seq, c_methylation_seq):
        """
        Transform sequences into one-hot + methylation channel.
        """
        module = np.array(
            [
                [1, 0, 0, 0, 0],  # A
                [0, 1, 0, 0, 0],  # T
                [0, 0, 1, 0, 0],  # C
                [0, 0, 0, 1, 0],  # G
                [0, 0, 1, 0, 1],  # Methylated C
            ],
            dtype=int,
        )

        onehot = np.zeros((len(dna_seq), self.max_sequence_length, 5), dtype=np.int32)

        for i, tmp_seq in enumerate(dna_seq):
            tmp_methylation = c_methylation_seq[i]
            for j in range(min(len(tmp_seq), self.max_sequence_length)):
                if tmp_methylation[j] == "1":
                    onehot[i, j] = module[4]
                elif tmp_seq[j] == "A":
                    onehot[i, j] = module[0]
                elif tmp_seq[j] == "T":
                    onehot[i, j] = module[1]
                elif tmp_seq[j] == "C":
                    onehot[i, j] = module[2]
                elif tmp_seq[j] == "G":
                    onehot[i, j] = module[3]
        return onehot

    def load_and_transform_input(
        self, data_path, return_grg_labels=False, soft_labels=False
    ):
        """
        Loads parquet data with columns: [input_ids, methylation_ids, label, (grg_label_column)]
        Then transforms sequences into one-hot + methylation.

        Args:
            data_path: Path to parquet file
            return_grg_labels: If True, also return GRG labels

        Returns:
            If return_grg_labels is False: (features, labels)
            If return_grg_labels is True: (features, labels, grg_labels)
        """
        data = pd.read_parquet(data_path)
        return self._transform_input_df(data, return_grg_labels, soft_labels)

    def _transform_input_df(
        self, data: pd.DataFrame, return_grg_labels=False, soft_labels=False
    ):
        """Transform dataframe into one-hot + methylation."""
        cols = list(data.columns)
        dna_col = resolve_column(cols, "input_ids")
        meth_col = resolve_column(cols, "methylation_ids")
        dna = data[dna_col]
        methylation = data[meth_col]
        if soft_labels:
            labels = np.array(data["soft_label"].to_list())
        else:
            labels = np.array(data["label"].to_list())
        features = self.conv_onehot(dna, methylation)

        if return_grg_labels:
            if self.grg_label_column is None:
                raise ValueError("grg_label_column must be set to return GRG labels")
            if self.grg_label_column not in data.columns:
                raise ValueError(
                    f"GRG label column '{self.grg_label_column}' not found in data"
                )
            grg_labels = data[self.grg_label_column]
            return features, labels, grg_labels

        return features, labels

    def train(
        self,
        train_dir="./",
        verbose=1,
        epochs=50,
        batch_size=32,
        patience=10,
        optimizer_type="SGD",
        lr=0.05,
        weight_decay=1e-6,
        momentum=0.9,
        nesterov=True,
        variable_length=False,
        reset_history=False,
        train_df=None,
        valid_df=None,
    ):
        """
        Enhanced training method with support for variable-length sequences.
        """
        if reset_history:
            self.history = []
        if variable_length:
            return self._train_variable_length(
                train_dir,
                verbose,
                epochs,
                batch_size,
                patience,
                optimizer_type,
                lr,
                weight_decay,
                momentum,
                nesterov,
                train_df,
                valid_df,
            )
        else:
            return self._train_fixed_length(
                train_dir,
                verbose,
                epochs,
                batch_size,
                patience,
                optimizer_type,
                lr,
                weight_decay,
                momentum,
                nesterov,
                train_df,
                valid_df,
            )

    def _train_fixed_length(
        self,
        train_dir,
        verbose,
        epochs,
        batch_size,
        patience,
        optimizer_type,
        lr,
        weight_decay,
        momentum,
        nesterov,
        train_df=None,
        valid_df=None,
    ):
        """
        Fixed-length training method with GRG support.
        """
        if verbose > 0:
            _module_logger.info("Preparing data for fixed-length training...")

        # Load data - with or without GRG labels
        use_grg = self.classifier_type == "grg_attention_based"

        def _get_data(df, path):
            if df is not None:
                return self._transform_input_df(
                    df, return_grg_labels=use_grg, soft_labels=self.soft_labels
                )
            return self.load_and_transform_input(
                path, return_grg_labels=use_grg, soft_labels=self.soft_labels
            )

        if use_grg:
            self.train_x, self.train_y, self.train_grg = _get_data(
                train_df, self.train_data_path
            )
            _module_logger.info(f"... Train is ready")
            self.valid_x, self.valid_y, self.valid_grg = _get_data(
                valid_df, self.valid_data_path
            )
            _module_logger.info(f"... Valid is ready")
        else:
            self.train_x, self.train_y = _get_data(train_df, self.train_data_path)
            _module_logger.info(f"... Train is ready")
            self.valid_x, self.valid_y = _get_data(valid_df, self.valid_data_path)
            _module_logger.info(f"... Valid is ready")

        # Convert features to torch.Tensor
        self.train_x = torch.tensor(self.train_x, dtype=torch.float32)
        self.valid_x = torch.tensor(self.valid_x, dtype=torch.float32)

        # Handle labels based on num_labels
        if self.num_labels == 1:
            self.train_y = torch.tensor(self.train_y, dtype=torch.float32).view(-1, 1)
            self.valid_y = torch.tensor(self.valid_y, dtype=torch.float32).view(-1, 1)
        else:
            dtype = torch.float32 if self.soft_labels else torch.long
            if not self.soft_labels:
                self.train_y = torch.tensor(self.train_y, dtype=dtype).squeeze()
                self.valid_y = torch.tensor(self.valid_y, dtype=dtype).squeeze()
            else:
                self.train_y = torch.tensor(self.train_y, dtype=dtype)
                self.valid_y = torch.tensor(self.valid_y, dtype=dtype)

        # Convert GRG labels if needed
        if use_grg:
            self.train_grg = torch.tensor(self.train_grg.values, dtype=torch.long)
            self.valid_grg = torch.tensor(self.valid_grg.values, dtype=torch.long)

        # Create optimizer
        optimizer = self._create_optimizer(
            optimizer_type, lr, weight_decay, momentum, nesterov
        )

        # DataLoaders
        if use_grg:
            train_dataset = torch.utils.data.TensorDataset(
                self.train_x, self.train_y, self.train_grg
            )
            valid_dataset = torch.utils.data.TensorDataset(
                self.valid_x, self.valid_y, self.valid_grg
            )
        else:
            train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
            valid_dataset = torch.utils.data.TensorDataset(self.valid_x, self.valid_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        return self._training_loop(
            train_loader,
            valid_loader,
            optimizer,
            self.criterion,
            epochs,
            patience,
            verbose,
            train_dir,
            "fixed",
        )

    def _train_variable_length(
        self,
        train_dir,
        verbose,
        epochs,
        batch_size,
        patience,
        optimizer_type,
        lr,
        weight_decay,
        momentum,
        nesterov,
        train_df=None,
        valid_df=None,
    ):
        """
        Variable-length training method with chunk-based weighted loss.
        """
        if self.classifier_type == "grg_attention_based":
            raise NotImplementedError(
                "Variable-length training is not yet supported for grg_attention_based classifier. "
                "Please use fixed-length training or implement VariableLengthDataset with GRG support."
            )

        if verbose > 0:
            _module_logger.info("Preparing data for variable-length training...")

        # Create variable-length datasets
        train_dataset = VariableLengthDataset(
            train_df if train_df is not None else self.train_data_path,
            self.max_sequence_length,
            self.conv_onehot,
        )
        valid_dataset = VariableLengthDataset(
            valid_df if valid_df is not None else self.valid_data_path,
            self.max_sequence_length,
            self.conv_onehot,
        )

        max_chunks_per_batch = batch_size

        if verbose > 0:
            _module_logger.info(f"Using max_chunks_per_batch: {max_chunks_per_batch}")
            _module_logger.info(
                f"Average chunks per read (train): {np.mean([train_dataset.get_chunk_count(i) for i in range(min(100, len(train_dataset)))]):.2f}"
            )

        train_batch_sampler = ChunkAwareBatchSampler(
            train_dataset, max_chunks_per_batch, shuffle=True
        )
        valid_batch_sampler = ChunkAwareBatchSampler(
            valid_dataset, max_chunks_per_batch, shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=variable_length_collate_fn,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_sampler=valid_batch_sampler,
            collate_fn=variable_length_collate_fn,
        )

        optimizer = self._create_optimizer(
            optimizer_type, lr, weight_decay, momentum, nesterov
        )

        return self._training_loop_variable_length(
            train_loader,
            valid_loader,
            optimizer,
            self.criterion,
            epochs,
            patience,
            verbose,
            train_dir,
        )

    def _compute_epoch_metrics(self, all_outputs, all_labels, prefix):
        """
        Compute sklearn metrics from accumulated epoch outputs.

        Delegates to compute_metrics or compute_metrics_soft_labels
        from syto.classification.evaluation, depending on self.soft_labels.

        Args:
            all_outputs: list of numpy arrays – model output probabilities
                         per batch, concatenated along dim 0.
            all_labels:  list of numpy arrays – ground-truth labels per batch.
            prefix:      'train' or 'val' – prepended to each metric key.

        Returns:
            dict with keys like '{prefix}_f1', '{prefix}_precision', etc.
        """
        outputs = np.concatenate(all_outputs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        eval_pred = (outputs, labels)

        if self.soft_labels:
            metrics = compute_metrics_soft_labels(eval_pred)
        else:
            metrics = compute_metrics(eval_pred)

        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def _create_optimizer(self, optimizer_type, lr, weight_decay, momentum, nesterov):
        """Create optimizer based on parameters."""
        if optimizer_type.upper() == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )
        elif optimizer_type.upper() == "ADAM":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("optimizer_type must be 'SGD' or 'Adam'")

    def _training_loop(
        self,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        epochs,
        patience,
        verbose,
        train_dir,
        mode,
    ):
        """Standard training loop for fixed-length sequences with GRG support."""
        best_val_loss = float("inf")
        patience_counter = 0
        session_start_time = time.time()
        use_grg = self.classifier_type == "grg_attention_based"

        if verbose > 0:
            _module_logger.info(f"Start {mode}-length training...")

        use_notebook = is_in_notebook()
        progress_tracker = TrainingProgressTracker(use_notebook=use_notebook)
        disable_tqdm = (verbose <= 0) or use_notebook

        epoch_iterator = tqdm(
            range(1, epochs + 1),
            desc="Training",
            disable=disable_tqdm,
            leave=True,
        )
        for epoch in epoch_iterator:
            # Training
            self.model.train()
            epoch_loss = 0.0
            correct, total = 0, 0
            train_all_outputs = []
            train_all_labels = []

            for batch in train_loader:
                if use_grg:
                    X_batch, y_batch, grg_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    grg_batch = grg_batch.to(self.device)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                if use_grg:
                    outputs, _ = self.model(X_batch, grg_ids=grg_batch)
                else:
                    outputs = self.model(X_batch)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

                # Collect outputs for metrics
                batch_outputs = outputs.detach()
                if use_grg and self.num_labels == 1:
                    batch_outputs = torch.sigmoid(batch_outputs)
                elif use_grg and self.num_labels > 1:
                    batch_outputs = torch.softmax(batch_outputs, dim=-1)
                train_all_outputs.append(batch_outputs.cpu().numpy())
                train_all_labels.append(y_batch.detach().cpu().numpy())

                # Prediction logic
                if self.num_labels == 1:
                    if use_grg:
                        preds = (
                            (torch.sigmoid(outputs.detach()) >= 0.5).float().squeeze()
                        )
                    else:
                        preds = (outputs.detach() >= 0.5).float().squeeze()
                    correct += (preds == y_batch.squeeze()).sum().item()
                else:
                    preds = outputs.detach().argmax(dim=1)
                    if self.soft_labels:
                        y_batch = y_batch.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

            train_loss = epoch_loss / len(train_loader.dataset)
            train_acc = correct / total
            train_metrics = self._compute_epoch_metrics(
                train_all_outputs, train_all_labels, "train"
            )

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            val_all_outputs = []
            val_all_labels = []

            with torch.no_grad():
                for batch in valid_loader:
                    if use_grg:
                        X_val, y_val, grg_val = batch
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)
                        grg_val = grg_val.to(self.device)
                        val_outputs, _ = self.model(X_val, grg_ids=grg_val)
                    else:
                        X_val, y_val = batch
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)
                        val_outputs = self.model(X_val)

                    v_loss = criterion(val_outputs, y_val)
                    val_loss += v_loss.item() * X_val.size(0)

                    # Collect outputs for metrics
                    batch_val_outputs = val_outputs.detach()
                    if use_grg and self.num_labels == 1:
                        batch_val_outputs = torch.sigmoid(batch_val_outputs)
                    elif use_grg and self.num_labels > 1:
                        batch_val_outputs = torch.softmax(batch_val_outputs, dim=-1)
                    val_all_outputs.append(batch_val_outputs.cpu().numpy())
                    val_all_labels.append(y_val.detach().cpu().numpy())

                    if self.num_labels == 1:
                        if use_grg:
                            val_preds = (
                                (torch.sigmoid(val_outputs) >= 0.5).float().squeeze()
                            )
                        else:
                            val_preds = (val_outputs >= 0.5).float().squeeze()
                        val_correct += (val_preds == y_val.squeeze()).sum().item()
                    else:
                        val_preds = val_outputs.argmax(dim=1)
                        if self.soft_labels:
                            y_val = y_val.argmax(dim=1)
                        val_correct += (val_preds == y_val).sum().item()
                    val_total += y_val.size(0)

            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / val_total
            val_metrics = self._compute_epoch_metrics(
                val_all_outputs, val_all_labels, "val"
            )
            epoch_time = time.time() - session_start_time

            epoch_record = {
                "session": len([h for h in self.history if h.get("epoch") == 1]) + 1,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_time": epoch_time,
            }
            epoch_record.update(train_metrics)
            epoch_record.update(val_metrics)
            self.history.append(epoch_record)

            if verbose > 0:
                progress_tracker.update(
                    epoch, epochs, train_loss, val_loss, val_metrics
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    self.model.state_dict(), os.path.join(train_dir, "weight.pt")
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose > 0:
                    _module_logger.info("Early stopping triggered.")
                break

    def _training_loop_variable_length(
        self,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        epochs,
        patience,
        verbose,
        train_dir,
    ):
        """Training loop for variable-length sequences with weighted chunk averaging."""
        # Note: GRG support not implemented for variable length yet
        best_val_loss = float("inf")
        patience_counter = 0
        session_start_time = time.time()
        if verbose > 0:
            _module_logger.info("Start variable-length training...")

        use_notebook = is_in_notebook()
        progress_tracker = TrainingProgressTracker(use_notebook=use_notebook)
        disable_tqdm = (verbose <= 0) or use_notebook

        epoch_iterator = tqdm(
            range(1, epochs + 1),
            desc="Training",
            disable=disable_tqdm,
            leave=True,
        )
        for epoch in epoch_iterator:
            # Training
            self.model.train()
            epoch_loss = 0.0
            correct, total = 0, 0
            train_all_outputs = []
            train_all_labels = []

            for batch_data in train_loader:
                all_chunks, all_weights, all_labels, chunk_to_read_mapping = batch_data
                all_chunks = all_chunks.to(self.device)
                all_weights = all_weights.to(self.device)
                chunk_to_read_mapping = chunk_to_read_mapping.to(self.device)

                if self.num_labels == 1:
                    all_labels = all_labels.to(self.device).float()
                else:
                    all_labels = all_labels.to(self.device).long()

                optimizer.zero_grad()

                chunk_outputs = self.model(all_chunks)

                batch_size = len(all_labels)
                read_losses = []
                read_preds = []
                read_outputs_for_metrics = []

                for read_idx in range(batch_size):
                    read_mask = chunk_to_read_mapping == read_idx
                    read_chunk_outputs = chunk_outputs[read_mask]
                    read_chunk_weights = all_weights[read_mask]
                    read_label = all_labels[read_idx]

                    if self.num_labels == 1:
                        read_chunk_outputs_1d = read_chunk_outputs.squeeze(-1)
                        chunk_labels = read_label.expand_as(read_chunk_outputs_1d)
                        chunk_losses = criterion(read_chunk_outputs_1d, chunk_labels)

                        weighted_loss = torch.sum(chunk_losses * read_chunk_weights)
                        weighted_pred = torch.sum(
                            read_chunk_outputs_1d * read_chunk_weights
                        )
                        read_losses.append(weighted_loss)
                        read_preds.append(weighted_pred)
                        read_outputs_for_metrics.append(weighted_pred.detach())
                    else:
                        chunk_labels = read_label.expand(read_chunk_outputs.size(0))

                        chunk_losses = torch.stack(
                            [
                                criterion(
                                    read_chunk_outputs[i : i + 1],
                                    chunk_labels[i : i + 1],
                                )
                                for i in range(len(chunk_labels))
                            ]
                        )

                        weighted_loss = torch.sum(chunk_losses * read_chunk_weights)
                        read_losses.append(weighted_loss)

                        weights_expanded = read_chunk_weights.unsqueeze(-1)
                        weighted_logits = torch.sum(
                            read_chunk_outputs * weights_expanded, dim=0
                        )
                        read_preds.append(weighted_logits.argmax())
                        read_outputs_for_metrics.append(weighted_logits.detach())

                total_loss = torch.stack(read_losses).mean()
                total_loss.backward()
                optimizer.step()

                # Collect for metrics
                train_all_outputs.append(
                    torch.stack(read_outputs_for_metrics).cpu().numpy()
                )
                train_all_labels.append(all_labels.detach().cpu().numpy())

                if self.num_labels == 1:
                    read_preds = torch.stack(read_preds)
                    binary_preds = (read_preds >= 0.5).float()
                    correct += (binary_preds == all_labels).sum().item()
                else:
                    read_preds = torch.stack(read_preds)
                    correct += (read_preds == all_labels).sum().item()

                total += batch_size
                epoch_loss += total_loss.item() * batch_size

            train_loss = epoch_loss / len(train_loader.dataset)
            train_acc = correct / total
            train_metrics = self._compute_epoch_metrics(
                train_all_outputs, train_all_labels, "train"
            )

            val_loss, val_acc, val_metrics = self._validate_variable_length(
                valid_loader, criterion
            )
            epoch_time = time.time() - session_start_time

            epoch_record = {
                "session": len([h for h in self.history if h.get("epoch") == 1]) + 1,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_time": epoch_time,
            }
            epoch_record.update(train_metrics)
            epoch_record.update(val_metrics)
            self.history.append(epoch_record)

            if verbose > 0:
                progress_tracker.update(
                    epoch, epochs, train_loss, val_loss, val_metrics
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    self.model.state_dict(), os.path.join(train_dir, "weight.pt")
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose > 0:
                    _module_logger.info("Early stopping triggered.")
                break

    def _validate_variable_length(self, valid_loader, criterion):
        """Validation for variable-length sequences."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_outputs = []
        val_all_labels = []

        with torch.no_grad():
            for batch_data in valid_loader:
                all_chunks, all_weights, all_labels, chunk_to_read_mapping = batch_data
                all_chunks = all_chunks.to(self.device)
                all_weights = all_weights.to(self.device)
                chunk_to_read_mapping = chunk_to_read_mapping.to(self.device)

                if self.num_labels == 1:
                    all_labels = all_labels.to(self.device).float()
                else:
                    all_labels = all_labels.to(self.device).long()

                chunk_outputs = self.model(all_chunks)

                batch_size = len(all_labels)
                read_losses = []
                read_preds = []
                read_outputs_for_metrics = []

                for read_idx in range(batch_size):
                    read_mask = chunk_to_read_mapping == read_idx
                    read_chunk_outputs = chunk_outputs[read_mask]
                    read_chunk_weights = all_weights[read_mask]
                    read_label = all_labels[read_idx]

                    if self.num_labels == 1:
                        read_chunk_outputs_1d = read_chunk_outputs.squeeze(-1)
                        chunk_labels = read_label.expand_as(read_chunk_outputs_1d)
                        chunk_losses = criterion(read_chunk_outputs_1d, chunk_labels)

                        weighted_loss = torch.sum(chunk_losses * read_chunk_weights)
                        weighted_pred = torch.sum(
                            read_chunk_outputs_1d * read_chunk_weights
                        )
                        read_losses.append(weighted_loss)
                        read_preds.append(weighted_pred)
                        read_outputs_for_metrics.append(weighted_pred)
                    else:
                        chunk_labels = read_label.expand(read_chunk_outputs.size(0))

                        chunk_losses = torch.stack(
                            [
                                criterion(
                                    read_chunk_outputs[i : i + 1],
                                    chunk_labels[i : i + 1],
                                )
                                for i in range(len(chunk_labels))
                            ]
                        )

                        weighted_loss = torch.sum(chunk_losses * read_chunk_weights)
                        read_losses.append(weighted_loss)

                        weights_expanded = read_chunk_weights.unsqueeze(-1)
                        weighted_logits = torch.sum(
                            read_chunk_outputs * weights_expanded, dim=0
                        )
                        read_preds.append(weighted_logits.argmax())
                        read_outputs_for_metrics.append(weighted_logits)

                total_loss = torch.stack(read_losses).mean()
                val_loss += total_loss.item() * batch_size

                # Collect for metrics
                val_all_outputs.append(
                    torch.stack(read_outputs_for_metrics).cpu().numpy()
                )
                val_all_labels.append(all_labels.cpu().numpy())

                if self.num_labels == 1:
                    read_preds = torch.stack(read_preds)
                    binary_preds = (read_preds >= 0.5).float()
                    val_correct += (binary_preds == all_labels).sum().item()
                else:
                    read_preds = torch.stack(read_preds)
                    val_correct += (read_preds == all_labels).sum().item()

                val_total += batch_size

        val_loss /= len(valid_loader.dataset)
        val_acc = val_correct / val_total
        val_metrics = self._compute_epoch_metrics(
            val_all_outputs, val_all_labels, "val"
        )
        return val_loss, val_acc, val_metrics

    def predict(
        self,
        dna_sequences,
        methylation_sequences,
        grg_ids=None,
        batch_size=128,
        threshold=0.5,
        parallel_scan=True,
        return_attention=False,
    ):
        """
        Predict on arbitrary sequences using the trained model.

        Args:
            dna_sequences: list (or array-like) of DNA strings
            methylation_sequences: list (or array-like) of methylation strings ("0"/"1")
            grg_ids: array-like of GRG labels (required for grg_attention_based classifier)
            batch_size: batch size for inference
            threshold: classification threshold for 'positive' label
            parallel_scan: whether to use parallel scan for MinGRU
            return_attention: if True and using grg_attention_based, also return attention weights

        Returns:
            For vanilla: (probabilities, predicted_labels)
            For grg_attention_based with return_attention=False: (probabilities, predicted_labels)
            For grg_attention_based with return_attention=True: (probabilities, predicted_labels, attention_weights)
        """
        self.model.eval()

        # Validate GRG IDs for attention-based classifier
        if self.classifier_type == "grg_attention_based":
            if grg_ids is None:
                raise ValueError(
                    "grg_ids must be provided for grg_attention_based classifier"
                )
            if len(grg_ids) != len(dna_sequences):
                raise ValueError("grg_ids must have the same length as dna_sequences")

        # Convert to one-hot + methylation
        onehot_data = self.conv_onehot(dna_sequences, methylation_sequences)

        # Create PyTorch dataset and dataloader
        X_tensor = torch.tensor(onehot_data, dtype=torch.float32)

        if self.classifier_type == "grg_attention_based":
            grg_tensor = torch.tensor(np.array(grg_ids), dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_tensor, grg_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # Perform forward passes in batches
        all_outputs = []
        all_attention_weights = (
            []
            if return_attention and self.classifier_type == "grg_attention_based"
            else None
        )

        with torch.no_grad():
            for batch in tqdm(loader, "predicting batches"):
                if self.classifier_type == "grg_attention_based":
                    X_batch, grg_batch = batch
                    X_batch = X_batch.to(self.device)
                    grg_batch = grg_batch.to(self.device)
                    outputs, attn_weights = self.model(
                        X_batch, parallel_scan=parallel_scan, grg_ids=grg_batch
                    )

                    # Apply sigmoid for probability output (model returns logits)
                    if self.num_labels == 1:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = torch.softmax(outputs, dim=-1)

                    if return_attention:
                        all_attention_weights.append(attn_weights.cpu().numpy())
                else:
                    X_batch = batch[0].to(self.device)
                    outputs = self.model(X_batch, parallel_scan=parallel_scan)

                outputs = outputs.squeeze(-1).cpu().numpy()
                all_outputs.extend(outputs)

        all_outputs = np.array(all_outputs)

        # Apply threshold for predictions
        if self.num_labels == 1:
            predicted_labels = (all_outputs >= threshold).astype(int)
        else:
            predicted_labels = all_outputs.argmax(axis=-1)

        if return_attention and self.classifier_type == "grg_attention_based":
            all_attention_weights = np.concatenate(all_attention_weights, axis=0)
            return all_outputs, predicted_labels, all_attention_weights

        return all_outputs, predicted_labels

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "Dismir":
        """
        Instantiate a DismirClassifier and load pre-trained weights
        from the specified checkpoint path.

        Args:
            path: path to the checkpoint file containing the model weights (.pt file)
            **kwargs: additional parameters required for instantiating the DismirClassifier:
                - seq_length: maximum sequence length (default: 150)
                - dismir_flavor: model architecture flavor (default: "lstm")
                - num_labels: number of output labels (required)
                - classifier_head_implementation: type of classifier head
                    (default: "grg_attention_based")
                - num_grg_labels: number of GRG labels (default: 39, only used if
                    classifier_head_implementation is "grg_attention_based")
                - grg_label_column: name of the GRG label column in the dataset
                    (required if classifier_head_implementation is "grg_attention_based")
        """
        classifier_head_implementation = kwargs.get(
            "classifier_head_implementation", "grg_attention_based"
        )
        grg_label_column = (
            kwargs.get("grg_label_column")
            or kwargs.get("grg_label_column", "grg_ctype_label")
            if classifier_head_implementation == "grg_attention_based"
            else None
        )
        instance = cls(
            max_sequence_length=kwargs.get("seq_length", 150),
            train_data_path="",
            test_data_path="",
            valid_data_path="",
            flavour=kwargs.get("dismir_flavor", "lstm"),
            num_labels=kwargs["num_labels"],
            classifier_type=kwargs.get(
                "classifier_head_implementation", "grg_attention_based"
            ),
            num_grg_labels=kwargs.get("num_grg_labels", 39),
            grg_label_column=grg_label_column,
            soft_labels=kwargs.get("soft_labels", False),
        )
        # Load pre-trained weights, if a checkpoint path was provided
        if path is not None:
            instance.model.load_state_dict(torch.load(path, weights_only=True))
            _module_logger.info("Dismir model loaded from checkpoint: %s", path)
        return instance

    def predict_split(self, split_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Predict labels for a given DataFrame split.

        Args:
            split_df: DataFrame containing the data for the split.
                      Must contain columns for DNA sequences, methylation patterns, and
                      GRG labels (if using attention-based classifier).
            **kwargs: additional parameters for prediction, such as:
                - batch_size: batch size for prediction (default: 2200)
        """
        batch_size = kwargs.get("batch_size", 2200)

        # Extract DNA and methylation sequences
        cols = list(split_df.columns)
        dna_col = resolve_column(cols, "input_ids")
        meth_col = resolve_column(cols, "methylation_ids")

        dna_sequences = split_df[dna_col].tolist()
        methylation_sequences = split_df[meth_col].tolist()

        # GRG ids if using attention-based classifier
        grg_ids = None
        if self.classifier_type == "grg_attention_based":
            grg_ids = split_df[self.grg_label_column].values

        # Run prediction
        probabilities, _ = self.predict(
            dna_sequences=dna_sequences,
            methylation_sequences=methylation_sequences,
            grg_ids=grg_ids,
            batch_size=batch_size,
        )

        # Build predictions DataFrame
        pred_cols = [f"prediction_{i}" for i in range(self.num_labels)]
        pred_df = pd.DataFrame(probabilities, columns=pred_cols)

        # Merge with original DataFrame
        result = split_df.copy()
        for col in pred_cols:
            result[col] = pred_df[col].values

        return result

    @mlflow_tracked_fit
    def fit_classificaton(
        self,
        train_df: pd.DataFrame,
        val_df: Union[pd.DataFrame, None] = None,
        output_dir: Union[str, Path, None] = None,
        **kwargs,
    ) -> "Dismir":
        """Fit the classifier on training data for compatibility with AbstractReadClassifier."""
        # Use kwargs or fall back to some defaults if not provided.
        # This matches train() signature defaults where possible
        train_dir = output_dir if output_dir else "./"
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.train(
            train_dir=str(train_dir),
            train_df=train_df,
            valid_df=val_df,
            epochs=kwargs.get("epochs", 50),
            batch_size=kwargs.get("batch_size", 32),
            patience=kwargs.get("patience", 10),
            optimizer_type=kwargs.get("optimizer_type", "SGD"),
            lr=kwargs.get("lr", 0.05),
            weight_decay=kwargs.get("weight_decay", 1e-6),
            momentum=kwargs.get("momentum", 0.9),
            nesterov=kwargs.get("nesterov", True),
            variable_length=kwargs.get("variable_length", False),
            reset_history=kwargs.get("reset_history", False),
            verbose=kwargs.get("verbose", 1),
        )
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted classifier to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        _module_logger.info("Saved Dismir model to %s", path)
