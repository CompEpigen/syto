import numpy as np
import pandas as pd
import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from methyldl.modelling.minirnns.minRNNs import BiMinGRU
from collections import defaultdict
import time


class DISMIRNet(nn.Module):
    """
    PyTorch model mirroring the structure of the original Keras model from the paper:
    1. Conv1D -> ReLU -> MaxPool
    2. Dropout
    3. Bidirectional LSTM
    4. Conv1D -> ReLU -> MaxPool
    5. Dropout
    6. Flatten
    7. Dense -> ReLU
    8. Dropout
    9. Dense -> ReLU
    10. Dense -> Sigmoid
    """
    def __init__(self, max_sequence_length, flavor ="lstm", num_labels=2):
        super(DISMIRNet, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.num_labels = num_labels
        
        # 1) First convolution block
        # Keras: Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')
        # In PyTorch, "same" padding for kernel_size=10 => padding=10//2=5 (assuming stride=1)
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=100, kernel_size=10, padding=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.2)
        
        # 2) Bidirectional LSTM
        #   input_size = 100 (from the 100 output channels of conv1)
        #   hidden_size = max_sequence_length//2 (same as the Keras code)
        #   batch_first=True => data shape = (batch_size, seq_len, features)
        if flavor == "lstm":
            rnn = nn.LSTM(input_size=100,
                            hidden_size=max_sequence_length//2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        elif flavor =="minigru":
            rnn = BiMinGRU(input_dim=100,hidden_dim=max_sequence_length//2,batch_first=True,use_init_hidden_state=False, num_layers=1)

        self.rnn = rnn
        
        # 3) Second convolution block
        # After bidir LSTM, the feature size becomes 2*(hidden_size) = max_sequence_length
        # So in_channels = max_sequence_length, out_channels=100
        # kernel_size=3 => padding=1 for "same"
        self.conv2 = nn.Conv1d(in_channels=max_sequence_length, out_channels=100, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.2)
        
        # 4) Fully connected layers
        # After the second pool, the time dimension is max_sequence_length/4 (halved twice).
        # The channels are 100. So flatten size = 100 * (max_sequence_length/4).
        # Then Dense(750), Dropout(0.2), Dense(300), Dense(1) => Sigmoid
        # Make sure the dimension is integer if max_sequence_length is divisible by 4.
        self.fc1 = nn.Linear(100 * (max_sequence_length // 4), 750)
        self.drop3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(750, 300)
        self.fc3 = nn.Linear(300, num_labels)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x,parallel_scan=True):
        """
        x shape expected: (batch_size, max_sequence_length, 5)
        PyTorch Conv1d expects: (batch_size, in_channels, seq_len)
        so we'll permute dimensions before/after the LSTM as needed.
        """
        # (batch, seq_len, channels=5) -> (batch, channels=5, seq_len)
        x = x.permute(0, 2, 1)
        
        # First conv block
        x = self.conv1(x)    # (batch, 100, seq_len)
        x = self.relu(x)
        x = self.pool1(x)    # (batch, 100, seq_len/2)
        x = self.drop1(x)
        
        # LSTM: expecting shape (batch, seq_len, features=100)
        x = x.permute(0, 2, 1)  # (batch, seq_len/2, 100)
        if isinstance(self.rnn, BiMinGRU) and not parallel_scan:
            x, _ = self.rnn(x, parallel_scan=False)     # (batch, seq_len/2, 2*hidden_size) = (batch, seq_len/2, max_sequence_length)
        else:
            x, _ = self.rnn(x)     # (batch, seq_len/2, 2*hidden_size) = (batch, seq_len/2, max_sequence_length)
        
        # Second conv block: shape -> (batch, max_sequence_length, seq_len/2)
        x = x.permute(0, 2, 1)  # (batch, in_channels=max_sequence_length, seq_len/2)
        x = self.conv2(x)       # (batch, 100, seq_len/2)
        x = self.relu(x)
        x = self.pool2(x)       # (batch, 100, seq_len/4)
        x = self.drop2(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)  # (batch, 100*(seq_len/4))
        
        # Fully connected
        x = self.fc1(x)       # (batch, 750)
        x = self.relu(x)
        x = self.drop3(x)
        x = self.fc2(x)       # (batch, 300)
        x = self.relu(x)
        x = self.fc3(x)       # (batch, num_labels)
        if self.num_labels>1:
            x = self.sigmoid(x)
        return x

class VariableLengthDataset(Dataset):
    """
    Custom dataset for variable-length sequences that handles chunking.
    TODO: Initialization via providing dataset from RAM instead of from disk
    """
    def __init__(self, data_path, max_sequence_length, conv_onehot_func):
        self.max_sequence_length = max_sequence_length
        self.conv_onehot = conv_onehot_func
        self.data = pd.read_parquet(data_path)
        
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
            read_id = idx  # Using index as read_id, you might have a specific read_id column
            
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
                cpg_count = self._count_cpg(chunk_dna[:end-start])  # Only count real sequence, not padding
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
            if sequence[i:i+2] == 'CG':
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
            if current_chunk_count + read_chunk_count > self.max_chunks_per_batch and batch:
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
        return (total_chunks + self.max_chunks_per_batch - 1) // self.max_chunks_per_batch


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

class Dismir:
    """
    Equivalent class to the Keras-based Dismir, but using PyTorch internally.
    1. Data loading and transformation (one-hot + methylation channel)
    2. Model creation (DISMIRNet)
    3. Training loop with a simplistic early-stopping approach
    """
    def __init__(self, max_sequence_length, train_data_path, test_data_path, valid_data_path, device=None, flavour="lstm", num_labels=2):
        self.max_sequence_length = max_sequence_length
        self.num_labels = num_labels
        
        # Use CUDA if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize the PyTorch model
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.valid_data_path = valid_data_path
        self.history = []
        
        self.model = DISMIRNet(max_sequence_length, flavour,num_labels).to(self.device)
        

    def conv_onehot(self, dna_seq, c_methylation_seq):
        """
        Transform sequences into one-hot + methylation channel.
        
        Original Keras code uses a 5-channel representation:
        - A: [1,0,0,0,0]
        - T: [0,1,0,0,0]
        - C: [0,0,1,0,0]
        - G: [0,0,0,1,0]
        - Methylated C: [0,0,1,0,1]  (the last index indicates methylation)
        """
        # module[i]:
        #   0 -> A
        #   1 -> T
        #   2 -> C
        #   3 -> G
        #   4 -> Methylated C
        # shape => (num_samples, max_sequence_length, 5)
        module = np.array([
            [1, 0, 0, 0, 0],  # A
            [0, 1, 0, 0, 0],  # T
            [0, 0, 1, 0, 0],  # C
            [0, 0, 0, 1, 0],  # G
            [0, 0, 1, 0, 1],  # Methylated C
        ], dtype=int)
        
        onehot = np.zeros((len(dna_seq), self.max_sequence_length, 5), dtype=np.int32)
        
        for i in range(len(dna_seq)):
            tmp_seq = dna_seq[i]
            tmp_methylation = c_methylation_seq[i]
            for j in range(min(len(tmp_seq),self.max_sequence_length)):
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
                # Else remain zeros if unexpected character
        return onehot

    def load_and_transform_input(self, data_path):
        """
        Loads CSV data with columns: [input_ids, methylation_ids, label]
        Then transforms sequences into one-hot + methylation.
        Returns (features, labels)
        """
        data = pd.read_parquet(data_path)
        # data_filtered = data.loc[data["sum_abs_areastat"]>4500,].reset_index() #TEMP --> Filter by Area Stat
        dna = data["input_ids"]
        methylation = data["methylation_ids"]
        labels = data["label"]
        features = self.conv_onehot(dna, methylation)
        return features, labels
    
    def train(self,
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
              reset_history=False):
        """
        Enhanced training method with support for variable-length sequences.
        
        :param variable_length: If True, use variable-length training mode
        """
        if reset_history:
            self.history = []
        if variable_length:
            return self._train_variable_length(
                train_dir, verbose, epochs, batch_size, patience,
                optimizer_type, lr, weight_decay, momentum, nesterov
            )
        else:
            return self._train_fixed_length(
                train_dir, verbose, epochs, batch_size, patience,
                optimizer_type, lr, weight_decay, momentum, nesterov
            )
    
    def _train_fixed_length(self, train_dir, verbose, epochs, batch_size, patience,
                           optimizer_type, lr, weight_decay, momentum, nesterov):
        """
        Original fixed-length training method.
        """
        if verbose > 0:
            print("Preparing data for fixed-length training...")
        
        # Load data
        self.train_x, self.train_y = self.load_and_transform_input(self.train_data_path)
        self.valid_x, self.valid_y = self.load_and_transform_input(self.valid_data_path)
        self.test_x, self.test_y = self.load_and_transform_input(self.test_data_path)
        
        # Convert to torch.Tensor
        self.train_x = torch.tensor(self.train_x, dtype=torch.float32)
        self.valid_x = torch.tensor(self.valid_x, dtype=torch.float32)
        self.test_x = torch.tensor(self.test_x, dtype=torch.float32)

        # Handle labels differently based on num_labels
        if self.num_labels == 1:
            # Binary classification with BCELoss - keep as (batch_size, 1)
            criterion = nn.BCELoss()
            self.train_y = torch.tensor(self.train_y.values, dtype=torch.float32).view(-1, 1)
            self.valid_y = torch.tensor(self.valid_y.values, dtype=torch.float32).view(-1, 1)
            self.test_y = torch.tensor(self.test_y.values, dtype=torch.float32).view(-1, 1)
        else:
            # Multi-class classification with CrossEntropyLoss - squeeze to (batch_size,)
            criterion = nn.CrossEntropyLoss()
            self.train_y = torch.tensor(self.train_y.values, dtype=torch.long).squeeze()
            self.valid_y = torch.tensor(self.valid_y.values, dtype=torch.long).squeeze()
            self.test_y = torch.tensor(self.test_y.values, dtype=torch.long).squeeze()
        
        # Create optimizer
        optimizer = self._create_optimizer(optimizer_type, lr, weight_decay, momentum, nesterov)
        # DataLoaders
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        valid_dataset = torch.utils.data.TensorDataset(self.valid_x, self.valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        return self._training_loop(train_loader, valid_loader, optimizer, criterion,
                                 epochs, patience, verbose, train_dir, "fixed")
    
    def _train_variable_length(self, train_dir, verbose, epochs, batch_size, patience,
                              optimizer_type, lr, weight_decay, momentum, nesterov):
        """
        Variable-length training method with chunk-based weighted loss.
        """
        if verbose > 0:
            print("Preparing data for variable-length training...")
        
        # Create variable-length datasets
        train_dataset = VariableLengthDataset(self.train_data_path, self.max_sequence_length, self.conv_onehot)
        valid_dataset = VariableLengthDataset(self.valid_data_path, self.max_sequence_length, self.conv_onehot)
        
        # batch_size now represents the maximum number of chunks per batch
        max_chunks_per_batch = batch_size
        
        if verbose > 0:
            print(f"Using max_chunks_per_batch: {max_chunks_per_batch}")
            print(f"Average chunks per read (train): {np.mean([train_dataset.get_chunk_count(i) for i in range(min(100, len(train_dataset)))]):.2f}")
        
        # Create data loaders with chunk-aware batch sampler
        train_batch_sampler = ChunkAwareBatchSampler(train_dataset, max_chunks_per_batch, shuffle=True)
        valid_batch_sampler = ChunkAwareBatchSampler(valid_dataset, max_chunks_per_batch, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, 
                                collate_fn=variable_length_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler,
                                collate_fn=variable_length_collate_fn)
        
        # Create optimizer
        optimizer = self._create_optimizer(optimizer_type, lr, weight_decay, momentum, nesterov)
        criterion = nn.BCELoss(reduction='none')  # Use 'none' to get per-sample losses
        
        return self._training_loop_variable_length(train_loader, valid_loader, optimizer, criterion,
                                                 epochs, patience, verbose, train_dir)
    
    def _create_optimizer(self, optimizer_type, lr, weight_decay, momentum, nesterov):
        """Create optimizer based on parameters."""
        if optimizer_type.upper() == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )
        elif optimizer_type.upper() == "ADAM":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError("optimizer_type must be 'SGD' or 'Adam'")
    
    def _training_loop(self, train_loader, valid_loader, optimizer, criterion,
                      epochs, patience, verbose, train_dir, mode):
        """Standard training loop for fixed-length sequences."""
        best_val_loss = float('inf')
        patience_counter = 0
        session_start_time = time.time()
        if verbose > 0:
            print(f"Start {mode}-length training...")
        
        for epoch in range(1, epochs+1):
            # Training
            self.model.train()
            epoch_loss = 0.0
            correct, total = 0, 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
                # Different prediction logic based on num_labels
                if self.num_labels == 1:
                    # Binary classification: threshold at 0.5
                    preds = (outputs.detach() >= 0.5).float().squeeze()
                    correct += (preds == y_batch.squeeze()).sum().item()
                else:
                    # Multi-class classification: use argmax
                    preds = outputs.detach().argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_val, y_val in valid_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_outputs = self.model(X_val)
                    v_loss = criterion(val_outputs, y_val)
                    
                    val_loss += v_loss.item() * X_val.size(0)
                    
                    if self.num_labels == 1:
                    # Binary classification: threshold at 0.5
                        val_preds = (val_outputs >= 0.5).float().squeeze()
                        val_correct += (val_preds == y_val.squeeze()).sum().item()
                    else:
                    # Multi-class classification: use argmax
                        val_preds = val_outputs.argmax(dim=1)
                        val_correct += (val_preds == y_val).sum().item()
                    val_total += y_val.size(0)
            
            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / val_total
            epoch_time = time.time() - session_start_time
            self.history.append({
                'session': len([h for h in self.history if h.get('epoch') == 1]) + 1,  # Optional session id
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'elapsed_time': epoch_time,
            })
            if verbose > 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(train_dir, "weight.pt"))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose > 0:
                    print("Early stopping triggered.")
                break
    
    def _training_loop_variable_length(self, train_loader, valid_loader, optimizer, criterion,
                                     epochs, patience, verbose, train_dir):
        """Training loop for variable-length sequences with weighted chunk averaging."""
        best_val_loss = float('inf')
        patience_counter = 0
        session_start_time = time.time()
        if verbose > 0:
            print("Start variable-length training...")
        
        for epoch in range(1, epochs+1):
            # Training
            self.model.train()
            epoch_loss = 0.0
            correct, total = 0, 0
            
            for batch_data in train_loader:
                all_chunks, all_weights, all_labels, chunk_to_read_mapping = batch_data
                all_chunks = all_chunks.to(self.device)
                all_weights = all_weights.to(self.device)
                all_labels = all_labels.to(self.device)
                chunk_to_read_mapping = chunk_to_read_mapping.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass on all chunks
                chunk_outputs = self.model(all_chunks).squeeze(-1)  # Shape: [num_chunks]
                
                # Calculate weighted loss for each read
                batch_size = len(all_labels)
                read_losses = []
                read_preds = []
                
                for read_idx in range(batch_size):
                    # Find chunks belonging to this read
                    read_mask = (chunk_to_read_mapping == read_idx)
                    read_chunk_outputs = chunk_outputs[read_mask]
                    read_chunk_weights = all_weights[read_mask]
                    
                    # Calculate individual chunk losses
                    read_label = all_labels[read_idx]
                    chunk_labels = read_label.expand_as(read_chunk_outputs)
                    chunk_losses = criterion(read_chunk_outputs, chunk_labels)
                    
                    # Weighted average loss for this read
                    weighted_loss = torch.sum(chunk_losses * read_chunk_weights)
                    read_losses.append(weighted_loss)
                    
                    # Weighted average prediction for this read
                    weighted_pred = torch.sum(read_chunk_outputs * read_chunk_weights)
                    read_preds.append(weighted_pred)
                
                # Total loss is average across reads in batch
                total_loss = torch.stack(read_losses).mean()
                total_loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                read_preds = torch.stack(read_preds)
                binary_preds = (read_preds >= 0.5).float()
                correct += (binary_preds == all_labels).sum().item()
                total += batch_size
                epoch_loss += total_loss.item() * batch_size
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # Validation
            val_loss, val_acc = self._validate_variable_length(valid_loader, criterion)
            epoch_time = time.time() - session_start_time
            self.history.append({
                'session': len([h for h in self.history if h.get('epoch') == 1]) + 1,  # Optional session id
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'elapsed_time': epoch_time,
            })            
            if verbose > 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(train_dir, "weight.pt"))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose > 0:
                    print("Early stopping triggered.")
                break
    
    def _validate_variable_length(self, valid_loader, criterion):
        """Validation for variable-length sequences."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data in valid_loader:
                all_chunks, all_weights, all_labels, chunk_to_read_mapping = batch_data
                all_chunks = all_chunks.to(self.device)
                all_weights = all_weights.to(self.device)
                all_labels = all_labels.to(self.device)
                chunk_to_read_mapping = chunk_to_read_mapping.to(self.device)
                
                # Forward pass
                chunk_outputs = self.model(all_chunks).squeeze(-1)
                
                # Calculate weighted loss and predictions for each read
                batch_size = len(all_labels)
                read_losses = []
                read_preds = []
                
                for read_idx in range(batch_size):
                    read_mask = (chunk_to_read_mapping == read_idx)
                    read_chunk_outputs = chunk_outputs[read_mask]
                    read_chunk_weights = all_weights[read_mask]
                    
                    # Weighted loss
                    read_label = all_labels[read_idx]
                    chunk_labels = read_label.expand_as(read_chunk_outputs)
                    chunk_losses = criterion(read_chunk_outputs, chunk_labels)
                    weighted_loss = torch.sum(chunk_losses * read_chunk_weights)
                    read_losses.append(weighted_loss)
                    
                    # Weighted prediction
                    weighted_pred = torch.sum(read_chunk_outputs * read_chunk_weights)
                    read_preds.append(weighted_pred)
                
                # Accumulate validation metrics
                total_loss = torch.stack(read_losses).mean()
                val_loss += total_loss.item() * batch_size
                
                read_preds = torch.stack(read_preds)
                binary_preds = (read_preds >= 0.5).float()
                val_correct += (binary_preds == all_labels).sum().item()
                val_total += batch_size
        
        val_loss /= len(valid_loader.dataset)
        val_acc = val_correct / val_total
        return val_loss, val_acc

    def evaluate(self, split='test', variable_length=False):
        """
        Evaluate on the test or validation split with support for both modes.
        """
        if variable_length:
            return self._evaluate_variable_length(split)
        else:
            return self._evaluate_fixed_length(split)
    
    def _evaluate_fixed_length(self, split):
        """Original fixed-length evaluation."""
        criterion = nn.BCELoss()
        if split == 'test':
            X_data, y_data = self.test_x, self.test_y
        else:
            X_data, y_data = self.valid_x, self.valid_y
        
        dataset = torch.utils.data.TensorDataset(X_data, y_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                
                preds = (outputs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _evaluate_variable_length(self, split):
        """Variable-length evaluation."""
        if split == 'test':
            dataset = VariableLengthDataset(self.test_data_path, self.max_sequence_length, self.conv_onehot)
        else:
            dataset = VariableLengthDataset(self.valid_data_path, self.max_sequence_length, self.conv_onehot)
        
        # Use same chunk limit as training for consistency
        max_chunks_per_batch = 32  # Conservative default for evaluation
        batch_sampler = ChunkAwareBatchSampler(dataset, max_chunks_per_batch, shuffle=False)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=variable_length_collate_fn)
        criterion = nn.BCELoss(reduction='none')
        
        return self._validate_variable_length(loader, criterion)
    
    def predict(self, dna_sequences, methylation_sequences, batch_size=128, threshold=0.5,parallel_scan=True):
        """
        Predict on arbitrary sequences using the trained model.
        :param dna_sequences: list (or array-like) of DNA strings
        :param methylation_sequences: list (or array-like) of methylation strings ("0"/"1")
        :param batch_size: batch size for inference
        :param threshold: classification threshold for 'positive' label
        :return: predictions as probability and binary label
        """
        self.model.eval()
        
        # 1) Convert to one-hot + methylation
        onehot_data = self.conv_onehot(dna_sequences, methylation_sequences)

        # 2) Create PyTorch dataset and dataloader
        X_tensor = torch.tensor(onehot_data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 3) Perform forward passes in batches
        all_outputs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model.forward(X_batch, parallel_scan=parallel_scan)   # shape: (batch_size, 1)
                # Squeeze to get shape [batch_size]
                outputs = outputs.squeeze(-1).cpu().numpy()
                all_outputs.extend(outputs)
        
        # 4) You can return probabilities or apply a threshold
        all_outputs = np.array(all_outputs)
        predicted_labels = (all_outputs >= threshold).astype(int)
        
        # Returns probabilities and the thresholded labels, 
        return all_outputs, predicted_labels

