import numpy as np
import pandas as pd
import os
import os.path
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras import optimizers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from methyldl.modelling.minirnns.minRNNs import BiMinGRU



# class Dismir_TF:
#     def __init__(self, max_sequence_length, train_data_path, test_data_path, valid_data_path):
#         self.max_sequence_length = max_sequence_length
#         self._init_model()
#         self._initialize_inputs(train_data_path, test_data_path, valid_data_path)

#     def _init_model(self):
#         model = Sequential()
#         model.add(layers.Convolution1D(input_shape=(self.max_sequence_length, 5),
#                                     filters=100,
#                                     kernel_size=10,
#                                     padding="same",
#                                     activation="relu"
#                                     ))
#         model.add(layers.MaxPooling1D(pool_size=2, strides=2))
#         model.add(layers.Dropout(0.2))
#         model.add(layers.Bidirectional(layers.LSTM(int(self.max_sequence_length/2), return_sequences=True)))
#         model.add(layers.Convolution1D(input_shape=(int(self.max_sequence_length/2), 132),
#                                     filters=100,
#                                     kernel_size=3,
#                                     padding="same",
#                                     activation="relu"
#                                     ))
#         model.add(layers.MaxPooling1D(pool_size=2, strides=2))
#         model.add(layers.Dropout(0.2))
#         model.add(layers.Flatten())
#         model.add(layers.Dense(750, activation='relu', kernel_regularizer=None, bias_regularizer=None))
#         model.add(layers.Dropout(0.2))
#         model.add(layers.Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None))
#         model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None))
#         sgd = optimizers.SGD(learning_rate=0.05, weight_decay=1e-6, momentum=0.9, nesterov=True)
#         model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
#         self.model = model

#     # transform sequence into one-hot code (0/1/2/3 to one-hot) and add methylation state channel
#     def conv_onehot(self,dna_seq, c_methylation_seq):
#         module = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
#         onehot = np.zeros((len(dna_seq), self.max_sequence_length, 5), dtype='int')
#         for i in range(len(dna_seq)):
#             tmp, tmp_methylation_seq = dna_seq[i], c_methylation_seq[i]
#             tmp_onehot = np.zeros((self.max_sequence_length, 5), dtype='int')
#             for j in range(len(tmp)):
#                 if tmp_methylation_seq[j] == "1":
#                     tmp_onehot[j] = module[4]
#                 elif tmp[j] == "A":
#                     tmp_onehot[j] = module[0]
#                 elif tmp[j] == "T":
#                     tmp_onehot[j] = module[1]
#                 elif tmp[j] == "C":
#                     tmp_onehot[j] = module[2]
#                 elif tmp[j] == "G":
#                     tmp_onehot[j] = module[3]

#             onehot[i] = tmp_onehot
#         return onehot
    
#     def load_and_transform_input(self, data_path):
#         data = pd.read_csv(data_path)
#         dna, methylation, labels = data["input_ids"],data["methylation_ids"], data["label"]
#         features = self.conv_onehot(dna, methylation)
#         return(features, labels)

    
#     def _initialize_inputs(self, train_data_path, test_data_path, valid_data_path):
#         self.train_x, self.train_y = self.load_and_transform_input(train_data_path)
#         self.validation_data = self.load_and_transform_input(valid_data_path)
#         self.test_data = self.load_and_transform_input(test_data_path)

#     def train(self,train_dir,verbose,epochs,batch_size):
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#         history = self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, validation_data=self.validation_data,
#                         callbacks=[EarlyStopping(patience=10), ModelCheckpoint(filepath=train_dir + 'weight.h5', save_best_only=True)],
#                         shuffle=True, verbose=verbose)
        




class DISMIRNet(nn.Module):
    """
    PyTorch model mirroring the structure of the Keras model:
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
    def __init__(self, max_sequence_length, flavor ="lstm"):
        super(DISMIRNet, self).__init__()
        self.max_sequence_length = max_sequence_length
        
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
            rnn = BiMinGRU(input_dim=100,hidden_dim=max_sequence_length//2,batch_first=True,use_init_hidden_state=False)

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
        self.fc3 = nn.Linear(300, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
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
        x = self.fc3(x)       # (batch, 1)
        x = self.sigmoid(x)
        return x


class Dismir:
    """
    Equivalent class to the Keras-based Dismir, but using PyTorch internally.
    1. Data loading and transformation (one-hot + methylation channel)
    2. Model creation (DISMIRNet)
    3. Training loop with a simplistic early-stopping approach
    """
    def __init__(self, max_sequence_length, train_data_path, test_data_path, valid_data_path, device=None, flavour="lstm"):
        self.max_sequence_length = max_sequence_length
        
        # Use CUDA if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize the PyTorch model
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.valid_data_path = valid_data_path
        
        self.model = DISMIRNet(max_sequence_length, flavour).to(self.device)
        

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
            for j in range(len(tmp_seq)):
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
        data = pd.read_csv(data_path)
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
              nesterov=True):
        """
        Train loop with early stopping & checkpointing.
        
        :param optimizer_type: "SGD" or "Adam"
        :param lr: Learning rate
        :param weight_decay: L2 regularization coefficient
        :param momentum: (only used if optimizer_type="SGD")
        :param nesterov: (only used if optimizer_type="SGD")
        """

        # Load data
        self.train_x, self.train_y = self.load_and_transform_input(self.train_data_path)
        self.valid_x, self.valid_y = self.load_and_transform_input(self.valid_data_path)
        self.test_x,  self.test_y  = self.load_and_transform_input(self.test_data_path)
        
        # Convert to torch.Tensor
        self.train_x = torch.tensor(self.train_x, dtype=torch.float32)
        self.train_y = torch.tensor(self.train_y.values, dtype=torch.float32).view(-1, 1)
        self.valid_x = torch.tensor(self.valid_x, dtype=torch.float32)
        self.valid_y = torch.tensor(self.valid_y.values, dtype=torch.float32).view(-1, 1)
        self.test_x  = torch.tensor(self.test_x,  dtype=torch.float32)
        self.test_y  = torch.tensor(self.test_y.values,  dtype=torch.float32).view(-1, 1)
        
        # Create the optimizer based on user choice
        if optimizer_type.upper() == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )
        elif optimizer_type.upper() == "ADAM":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError("optimizer_type must be 'SGD' or 'Adam'")
        
        criterion = nn.BCELoss()

        # DataLoaders
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        valid_dataset = torch.utils.data.TensorDataset(self.valid_x, self.valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs+1):
            # ---- Training ----
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
                preds = (outputs.detach() >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # ---- Validation ----
            self.model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_val, y_val in valid_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_outputs = self.model(X_val)
                    v_loss = criterion(val_outputs, y_val)
                    
                    val_loss += v_loss.item() * X_val.size(0)
                    
                    val_preds = (val_outputs >= 0.5).float()
                    val_correct += (val_preds == y_val).sum().item()
                    val_total += y_val.size(0)
            
            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / val_total
            
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

    def evaluate(self, split='test'):
        """
        Evaluate on the test or validation split. 
        Returns (loss, accuracy).
        """
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
    
    def predict(self, dna_sequences, methylation_sequences, batch_size=128, threshold=0.5):
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
                outputs = self.model(X_batch)   # shape: (batch_size, 1)
                # Squeeze to get shape [batch_size]
                outputs = outputs.squeeze(-1).cpu().numpy()
                all_outputs.extend(outputs)
        
        # 4) You can return probabilities or apply a threshold
        all_outputs = np.array(all_outputs)
        predicted_labels = (all_outputs >= threshold).astype(int)
        
        # Returns probabilities and the thresholded labels, 
        return all_outputs, predicted_labels

