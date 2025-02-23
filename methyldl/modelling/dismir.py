import numpy as np
import pandas as pd
import random
import gc
import os
import os.path
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf



class Dismir():
    def __init__(self, max_sequence_length, train_data_path, test_data_path, valid_data_path):
        self.max_sequence_length = max_sequence_length
        self._init_model()
        self._initialize_inputs(train_data_path, test_data_path, valid_data_path)

    def _init_model(self):
        model = Sequential()
        model.add(layers.Convolution1D(input_shape=(self.max_sequence_length, 5),
                                    filters=100,
                                    kernel_size=10,
                                    padding="same",
                                    activation="relu"
                                    ))
        model.add(layers.MaxPooling1D(pool_size=2, strides=2))
        model.add(layers.Dropout(0.2))
        model.add(layers.Bidirectional(layers.LSTM(int(self.max_sequence_length/2), return_sequences=True)))
        model.add(layers.Convolution1D(input_shape=(int(self.max_sequence_length/2), 132),
                                    filters=100,
                                    kernel_size=3,
                                    padding="same",
                                    activation="relu"
                                    ))
        model.add(layers.MaxPooling1D(pool_size=2, strides=2))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(750, activation='relu', kernel_regularizer=None, bias_regularizer=None))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None))
        model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None))
        sgd = optimizers.SGD(learning_rate=0.05, weight_decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    # transform sequence into one-hot code (0/1/2/3 to one-hot) and add methylation state channel
    def conv_onehot(self,dna_seq, c_methylation_seq):
        module = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
        onehot = np.zeros((len(dna_seq), self.max_sequence_length, 5), dtype='int')
        for i in range(len(dna_seq)):
            tmp, tmp_methylation_seq = dna_seq[i], c_methylation_seq[i]
            tmp_onehot = np.zeros((self.max_sequence_length, 5), dtype='int')
            for j in range(len(tmp)):
                if tmp_methylation_seq[j] == "1":
                    tmp_onehot[j] = module[4]
                elif tmp[j] == "A":
                    tmp_onehot[j] = module[0]
                elif tmp[j] == "T":
                    tmp_onehot[j] = module[1]
                elif tmp[j] == "C":
                    tmp_onehot[j] = module[2]
                elif tmp[j] == "G":
                    tmp_onehot[j] = module[3]

            onehot[i] = tmp_onehot
        return onehot
    
    def load_and_transform_input(self, data_path):
        data = pd.read_csv(data_path)
        dna, methylation, labels = data["input_ids"],data["methylation_ids"], data["label"]
        features = self.conv_onehot(dna, methylation)
        return(features, labels)

    
    def _initialize_inputs(self, train_data_path, test_data_path, valid_data_path):
        self.train_x, self.train_y = self.load_and_transform_input(train_data_path)
        self.validation_data = self.load_and_transform_input(valid_data_path)
        self.test_data = self.load_and_transform_input(test_data_path)