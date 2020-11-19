#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""# Imports"""

import os
import pickle
import pandas as pd
from Bio import SeqIO
from warnings import simplefilter
import numpy as np
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# print(tf.__version__)
# Files
dataset = 'nadd_ml_df_train_set_00.01.csv'
model_wb = "model_nadd_cnn_wb.hdf5"
model_final = "model_nadd_cnn.hdf5"
model_history = "model_nadd_cnn_saved_history"
model_plot = 'model_nadd_cnn.png'

# Functions
def fasta_frame(fasta_file,label):
  identifiers = []
  sequences = []
  with open(fasta_file) as f_f:
    for seq_record in SeqIO.parse(f_f, 'fasta'):
        identifiers.append(seq_record.id)
        sequences.append(str(seq_record.seq.lower()))
  s1 = pd.Series(identifiers, name = 'ID')
  s2 = pd.Series(sequences, name = 'sequence')
  fasta_frame = pd.DataFrame(dict(ID = s1, sequence = s2))
  fasta_frame['label'] = label
  return(fasta_frame)

def ohe_fun(coluna):
  integer_encoder = LabelEncoder()  
  one_hot_encoder = OneHotEncoder(categories='auto')   
  input_features = []

  for linha in coluna[coluna.columns[1]]:
    integer_encoded = integer_encoder.fit_transform(list(linha))
    integer_encoded = np.array(integer_encoded).reshape(-1, 1)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    input_features.append(one_hot_encoded.toarray())
  input_features=pad_sequences(input_features, padding='post')
  input_features = np.stack(input_features)
  return(input_features)

def flatten_sequence(pred_fasta_flat):
  dimensoes=pred_fasta_flat.shape
  n_samples=dimensoes[0]
  n_x=dimensoes[1]
  n_y=dimensoes[2]
  n_xy=(n_x * n_y)
  pred_fasta_flat=pred_fasta_flat.reshape(n_samples,n_xy)
  return(pred_fasta_flat)


# Seeds for model replication
SEED = 13
tf.random.set_random_seed(SEED)
np.random.seed(SEED)
# Load saved dataframe
seq_df = pd.read_csv(dataset)
# Transform sequences and labels
## Sequences
x_sequence_arrays = ohe_fun(seq_df)
x_flat_2d = flatten_sequence(x_sequence_arrays)
## Labels
y_str = seq_df['label']
lbenc = preprocessing.LabelBinarizer()
ynn = lbenc.fit_transform(y_str)
encoded = to_categorical(ynn)
# Split dataset in training and test sets
x_train, x_test, ynn_train, ynn_test = train_test_split(x_flat_2d,
                                                        encoded,
                                                        test_size = 0.20,
                                                        random_state = SEED,
                                                        stratify = y_str)
# Expand dimensions for deep learning model
x_train_3d = np.expand_dims(x_train, axis=2)
x_test_3d = np.expand_dims(x_test, axis=2)
# np.random.seed(SEED)
# tf.random.set_random_seed(SEED)
# CNN model
model_cnn = Sequential()
# CNN layers
## Input and first hidden layer
model_cnn.add(Conv1D(filters=32,
                    kernel_size=8,
                    input_shape=(x_train.shape[1], 1),
                    activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=4))
## Second hidden layer
model_cnn.add(Conv1D(filters=24,
                    kernel_size=8,
                    activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=4))
## Third hidden layer
model_cnn.add(Conv1D(filters=24,
                    kernel_size=8,
                    activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=4))
# Flatten data for dense layers
model_cnn.add(Flatten())
# Dense layers
## Fourth hidden layer
model_cnn.add(Dense(24, activation='relu'))
model_cnn.add(Dropout(0.4))
## Output layer
model_cnn.add(Dense(2, activation='softmax'))
# Compile model
model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
# Checkpoint best weights
filepath = model_wb
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]
# Fit the model
history_cnn = model_cnn.fit(x_train_3d, ynn_train,
                            epochs = 15,
                            batch_size = 10,
                            verbose = 1,
                            callbacks = callbacks_list,
                            validation_split = 0.0,
                            validation_data = (x_test_3d,ynn_test),
                            shuffle = True)
# Save model
model_cnn.save(model_final)
# Save history
with open(model_history,"wb") as file_pi:
  pickle.dump(history_cnn.history, file_pi)
# Plot model
tf.keras.utils.plot_model(model_cnn,
                          show_shapes = True,
                          show_layer_names = True,
                          to_file = model_plot)