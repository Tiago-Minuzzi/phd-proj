#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import os
import pickle
import pandas as pd
from warnings import simplefilter
from model_funs import fasta_frame, ohe_fun, flatten_sequence
import numpy as np
from numpy import array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.util import deprecation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
# Suppress warnings 
deprecation._PRINT_DEPRECATION_WARNINGS = False
simplefilter(action = 'ignore', category = FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Files
dataset = 'nadd_ml_df_train_set_00.01.csv'
model_wb = "model_nadd_cnn_lstm_wb.hdf5"
model_final = "model_nadd_cnn_lstm.hdf5"
model_history = "model_nadd_cnn_lstm_saved_history"
model_plot = 'model_nadd_cnn_lstm.png'
# Set seed for model reproducibility
SEED = 13
tf.compat.v1.random.set_random_seed(SEED)
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
# CNN + LSTM model
model_cnn_lstm = Sequential()
# CNN layers
## Input and first hidden layer
model_cnn_lstm.add(Conv1D(filters=32,
                    kernel_size=8,
                    input_shape=(x_train.shape[1], 1),
                    activation='relu'))
model_cnn_lstm.add(MaxPooling1D(pool_size=4))
## Second hidden layer
model_cnn_lstm.add(Conv1D(filters=64,
                    kernel_size=8,
                    activation='relu'))
model_cnn_lstm.add(MaxPooling1D(pool_size=4))
# LSTM layer
## Third hidden layer
model_cnn_lstm.add(LSTM(units = 64,
					return_sequences = True))
# Flatten data for dense layers
model_cnn_lstm.add(Flatten())
# Dense layers
## Fourth hidden layer
model_cnn_lstm.add(Dense(24, activation='relu'))
model_cnn_lstm.add(Dropout(0.4))
## Output layer
model_cnn_lstm.add(Dense(2, activation='softmax'))
# Compile model
model_cnn_lstm.compile(loss='categorical_crossentropy',
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
history_cnn_lstm = model_cnn_lstm.fit(x_train_3d, ynn_train,
                            epochs = 15,
                            batch_size = 25,
                            verbose = 1,
                            callbacks = callbacks_list,
                            validation_split = 0.0,
                            validation_data = (x_test_3d,ynn_test),
                            shuffle = True)
# Save model
model_cnn_lstm.save(model_final)
# Save history
with open(model_history,"wb") as file_pi:
  pickle.dump(history_cnn_lstm.history, file_pi)
# Plot model
tf.keras.utils.plot_model(model_cnn_lstm,
                          show_shapes = True,
                          show_layer_names = True,
                          to_file = model_plot)
