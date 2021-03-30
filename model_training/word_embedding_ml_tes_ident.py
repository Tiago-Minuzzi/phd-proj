#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from contextlib import redirect_stderr
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import argmax
from numpy import array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Embedding, Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from warnings import simplefilter

# Hide warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
with redirect_stderr(open(os.devnull, "w")):
  from keras.preprocessing.sequence import pad_sequences

# Set seed
SEED = 13
tf.compat.v1.random.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# Files
TDS = 'tds_te_orders_estrat_20210308_v1.csv'
MODELWB = 'model_cnn_embedding_wb.hdf5'
MODELF = 'model_cnn_embedding.hdf5'
MODELHIST = 'model_cnn_embedding_history.pkl'
# TDS values
FIDENT = 'id'
CLASSI = 'order'
TARGET = 'sequence'
# Read training dataset
seq_df = pd.read_csv(TDS, usecols=(FIDENT, CLASSI, TARGET))
# Tokenize sequences
tokenizer = Tokenizer(num_words=None,split=' ', char_level=True, lower=True)
sequencias = seq_df[TARGET]
tokenizer.fit_on_texts(sequencias)
vocab_size_seq = len(tokenizer.word_index) + 1
x_sequence_arrays = tokenizer.texts_to_sequences(sequencias)
# Sequence padding
padded_seqs = pad_sequences(x_sequence_arrays, padding='post')
# Label tokenization
y_str = seq_df[CLASSI]
labtok = Tokenizer()
labtok.fit_on_texts(y_str)
vocab_size = len(labtok.word_index) + 1
toklabs = labtok.texts_to_sequences(y_str)
lbenc = preprocessing.LabelBinarizer()
toklabs = to_categorical(toklabs)
# Train-test split
x_train, x_test, ynn_train, ynn_test = train_test_split(padded_seqs,
                                                        toklabs,
                                                        test_size = 0.20,
                                                        random_state = SEED,
                                                        stratify = toklabs,
                                                        shuffle = True)
# CNN model
## input shape and size
XINP = x_train.shape[1]
## embedding dimensions
EMB_DIM = 12
## CNN hyperparamenters
### kernel and pooling layer
NKERNEL = 12
NPOOL = 7
### conv layers
ICNN = 24
HCNN = 16
### dense layer
DLAY = 16
DROP = 0.4
### ouput layer
NCLASS = vocab_size
### number of epochs and batch size
EPOCH = 15
BSIZE = 25
# Set sequential model
model_var01_05 = Sequential()
# Embedding layer
model_var01_05.add(Embedding(input_dim=vocab_size_seq,
                             output_dim=EMB_DIM,
                             input_length=XINP,
                             mask_zero = False))
# CNN layers
model_var01_05.add(Conv1D(filters=ICNN, kernel_size=NKERNEL,
                          input_shape=(XINP, 1),
                          activation='relu'))
model_var01_05.add(MaxPooling1D(pool_size=NPOOL))
## H1
model_var01_05.add(Conv1D(filters=HCNN, kernel_size=NKERNEL,activation='relu'))
model_var01_05.add(MaxPooling1D(pool_size=NPOOL))
## H2
model_var01_05.add(Conv1D(filters=HCNN, kernel_size=NKERNEL,activation='relu'))
model_var01_05.add(MaxPooling1D(pool_size=NPOOL))
## H3
model_var01_05.add(Conv1D(filters=HCNN, kernel_size=NKERNEL,activation='relu'))
model_var01_05.add(MaxPooling1D(pool_size=NPOOL))
# Flattening
model_var01_05.add(Flatten())
# Dense layer
model_var01_05.add(Dense(DLAY, activation='relu'))
model_var01_05.add(Dropout(DROP))
# Output layer
model_var01_05.add(Dense(NCLASS, activation='softmax'))
# Compiling
model_var01_05.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
# model_var01_05.summary()
# Checkpoint
checkpoint = ModelCheckpoint(MODELWB,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]
# Fit the model
history_var01_05 = model_var01_05.fit(x_train, ynn_train,
                                epochs = EPOCH,
                                batch_size = BSIZE,
                                verbose = 1,
                                callbacks = callbacks_list,
                                validation_split = 0.0,
                                validation_data = (x_test, ynn_test),
                                shuffle = True)
# Save model
model_var01_05.save(MODELF)
# Save history binary
with open(MODELHIST,"wb") as file_pi:
  pickle.dump(history_var01_05.history, file_pi)
