#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from numpy import array
from Bio import SeqIO
from warnings import simplefilter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
# Suppress warnings etc
simplefilter(action = 'ignore', category = FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'


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
  input_features = pad_sequences(input_features, padding = 'post')
  input_features = np.stack(input_features)
  return(input_features)


def flatten_sequence(pred_fasta_flat):
  dimensoes = pred_fasta_flat.shape
  n_samples = dimensoes[0]
  n_x = dimensoes[1]
  n_y = dimensoes[2]
  n_xy = (n_x * n_y)
  pred_fasta_flat = pred_fasta_flat.reshape(n_samples,n_xy)
  return(pred_fasta_flat)


def te_hype(x_train, y_train, x_val, y_val, params):
	model = Sequential()
	model.add(Conv1D(filters=params['first_neuron'], kernel_size=params['kernel_size'], 
	                input_shape=(x_train.shape[1], 1),activation="relu"))
	model.add(MaxPooling1D(pool_size=params['pool_size']))
	model.add(Conv1D(filters=params['second_neuron'], kernel_size=params['kernel_size'],activation="relu"))
	model.add(MaxPooling1D(pool_size=params['pool_size']))
	model.add(Conv1D(filters=params['third_neuron'], kernel_size=params['kernel_size'],activation="relu"))
	model.add(MaxPooling1D(pool_size=params['pool_size']))
	model.add(Flatten())
	model.add(Dense(params['dense01_neuron'], activation="relu"))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss="categorical_crossentropy",
	              optimizer='adam',
	              metrics=['acc'])
	history = model.fit(x_train, y_train, 
	                  epochs=params['epochs'],
	                  batch_size=params['batch_size'],
	                  verbose=1,
	                  validation_split=0.0,
	                  validation_data=(x_val, y_val))	
	return(history, model)