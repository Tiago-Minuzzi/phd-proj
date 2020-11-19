#!/usr/bin/env python3
# Imports
import os
import talos as ta
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
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
simplefilter(action = 'ignore', category = FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# Files
dataset = 'nadd_ml_df_train_set_00.01.csv'
# Set seed for model reproducibility
SEED = 13
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
	history = model.fit(x_train_3d, ynn_train, 
	                  epochs=params['epochs'],
	                  batch_size=params['batch_size'],
	                  verbose=1,
	                  validation_split=0.0,
	                  validation_data=(x_test_3d,ynn_test))
	
	return(history, model)


# SEED for model reproducibility
np.random.seed(SEED)
if tensorflow.__version__ == '2.3.1':
  tensorflow.random.set_seed(SEED)
else:
  #tensorflow.random.set_random_seed(SEED)
  tensorflow.compat.v1.random.set_random_seed(SEED)
# Import from csv
seq_df = pd.read_csv(dataset)
# Preprocessing
## Sequences
x_sequence_arrays = ohe_fun(seq_df)
x_flat_2d = flatten_sequence(x_sequence_arrays)
## Labels
y_str=seq_df['label']
lbenc = preprocessing.LabelBinarizer()
ynn = lbenc.fit_transform(y_str)
encoded = to_categorical(ynn)
# Train and test splits
x_train, x_test, ynn_train, ynn_test = train_test_split(x_flat_2d, 
                                                    encoded,
                                                    test_size = 0.20, 
                                                    random_state = SEED,
                                                    stratify = y_str)
# Expand dimensions
x_train_3d = np.expand_dims(x_train, axis=2)
x_test_3d = np.expand_dims(x_test, axis=2)
# Hyperparameter testing
p = {'lr': (0.5, 5, 10),
     'first_neuron': [32],
     'second_neuron': [24],
     'third_neuron': [24],
     'dense01_neuron': [24],
     'kernel_size': [7,8,9,12],
     'pool_size': [3,4,6,7],
     'batch_size': [10,25],
     'epochs': [15,20]}
# Run scanning
teste = ta.Scan(x=x_train_3d,
            y=ynn,
            model=te_hype,
            fraction_limit=0.01, 
            params=p,
            experiment_name='te_identification')
