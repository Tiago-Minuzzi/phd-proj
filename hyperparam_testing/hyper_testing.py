#!/usr/bin/env python3
# Imports
import os
import talos as ta
import pandas as pd
import numpy as np
from warnings import simplefilter
import tensorflow
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from hyper_fun import fasta_frame, ohe_fun, flatten_sequence, te_hype
# Suppress warnings etc
simplefilter(action = 'ignore', category = FutureWarning)
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Files
dataset = f'../tables_etc/ml_df_train_set_00.01.csv'
# dataset = '/home/tiago/Desktop/training_step/mini_test.csv'
# Hyperparameter settings
experiment_name = 'te_identification'
parameters = {'lr': [0.5, 5, 10],
              'hidden_layers': [0,1,2,3],
              'first_neuron': [16,24,32],
              'hidden_neuron': [16,24,32],
              'kernel_size': [7,8,9,12],
              'pool_size': [3,4,6,7],
              #'third_neuron': [24],
              'dense_neuron': [16,24],
              'dropout': [0,0.2,0.4],
              'batch_size': [10,25],
              'epochs': [15,20]}
# Set seed for model reproducibility
SEED = 13
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
# Run scanning
teste = ta.Scan(x=x_train_3d,
                y=ynn_train,
                x_val=x_test_3d,
                y_val=ynn_test,
                model=te_hype,
                fraction_limit=None,
                round_limit=50
                params=parameters,
                experiment_name=experiment_name,
                seed=SEED)
