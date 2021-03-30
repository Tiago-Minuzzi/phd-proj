#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from numpy import array
from numpy import argmax
from warnings import simplefilter
from contextlib import redirect_stderr
from keras.preprocessing.text import Tokenizer

# Hide warning messages
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with redirect_stderr(open(os.devnull, "w")):
  from tensorflow.keras.models import load_model
  from keras.preprocessing.sequence import pad_sequences
# Show full array
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)
# IO files
INFASTA = sys.argv[1]
RESCSV = sys.argv[2] if len(sys.argv) >=3 else None
# Get model
MODELO = 'model_embedded_order_wb.hdf5'
PADVALUE = 38797


def fasta_frame(fasta_file):
    fids = []
    fseq = []
    with open(fasta_file) as fasta:
        for record in SeqIO.parse(fasta, 'fasta'):
            fids.append(record.id)
            fseq.append(str(record.seq).lower())
    s1 = pd.Series(fids, name = 'id')
    s2 = pd.Series(fseq, name = 'sequence')
    data = {'id':s1, 'sequence':s2}
    df = pd.concat(data, axis=1)
    return df


# Read fasta as dataframe
fas_df = fasta_frame(INFASTA)
identifiers = fas_df['id']
sequences = fas_df['sequence']
# Labels
te_labels = {'te': 1, 'nt': 2}
# Tokenize sequences
tkz_seq = Tokenizer(num_words = None, split = ' ', char_level = True, lower = True)
tkz_seq.fit_on_texts(sequences)
x_seq_arrays = tkz_seq.texts_to_sequences(sequences)
vocab_size_seq = len(tkz_seq.word_index) + 1
# Pad sequences
padded_seqs = pad_sequences(x_seq_arrays, padding='post', maxlen = PADVALUE)
# Load model
modelo = load_model(MODELO)
# Predict labels
pred_labels = modelo.predict_classes(padded_seqs, batch_size = 2)
mapped_labels = [k for label in pred_labels for k, v in te_labels.items() if v == label]
# Results
mapped_series = pd.Series(mapped_labels)
results_dict = {"id": identifiers, "classification": mapped_series}
results_df = pd.DataFrame(results_dict)
# Results to stdout
print('\n')
print('# RESULTS')
print(results_df)
print('\n')
# Write results to file
if RESCSV:
    results_df.to_csv(RESCSV, index = False)
    print(f"Results saved as {RESCSV}.")
else:
    print("No ouput file created")
