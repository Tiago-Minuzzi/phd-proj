#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from numpy import save
from numpy import array
from numpy import argmax
from warnings import simplefilter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Fasta file to dataframe
def fasta_frame(fasta_file, label):
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


# Sequence padding
def padding(dataframe, padded):
    # Get max len
    df = pd.read_csv(dataframe)
    max_len = df[df.columns[1]].map(len).max()
    # Pad sequences to max len
    with open(padded,'w') as padded:
        for line in df[df.columns[1]]:
            line = line.strip()
            line = line.lower().ljust(max_len, 'p') + 'acgtn'
            padded.write(line + "\n")


# Sequence one-hot encoding
def ohe(padded_file): # text file containing only padded sequences, no identifier
    integer_encoder = LabelEncoder()  
    one_hot_encoder = OneHotEncoder(categories = 'auto')   
    # Create folder to store npy files
    if not os.path.exists('np_arrays'):
        os.mkdir('np_arrays')
    # Open file with padded sequences
    with open(padded_file, 'r') as padded:
        for i, linha in enumerate(padded):
            arr_file = os.path.join('np_arrays', f'arr_{i}.npy')
            linha = linha.strip()
            integer_encoded = integer_encoder.fit_transform(list(linha))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            stacked = np.stack(one_hot_encoded.toarray())
            save(arr_file, stacked)