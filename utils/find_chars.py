#!/usr/bin/env python3
# coding: utf-8

from Bio import SeqIO
import pandas as pd
import sys

with open(sys.argv[1]) as fasta_file:  # Will close handle cleanly
    identifiers = []
    sequences = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        identifiers.append(seq_record.id)
        sequences.append(str(seq_record.seq).lower())

dict_seqs=dict(zip(identifiers,sequences))
unwt = ['r','y','k','m','s','w','b','d','h','v','n']

with open('outfile.txt','w') as outfile:
    for k, v in dict_seqs.items():
        for char in unwt:
            if char in v:
#                 print(k)
                outfile.write('{}\n'.format(k))
