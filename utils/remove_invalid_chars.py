#!/usr/bin/env python3

import sys
from Bio import SeqIO
# User input
in_fasta = sys.argv[1]
# Empty dictionary to store sequences
seqs_by_ids = {}
# Invalid characters
nts = "rykmswbdhvn"
# Open fasta and write to files
with open(in_fasta, "r") as fasta, open(f"valid.fasta", "w") as val_fasta, open("invalid.fasta", "w") as inv_fasta:
    for record in SeqIO.parse(fasta, "fasta"):
        sid = record.description
        sseq = record.seq.lower()
        if not any(nt in sseq for nt in nts):
            val_fasta.write(f'>{sid}\n{sseq}\n')
        else:
            inv_fasta.write(f'>{sid}\n{sseq}\n')