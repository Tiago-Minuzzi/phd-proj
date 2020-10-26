#!/usr/bin/env python3
# Pad sequences
import sys
in_fasta = sys.argv[1]
# Pad sequences to the leght of the longest sequence
maxLen=len(max(open(in_fasta),key=len).strip())
# Pad to the end
with open(in_fasta, 'r') as fasta, open("pad_out.fasta","w") as pout:
    for line in fasta:
        line = line.strip()
        if not line.startswith('>'):
            line = line.upper().ljust(maxLen, 'P') + 'TTTAAAGGGCCCNNN'
        pout.write(line+"\n")
        print(line)
    print(f'Longest sequence is {maxLen} nucleotides.')