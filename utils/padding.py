#!/usr/bin/env python3
# Pad sequences
import sys
in_fasta = sys.argv[1]
out_fasta = sys.argv[2]

def padding(in_fasta, out_fasta):
# Pad sequences to the leght of the longest sequence
    maxLen=len(max(open(in_fasta),key=len).strip())
    # Pad to the end
    with open(in_fasta, 'r') as fasta, open(out_fasta,"w") as pout:
        for line in fasta:
            line = line.strip()
            if not line.startswith('>'):
                line = line.upper().ljust(maxLen, 'P') + 'TTTAAAGGGCCCNNN'
            pout.write(line+"\n")
            # print(line)
        print(f'Longest sequence is {maxLen} nucleotides.')

padding(in_fasta, out_fasta)