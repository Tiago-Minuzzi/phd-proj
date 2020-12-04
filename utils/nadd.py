#!/usr/bin/env python3
import sys
in_fasta = sys.argv[1]
out_fasta = sys.argv[2]
ntail = 'n'*6


def nadd(in_fasta,out_fasta):
    with open(in_fasta, 'r') as fasta, open(out_fasta,"w") as fas_out:
        for line in fasta:
            line = line.strip()
            if not line.startswith('>'):
                line = line.lower() + ntail
            fas_out.write(line + "\n")


if __name__ == '__main__':
	nadd(in_fasta, out_fasta)
