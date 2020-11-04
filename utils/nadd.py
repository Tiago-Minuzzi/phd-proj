#!/usr/bin/env python3
import sys
in_fasta = sys.argv[1]
out_fasta = sys.argv[2]


def nadd(in_fasta,out_fasta):
    with open(in_fasta, 'r') as fasta, open(out_fasta,"w") as fas_out:
        for line in fasta:
            line = line.strip()
            if not line.startswith('>'):
                line = 'nnnnnnnnn' + line.lower()
            fas_out.write(line + "\n")


if __name__ == '__main__':
	nadd(in_fasta, out_fasta)
