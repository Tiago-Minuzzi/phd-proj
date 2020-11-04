#!/usr/bin/env python3
import sys
from Bio import SeqIO
# Input file
in_fasta = sys.argv[1]


# Get longest sequence to use as padding reference
def get_longest(in_fasta):
	fas_dict = {}
	with open(in_fasta) as fasta, open("ref_len.fasta","w") as out_fasta:
		for record in SeqIO.parse(in_fasta, 'fasta'):
			name = record.description
			sequence = str(record.seq.lower())
			fas_dict[name] = sequence
		seq_len, seq_id, seq = max((len(v), k, v) for k, v in fas_dict.items())
		out_fasta.write(f'>{seq_id}\n{seq}')
		print(f"Longest sequence is:\nID: {seq_id}\nLength: {seq_len}")


if __name__ == '__main__':
	get_longest(in_fasta)