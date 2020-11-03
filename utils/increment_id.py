#!/usr/bin/env python3
# If there is a repeated id, increment name with copy number
import sys
# Get fasta file from user input
arq = sys.argv[1]


def increment_id(fasta_file):
    # Create empty dictionaries
    count_ids = {} # Id counting
    fas_dict = {} # Dictionary for sequences
    # Read fasta file
    with open(fasta_file) as fasta:
        fasta = fasta.readlines()
        for line in fasta:
            line = line.strip()
            if line.startswith('>'):
                f_id = line
                # Count Ids
                if f_id not in count_ids:
                    count_ids[f_id] = 0
                count_ids[f_id] += 1
                # Create Id with copy number
                incr_id = f'{f_id} seq{count_ids[f_id]}'
                # Store new ids in dictionary
                fas_dict[incr_id] = ''
            else:
                # Store sequences in dictionary
                fas_dict[incr_id] += line
        # Print sequences in fasta format
        for k, v in fas_dict.items():
            print(f'{k}\n{v}')


if __name__ == '__main__':
    increment_id(arq)