from Bio import SeqIO
seqs_by_ids = {}
nts = "rykmswbdhvn"
with open("/home/tiago/db_TE.fasta", "r") as fasta, open("db_te_valid.fasta", "w") as te_fastainv_fasta, open("db_te_invalid.fasta", "w") as inv_fasta:
    for record in SeqIO.parse(fasta, "fasta"):
        sid = record.description
        sseq = record.seq
        if not any(nt in sseq for nt in nts):
            te_fastainv_fasta.write(f'>{sid}\n{sseq}\n')
        else:
            inv_fasta.write(f'>{sid}\n{sseq}\n')