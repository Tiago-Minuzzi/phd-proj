#!/usr/bin/env bash

grep ',RD' $1 > nt_seqs.csv
grep ',TE' $1 > te_seqs.csv

shuf -n 4000 nt_seqs.csv > nt.4k.sample.csv
shuf -n 4000 te_seqs.csv > te.4k.sample.csv
cat *.4k.sample.csv > tds01.01.8k.csv &&\
rm *.4k.sample.csv

shuf -n 5000 nt_seqs.csv > nt.5k.sample.csv
shuf -n 5000 te_seqs.csv > te.5k.sample.csv
cat *.5k.sample.csv > tds01.01.10k.csv &&\
rm *.5k.sample.csv

shuf -n 6000 nt_seqs.csv > nt.6k.sample.csv
shuf -n 6000 te_seqs.csv > te.6k.sample.csv
cat *.6k.sample.csv > tds01.01.12k.csv &&\
rm *.6k.sample.csv &&\

rm *_seqs.csv

