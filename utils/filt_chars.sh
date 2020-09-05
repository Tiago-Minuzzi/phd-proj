#!/usr/bin/env bash

FILE=$1
python3 find_chars.py $FILE
uniq outfile.txt > tmp.ids.txt &&\
rm outfile.txt &&\
grep -A1 -f tmp.ids.txt $FILE > tmp.fasta &&\
rm tmp.ids.txt &&\
grep -v -f tmp.fasta $FILE > filtered.$FILE &&\
rm tmp.fasta &&\
echo $(grep -c '>' filtered.$FILE $FILE)
