#!/bin/env zsh
# Add a 'N' to the end of line if line does not start with '>'

file=$1
while IFS= read line
do
	if [[ "$line" == \>* ]];then
		echo "$line"
	else       
		echo "$line""n"
	fi
done <"$file"