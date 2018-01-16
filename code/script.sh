#!bin/bash

filename="$1"
while read -r line
do
	name="$line"
	echo $name
	python rnn-pytorch-tree-teacher-Yiling.py 200 200 1 0.6 1 $name
done < "$filename"
