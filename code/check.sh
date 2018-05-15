#!bin/bash
while true
do
	now=$(date)
	echo $now
	python check.py
	sleep 200
done
