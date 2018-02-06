#!bin/bash

for dataset in 1 2
do
	for fold_num in 0 1 2 3 4
	do
		for num_latent in {1..50}
		do
			python baseline-sc-without-disc-nested.py $dataset $fold_num $num_latent
			for num_iters in {10..100..10}
			do
				python baseline-sc-with-disc-nested.py $dataset $fold_num $num_latent $num_iters
			done
		done
	done
done