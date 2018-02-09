#!bin/bash

for dataset in 2
do
	for fold_num in 3
	do
		for num_latent in {1..50}
		do
			for num_iters in {10..100..10}	
			do
				echo $dataset $fold_num $num_latent $num_iters
				python baseline-sc-with-disc-nested.py $dataset $fold_num $num_latent $num_iters
			done
		done
	done
done
