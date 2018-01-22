#!bin/bash

for appliance in 'dr'
do
	echo $appliance
	for cell_type in 'GRU' 'LSTM' 'RNN'
	do
		for hidden_size in 20 50 100 150
		do
			for num_layers in 1 2 3 4
			do 
				for bidirectional in 'True' 'False'
				do
					for lr in 0.001 0.01 0.1 1 2
					do
						for num_iterations in 200 400 600 800
						do
							echo $appliance $cell_type $hidden_size $num_layers $bidirectional $lr $num_iterations
							CUDA_VISIBLE_DEVICES=1 python baseline-rnn-individual.py $appliance $cell_type $hidden_size $num_layers $bidirectional $lr $num_iterations
						done
					done
				done
			done
		done
	done
done


