#!bin/bash

appliance='mw'
fold=4

for dataset in 1 2
do
	for cell_type in 'GRU' 'LSTM' 'RNN'
	do
		for hidden_size in 20 50 100
		do
			for num_layers in 1 2 3 4
			do
				for bidirectional in 'True' 'False'
				do
					for lr in 0.01 0.1 1
					do
						for iterations in 1000 2000 3000
						do
							CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
						done
					done
				done
			done
		done
	done
done