#!bin/bash

appliance='mw hvac'
fold=1

for dataset in 1
do
	for cell_type in 'GRU' 'LSTM' 'RNN'
	do
		for hidden_size in 20 50 100
		do
			for num_layers in 1 2 3
			do
				for bidirectional in 'True'
				do
					for lr in 0.01 0.1 1
					do
						for iterations in 3000
						do
						    echo $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
							CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv-new.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
						done
					done
				done
			done
		done
	done
done

appliance='dw hvac'
fold = 3
for dataset in 1
do
	for cell_type in 'GRU' 'LSTM' 'RNN'
	do
		for hidden_size in 20 50 100
		do
			for num_layers in 1 2 3
			do
				for bidirectional in 'True'
				do
					for lr in 0.01 0.1 1
					do
						for iterations in 3000
						do
						    echo $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
							CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv-new.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
						done
					done
				done
			done
		done
	done
done

fold = 0
for dataset in 2
do
	for cell_type in 'GRU' 'LSTM' 'RNN'
	do
		for hidden_size in 20 50 100
		do
			for num_layers in 1 2 3
			do
				for bidirectional in 'True'
				do
					for lr in 0.01 0.1 1
					do
						for iterations in 3000
						do
						    echo $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
							CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv-new.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
						done
					done
				done
			done
		done
	done
done