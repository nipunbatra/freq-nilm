#!bin/bash

appliance='mw'
fold=1

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
							CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
						done
					done
				done
			done
		done
	done
done


# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw hvac dw dr fridge
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw fridge hvac dr dw
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw fridge hvac dw dr
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw fridge dr hvac dw
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw fridge dr dw hvac


# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 20 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 50 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 100 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 20 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 50 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 100 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 20 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 50 2 True 1 2000 0 hvac
# CUDA_VISIBLE_DEVICES=1 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 100 2 True 1 2000 0 hvac