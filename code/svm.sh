#!bin/bash

# for Fridge and mw

for lr in [0.001, 0.01, 0.1, 1, 2]:
	for bidirectional in [True, False]:
		echo $lr $bidirectional
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py mw GRU 100 3 $bidirectional $lr 800
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py mw GRU 100 4 $bidirectional $lr 800
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py fridge LSTM 100 4 $bidirectional $lr 800
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py dw LSTM 100 4 $bidirectional $lr 800
