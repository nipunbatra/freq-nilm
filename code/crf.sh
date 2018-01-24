#!bin/bash


for lr in [0.001, 0.01, 0.1, 1, 2]:
	for bidirectional in [True, False]:
		echo $lr $bidirectional
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py dw GRU 100 3 $bidirectional $lr 800
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py dw GRU 100 4 $bidirectional $lr 800
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py dr GRU 100 3 $bidirectional $lr 800
		CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cuda_cache' python baseline-rnn-individual-new.py dr GRU 100 4 $bidirectional $lr 800