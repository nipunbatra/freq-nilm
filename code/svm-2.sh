#!bin/bash

# svm gpu2

CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 2 True 1 400 0
CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 2 True 1 800 0

CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 4 True 1 400 0
CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 4 True 1 800 0

CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 2 False 1 400 0
CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 2 False 1 800 0

CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 4 False 1 400 0
CUDA_VISIBLE_DEVICES=2 python rnn-tree-greedy.py 2 GRU 100 4 False 1 800 0