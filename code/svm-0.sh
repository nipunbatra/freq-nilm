#!bin/bash

# svm gpu0

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 2 True 1 200 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 2 True 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 2 True 1 600 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 2 True 1 800 0

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 4 True 1 200 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 4 True 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 4 True 1 600 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py GRU 20 4 True 1 800 0