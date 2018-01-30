#!bin/bash

# svm 

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 2 True 1 200 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 2 True 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 2 True 1 600 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 2 True 1 800 0

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 4 True 1 200 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 4 True 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 4 True 1 600 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py RNN 100 4 True 1 800 0