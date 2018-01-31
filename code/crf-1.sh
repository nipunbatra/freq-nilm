#!bin/bash

# svm 

CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 2 True 1 400 0
CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 2 True 1 800 0

CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 4 True 1 400 0
CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 4 True 1 800 0

#!bin/bash

# svm 

CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 2 False 1 400 0
CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 2 False 1 800 0

CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 4 False 1 400 0
CUDA_VISIBLE_DEVICES=1 python rnn-tree-greedy.py 2 LSTM 50 4 False 1 800 0