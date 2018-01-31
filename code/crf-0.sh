#!bin/bash

# svm 

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 2 True 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 2 True 1 800 0

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 4 True 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 4 True 1 800 0


#!bin/bash

# svm 

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 2 False 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 2 False 1 800 0

CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 4 False 1 400 0
CUDA_VISIBLE_DEVICES=0 python rnn-tree-greedy.py 2 LSTM 20 4 False 1 800 0