#!bin/bash


CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 dw dr fridge hvac mw
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 dw dr fridge mw hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 dw dr mw hvac fridge
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 dw dr mw fridge hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 dw mw hvac fridge dr
