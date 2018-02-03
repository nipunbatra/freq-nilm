#!bin/bash

# CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dr mw hvac fridge dw
# CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dr mw hvac dw fridge
# CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dr mw fridge hvac dw
# CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dr mw fridge dw hvac
# CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dr mw dw hvac fridge

CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 20 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 50 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 100 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 20 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 50 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 100 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 20 4 False 0.1 800 0 hvac
CUDA_VISIBLE_DEVICES=3 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 50 4 False 0.1 800 0 hvac