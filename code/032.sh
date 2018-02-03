#!bin/bash


# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 3 True 0.1 5000 0 dw dr fridge hvac mw
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 3 True 0.1 5000 0 dw dr fridge mw hvac
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 3 True 0.1 5000 0 dw dr mw hvac fridge
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 3 True 0.1 5000 0 dw dr mw fridge hvac
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 3 True 0.1 5000 0 dw mw hvac fridge dr

CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 20 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 50 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 100 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 20 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 50 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 100 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 20 3 True 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 50 3 True 0.1 2000 0 hvac
