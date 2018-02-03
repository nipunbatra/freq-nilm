#!bin/bash

# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw dw fridge hvac dr
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw dw fridge dr hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw dw dr hvac fridge
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 1 5000 0 mw dw dr fridge hvac

CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 20 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 50 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 100 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 20 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 50 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 100 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 20 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 50 1 True 1 800 0 hvac
CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 100 1 True 1 800 0 hvac