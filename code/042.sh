#!bin/bash

# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dw fridge dr mw hvac
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dw fridge mw hvac dr
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dw fridge mw dr hvac
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dw dr hvac fridge mw
# CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 False 0.1 5000 0 dw dr hvac mw fridge



CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 20 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 50 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 100 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 20 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 50 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 100 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 20 3 False 0.1 2000 0 hvac
CUDA_VISIBLE_DEVICES=2 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 50 3 False 0.1 2000 0 hvac