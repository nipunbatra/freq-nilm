#!bin/bash

CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr dw hvac fridge mw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr dw hvac mw fridge
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr dw fridge hvac mw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr dw fridge mw hvac
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr dw mw hvac fridge
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr dw mw fridge hvac
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr mw hvac fridge dw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr mw hvac dw fridge
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr mw fridge hvac dw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr mw fridge dw hvac
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr mw dw hvac fridge
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dr mw dw fridge hvac
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw hvac fridge dr mw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw hvac fridge mw dr
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw hvac dr fridge mw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw hvac dr mw fridge
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw hvac mw fridge dr
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw hvac mw dr fridge
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw fridge hvac dr mw
CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 dw fridge hvac mw dr

# appliance='fridge hvac dw dr mw'
# for fold in 0
# do
#     for dataset in 1
#     do
#         for lr in 0.01 0.1 1
#         do
#             for iters in 1000 2000 3000
#             do
#                 echo $appliance $fold $dataset $lr $iters
#                 CUDA_VISIBLE_DEVICES=1 python dnn-nested-cv.py $fold $dataset $lr $iters 0 $appliance
#             done
#         done
#     done
# done




# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 mw dw fridge hvac dr
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 mw dw fridge dr hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 mw dw dr hvac fridge
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_cv_Yiling.py 2 LSTM 100 1 True 0.01 5000 0 mw dw dr fridge hvac

# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 20 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 50 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 GRU 100 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 20 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 50 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 LSTM 100 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 20 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 50 1 True 0.01 2000 0 hvac
# CUDA_VISIBLE_DEVICES=0 python rnn_pytorch_tree_teacher_reduced_p.py 2 RNN 100 1 True 0.01 2000 0 hvac

# hvac fold 0

# appliance='fridge hvac dw dr mw'
# #fold=0
# for fold in 0
# do
#     for dataset in 1
#     do
#         for cell_type in 'GRU' 'LSTM' 'RNN'
#         do
#             for hidden_size in 20 50 100
#             do
#                 for num_layers in 1 2 3
#                 do
#                     if [ $hidden_size -eq 100 -a $num_layers -eq 2 ]
#                     then
#                         continue
#                     fi

#                     if [ $hidden_size -eq 100 -a $num_layers -eq 3 ]
#                     then
#                         continue
#                     fi

#                     for bidirectional in 'True'
#                     do
#                         for lr in 0.01 0.1 1
#                         do
#                             for iterations in 3000
#                             do
#                                 echo $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
#                                 CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv-new.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
