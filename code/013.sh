#!bin/bash

appliance='fridge hvac dr dw mw'
for fold in 4
do
    for dataset in 1
    do
        for lr in 0.01 0.1 1
        do
            for iters in 1000 2000 3000
            do
                echo $appliance $fold $dataset $lr $iters
                CUDA_VISIBLE_DEVICES=3 python dnn-nested-cv.py $fold $dataset $lr $iters 0 $appliance
            done
        done
    done
done




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
