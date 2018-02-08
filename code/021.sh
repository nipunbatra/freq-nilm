#!bin/bash


appliance='fridge dr'


for fold in 0 2 4
do
    for dataset in 1
    do
        for cell_type in 'GRU' 'LSTM' 'RNN'
        do
            for hidden_size in 20 50 100
            do
                for num_layers in 1 2 3
                do
                    for bidirectional in 'True'
                    do
                        for lr in 0.01 0.1 1
                        do
                            for iterations in 3000
                            do
                                CUDA_VISIBLE_DEVICES=1 python rnn-nested-cv-new.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
                            done
                        done
                    done
                done
            done
        done
    done
done
