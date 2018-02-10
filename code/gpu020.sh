#!bin/bash
<<COMMENT1
appliance='fridge hvac dr dw mw'
for fold in 2
do
    for dataset in 1
    do
        for lr in 0.01 0.1 1
        do
            for iters in 1000 2000 3000
            do
                echo $appliance $fold $dataset $lr $iters
                CUDA_VISIBLE_DEVICES=0 python dnn-nested-cv.py $fold $dataset $lr $iters 0 $appliance
            done
        done
    done
done
COMMENT1


appliance='fridge hvac dw'
 #fold=0
for fold in 3
do
    for dataset in 2
    do
        for cell_type in 'LSTM' 'RNN'
        do
            for hidden_size in 20 50 100
            do
                for num_layers in 1 2 3
                do
                    if [ $hidden_size -eq 100 -a $num_layers -eq 2 ]
                    then
                        continue
                    fi

                    if [ $hidden_size -eq 100 -a $num_layers -eq 3 ]
                    then
                        continue
                    fi

                    for bidirectional in 'True'
                    do
                        for lr in 0.01 0.1 1
                        do
                            for iterations in 3000
                            do
                                echo $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
                                CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv-new.py $fold $dataset $cell_type $hidden_size $num_layers $bidirectional $lr $iterations 0 $appliance
                            done
                        done
                    done
                done
            done
        done
    done
done
