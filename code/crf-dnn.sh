#!bin/bash


for appliance in 'hvac' 'fridge' 'dr' 'dw' 'mw'
do
    for fold_num in 0 1 2 3 4
    do
        for dataset in 1 2
        do
            for lr in 0.01 0.1 1
            do
                for iters in 1000 2000 3000
                do
                    python dnn-nested-cv.py $fold_num $dataset $lr $iters 0 $appliance
                done
            done
        done
    done
done
