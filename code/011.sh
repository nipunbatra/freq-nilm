#!bin/bash
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.01 20000 0.0 3 0.2 2 fridge hvac dr mw dw 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.01 20000 0.0 3 1.0 4 fridge hvac dr mw dw 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.01 20000 0.0 3 2.0 3 fridge hvac dr mw dw 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.01 20000 0.0 3 5.0 0 fridge hvac dr mw dw 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.01 20000 0.0 3 5.0 3 fridge hvac dr mw dw 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.2 0 dr hvac mw dw fridge 
