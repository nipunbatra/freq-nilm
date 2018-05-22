#!bin/bash
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 1 5.0 0 dw hvac fridge mw dr 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 1 5.0 3 dw hvac fridge mw dr 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 1 5.0 4 dw hvac fridge mw dr 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.01 20000 0.0 2 2.0 1 fridge hvac mw dw dr 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.01 20000 0.0 2 2.0 3 fridge hvac mw dw dr 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.01 20000 0.0 2 5.0 4 fridge hvac mw dw dr 
