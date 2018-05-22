#!bin/bash
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 4 1.0 1 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 4 2.0 1 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 4 2.0 2 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 4 2.0 3 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=0 python cnn-tree-aug.py 3 0.1 20000 0.0 4 5.0 3 dr hvac mw dw fridge 
