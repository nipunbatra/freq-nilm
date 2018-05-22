#!bin/bash
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.2 1 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.2 3 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.2 4 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.5 0 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.5 1 dr hvac mw dw fridge 
CUDA_VISIBLE_DEVIECS=1 python cnn-tree-aug.py 3 0.1 20000 0.0 4 0.5 2 dr hvac mw dw fridge 
