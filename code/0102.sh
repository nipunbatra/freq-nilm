#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.4 4 residual mw dw fridge hvac dr 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.8 4 residual mw dw fridge hvac dr 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.2 4 residual mw dw fridge dr hvac 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.4 4 residual mw dw fridge dr hvac 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.8 4 residual mw dw fridge dr hvac 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.2 4 residual mw dw dr hvac fridge 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.4 4 residual mw dw dr hvac fridge 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.8 4 residual mw dw dr hvac fridge 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.2 4 residual mw dw dr fridge hvac 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.4 4 residual mw dw dr fridge hvac 
#!bin/bash 
CUDA_VISIBLE_DEVICES=2 python cnn-tree.py 5 0.01 20000 0.8 4 residual mw dw dr fridge hvac 
