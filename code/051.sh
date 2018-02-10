#!bin/bash


CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 hvac dw fridge dr mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 hvac dw fridge mw dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 1 1 GRU 50 1 True 0.1 2000 0 hvac mw dr fridge dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw hvac dr dw fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw fridge hvac dw dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw fridge dr hvac dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw fridge dw dr hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw dr hvac fridge dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw dw hvac fridge dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 3 1 GRU 50 1 True 0.1 2000 0 mw dw fridge hvac dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac fridge dw mw dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac fridge mw dr dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac dr fridge dw mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac dr fridge mw dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac dr dw mw fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac dr mw fridge dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac dw fridge dr mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 hvac dw fridge mw dr
