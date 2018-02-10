#!bin/bash

CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw hvac mw fridge dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw hvac mw dr fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw fridge hvac dr mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw fridge hvac mw dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw fridge dr hvac mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw fridge dr mw hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw fridge mw hvac dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw fridge mw dr hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw dr hvac fridge mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw dr hvac mw fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw dr fridge hvac mw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw dr fridge mw hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw dr mw hvac fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw dr mw fridge hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw mw hvac fridge dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw mw hvac dr fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw mw fridge hvac dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw mw fridge dr hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw mw dr hvac fridge
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 dw mw dr fridge hvac
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 mw hvac fridge dr dw
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 mw hvac fridge dw dr
CUDA_VISIBLE_DEVICES=0 python rnn-nested-cv.py 4 1 GRU 50 1 True 0.1 2000 0 mw hvac dr fridge dw
