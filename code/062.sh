#!bin/bash


CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw hvac dr dw fridge
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw hvac dw fridge dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw hvac dw dr fridge
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw fridge hvac dr dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw fridge hvac dw dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw fridge dr hvac dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw fridge dr dw hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw fridge dw hvac dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw fridge dw dr hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dr hvac fridge dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dr hvac dw fridge
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dr fridge hvac dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dr fridge dw hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dr dw hvac fridge
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dr dw fridge hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dw hvac fridge dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dw hvac dr fridge
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dw fridge hvac dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dw fridge dr hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dw dr hvac fridge
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 mw dw dr fridge hvac


