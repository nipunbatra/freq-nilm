#!bin/bash


CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge hvac dr mw dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge hvac dw dr mw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge hvac dw mw dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge hvac mw dr dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge hvac mw dw dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dr hvac dw mw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dr hvac mw dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dr dw hvac mw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dr dw mw hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dr mw hvac dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dr mw dw hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dw hvac dr mw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dw hvac mw dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dw dr hvac mw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dw dr mw hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dw mw hvac dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge dw mw dr hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge mw hvac dr dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge mw hvac dw dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge mw dr hvac dw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge mw dr dw hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge mw dw hvac dr
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 fridge mw dw dr hvac
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 dr hvac fridge dw mw
CUDA_VISIBLE_DEVICES=2 python rnn-nested-cv.py 2 1 GRU 50 1 True 0.1 2000 0 dr hvac fridge mw dw

