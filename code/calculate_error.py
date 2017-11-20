import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import pickle
from tensor_custom_core import *
from common import hourly_4d

freq = sys.argv[1]
#freq = '2H'
tensor = pickle.load(open('{}-input.pkl'.format(freq),'r'))
error_rank = {}
for r in range(1, 10):
	print("*" * 20)

	pred = pickle.load(open("{}-{}-pred-hourly.pkl".format(freq, r), 'r'))

	error_rank[r] = hourly_4d(tensor, pred)