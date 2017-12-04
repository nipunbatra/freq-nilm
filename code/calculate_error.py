import numpy as np

from tensor_custom_core import *
from common import hourly_4d

freq = '1H'
error_rank = {}
for tf_type in ['MTF','STF']:
	error_rank[tf_type] = {}

	#freq = '2H'
	tensor = np.load('../{}-input.npy'.format(freq))

	for r in range(1, 10):
		print("*" * 20)
		print(tf_type, r)

		pred = np.load("../{}-{}-{}-pred-hourly.npy".format(tf_type, freq, r))

		error_rank[tf_type][r] = hourly_4d(tensor, pred)