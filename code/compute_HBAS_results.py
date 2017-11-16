"""
Run all the code on HCDM

"""

import os
import sys
import pickle
import pandas as pd
from common import APPLIANCES_ORDER, compute_rmse_fraction
source, target, start, stop = sys.argv[1:]
start = int(start)
stop = int(stop)
out = {}
params = {}

for constant_use in ['True', 'False']:
	out[constant_use] = {}
	params[constant_use] = {}
	
		
	for setting in ['normal' , 'transfer']:
		print setting, constant_use
		out[constant_use][setting] = {}
		params[constant_use][setting] = {}
		for train_percentage in [6., 8., 10., 15., 20., 30., 40.,50.,60.,70.,80.,90.,100.]:
			out[constant_use][setting][train_percentage] = {}
			params[constant_use][setting][train_percentage] = {}
			for random_seed in range(5):
				if setting == "transfer":
					name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
				else:
					name = "{}-{}-{}".format(target, random_seed, train_percentage)

				directory = os.path.expanduser(
					'~/git/scalable-nilm/aaai18/predictions/HBAS/TF-all/{}/{}'.format(setting, constant_use))
				if not os.path.exists(directory):
					os.makedirs(directory)
				filename = os.path.join(directory, name + '.pkl')
				try:
					out[constant_use][setting][train_percentage][random_seed]={}
					params[constant_use][setting][train_percentage][random_seed] = {}
					pr = pickle.load(open(filename, 'r'))
					pred = pr['Predictions']
					parameter_data = pr['Learning Params']
					params[constant_use][setting][train_percentage][random_seed] = parameter_data
					for appliance in APPLIANCES_ORDER[1:]:
						prediction = pred[appliance]
						if appliance == 'hvac':
							prediction = prediction[range(5-start, 11-start)]
						out[constant_use][setting][train_percentage][random_seed][appliance] = \
							compute_rmse_fraction(appliance, prediction, target, start, stop)[2]
					print("Computed for: {}".format(name))
					# print case, constant_use, static_use, setting, train_percentage, random_seed
					# print out[constant_use][setting][train_percentage][random_seed]

				except Exception, e:
					print(e)
					print("Exception")
			out[constant_use][setting][train_percentage] = pd.DataFrame(out[constant_use][setting][train_percentage]).mean(axis=1)
pickle.dump(out, open('../predictions/lr-tf-{}-{}-HBAS.pkl'.format(source, target), 'w'))
pickle.dump(params, open('../predictions/params-lr-tf-{}-{}-HBAS.pkl'.format(source, target), 'w'))
