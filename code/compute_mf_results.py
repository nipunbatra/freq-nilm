"""
Run all the code on HCDM

"""

import os
import sys
import pickle
import pandas as pd
from common import APPLIANCES_ORDER, compute_rmse_fraction
source, target = sys.argv[1:]

out = {}
params = {}
features = 'energy'
out = {}
for setting in ['transfer', 'normal']:
	out[setting] = {}
	for train_percentage in [6., 7., 8., 9., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]:
		out[setting][train_percentage] = {}
		for random_seed in range(4):
			out[setting][train_percentage][random_seed] = {}
			for appliance in ['hvac','fridge','mw','dw','oven','wm']:




				import os
				import pickle
				try:

					if setting == "normal":
						name = "{}-{}-{}-{}-{}".format(target, appliance, features, random_seed, train_percentage)
					else:
						name = "{}-{}-{}-{}-{}-{}".format(source, target, appliance, features, random_seed,
						                                  train_percentage)

					directory = os.path.expanduser('~/git/scalable-nilm/aaai18/predictions/MF/{}'.format(setting))
					if not os.path.exists(directory):
						os.makedirs(directory)
					filename = os.path.join(directory, name + '.pkl')
					pred = pickle.load(open(filename, 'r'))['Prediction']
					print(filename)
					out[setting][train_percentage][random_seed][appliance] = \
						compute_rmse_fraction(appliance, pred, target)[2]
				except:
					pass
		out[setting][train_percentage] = pd.DataFrame(out[setting][train_percentage]).mean(axis=1)

save_loc = os.path.expanduser("~/git/scalable-nilm/aaai18/predictions/mf-{}-{}.pkl".format(source, target))
pickle.dump(out, open(save_loc, 'w'))

pickle.dump(out, open('../../../predictions/mf-{}-{}.pkl'.format(source, target), 'w'))
#pickle.dump(params, open('../predictions/params-lr-tf-{}-{}.pkl'.format(source, target), 'w'))
