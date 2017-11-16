"""

This module saves the A matrix learnt using P = YX, for various
combinations of parameters, such as number of iterations,
number of season and home factors, Lambda

It can be run as:

### Template
>>>python save_A_mf.py case region

### Example
>>> python save_A_mf.py 2 Austin
"""

from common import create_region_df_dfc_static
from create_matrix import *
from tensor_custom_core import *
import datetime
from mf_core import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014

source = sys.argv[1]

n_splits = 10


X_store = {}

for appliance in APPLIANCES_ORDER[1:]:
	X_store[appliance] = {}
	for features in ['energy']:
		X_store[appliance][features] = {}
		for iterations in range(10, 20, 4):
			X_store[appliance][features][iterations] = {}
			for lat in range(3, 8):

				source_df, source_dfc, source_static_df, source_X_matrix, source_X_normalised, source_matrix_max, source_matrix_min, source_appliance_cols, source_aggregate_cols, source_idx_user, source_data_user = create_df_dfc_static(
					source, 2014, appliance, features)



				X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(source_df,
				                                                                                            source_dfc,
				                                                                                            appliance,
				                                                                                            source_matrix_max,
				                                                                                            source_matrix_min,
				                                                                                            False)

				static_features = get_static_features(source_dfc, X_normalised)
				if features == "energy":
					feature_comb = ['None']
				else:
					feature_comb = ['occ', 'area', 'rooms']
				idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

				# Static features can only be used if we have atleast some values from the train homes
				if idx_user is not None:
					if min([len(x) for x in idx_user.values()]) == 0:
						idx_user = None
						data_user = None

				A = create_matrix_factorised(appliance, [], X_normalised)
				X, Y, res = nmf_features(A=A, k=lat, constant=0.01, regularisation=False,
				                         idx_user=idx_user, data_user=data_user,
				                         MAX_ITERS=iterations,
				                         cost='absolute')
				X_store[appliance][features][iterations][lat] = X
				print(appliance, features, iterations, lat)
pickle.dump(X_store, open('../predictions/{}-X.pkl'.format(source), 'w'))

