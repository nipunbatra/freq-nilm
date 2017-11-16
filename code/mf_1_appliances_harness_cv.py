from sklearn.model_selection import KFold, train_test_split

from common import compute_rmse_fraction
from create_matrix import *
from mf_core import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

import sys

n_splits = 10

region, appliance, features, cost, random_seed, train_percentage = sys.argv[1:]
random_seed = int(random_seed)
train_percentage = float(train_percentage)


year = 2014
target = region
target_df, target_dfc = create_matrix_single_region(region, year)
start, stop = 1, 13
energy_cols = np.array(
	[['%s_%d' % (ap, month) for month in range(start, stop)] for ap in ['aggregate', appliance]]).flatten()

static_cols = ['area', 'total_occupants', 'num_rooms']
static_df = target_df[static_cols]
static_df = static_df.div(static_df.max())

target_dfc = target_df.copy()

target_df = target_dfc[energy_cols]
col_max = target_df.max().max()
col_min = target_df.min().min()

X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(target_df, target_dfc,
                                                                                            appliance,
                                                                                            col_max,
                                                                                            col_min, False)
static_features = get_static_features(target_dfc, X_normalised)
if features == "energy":
	feature_comb = ['None']
else:
	feature_comb = ['occ', 'area', 'rooms']
idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

kf = KFold(n_splits=n_splits)

best_params_global = {}
pred_df = []
for outer_loop_iteration, (train_max, test) in enumerate(kf.split(target_df)):
	# Just a random thing
	np.random.seed(10 * random_seed + 7 * outer_loop_iteration)
	np.random.shuffle(train_max)
	print("-" * 80)
	print("Progress: {}".format(100.0 * outer_loop_iteration / n_splits))
	num_train = int((train_percentage * len(train_max) / 100) + 0.5)
	if train_percentage == 100:
		train = train_max
		train_ix = target_df.index[train]
		# print("Train set {}".format(train_ix.values))
		test_ix = target_df.index[test]
	else:
		# Sample `train_percentage` homes
		# An important condition here is that all homes should have energy data
		# for all appliances for atleast one month.
		SAMPLE_CRITERION_MET = False

		train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
		train_ix = target_df.index[train]
		# print("Train set {}".format(train_ix.values))
		test_ix = target_df.index[test]
		a = target_df.loc[train_ix]
		count_condition_violation = 0

	print("-" * 80)

	print("Test set {}".format(test_ix.values))

	print("-" * 80)
	print("Current Error, Least Error, #Iterations")

	### Inner CV loop to find the optimum set of params. In this case: the number of iterations
	inner_kf = KFold(n_splits=2)

	best_num_iterations = 0
	best_num_season_factors = 0
	best_num_home_factors = 0
	least_error = 1e6

	overall_df_inner = target_df.loc[train_ix]

	best_params_global[outer_loop_iteration] = {}
	for num_iterations_cv in range(5, 10):
		for num_latent_factors_cv in range(3, 9):
			pred_inner = {}
			for train_inner, test_inner in inner_kf.split(overall_df_inner):

				train_ix_inner = overall_df_inner.index[train_inner]
				test_ix_inner = overall_df_inner.index[test_inner]
				train_test_ix_inner = np.concatenate([test_ix_inner, train_ix_inner])
				df_t_inner, dfc_t_inner = target_df.loc[train_test_ix_inner], target_dfc.loc[train_test_ix_inner]
				X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(df_t_inner,
				                                                                                            dfc_t_inner,
				                                                                                            appliance,
				                                                                                            col_max,
				                                                                                            col_min,
				                                                                                            False)

				static_features = get_static_features(dfc_t_inner, X_normalised)
				if features == "energy":
					feature_comb = ['None']
				else:
					feature_comb = ['occ', 'area', 'rooms']
				idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

				# Static features can only be used if we have atleast some values from the train homes
				if idx_user is not None:
					if min([len(x) for x in idx_user.values()])==0:
						idx_user = None
						data_user =None

				A = create_matrix_factorised(appliance, test_ix_inner, X_normalised)
				X, Y, res = nmf_features(A=A, k=num_latent_factors_cv, constant=0.01, regularisation=False,
				                         idx_user=idx_user, data_user=data_user,
				                         idx_item=None, data_item=None, MAX_ITERS=num_iterations_cv, cost=cost)

				pred_inner = []

				pred_inner.append(create_prediction(test_ix_inner, X, Y, X_normalised, appliance,
				                                    col_max, col_min, appliance_cols))

			err = {}

			pred_inner = pd.DataFrame(pd.concat(pred_inner))

			try:
				if appliance == "hvac":
					err[appliance] = \
						compute_rmse_fraction(appliance, pred_inner[['hvac_{}'.format(month) for month in range(5, 11)]],
						                      'SanDiego')[2]
				else:
					err[appliance] = compute_rmse_fraction(appliance, pred_inner, 'SanDiego')[2]
			except Exception, e:
				# This appliance does not have enough samples. Will not be
				# weighed
				print(e)
				print(appliance)
			err_weight = {}

			mean_err = pd.Series(err).sum()
			if mean_err < least_error:
				best_num_iterations = num_iterations_cv
				best_num_latent_factors = num_latent_factors_cv
				least_error = mean_err
			print(mean_err, least_error, num_iterations_cv, num_latent_factors_cv)
	best_params_global[outer_loop_iteration] = {'Iterations': best_num_iterations,
	                                            "Appliance Train Error": err,
	                                            'Num latent factors': best_num_latent_factors,
	                                            "Least Train Error": least_error}

	print("******* BEST PARAMS *******")
	print(best_params_global[outer_loop_iteration])
	print("******* BEST PARAMS *******")
	# Now we will be using the best parameter set obtained to compute the predictions



	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(df_t,
	                                                                                            dfc_t,
	                                                                                            appliance,
	                                                                                            col_max,
	                                                                                            col_min,
	                                                                                            False)

	static_features = get_static_features(dfc_t, X_normalised)
	if features == "energy":
		feature_comb = ['None']
	else:
		feature_comb = ['occ', 'area', 'rooms']
	idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)
	# Static features can only be used if we have atleast some values from the train homes for each of the features
	if idx_user is not None:
		if min([len(x) for x in idx_user.values()]) == 0:
			idx_user = None
			data_user = None

	A = create_matrix_factorised(appliance, test_ix, X_normalised)
	X, Y, res = nmf_features(A=A, k=best_num_latent_factors, constant=0.01, regularisation=False,
	                         idx_user=idx_user, data_user=data_user,
	                         idx_item=None, data_item=None, MAX_ITERS=best_num_iterations, cost=cost)

	pred_df.append(create_prediction(test_ix, X, Y, X_normalised, appliance,
	                                 col_max, col_min, appliance_cols, normalisation=True))


out = {'Prediction': pd.concat(pred_df), 'Parameters': best_params_global}
import os
import pickle
name = "{}-{}-{}".format(features, random_seed, train_percentage)
directory = os.path.expanduser('~/aaai2017/Normal-MF_{}/{}'.format(target, appliance))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = '{}'.format(os.path.join(directory, name + '.pkl'))
pickle.dump(out, open(filename,'w'))


