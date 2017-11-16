from copy import deepcopy

import cvxpy as cvx
import numpy as np
import pandas as pd

from common import APPLIANCES_ORDER
from create_matrix import create_matrix_single_region

def create_df_dfc_static(region, year, appliance, features):
	df, dfc = create_matrix_single_region(region, year)
	start, stop = 1, 13
	energy_cols = np.array(
		[['%s_%d' % (ap, month) for month in range(start, stop)] for ap in ['aggregate', appliance]]).flatten()

	static_cols = ['area', 'total_occupants', 'num_rooms']
	static_df = df[static_cols]
	static_df = static_df.div(static_df.max())

	dfc = df.copy()

	df = dfc[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()

	X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(df, dfc,
                                                                                            appliance,
                                                                                            col_max,
                                                                                            col_min, False)
	static_features = get_static_features(dfc, X_normalised)
	if features == "energy":
		feature_comb = ['None']
	else:
		feature_comb = ['occ', 'area', 'rooms']
	idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)
	return df, dfc, static_df, X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols, idx_user, data_user


def nmf_features(A, k, constant=0.01, regularisation=False, idx_user=None, data_user=None,
                  MAX_ITERS=30, cost='absolute', X_known=None):
	np.random.seed(0)
	# print idx_user, idx_item, data_user, data_item

	# Generate random data matrix A.
	m = len(A)
	n = len(A.columns)
	mask = A.notnull().values
	if X_known:
		Y = cvx.Variable(m, k)
		constraint = [Y >= 0]
		if mask is None:
			obj = cvx.Minimize(cvx.norm(A.values - (Y * X_known), 'fro'))
		else:
			obj = cvx.Minimize(cvx.norm(A.values[mask] - (Y * X_known)[mask], 'fro'))

		prob = cvx.Problem(obj, constraint)
		prob.solve(solver=cvx.SCS)
		return X_known, Y.value, []

	"""

	Parameters
	----------
	A: matrix to be decomposed (m rows and n columns)
	k: number of latent factors
	idx_user: index of entries to be directly passed as fixed features
	data_user: data corresponding to user feature to be directly passed
	idx_item
	data_item
	MAX_ITERS

	Returns
	-------
	X:
	Y:
	Residual

	"""


	one_A = cvx.Constant(1.0 / (A.values[mask] + 1e-3))

	# Initialize Y randomly.
	Y_init = np.random.rand(m, k)
	Y = Y_init

	# Perform alternating minimization.

	residual = np.zeros(MAX_ITERS)
	for iter_num in xrange(1, 1 + MAX_ITERS):
		#print "######## Iteration {}##########".format(iter_num)
		# print iter_num
		# At the beginning of an iteration, X and Y are NumPy
		# array types, NOT CVXPY variables.

		# For odd iterations, treat Y constant, optimize over X.
		if iter_num % 2 == 1:
			X = cvx.Variable(k, n)
			constraint = [X >= 0]
			"""
			for ap in range(n / 2):
				# Put constraints that aggregate factor be more than
				# appliance factor
				constraint.append(X[:, ap] < X[:, ap + n / 2])
			"""


		# For even iterations, treat X constant, optimize over Y.
		else:
			Y = cvx.Variable(m, k)

			constraint = [Y >= 0]
			if idx_user is not None:
				# print np.size(idx_user)
				num_cols = len(idx_user)
				# print num_cols
				for index_feature, fe_name in enumerate(idx_user):
					constraint.append(Y[:, index_feature][idx_user[fe_name]] == data_user[fe_name])
				# return constraint
		#print(constraint)
		#print "----------X--------"
		#print X
		#print "----------Y--------"
		#print Y
		#print "######## Iteration ##########"

				# Y.value[0]=f

		if cost == 'absolute':
			error = A.values[mask] - (Y * X)[mask]
		elif cost =='relative':
			error = cvx.mul_elemwise(one_A, A.values[mask] - (Y * X)[mask])
		else:
			print("NO COST DEFINED. BREAKING")
			import sys
			sys.exit(0)
		# Solve the problem.
		if not regularisation:
			obj = cvx.Minimize(cvx.norm(error, 'fro'))

		else:
			# obj = cvx.Minimize(cvx.norm(A.values[mask] - (Y*X)[mask], 'fro')+constant*(cvx.norm(X, 'fro')+cvx.norm(Y,'fro')))


			if iter_num % 2 == 1:
				obj = cvx.Minimize(cvx.norm(A.values[mask] - (Y * X)[mask], 'fro'))
			else:
				obj = cvx.Minimize(cvx.norm(A.values[mask] - (Y * X)[mask], 'fro'))

		prob = cvx.Problem(obj, constraint)
		prob.solve(solver=cvx.SCS)

		if prob.status != cvx.OPTIMAL:
			pass
		# return X.value, Y.value, residual
		# raise Exception("Solver did not converge!")

		# print 'Iteration {}, residual norm {}'.format(iter_num, prob.value)
		residual[iter_num - 1] = prob.value
		# print prob.value
		# Convert variable to NumPy array constant for next iteration.
		if iter_num % 2 == 1:
			X = X.value
		else:
			Y = Y.value
	return X, Y, residual


def transform_2(pred_df, appliance, matrix_max, matrix_min, normalisation=False):
	pred_df_copy = pred_df.copy()
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13

	if not normalisation:
		for month in range(start, stop):
			pred_df_copy['%s_%d' % (appliance, month)] = pred_df[
				'%s_%d' % (appliance, month)]
	else:
		for month in range(start, stop):
			pass
			#pred_df_copy['%s_%d' % (appliance, month)] = (matrix_max * 1. - matrix_min * 1.) * pred_df[
			#'%s_%d' % (appliance, month)] * 1. + matrix_min * 1.


	return pred_df_copy


def transform(pred_df, appliance, col_max, col_min):
	pred_df_copy = pred_df.copy()
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13

	for month in range(start, stop):
		pred_df_copy['%s_%d' % (appliance, month)] = (col_max['%s_%d' % (appliance, month)] - col_min[
			'%s_%d' % (appliance, month)]) * pred_df['%s_%d' % (appliance, month)] + col_min[
			                                             '%s_%d' % (appliance, month)]
	return pred_df_copy


def preprocess(df, dfc, appliance, matrix_max=None, matrix_min=None, use_all=True):
	if appliance == "hvac":
		start, end = 5, 11

	else:
		start, end = 1, 13
		max_cols = 24

	appliance_cols = ['%s_%d' % (appliance, month) for month in range(start, end)]
	aggregate_cols = ['aggregate_%d' % month for month in range(start, end)]

	all_cols = deepcopy(appliance_cols)
	all_cols.extend(aggregate_cols)

	# X_matrix = dfc[all_cols].dropna()
	# X_matrix = dfc[all_cols].ix[dfc[appliance_cols].dropna().index]
	temp = dfc[all_cols].copy()

	if matrix_max is None:
		matrix_max = temp.max().max()
	if matrix_min is None:
		matrix_min = temp.min().min()

	#ix_use = df[((df[all_cols] >= matrix_max).sum(axis=1) == 0)].index
	#df = df.ix[ix_use]
	#dfc = dfc.ix[ix_use]

	if use_all:
		static_cols = ['area', 'num_rooms', 'total_occupants']
		all_cols_with_static = np.append(aggregate_cols, appliance_cols)
		all_cols_with_static = np.append(all_cols_with_static, static_cols)
		ix_use = df[all_cols_with_static].dropna().index
		df = df.ix[ix_use]
		dfc = dfc.ix[ix_use]

	X_matrix = dfc[all_cols].copy()
	X_normalised = X_matrix.copy()
	# for col in X_matrix.columns:
	#    X_normalised[col] = (X_matrix[col]-col_min[col])/(col_max[col]-col_min[col])
	#for col in X_matrix.columns:
	#	X_normalised[col] = (X_matrix[col] - matrix_min) / (matrix_max - matrix_min)
	df = pd.DataFrame(X_normalised)
	return X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols


def get_static_features(dfc, X_normalised):
	area_col = [x for x in dfc.columns if "area" in x][0]
	rooms_col = [x for x in dfc.columns if "rooms" in x][0]
	occ_col = [x for x in dfc.columns if "occupants" in x][0]
	area = dfc.ix[X_normalised.index][area_col].div(dfc.ix[X_normalised.index][area_col].max()).values
	occ = dfc.ix[X_normalised.index][occ_col].div(dfc.ix[X_normalised.index][occ_col].max()).values
	rooms = dfc.ix[X_normalised.index][rooms_col].div(dfc.ix[X_normalised.index][rooms_col].max()).values
	return {"area": area, "occ": occ, "rooms": rooms}


def get_static_features_region_level(dfc, X_normalised):
	area = dfc.ix[X_normalised.index].area.div(dfc.ix[X_normalised.index].area.max()).values
	occ = dfc.ix[X_normalised.index].num_occupants.div(dfc.ix[X_normalised.index].num_occupants.max()).values
	rooms = dfc.ix[X_normalised.index].house_num_rooms.div(dfc.ix[X_normalised.index].house_num_rooms.max()).values
	dd_keys = ['dd_' + str(x) for x in range(1, 13)]
	out = {"area": area, "occ": occ, "rooms": rooms}
	for dd_k in dd_keys:
		out[dd_k] = dfc.ix[X_normalised.index][dd_k].div(dfc.ix[X_normalised.index][dd_k].max()).values
	return out


def preprocess_all_appliances(df, dfc):
	all_appliances = APPLIANCES_ORDER[1:]
	all_appliance_cols = []
	for appliance in all_appliances:

		start, end = 1, 13

		appliance_cols = ['%s_%d' % (appliance, month) for month in range(start, end)]
		all_appliance_cols.append(appliance_cols)

	aggregate_cols = ['aggregate_%d' % month for month in range(1, 13)]

	all_appliance_cols_flat = []
	for y in all_appliance_cols:
		for x in y:
			all_appliance_cols_flat.append(x)
	all_cols = deepcopy(aggregate_cols)
	all_cols.extend(all_appliance_cols_flat)
	X_matrix = dfc[all_cols]

	columns_max = {}
	columns_min = {}
	col_max = X_matrix.max()
	columns_max[appliance] = col_max
	col_min = X_matrix.min()
	columns_min[appliance] = col_min
	X_normalised = X_matrix.copy()
	# for col in X_matrix.columns:
	#    X_normalised[col] = (X_matrix[col]-col_min[col])/(col_max[col]-col_min[col])
	"""
	for col in X_matrix.columns:
		X_normalised[col] = (X_matrix[col] - col_min.min()) / (col_max.max() - col_min.min())
	
	"""
	df = pd.DataFrame(X_normalised)
	return X_matrix, X_normalised, col_max, col_min

def transform_all_appliances(pred_df, all_appliances, col_max, col_min):
	pred_df_copy = pred_df.copy()

	for appliance in all_appliances:

		if appliance == "hvac":
			start, stop = 5, 11
		else:
			start, stop = 1, 13

		for month in range(start, stop):
			pred_df_copy['%s_%d' % (appliance, month)] = (col_max.max() - col_min.min()) * pred_df[
				'%s_%d' % (appliance, month)] + col_min.min()
	return pred_df_copy


def prepare_df_factorisation(appliance, year, train_regions, train_fraction_dict,
                             test_region, test_home_list, feature_list, seed,
                             matrix_max=None, matrix_min=None, use_all=True):
	print(test_home_list)
	df, dfc = create_df_main(appliance, year, train_regions, train_fraction_dict,
	                         test_region, test_home_list, feature_list, seed)

	X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance,
	                                                                                            matrix_max, matrix_min,
	                                                                                            use_all)

	# Only use homes that are within min and max
	# X_normalised = [(X_normalised > 1).sum(axis=1) == 0]
	# X_matrix = X_matrix.ix[X_normalised.index]
	# df = df.ix[X_normalised.index]
	# dfc = dfc.ix[X_normalised.index]

	if "region" in feature_list:
		static_features = get_static_features_region_level(dfc, X_normalised)
	else:
		static_features = get_static_features(dfc, X_normalised)
	if "region" in feature_list:
		max_f = 20
	else:
		max_f = 3
	return df, dfc, X_matrix, X_normalised, matrix_max, matrix_min, \
	       appliance_cols, aggregate_cols, static_features, max_f


def prepare_known_features(feature_comb, static_features, X_normalised):
	"""
	Example of idx_user: {'occ': array([ 0,  4,  7,  8,  9, 12, 14, 15, 16, 17, 20, 22, 23, 25, 31, 34, 37,
		41, 50, 51, 55, 59])}
	Example of data_user: {'occ': array([ 0.5 ,  0.5 ,  0.5 ,  0.25,  0.75,  1.  ,  0.75,  0.5 ,  0.5 ,
		 0.5 ,  1.  ,  0.75,  1.  ,  0.75,  0.25,  0.5 ,  0.75,  0.25,
		 0.25,  0.5 ,  0.5 ,  0.75])}

	"""
	if 'None' in feature_comb:
		idx_user = None
		data_user = None
	else:
		idx_user = {}
		data_user = {}
		dictionary_static = {}
		for feature in feature_comb:
			dictionary_static[feature] = static_features[feature]
		static_features_df = pd.DataFrame(dictionary_static, index=range(len(X_normalised.index)))
		for fe in static_features_df.columns:
			idx_user[fe] = np.where(static_features_df[fe].notnull())[0]
			data_user[fe] = static_features_df[fe].dropna().values
	return idx_user, data_user


def create_matrix_factorised(appliance, test_home_list, X_normalised):
	X_home = X_normalised.copy()
	if appliance == "hvac":
		start, end = 5, 11
	else:
		start, end = 1, 13
	for test_home in test_home_list:
		for month in range(start, end):
			X_home.loc[test_home, '%s_%d' % (appliance, month)] = np.NAN
	mask = X_home.notnull().values
	# Ensure repeatably random problem data.
	A = X_home.copy()
	return A

def create_matrix_factorised_all_appliances(test_home_list, X_normalised):
	X_home = X_normalised.copy()

	start, end = 1, 13
	for test_home in test_home_list:
		for appliance in ['fridge','hvac','wm','mw','dw','oven']:
			for month in range(start, end):
				X_home.loc[test_home, '%s_%d' % (appliance, month)] = np.NAN
	# Ensure repeatably random problem data.
	A = X_home.copy()
	return A


def create_prediction(test_home_list, X, Y, X_normalised, appliance, matrix_max, matrix_min, appliance_cols, normalisation=False):
	pred_df = pd.DataFrame(Y * X)
	pred_df.columns = X_normalised.columns
	pred_df.index = X_normalised.index
	pred_df = transform_2(pred_df.ix[test_home_list], appliance, matrix_max, matrix_min, normalisation)[appliance_cols]
	return pred_df

def create_prediction_all_appliances(test_home, X, Y, X_normalised, appliance, matrix_max, matrix_min, appliance_cols):
	pred_df = pd.DataFrame(Y * X)
	pred_df.columns = X_normalised.columns
	pred_df.index = X_normalised.index
	pred_df = transform_2(pred_df.ix[test_home], appliance, matrix_max, matrix_min)[appliance_cols]
	return pred_df
