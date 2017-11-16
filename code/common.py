from sklearn.metrics import mean_squared_error

from create_matrix import *


def get_tensor(df, start=1, stop=13):
	# start, stop = 1, 13
	energy_cols = np.array(
		[['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

	dfc = df.copy()

	df = dfc[energy_cols]

	tensor = df.values.reshape((len(df), 7, stop - start))
	return tensor


def create_region_df_dfc_static(region, year, start=1, stop=13):
	df, dfc = create_matrix_single_region(region, year)
	tensor = get_tensor(df, start, stop)
	static_region = df[['area', 'total_occupants', 'num_rooms']].copy()
	static_region['area'] = static_region['area'].div(4000)
	static_region['total_occupants'] = static_region['total_occupants'].div(8)
	static_region['num_rooms'] = static_region['num_rooms'].div(8)
	static_region = static_region.values
	return df, dfc, tensor, static_region


contri = {}

for region in ['SanDiego', 'Austin', 'Boulder']:

	temp = {}
	r_df = create_matrix_single_region(region, 2014)[0]
	for appliance in APPLIANCES_ORDER[1:]:
		df_app = r_df[['{}_{}'.format(appliance, month) for month in range(5, 11)]]
		df_agg = r_df[['{}_{}'.format("aggregate", month) for month in range(5, 11)]]
		df_app.columns = df_agg.columns
		temp[appliance] = df_app.div(df_agg).mean().mean()
	error_weights = pd.Series(temp).div(pd.Series(temp).sum()).to_dict()
	contri[region] = error_weights


def compute_rmse(appliance, pred_df, region='Austin', year=2014):
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)

	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred_df = pred_df.copy()
	pred_df.columns = [['%s_%d' % (appliance, month) for month in range(start, stop)]]
	gt_df = appliance_df[pred_df.columns].ix[pred_df.index]

	gt_df = gt_df.unstack().dropna()
	pred_df = pred_df.unstack().dropna()
	index_intersection = gt_df.index.intersection(pred_df.index)
	gt_df = gt_df.ix[index_intersection]
	pred_df = pred_df.ix[index_intersection]

	rms = np.sqrt(mean_squared_error(gt_df, pred_df))
	return gt_df, pred_df, rms, (gt_df - pred_df).abs()


def compute_rmse_fraction(appliance, pred_df, region='Austin', start=1, stop=13, year=2014):
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)

	if appliance == "hvac":
		start, stop = 5, 11
	
	pred_df = pred_df.copy()
	pred_df.columns = [['%s_%d' % (appliance, month) for month in range(start, stop)]]
	gt_df = appliance_df[pred_df.columns].ix[pred_df.index]

	aggregate_df = appliance_df.ix[pred_df.index][['aggregate_%d' % month for month in range(start, stop)]]

	aggregate_df.columns = gt_df.columns
	rows, cols = np.where((aggregate_df < 100))
	for r, c in zip(rows, cols):
		r_i, c_i = aggregate_df.index[r], aggregate_df.columns[c]
		aggregate_df.loc[r_i, c_i] = np.NaN

	gt_fraction = gt_df.div(aggregate_df) * 100
	pred_fraction = pred_df.div(aggregate_df) * 100

	# Capping it to 100%
	pred_fraction[pred_fraction > 100] = 100.

	gt_fraction_dropna = gt_fraction.unstack().dropna()
	pred_fraction_dropna = pred_fraction.unstack().dropna()
	index_intersection = gt_fraction_dropna.index.intersection(pred_fraction_dropna.index)
	gt_fraction_dropna = gt_fraction_dropna.ix[index_intersection]
	pred_fraction_dropna = pred_fraction_dropna.ix[index_intersection]
	difference_error = (gt_fraction_dropna - pred_fraction_dropna).abs()

	rms = np.sqrt(mean_squared_error(gt_fraction_dropna, pred_fraction_dropna))
	return gt_fraction_dropna, pred_fraction_dropna, rms, difference_error


import os


def get_directory_structure(rootdir):
	"""
	Creates a nested dictionary that represents the folder structure of rootdir
	"""
	dir = {}
	rootdir = rootdir.rstrip(os.sep)
	start = rootdir.rfind(os.sep) + 1
	for path, dirs, files in os.walk(rootdir):
		folders = path[start:].split(os.sep)
		print path, dirs, files, folders

		if len(files):
			subdir = pd.read_csv(os.path.join(path, files[0]), index_col=0)
		else:
			subdir = dict.fromkeys(files)
		parent = reduce(dict.get, folders[:-1], dir)
		parent[folders[-1]] = subdir
	return dir
