import pandas as pd
import numpy as np
import sys
import os
import pickle

### FIX This
data_path = os.path.expanduser("~/git/scalable-nilm/create_dataset/metadata/all_regions_years_cleaned_z_score_5.pkl")

out_overall = pickle.load(open(data_path, 'r'))



APPLIANCES_ORDER = ['aggregate', 'hvac', 'fridge', 'mw', 'dw', 'wm', 'oven']

upper_limit = {
	'hvac': 40.,
	'fridge': 10.,
	'wm': 1.,
	'oven':1.,
	'mw':1.,
	'dw':1.

}

only_monthly_features = np.hstack([["aggregate_%d" % i for i in range(1, 13)],
                                   'skew', 'kurtosis', 'p_25', 'p_50', 'p_75',
                                   ['ratio_min_max',
                                    # 'difference_min_max',
                                    'ratio_difference_min_max']])

def remove_hvac_features(fe):
    hvac_all_features = [x for x in fe]
    hvac_all_features = [x for x in hvac_all_features if 'stdev_trend' not in x]
    hvac_all_features = [x for x in hvac_all_features if 'stdev_seasonal' not in x]
    hvac_all_features = [x for x in hvac_all_features if 'variance' not in x]
    hvac_all_features = [x for x in hvac_all_features if 'mins_hvac' not in x]
    # hvac_all_features = [x for x in hvac_all_features if 'fraction' not in x]
    return hvac_all_features

feature_map = {
    "Monthly": remove_hvac_features(only_monthly_features.tolist()),

}



def create_region_df(region, year=2014):
	df = out_overall[year][region]

	df_copy = df.copy()
	# drop_rows_having_no_data
	o = {}
	for h in df.index:
		o[h] = len(df.ix[h][feature_map['Monthly']].dropna())
	num_features_ser = pd.Series(o)
	drop_rows = num_features_ser[num_features_ser == 0].index

	df = df.drop(drop_rows)
	dfc = df.copy()

	df = df.rename(columns={'house_num_rooms': 'num_rooms',
	                        'num_occupants': 'total_occupants',
	                        'difference_ratio_min_max': 'ratio_difference_min_max'})
	return df, dfc

def create_matrix_all_appliances(region, year, all_features=False):
	df, dfc = create_matrix_single_region(region, year)
	start, stop=1, 13
	energy_cols = np.array([['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()
	static_cols = np.array(['area', 'total_occupants', 'num_rooms'])
	all_cols = np.concatenate(np.array([energy_cols, static_cols]).flatten())
	if all_features:
		df = df[all_cols].dropna()

	else:
		df = df[all_cols]
	return df



def create_matrix_single_region(region, year):
	temp_df, temp_dfc = create_region_df(region, year)
	return temp_df, temp_dfc


def create_matrix_region_appliance_year(region, year, appliance, all_features=False):
	df, dfc = create_matrix_single_region(region, year)
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	appliance_cols = ['%s_%d' % (appliance, month) for month in range(start, stop)]
	aggregate_cols = ['%s_%d' % ("aggregate", month) for month in range(start, stop)]
	static_cols = ['area', 'total_occupants', 'num_rooms']
	all_cols = np.concatenate(np.array([appliance_cols, aggregate_cols, static_cols]).flatten())
	if all_features:
		df = df[all_cols].dropna()
		df = df[df["total_occupants"] > 0]
		df = df[df["area"] > 100]
		df = df[~(df[aggregate_cols] < 200).sum(axis=1).astype('bool')]
		ul_appliance = upper_limit[appliance]
		df = df[~(df[appliance_cols] < ul_appliance).sum(axis=1).astype('bool')]
	else:
		df = df[all_cols]


	return df

