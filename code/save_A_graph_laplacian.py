"""

This module saves the A matrix learnt using P = HAS, for various
combinations of parameters, such as number of iterations,
number of season and home factors, Lambda

It can be run as:

### Template
>>>python save_A_graph_laplacian.py case region

### Example
>>> python save_A_graph_laplacian.py 2 Austin
"""

from common import create_region_df_dfc_static
from create_matrix import *
from tensor_custom_core import *
import datetime
from common import compute_rmse_fraction, contri, get_tensor, create_region_df_dfc_static

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014

case, source, constant_use, start, stop = sys.argv[1:]
case = int(case)
start = int(start)
stop = int(stop)
source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year, start, stop)

# # using cosine similarity to compute L
source_L = get_L(source_static)

# Seasonal constant constraints
if constant_use == 'True':
	T_constant = np.ones(stop-start).reshape(-1,1)
else:
	T_constant = None
# End

pred = {}
n_splits = 10

algo = 'adagrad'
cost = 'l21'

for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
best_params_global = {}
A_store = {}
err_store = {}

max_num_iterations = 1300
for learning_rate_cv in [0.1, 0.5, 1, 2]:
	A_store[learning_rate_cv] = {}
	err_store[learning_rate_cv] = {}
	for num_season_factors_cv in range(2, 6):

		A_store[learning_rate_cv][num_season_factors_cv] = {}
		err_store[learning_rate_cv][num_season_factors_cv] = {}
		for num_home_factors_cv in range(2, 6):
			if case == 4:
				if num_home_factors_cv != num_season_factors_cv:
					print("Case 4 needs equal # dimensions. Skipping")
					continue
			A_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv] = {}
			err_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv] = {}
			for lam_cv in [0.001, 0.01, 0.1, 0, 1]:
				print("-"*80)
				print(datetime.datetime.now())
				print("-" * 80)
				sys.stdout.flush()
				A_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][lam_cv] = {}
				err_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][lam_cv] = {}

				H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, source_tensor,
				                                                                                source_L,
				                                                                                num_home_factors_cv,
				                                                                                num_season_factors_cv,
				                                                                                num_iter=max_num_iterations,
				                                                                                lr=learning_rate_cv, dis=False,
				                                                                                lam=lam_cv,
				                                                                                T_known=T_constant)
				
				for num_iterations in range(100, 1400, 200):
					A_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][lam_cv][num_iterations] = As[num_iterations]
					print(learning_rate_cv, num_season_factors_cv, num_home_factors_cv, lam_cv, num_iterations, costs[num_iterations])
					sys.stdout.flush()

				err = {}
				for num_iterations in range(100, 1400, 200):
					err[num_iterations] = {}
					for appliance in APPLIANCES_ORDER:
						pred[appliance] = []
					for appliance in APPLIANCES_ORDER:
						pred[appliance].append(pd.DataFrame(HATs[num_iterations][:, appliance_index[appliance], :], index=source_df.index))
					for appliance in APPLIANCES_ORDER:
						pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

					for appliance in APPLIANCES_ORDER[1:]:
						prediction = pred[appliance]
						if appliance == "hvac":
							prediction = prediction[range(5-start, 11-start)]
						err[num_iterations][appliance] = \
							compute_rmse_fraction(appliance, prediction, source, start, stop)[2]
					# print("Computed for: {}".format())

					err_weight = {}
					for appliance in APPLIANCES_ORDER[1:]:
						err_weight[appliance] = err[num_iterations][appliance]*contri[source][appliance]
					mean_err = pd.Series(err_weight).sum()
					err_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][lam_cv][num_iterations] = mean_err
																																
pickle.dump(A_store, open('../predictions/case-{}-graph_{}_{}_{}_{}_As.pkl'.format(case, source, constant_use, start, stop), 'w'))
pickle.dump(err_store, open('../predictions/case-{}-graph_{}_{}_{}_{}_errs.pkl'.format(case, source, constant_use, start, stop), 'w'))
