"""
This module computes results for transfer learning

>>>python graph_laplacian.py setting case static_use source target random_seed train_percentage
setting: transfer or normal
case: 1, 2, 3, 4; 2 is for our proposed approach, 4 is for standard TF
static_use: "True" or "False"- If False, we don't use static household properties 
and the corresponding laplacian penalty term is set to 0
constant_use: "True" or "False" - If True, we add the constraint that one column of seasonal factors to be 1.
source:
target:
random_seed:
train_percentage:

NB: Prediction region is always called target. So, if we are doing n
normal learning on SD, we don't care about source, but target will be SD

Example:
# Transfer learning from Austin -> SD, case 2, 10% data used, 0th random seed, static_data used
>>> python graph_laplacian.py transfer 2 True Austin SanDiego 0 10

# Normal training in SD, case 2, 10% data used, 0th random seed, static data used
>>> python graph_laplacian.py normal 2 True None SanDiego 0 10

TODO: mention the objective being solved here

"""

import datetime
from sklearn.model_selection import train_test_split, KFold
from common import compute_rmse_fraction, contri, get_tensor, create_region_df_dfc_static
from create_matrix import *
from tensor_custom_core import *
import multiprocessing as mp

global source, target
global case
global source_df, source_dfc, source_tensor, source_static
global target_df, target_dfc, target_tensor, target_static
global source_L, target_L
global T_constant
global start, stop

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014

# setting, case, constant_use, static_use, source, target, random_seed, train_percentage, start, stop = sys.argv[1:]
# case = int(case)
# train_percentage = float(train_percentage)
# random_seed = int(random_seed)
# start = int(start)
# stop = int(stop)

setting, constant_use, source, target, random_seed, train_percentage, start, stop = sys.argv[1:]
random_seed = int(random_seed)
train_percentage = float(train_percentage)
start = int(start)
stop = int(stop)
# constant_use = 'False'

# if static_use == "True":
#   # Use non-zero value of penalty
#   lambda_cv_range = [0, 0.001, 0.01, 0.1]
# else:
#   lambda_cv_range = [0]

A_store = pickle.load(open(os.path.expanduser('~/git/scalable-nilm/aaai18/predictions/HBAS_{}_As.pkl'.format(source)), 'r'))
B_store = pickle.load(open(os.path.expanduser('~/git/scalable-nilm/aaai18/predictions/HBAS_{}_Bs.pkl'.format(source)), 'r'))

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year)

# # using cosine similarity to compute L
# source_L = get_L(source_static)
# target_L = get_L(target_static)

if setting=="transfer":
    name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
else:
    name = "{}-{}-{}".format(target, random_seed, train_percentage)

# Seasonal constant constraints
if constant_use == 'True':
  T_constant = np.ones(stop-start).reshape(-1 , 1)
else:
  T_constant = None
# End

directory = os.path.expanduser('~/git/scalable-nilm/aaai18/predictions/HBAS/TF-all/{}/{}'.format(setting, constant_use))
if not os.path.exists(directory):
    os.makedirs(directory)
filename = os.path.join(directory, name + '.pkl')

if os.path.exists(filename):
    print("File already exists. Quitting.")

def multiply_HBAT(H, B, A, T):
        return np.einsum('mh, hn, ns, ts ->mnt', H, B, A, T)

def compute_inner_error(overall_df_inner, learning_rate_cv, num_iterations_cv, num_season_factors_cv,num_home_factors_cv, B_source, A_source):
    # overall_df_inner, num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv = param
    # print num_iterations_cv, num_season_factors_cv,num_home_factors_cv
    inner_kf = KFold(n_splits=2)
    pred_inner = {}
    for train_inner, test_inner in inner_kf.split(overall_df_inner):

        train_ix_inner = overall_df_inner.index[train_inner]
        test_ix_inner = overall_df_inner.index[test_inner]

        # H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, source_tensor, source_L, 
        #                                                                                 num_home_factors_cv, num_season_factors_cv, 
        #                                                                                 num_iter=num_iterations_cv, lr=1, dis=False, 
        #                                                                                 lam=lam_cv, T_known = np.ones(12).reshape(-1, 1))

        train_test_ix_inner = np.concatenate([test_ix_inner, train_ix_inner])
        df_t_inner, dfc_t_inner = target_df.loc[train_test_ix_inner], target_dfc.loc[train_test_ix_inner]
        tensor_inner = get_tensor(df_t_inner, start, stop)
        tensor_copy_inner = tensor_inner.copy()
        # First n
        tensor_copy_inner[:len(test_ix_inner), 1:, :] = np.NaN
        # L_inner = target_L[np.ix_(np.concatenate([test_inner, train_inner]), np.concatenate([test_inner, train_inner]))]

        if setting=="transfer":
            A_source = A_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][num_iterations_cv]
            B_source = B_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][num_iterations_cv]
        else:
            A_source = None
            B_source = None
        
    
        H, B, A, T, Hs, Bs, As, Ts, HBATs, costs = learn_HBAT_adagrad_graph(tensor_copy_inner,
                                                                            num_home_factors_cv,
                                                                            num_season_factors_cv,
                                                                            num_iter = num_iterations_cv,
                                                                            lr = learning_rate_cv, 
                                                                            B_known = B_source,
                                                                            A_known = A_source,
                                                                            T_known = T_constant)

        # print "After learning"
        HBAT = multiply_HBAT(H, B, A, T)
        for appliance in APPLIANCES_ORDER:
            if appliance not in pred_inner:
                pred_inner[appliance] = []

            pred_inner[appliance].append(pd.DataFrame(HBAT[:len(test_ix_inner), appliance_index[appliance], :], index=test_ix_inner))

    err = {}
    appliance_to_weight = []
    for appliance in APPLIANCES_ORDER[1:]:
        pred_inner[appliance] = pd.DataFrame(pd.concat(pred_inner[appliance]))

        try:
            if appliance == "hvac":
                err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance][range(5-start, 11-start)], target, start, stop)[2]
            else:
                err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance], target, start, stop)[2]
            appliance_to_weight.append(appliance)

        except Exception, e:
            # This appliance does not have enough samples. Will not be
            # weighed
            print 'here'
            print(e)
            print(appliance)
    err_weight = {}
    for appliance in appliance_to_weight:
        err_weight[appliance] = err[appliance]*contri[target][appliance]
    mean_err = pd.Series(err_weight).sum()
    # print num_home_factors_cv,num_season_factors_cv,num_iterations_cv, learning_rate_cv, mean_err
    # error[num_iterations_cv][num_season_factors_cv][num_home_factors_cv][lam_cv] = mean_err
    return mean_err

pred = {}
n_splits = 10

for appliance in APPLIANCES_ORDER:
    pred[appliance] = []
best_params_global = {}
kf = KFold(n_splits=n_splits)



for outer_loop_iteration, (train_max, test) in enumerate(kf.split(target_df)):
    # Just a random thing
    np.random.seed(10 * random_seed + 7 * outer_loop_iteration)
    np.random.shuffle(train_max)
    print("-" * 80)
    print("Progress: {}".format(100.0 * outer_loop_iteration / n_splits))
    print(datetime.datetime.now())
    sys.stdout.flush()
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
        train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
        train_ix = target_df.index[train]
        test_ix = target_df.index[test]


    print("-" * 80)
    print("Test set {}".format(test_ix.values))
    print("-" * 80)
    print("Current Error, Least Error, #Iterations")

    ### Inner CV loop to find the optimum set of params. In this case: the number of iterations
    inner_kf = KFold(n_splits=2)

    best_learning_rate = 0.1
    best_num_iterations = 1300
    best_num_season_factors = 2
    best_num_home_factors = 3
    best_lam = 0
    least_error = 1e6

    overall_df_inner = target_df.loc[train_ix]
    best_params_global[outer_loop_iteration] = {}

    params = {}
    count = 0


    ##############################################################
    # Parallel part
    results = []
    cpus = mp.cpu_count()
    pool = mp.Pool(64)
    for learning_rate_cv in [0.1,0.5, 1]:
        for num_iterations_cv in [1300, 700, 100][:]:
            for num_season_factors_cv in range(2, 5)[:]:
                for num_home_factors_cv in range(3, 6)[:]:
                    
                    if setting == 'transfer':
                        A_source = A_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][num_iterations_cv]
                        B_source = B_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][num_iterations_cv]
                    else: 
                        A_source = None
                        B_source = None

                    
                    params[count] = []
                    params[count].extend((overall_df_inner, learning_rate_cv, num_iterations_cv, num_season_factors_cv, num_home_factors_cv, B_source, A_source))
                    count += 1
    for i in range(count): 
        result = pool.apply_async(compute_inner_error, params[i])
        results.append(result)
    pool.close()
    pool.join()
    # End of parallel part
    ###############################################################

    # get the results of all processes
    error = []
    for result in results:
        error.append(result.get())
    # get the parameters for the best setting
    best_idx = np.argmin(error)
    overall_df_inner, best_learning_rate, best_num_iterations, best_num_season_factors, best_num_home_factors, B, A = params[best_idx]
    least_error = error[best_idx]

    best_params_global[outer_loop_iteration] = {'Learning Rate': best_learning_rate,
                                                'Iterations': best_num_iterations,
                                                'Num season factors': best_num_season_factors,
                                                'Num home factors': best_num_home_factors,
                                                'Least Train Error': least_error}

    print("******* BEST PARAMS *******")
    print(best_params_global[outer_loop_iteration])
    print("******* BEST PARAMS *******")
    sys.stdout.flush()
    # Now we will be using the best parameter set obtained to compute the predictions
    if setting=="transfer":
        A_source = A_store[best_learning_rate][best_num_season_factors][best_num_home_factors][best_num_iterations]
        B_source = B_store[best_learning_rate][best_num_season_factors][best_num_home_factors][best_num_iterations]
    else:
        A_source = None
        B_source = None
    num_test = len(test_ix)
    train_test_ix = np.concatenate([test_ix, train_ix])
    df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
    tensor = get_tensor(df_t, start, stop)
    tensor_copy = tensor.copy()
    # First n
    tensor_copy[:num_test, 1:, :] = np.NaN

    # L = target_L[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]

    H, B, A, T, Hs, Bs, As, Ts, HBATs, costs = learn_HBAT_adagrad_graph(tensor_copy,
                                                                        best_num_home_factors,
                                                                        best_num_season_factors,
                                                                        num_iter = best_num_iterations,
                                                                        lr = best_learning_rate, 
                                                                        B_known = B_source,
                                                                        A_known = A_source,
                                                                        T_known = T_constant)


    HBAT = multiply_HBAT(H, B, A, T)
    for appliance in APPLIANCES_ORDER:
        pred[appliance].append(pd.DataFrame(HBAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
    pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

out = {'Predictions': pred, 'Learning Params': best_params_global}

with open(filename, 'wb') as f:
    pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
