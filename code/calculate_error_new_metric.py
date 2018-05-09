import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets,transforms
import torch.nn.functional as F
import sys
sys.path.append("../code/")
from dataloader import APPLIANCE_ORDER, get_train_test, ON_THRESHOLD
from sklearn.metrics import mean_absolute_error
import os
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../code/')
%matplotlib inline
import itertools
from pathlib import Path

def onoff_error(pred, gt, threshold):
    abs_error = np.abs(pred-gt)
    error = [x for x in abs_error.reshape(1, -1).tolist()[0] if x >= threshold]
    
    return np.mean(error)

def calculate_error(pred, gt, threshold):
    error = {}
    overall = {}
    num_homes = {}
    
    # calculte number of homes in each fold
    for fold_num in range(5):
        num_homes[fold_num] = gt[fold_num]['hvac'].reshape(-1, 1, 112,24).shape[0]
    homes = pd.Series(num_homes).sum()

    for appliance in ['hvac', 'fridge', 'dr', 'dw', 'mw']:
        error[appliance] = {}
        overall[appliance] = 0                                                                
        for fold_num in range(5):
            error[appliance][fold_num] = onoff_error(pred[fold_num][appliance].reshape(-1, 24), 
                                                     gt[fold_num][appliance].reshape(-1, 24), threshold[appliance])
            overall[appliance] += error[appliance][fold_num]*num_homes[fold_num]
        overall[appliance] /= homes
    
    return error, overall

def calculate_tf_valid_error(method, threshold):
	# load the results
	r = {}
	mean_r = {}
	for dataset in [1, 3]:
	    r[dataset] = {}
	    mean_r[dataset] = {}
	    for cur_fold in range(5):
	        r[dataset][cur_fold] = {}
	        mean_r[dataset][cur_fold] = {}
	        for num_latent in range(1, 21):
	            r[dataset][cur_fold][num_latent] = {}
	            mean_r[dataset][cur_fold][num_latent] = {}
	            for lr in [0.01, 0.1 ,1 ,2]:
	                lr = float(lr)
	                r[dataset][cur_fold][num_latent][lr] = {}
	                mean_r[dataset][cur_fold][num_latent][lr] = {}
	                for iters in range(100, 2600, 400):
	                    r[dataset][cur_fold][num_latent][lr][iters] = np.load("../code/baseline/{}/{}/valid/{}-pred-{}-{}-{}-{}-{}.npy".format(method, dataset, method, dataset, cur_fold, num_latent, lr, iters))

	# calculate error
	error = {}
	best_p = {}
	errors = {}
	for dataset in [1, 3]:
	    error[dataset] = {}
	    errors[dataset] = {}
	    best_p[dataset] = {}
	    for cur_fold in range(5):
	        error[dataset][cur_fold] = np.inf
	        errors[dataset][cur_fold] = {}
	        best_p[dataset][cur_fold] = {}

	        for num_latent in range(1, 21):
	        	errors[dataset][cur_fold][num_latent] = {}
	            for lr in [0.01, 0.1, 1, 2]:
	            	errors[dataset][cur_fold][num_latent][lr] = {}
	                for iters in range(100, 2600, 400):
	                	errors[dataset][cur_fold][num_latent][lr][iters] = {}
	                    cur_error = 0
	                    for idx, appliance in enumerate(APPLIANCE_ORDER[1:-1]):

	                    	errors[dataset][cur_fold][num_latent][lr][iters][appliance] = onoff_error(r[dataset][cur_fold][num_latent][lr][iters][:, idx].reshape(-1, 1), 
	                                                        valid_gt[cur_fold][appliance].reshape(-1, 1), threshold[cur_fold][appliance])
	                    	
	                        cur_error += errors[dataset][cur_fold][num_latent][lr][iters][appliance]
	                    if cur_error < error[dataset][cur_fold]:
	                        error[dataset][cur_fold] = cur_error
	                        best_p[dataset][cur_fold]['num_latent'] = num_latent
	                        best_p[dataset][cur_fold]['lr'] = lr
	                        best_p[dataset][cur_fold]['iters'] = iters                

	return errors, error, best_p

def calculate_cnn_valid_error(dataset, threshold):
	cnn_individual_valid_pred = {}
	for fold_num in range(5):
	    cnn_individual_valid_pred[fold_num] = {}
	    for appliance in ['hvac', 'fridge', 'dr', 'dw', 'mw']:
	        cnn_individual_valid_pred[fold_num][appliance] = {}
	        for lr in [0.001, 0.01, 0.1]:
	            cnn_individual_valid_pred[fold_num][appliance][lr] = {}
	            for iters in [200000]:
	            
	                directory = "../code/baseline/cnn-tree/{}/{}/{}/{}/0.0/".format(dataset, fold_num, lr, iters)
	                filename = "valid-pred-[\'{}\'].npy".format(appliance)
	                
	                full_path = directory + filename
	                my_file = Path(full_path)
                    k = np.load(full_path).item()

                    for it in range(1000, 200000+1, 1000):
                        cnn_individual_valid_pred[fold_num][appliance][lr][it] = k[it][0]
    cnn_errors = {}
    cnn_individual_best_param = {}
    min_error = {}
	for fold_num in range(5):
	    cnn_individual_best_param[fold_num] = {}
	    cnn_errors[fold_num] = {}
	    min_error[fold_num] = {}
	    for appliance in ['hvac', 'fridge', 'dr', 'dw', 'mw']:
	        cnn_individual_best_param[fold_num][appliance] = {}
	        cnn_errors[fold_num][appliance] = {}
	        min_error[fold_num][appliance] = np.inf
	        for lr in [0.001, 0.01, 0.1]:
	        	cnn_errors[fold_num][appliance][lr] = {}
	            for it in range(1000, 200000+1, 1000):
	                error = onoff_error(cnn_individual_valid_pred[fold_num][appliance][lr][it].reshape(-1, 24), valid_gt[fold_num][appliance].reshape(-1, 24), threshold[appliance])
	                cnn_errors[fold_num][appliance][lr][it] = error
	                if error < min_error[fold_num][appliance]:
	                    cnn_individual_best_param[fold_num][appliance]['lr'] = lr
	                    cnn_individual_best_param[fold_num][appliance]['iters'] = it
	                    min_error[fold_num][appliance] = error

	return cnn_errors, min_error, cnn_individual_best_param

	# calculate gt

def calculate_cnn_tree_valid_error(dataset, threshold):

	cnn_tree_valid_pred = {}
	num_iterations = 20000

	for fold_num in range(5):
	    cnn_tree_valid_pred[fold_num] = {}
	    for lr in [0.01]:
	        cnn_tree_valid_pred[fold_num][lr] = {}
	        for order in list(itertools.permutations(['hvac', 'fridge', 'dr', 'dw', 'mw'])):
	            
	            if order[0] == 'hvac':
	                continue
	            
	            cnn_tree_valid_pred[fold_num][lr][order] = {}


	            o = "\', \'".join(str(x) for x in order)
	            directory = "../code/baseline/cnn-tree/{}/{}/{}/20000/0.0/".format(dataset, fold_num, lr)
	            filename = "valid-pred-[\'{}\'].npy".format(o)

	            full_path = directory + filename
	            my_file = Path(full_path)
	            
                k = np.load(full_path).item()
                for it in range(1000, 20001, 1000):
                    cnn_tree_valid_pred[fold_num][lr][order][it] = {}
                    for idx, appliance in enumerate(order):
                        cnn_tree_valid_pred[fold_num][lr][order][it][appliance] = k[it][idx]

    cnn_tree_best_param = {}
    cnn_tree_errors = {}
    min_error = {}
	for fold_num in range(5):
	    cnn_tree_best_param[fold_num] = {}
	    min_error[fold_num] = np.inf
	    cnn_tree_errors[fold_num] = {}
	    for lr in [0.01]:
	    	cnn_tree_errors[fold_num][lr] = {}
	        for order in list(itertools.permutations(['hvac', 'fridge', 'dr', 'dw', 'mw'])):
	            if order[0] == 'hvac':
	                continue
	            cnn_tree_errors[fold_num][lr][order] = {}
	            for it in range(1000, 20001, 1000):
	                error = 0
	                cnn_tree_errors[fold_num][lr][order][it] = {}
	                for idx, appliance in enumerate(order):
	                	cnn_tree_errors[fold_num][lr][order][it][appliance] = onoff_error(cnn_tree_valid_pred[fold_num][lr][order][it][appliance].reshape(-1, 24),
	                                                valid_gt[fold_num][appliance].reshape(-1, 24), threshold[appliance])
	                	error += cnn_tree_errors[fold_num][lr][order][it][appliance]
	                if error < min_error[fold_num]:
	                    min_error[fold_num] = error
	                    cnn_tree_best_param[fold_num]['lr'] = lr
	                    cnn_tree_best_param[fold_num]['order'] = order
	                    cnn_tree_best_param[fold_num]['iters'] = it

    return cnn_tree_errors, min_error, cnn_tree_best_param        
                    

tensor = np.load("../2015-5appliances.numpy.npy")
test_gt = {}
valid_gt = {}
for fold_num in range(5):
    test_gt[fold_num] = {}
    valid_gt[fold_num] = {}
    train, test = get_train_test(1, 5, fold_num)
    valid = train[int(0.8*len(train)):].copy()
    for idx, appliance in enumerate(APPLIANCE_ORDER[1:-1]):
        test_gt[fold_num][appliance] = test[:, idx+1]
        valid_gt[fold_num][appliance] = valid[:, idx+1]

threshold = {}
for appliance in ['hvac', 'fridge', 'dr', 'dw', 'mw']:
    sample_list = []
    for fold_num in range(5):
        sample_list = np.append(sample_list, [x for x in test_gt[fold_num][appliance].reshape(1, -1).tolist()[0] if x > ON_THRESHOLD[appliance]])
    mean = np.mean(sample_list)
    print(appliance, mean)
    threshold[appliance] = 0.1*mean

x = sys.argv[1]
directory = "./baseline/{}".format(x)

if not os.path.exists(directory):
    os.makedirs(directory)

np.save("{}/on_threshold.txt".format(directory), ON_THRESHOLD)
np.save("{}/threshold.txt".format(directory), threshold)

stf_errors, stf_best_error, stf_best_param = calculate_tf_valid_error('stf', threshold)
mtf_errors, mtf_best_error, mtf_best_param = calculate_tf_valid_error('mtf', threshold)

cnn_ind_errors = {}
cnn_ind_best_error = {}
cnn_ind_best_param = {}
cnn_tree_errors = {}
cnn_tree_best_error = {}
cnn_tree_best_param = {}
for dataset in [1, 3]:
	cnn_ind_errors[dataset], cnn_ind_best_error[dataset], cnn_ind_best_param[dataset] = calculate_cnn_valid_error(dataset, threshold)
	cnn_tree_errors[dataset], cnn_tree_best_error[dataset], cnn_tree_best_param[dataset] = calculate_cnn_tree_valid_error(dataset, threshold)

np.save("{}/stf_errors.npy".format(directory), stf_errors)
np.save("{}/stf_best_error.npy".format(directory), stf_best_error)
np.save("{}/stf_best_param.npy".format(directory), stf_best_param)
np.save("{}/mtf_errors.npy".format(directory), mtf_errors)
np.save("{}/mtf_best_error.npy".format(directory), mtf_best_error)
np.save("{}/mtf_best_param.npy".format(directory), mtf_best_param)
np.save("{}/cnn_ind_errors.npy".format(directory), cnn_ind_errors)
np.save("{}/cnn_ind_best_error.npy".format(directory), cnn_ind_best_error)
np.save("{}/cnn_ind_best_param.npy".format(directory), cnn_ind_best_param)
np.save("{}/cnn_tree_errors.npy".format(directory), cnn_tree_errors)
np.save("{}/cnn_tree_best_error.npy".format(directory), cnn_tree_best_error)
np.save("{}/cnn_tree_best_param.npy".format(directory), cnn_tree_best_param)



