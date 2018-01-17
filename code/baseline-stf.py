import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from tensor_custom_core import stf_4dim, stf_4dim_time
num_folds = 5

def stf(r=2, lr=1, num_iter=100):
    out = []
    for cur_fold in range(5):
        train, test = get_train_test(num_folds=num_folds, fold_num=cur_fold)
        test_copy = test.copy()
        test_copy[:, 1:, :, :] = np.NaN
        train_test = np.concatenate([train, test_copy])
        H, A, D, T = stf_4dim(tensor=train_test,r=r,lr=lr, num_iter=num_iter)
        pred = np.einsum("Hr, Ar, Dr, Tr ->HADT", H, A, D, T)[len(train):, 1:, :, :]
        out.append(pred)
    return np.concatenate(out)


tensor = np.load('../2015-5appliances.numpy.npy')
from sklearn.metrics import mean_absolute_error

err_stf ={}
gt = tensor[:, 1:, :, :]
for num_latent in range(1, 2):
	err_stf[num_latent] = {}
	for lr in [0.01, 0.1, 1, 2]:
		err_stf[num_latent][lr] = {}
		for iters in range(100, 2500, 400):
			print (num_latent, lr, iters)
			pred = stf(num_latent, lr, iters)
			pred = np.minimum(pred, tensor[:, 0:1, :, :])
			err_stf[num_latent][lr][iters] = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}

import pickle
pickle.dump(err_stf, open("./baseline-stf.pkl", 'w'))