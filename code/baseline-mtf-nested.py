import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from tensor_custom_core import stf_4dim, stf_4dim_time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

num_folds = 5


def nested_stf(dataset, cur_fold, r, lr, num_iter):

    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
    train, valid = train_test_split(train, test_size=0.2, random_state=0)
    valid_gt = valid[:, 1:, :, :]


    valid_copy = valid.copy()
    valid_copy[:, 1:, :, :] =np.NaN
    train_valid = np.concatenate([train, valid_copy])
    H, A, D, T = stf_4dim_time(tensor=train_valid, r=r, lr=lr, num_iter=num_iter)
    pred = np.einsum("Hr, Ar, Dr, ATr -> HADT", H, A, D, T)[len(train):, 1:, :, :]
    valid_error[cur_fold][r][lr][num_iter] = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       valid_gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}
    
    return pred, valid_error


dataset, cur_fold, r, lr, num_iter= sys.argv[1:]
dataset = int(dataset)
cur_fold = int(cur_fold)
r = int(r)
lr = float(lr)
num_iter = int(num_iter)


pred, error = nested_stf(dataset, cur_fold, r, lr, num_iter)

np.save("./baseline/mtf-nested/mtf-pred-{}-{}-{}-{}-{}.npy".format(dataset, cur_fold, r, lr, num_iter), pred)
np.save("./baseline/mtf-nested/mtf-error-{}-{}-{}-{}-{}.npy".format(dataset, cur_fold, r, lr, num_iter), error)

# import pickle
#ipickle.dump(err_stf, open("./baseline-stf-{}-{}-{}.pkl".format(num_latent, lr, iters), 'wb'))
