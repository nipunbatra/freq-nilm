import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from tensor_custom_core import stf_4dim, stf_4dim_time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

num_folds = 5


def nested_stf():
    valid_error = {}
    out = []
    for cur_fold in range(5):
        valid_error[cur_fold] = {}
        train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
        train, valid = train_test_split(train, test_size=0.2)
        valid_gt = valid[:, 1:, :, :]

        
        for r in range(1, 21):
            valid_error[cur_fold][r] = {}
            for lr in [0.01, 0.1, 1, 2]:
                valid_error[cur_fold][r][lr] = {}
                for num_iter in range(100, 2600, 400):
                    print ("fold: ", cur_fold, " num_latent: ", r, " lr: ", lr, " num_iter: ", num_iter)
                    valid_copy = valid.copy()
                    valid_copy[:, 1:, :, :] =np.NaN
                    train_valid = np.concatenate([train, valid_copy])
                    H, A, D, T = stf_4dim_time(tensor=train_valid, r=r, lr=lr, num_iter=num_iter)
                    pred = np.einsum("Hr, Ar, Dr, ATr -> HADT", H, A, D, T)[len(train):, 1:, :, :]
                    valid_error[cur_fold][r][lr][num_iter] = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       valid_gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}

        min_error = np.inf
        for r in range(1, 21):
            for lr in [0.01, 0.1, 1, 2]:
                for num_iter in range(100, 2600, 400):
                    error = pd.Series(valid_error[cur_fold][r][lr][num_iter]).mean()
                    if min_error > error:
                        min_error = error
                        best_r = r
                        best_lr = lr
                        best_num_iter = num_iter
                    print (best_r, best_lr, best_num_iter, min_error)

        # use the best parameters to train model and test on test homes
        train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
        test_copy = test.copy()
        test_copy[:, 1:, :, :] = np.NaN
        train_test = np.concatenate([train, test_copy])
        H, A, D, T = stf_4dim_time(tensor=train_test,r=best_r,lr=best_lr, num_iter=best_num_iter)
        pred = np.einsum("Hr, Ar, Dr, Tr ->HADT", H, A, D, T)[len(train):, 1:, :, :]
        out.append(pred)

    return np.concatenate(out)


dataset= sys.argv[1]
dataset = int(dataset)


if dataset == 1:
    tensor = np.load('../2015-5appliances.numpy.npy')
if dataset == 2:
    tensor = np.load('../2015-5appliances-true-agg.npy')
gt = tensor[:, 1:, :, :]

pred = nested_stf()
pred = np.minimum(pred, tensor[:, 0:1, :, :])
err_stf = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}

np.save("./baseline/mtf-nested/baseline-mtf-{}.npy".format(dataset), err_stf)

# import pickle
#ipickle.dump(err_stf, open("./baseline-stf-{}-{}-{}.pkl".format(num_latent, lr, iters), 'wb'))
