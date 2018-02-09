import numpy as np
import pandas as pd

err = {}
for dataset in [1, 2]:
    err[dataset] = {}
    for cur_fold in range(5):
        err[dataset][cur_fold] = {}
        for r in range(1 ,21):
            err[dataset][cur_fold][r] = {}
            for lr in [0.01, 0.1, 1.0, 2.0]:
                err[dataset][cur_fold][r][lr] = {}
                for num_iter in range(100, 2600, 400):
                    err[dataset][cur_fold][r][lr][num_iter] = \
                    np.load("../code/baseline/stf-nested/stf-error-{}-{}-{}-{}-{}.npy".format(dataset, cur_fold, r, lr, num_iter)).item()


best_r = {}
best_lr = {}
best_iter = {}
for dataset in [1, 2]:
    best_r[dataset] = {}
    best_lr[dataset] = {}
    best_iter[dataset] = {}
    for cur_fold in range(5):
        min_error = np.inf
        for r in range(1, 21):
            for lr in [0.01, 0.1, 1.0, 2.0]:
                iter = pd.DataFrame(err[dataset][cur_fold][r][lr]).mean().idxmin()
                error = pd.DataFrame(err[dataset][cur_fold][r][lr]).mean()[iter]
                if error < min_error:
                    min_error = error
                    best_r[dataset][cur_fold] = r
                    best_lr[dataset][cur_fold] = lr
                    best_iter[dataset][cur_fold] = iter



import sys
sys.path.append("../code/")
from dataloader import APPLIANCE_ORDER, get_train_test
from tensor_custom_core import stf_4dim, stf_4dim_time

def stf(dataset, cur_fold, r=2, lr=1, num_iter=100):
    num_folds=5
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
    test_copy = test.copy()
    test_copy[:, 1:, :, :] = np.NaN
    train_test = np.concatenate([train, test_copy])
    H, A, D, T = stf_4dim(tensor=train_test, r=r, lr=lr, num_iter=num_iter)
    pred = np.einsum("Hr, Ar, Dr, Tr ->HADT", H, A, D, T)[len(train):, 1:, :, :]
    
    return pred

dataset = 2
out = []
num_fold=5
for cur_fold in range(5):
    b_r= best_r[dataset][cur_fold]
    b_lr = best_lr[dataset][cur_fold]
    b_iter = best_iter[dataset][cur_fold]
    
    pred = stf(dataset, cur_fold, b_r, b_lr, b_iter)
    
    out.append(pred)


cont_out = np.concatenate(out)
tensor = np.load('../2015-5appliances.numpy.npy')
gt = tensor[:, 1:, :, :]


np.save("./stf-nested-pred-2.npy", cont_out)
