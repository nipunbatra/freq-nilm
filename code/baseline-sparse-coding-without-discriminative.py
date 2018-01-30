import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from ddsc import SparseCoding, reshape_for_sc
from sklearn.metrics import mean_absolute_error
num_folds = 5

def non_discriminative(num_latent):
    out = []
    for cur_fold in range(5):
        train, test = get_train_test(num_folds=num_folds, fold_num=cur_fold)
        train_sc, test_sc = reshape_for_sc(train), reshape_for_sc(test)
        train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
        c = SparseCoding()
        c.train(train_data, num_latent=num_latent)
        pred = c.disaggregate(test_sc[:, :, 0].swapaxes(0, 1)).swapaxes(0, 2).swapaxes(1, 2)
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 24)
        out.append(pred)
    return np.concatenate(out)

tensor = np.load('../2015-5appliances-true-agg.npy')


err_non_disc ={}
gt = tensor[:, 1:, :, :]
for num_latent in range(1, 50):
    print(num_latent)
    pred = non_discriminative(num_latent)
    # Clamping prediction to aggregate
    pred = np.minimum(pred, tensor[:, 0:1, :,:])
    err_non_disc[num_latent] = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}
np.save("./baseline/baseline-sparse-coding-non-disc-set2.npy", err_non_disc)

