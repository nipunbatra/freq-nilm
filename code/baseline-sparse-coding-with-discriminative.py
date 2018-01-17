import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from ddsc import SparseCoding, reshape_for_sc

num_folds = 5

def discriminative(num_latent, num_iterations):
    print(num_iterations)
    out = []
    for cur_fold in range(5):
        train, test = get_train_test(num_folds=num_folds, fold_num=cur_fold)
        train_sc, test_sc = reshape_for_sc(train), reshape_for_sc(test)
        train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
        c = SparseCoding()
        c.train(train_data, num_latent=num_latent)
        pred = c.disaggregate_discriminative(train_sc[:, :, 0].swapaxes(0, 1), 
                                             test_sc[:, :, 0].swapaxes(0, 1),
                                             num_iter=num_iterations)
        pred = pred[-1, :, :, :]
        pred = pred.swapaxes(0, 2).swapaxes(1, 2)
        print(pred.shape)
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 24)
        out.append(pred)
    return np.concatenate(out)


tensor = np.load('../2015-5appliances.numpy.npy')
from sklearn.metrics import mean_absolute_error

err_disc ={}
gt = tensor[:, 1:, :, :]
# should be varied from [1, 30]
for num_latent in range(1, 50):
    err_disc[num_latent] = {}
    # Should be upto 100 or so iterations.
    for num_iterations in range(10, 110, 10):
        print(num_latent, num_iterations)
        pred = discriminative(num_latent, num_iterations)
        # Clamping prediction to aggregate
        pred = np.minimum(pred, tensor[:, 0:1, :,:])
        err_disc[num_latent][num_iterations] = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}

import pickle
pickle.dump(err_non_disc, open("./baseline-sparse-coding-disc.pkl"), 'w')