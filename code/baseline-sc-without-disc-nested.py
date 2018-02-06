import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from ddsc import SparseCoding, reshape_for_sc
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

num_folds = 5

def non_discriminative(dataset, cur_fold, num_latent):
    out = []
    # valid_error
    # for cur_fold in range(5):
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
    train, valid = train_test_split(train, test_size=0.2, random_state=0)

    valid_gt = valid[:, 1:, :, :]


    train_sc, valid_sc = reshape_for_sc(train), reshape_for_sc(valid)
    train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
    c = SparseCoding()
    c.train(train_data, num_latent=num_latent)
    pred = c.disaggregate(valid_sc[:, :, 0].swapaxes(0, 1)).swapaxes(0, 2).swapaxes(1, 2)
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 24)

    pred = np.minimum(pred, valid_gt[:, 0:1, :, :])

    valid_error = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       valid_gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}
    # out.append(pred)

    return pred, valid_error

dataset, cur_fold, num_latent = sys.argv[1:]
dataset = int(dataset)
cur_fold = int(cur_fold)
num_latent = int(num_latent)

pred, error = non_discriminative(dataset, cur_fold, num_latent)

np.save("./baseline/sc-non-nested/sc-non-pred-{}-{}-{}.npy".format(dataset, cur_fold, num_latent), pred)
np.save("./baseline/sc-non-nested/sc-non-error-{}-{}-{}.npy".format(dataset, cur_fold, num_latent), error)

