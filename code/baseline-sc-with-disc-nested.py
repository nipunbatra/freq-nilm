import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from ddsc import SparseCoding, reshape_for_sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



num_folds = 5

def discriminative(dataset, cur_fold, num_latent, num_iterations):

    # for cur_fold in range(5):
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
    train, valid = train_test_split(train, test_size=0.2, random_state=0)

    valid_gt = valid[:, 1:, :, :]

    train_sc, valid_sc = reshape_for_sc(train), reshape_for_sc(valid)
    train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
    c = SparseCoding()
    c.train(train_data, num_latent=num_latent)
    pred = c.disaggregate_discriminative(train_sc[:, :, 0].swapaxes(0, 1), 
                                         valid_sc[:, :, 0].swapaxes(0, 1),
                                         num_iter=num_iterations)
    pred = pred[-1, :, :, :]
    pred = pred.swapaxes(0, 2).swapaxes(1, 2)
    # print(pred.shape)
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 24)
    
    pred = np.minimum(pred, valid_gt[:, 0:1, :, :])

    valid_error = {APPLIANCE_ORDER[i+1]:mean_absolute_error(pred[:, i,:,:].flatten(), 
                                                                       valid_gt[:, i, :, :].flatten()) for i in range(pred.shape[1])}

    return pred, valid_error


dataset, cur_fold, num_latent, num_iterations = sys.argv[1:]
dataset = int(dataset)
cur_fold = int(cur_fold)
num_latent = int(num_latent)
num_iterations = int(num_iterations)

pred, error = discriminative(dataset, cur_fold, num_latent, num_iterations)

np.save("./baseline/sc-with-nested/sc-with-pred-{}-{}-{}-{}.npy".format(dataset, cur_fold, num_latent, num_iterations), pred)
np.save("./baseline/sc-with-nested/sc-with-error-{}-{}-{}-{}.npy".format(dataset, cur_fold, num_latent, num_iterations), error)
