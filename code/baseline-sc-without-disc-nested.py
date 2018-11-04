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
    #train, valid = train_test_split(train, test_size=0.2, random_state=0)
    valid = train[int(0.8*len(train)):].copy()
    train = train[:int(0.8 * len(train))].copy()

    valid_gt = valid[:, 1:, :, :]
    test_gt = test[:, 1:, :, :]


    train_sc, valid_sc = reshape_for_sc(train), reshape_for_sc(valid)
    train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
    c = SparseCoding()
    c.train(train_data, num_latent=num_latent)
    valid_pred = c.disaggregate(valid_sc[:, :, 0].swapaxes(0, 1)).swapaxes(0, 2).swapaxes(1, 2)
    valid_pred = valid_pred.reshape(valid_pred.shape[0], valid_pred.shape[1], -1, 24)

    valid_pred = np.minimum(valid_pred, valid_gt[:, 0:1, :, :])

    valid_error = {APPLIANCE_ORDER[i+1]:mean_absolute_error(valid_pred[:, i,:,:].flatten(), 
                                                                       valid_gt[:, i, :, :].flatten()) for i in range(valid_pred.shape[1])}
    
    
    train_sc, test_sc = reshape_for_sc(train), reshape_for_sc(test)
    train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
    c = SparseCoding()
    c.train(train_data, num_latent=num_latent)
    test_pred = c.disaggregate(test_sc[:, :, 0].swapaxes(0, 1)).swapaxes(0, 2).swapaxes(1, 2)
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1], -1, 24)

    #test_pred = np.minimum(test_pred, test_gt[:, 0:1, :, :])

    test_error = {APPLIANCE_ORDER[i+1]:mean_absolute_error(test_pred[:, i,:,:].flatten(), 
                                                                       test_gt[:, i, :, :].flatten()) for i in range(test_pred.shape[1])}

    return valid_pred, valid_error, test_pred, test_error

dataset, cur_fold, num_latent = sys.argv[1:]
dataset = int(dataset)
cur_fold = int(cur_fold)
num_latent = int(num_latent)

directory = "./baseline/sc-non-nested/{}/".format(dataset)
if not os.path.exists(directory):
    os.makedirs(directory)
    
valid_pred, valid_error, test_pred, test_error = non_discriminative(dataset, cur_fold, num_latent)

np.save("./baseline/sc-non-nested/{}/sc-non-valid-pred-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent), valid_pred)
np.save("./baseline/sc-non-nested/{}/sc-non-valid-error-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent), valid_error)

np.save("./baseline/sc-non-nested/{}/sc-non-test-pred-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent), test_pred)
np.save("./baseline/sc-non-nested/{}/sc-non-test-error-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent), test_error)
