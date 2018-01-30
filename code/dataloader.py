import numpy as np
from sklearn.model_selection import KFold

np.random.seed(0)

# tensor = np.load('../2015-5appliances.numpy.npy')

APPLIANCE_ORDER = ['aggregate', 'hvac', 'fridge', 'dr', 'dw', 'mw']


def get_train_test(dataset, num_folds=5, fold_num=0):
    """

    :param num_folds: number of folds
    :param fold_num: which fold to return
    :return:
    """

    if dataset == 1:
    	tensor = np.load('../2015-5appliances.numpy.npy')
    if dataset == 2:
    	tensor = np.load('../2015-5appliances-true-agg.npy')

    num_homes = tensor.shape[0]
    
    k = KFold(n_splits=num_folds)
    train, test = list(k.split(range(0, num_homes)))[fold_num]
    return tensor[train, :, :, :], tensor[test, :, :, :]
