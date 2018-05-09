import numpy as np
from sklearn.model_selection import KFold

np.random.seed(0)

# tensor = np.load('../2015-5appliances.numpy.npy')

APPLIANCE_ORDER = ['aggregate', 'hvac', 'fridge', 'dr', 'dw', 'mw', 'residual']
#ON_THRESHOLD = {'dr': 149.88716870476219,'dw': 33.557622495205969,'fridge': 12.55968132019043,'hvac': 136.5926835004021,'mw': 15.73143377865062}
#ON_THRESHOLD = {'dr': 299.77433740952438,'dw': 67.115244990411938,'fridge': 25.11936264038086,'hvac': 273.18536700080421,'mw': 31.462867557301241}
ON_THRESHOLD = {'dr': 419.68407237333417,'dw': 93.96134298657671,'fridge': 35.167107696533208,'hvac': 382.45951380112592,'mw': 44.048014580221739}


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
    if dataset == 3:
        tensor = np.load('../2015-5appliances-subtract-true-agg.npy')
    if dataset == 4:
        tensor = np.load('../2015-5appliances-sum-true-agg.npy')
    if dataset == 5:
        tensor = np.load('../2015-5appliances-true-agg-residual.npy')
        
    num_homes = tensor.shape[0]
    
    k = KFold(n_splits=num_folds)
    train, test = list(k.split(range(0, num_homes)))[fold_num]
    return tensor[train, :, :, :], tensor[test, :, :, :]


def get_train_test_tensor(tensor, num_folds=5, fold_num=0):
    
    num_homes= tensor.shape[0]
    print(tensor.shape)
    
    k = KFold(n_splits=num_folds)
    train, test = list(k.split(range(0, num_homes)))[fold_num]
    return tensor[train, :, :, :], tensor[test, :, :, :]
    
    
