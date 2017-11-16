import pickle
from tensor_custom_core import *



tensor = pickle.load(open('hourly.pkl','r'))



home, appliance, day, hour = stf_4dim(tensor=tensor[:20, :, :, :], r=2, num_iter=500, lr=2)
from common import APPLIANCES_ORDER

t = tensor[:, :, :, :]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
pred = np.zeros_like(t)
pred[:] = np.nan

for train, test in kf.split(t):
    print "*"*20, train, test
    t_copy = t.copy()
    t_copy[test][1:, :, :] = np.nan
    home, appliance, day, hour = stf_4dim(tensor=t_copy, r=2, num_iter=1000, lr=2)
    pred[test] = np.einsum("Hr, Ar, Dr, Tr ->HADT", home, appliance, day, hour)[test]



