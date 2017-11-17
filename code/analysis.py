import pickle
from tensor_custom_core import *
import sys

r = int(sys.argv[1])

tensor = pickle.load(open('../hourly.pkl','r'))

t = tensor[:, :, :, :]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
pred = np.zeros_like(t)
pred[:] = np.nan

for train, test in kf.split(t):
    print "*"*20, train, test
    sys.stdout.flush()
    t_copy = t.copy()
    t_copy[test][1:, :, :] = np.nan
    home, appliance, day, hour = stf_4dim(tensor=t_copy, r=r, num_iter=50, lr=2)
    pred[test] = np.einsum("Hr, Ar, Dr, Tr ->HADT", home, appliance, day, hour)[test]

pickle.dump(pred, open("../{}-pred-hourly.pkl".format(r),'w'))

