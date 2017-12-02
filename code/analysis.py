import numpy as np
from tensor_custom_core import *
import sys

tf_type, freq, r = sys.argv[1:]
r = int(r)

tensor = np.load('../{}-input.npy'.format(freq))

t = tensor[:, :, :, :]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
pred = np.zeros_like(t)
pred[:] = np.nan

for train, test in kf.split(t):
    print("*"*20, train, test)
    sys.stdout.flush()
    t_copy = t.copy()
    t_copy[test][1:, :, :] = np.nan
    if tf_type=="STF":
        home, appliance, day, hour = stf_4dim(tensor=t_copy, r=r, num_iter=50, lr=2)
        pred[test] = np.einsum("Hr, Ar, Dr, Tr ->HADT", home, appliance, day, hour)[test]
    elif tf_type=="MTF":
        home, appliance, day, hour = stf_4dim_time(tensor=t_copy, r=r, num_iter=50, lr=2)
        pred[test] = np.einsum("Hr, Ar, Dr, ATr ->HADT", home, appliance, day, hour)[test]


np.save("../{}-{}-{}-pred-hourly.npy".format(tf_type, freq, r), pred)

