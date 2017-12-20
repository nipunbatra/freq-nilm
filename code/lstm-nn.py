from common import APPLIANCES_ORDER
import sys

appliance = sys.argv[1]
appliance_num = APPLIANCES_ORDER.index(appliance)

import numpy as np
import pandas as pd

tensor = np.load('../1H-input.npy')


def create_subset_dataset(tensor):
    t_subset = tensor[:, :, 180:194, :]
    all_indices = np.array(list(range(320)))
    for i in range(1, 7):
        valid_homes = pd.DataFrame(t_subset[:, i, :].reshape(320, 14 * 24)).dropna().index
        all_indices = np.intersect1d(all_indices, valid_homes)
    t_subset = t_subset[all_indices, :, :, :].reshape(52, 7, 14 * 24)

    # Create artificial aggregate
    t_subset[:, 0, :] = 0.0
    for i in range(1, 7):
        t_subset[:, 0, :] = t_subset[:, 0, :] + t_subset[:, i, :]
    # t_subset is of shape (#home, appliance, days*hours)
    return t_subset, all_indices

t_all, valid_homes = create_subset_dataset(tensor)


train_agg = t_all[:30, 0, :].reshape(30*14, 24, 1)
train_appliance = t_all[:30, appliance_num, :].reshape(30*14, 24, 1)

test_appliance = t_all[30:, appliance_num, :].reshape(22*14, 24, 1)
test_agg = t_all[30:, 0, :].reshape(22*14, 24, 1)

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

n_features = 1
n_timesteps_in = 24
n_timesteps_out = 24
# define model
model = Sequential()
model.add(LSTM(250, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(250, return_sequences=True))
model.add(TimeDistributed(Dense(n_features, activation='linear')))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['acc'])

hist = model.fit(train_agg, train_appliance, epochs=400, verbose=2, validation_split=0.2)

pred = model.predict(test_agg)
pred = pred.reshape(308, 24)
test_appliance = test_appliance.reshape(308, 24)
pred[pred > test_appliance] = test_appliance[pred > test_appliance]

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(test_appliance, pred))