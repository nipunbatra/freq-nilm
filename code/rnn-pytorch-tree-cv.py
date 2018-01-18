import sys

num_hidden, num_iterations, num_layers, p, num_directions = sys.argv[1:6]
num_hidden = int(num_hidden)
num_layers = int(num_layers)
num_iterations = int(num_iterations)
p = float(p)
num_directions = int(num_directions)
ORDER = sys.argv[6:len(sys.argv)]

from sklearn.metrics import mean_absolute_error

import numpy as np

import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
num_folds = 5
train, test = get_train_test(num_folds=num_folds, fold_num=0)

train_agg = train[:, 0, :].reshape(-1, 24)
test_agg = test[:, 0, :].reshape(-1, 24)

import torch
import torch.nn as nn
from torch.autograd import Variable

cuda_av = False
if torch.cuda.is_available():
    cuda_av = True


torch.manual_seed(0)
np.random.seed(0)

input_dim = 1
hidden_size = num_hidden
num_layers = num_layers
if num_directions == 1:
    bidirectional = False
else:
    bidirectional = True


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=0.1, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * num_directions, output_size, )

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
        pred = torch.clamp(pred, min=0.)
        pred = torch.min(pred, x)

        return pred


# ORDER = ['hvac','fridge','oven','dw','mw','wm'][:3]
# ORDER = ['oven','fridge','hvac','dw','mw','wm'][:4]

class AppliancesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_appliance):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(input_size, hidden_size, output_size).cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(input_size, hidden_size, output_size))

    def forward(self, *args):
        agg_current = args[0]
        flag = False
        if np.random.random() > args[1]:
            flag = True
            # print("Subtracting prediction")
        else:
            pass
            # print("Subtracting true")
        for appliance in range(self.num_appliance):
            # print(agg_current.mean().data[0])
            # print appliance
            # print self.order[appliance]
            # print args[2+appliance]
            self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
            if flag:
                agg_current = agg_current - self.preds[appliance]

            else:
                agg_current = agg_current - args[2 + appliance]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])


a = AppliancesRNN(input_dim, hidden_size, 1, len(ORDER))
# print(cuda_av)
if cuda_av:
    a = a.cuda()
# print(a)
# Storing predictions per iterations to visualise later
predictions = []

optimizer = torch.optim.Adam(a.parameters(), lr=2)
loss_func = nn.L1Loss().cuda()

out_train = {}
for appliance in ORDER:
    out_train[appliance] = Variable(
        torch.Tensor(train[APPLIANCE_ORDER.index(appliance)].reshape((train_agg.shape[0], -1, 1))))
    if cuda_av:
        out_train[appliance] = out_train[appliance].cuda()

inp = Variable(torch.Tensor(train_agg.reshape((train_agg.shape[0], -1, 1))).type(torch.FloatTensor), requires_grad=True)
if cuda_av:
    inp = inp.cuda()
for t in range(num_iterations):
    import pdb

    # pdb.set_trace()
    out = torch.cat([out_train[appliance] for appliance in ORDER])

    # pred = a(inp, p)
    params = [inp, p]
    for appliance in ORDER:
        params.append(out_train[appliance])

    pred = a(*params)
    # pred = a(inp, p, out_train['oven'], out_train['fridge'], out_train['hvac'], out_train['dw'])

    optimizer.zero_grad()
    # predictions.append(pred.data.numpy())
    loss = loss_func(pred, out)
    # loss_0 = torch.split(pred, train_agg.shape[0])[0].mean()
    # loss = loss - loss_0
    if t % 1 == 0:
        print(t, loss.data[0])
    if not cuda_av:
        if t % 2 == 0:

            test_inp = Variable(torch.Tensor(test_agg.reshape((test_agg.shape[0], -1, 1))), requires_grad=True)
            params = [test_inp, -2]
            for i in range(len(ORDER)):
                params.append(None)
            test_pred = torch.split(a(*params), test_agg.shape[0])

            preds = {k: test_pred[i].data.numpy().reshape(-1, 24) for i, k in enumerate(ORDER)}
            errors = {}
            for appliance in ORDER:
                errors[appliance] = mean_absolute_error(eval("test_" + appliance), preds[appliance])
            print(pd.Series(errors))

    loss.backward()
    optimizer.step()

if cuda_av:
    test_inp = Variable(torch.Tensor(test_agg.reshape((test_agg.shape[0], -1, 1))).cuda(), requires_grad=True)
    params = [test_inp, -2]
    for i in range(len(ORDER)):
        params.append(None)
    test_pred = torch.split(a(*params), test_agg.shape[0])
    preds = {k: test_pred[i].cpu().data.numpy().reshape(-1, 24) for i, k in enumerate(ORDER)}
errors = {}
for appliance in ORDER:
    # print appliance
    errors[appliance] = mean_absolute_error(eval("test_" + appliance), preds[appliance])
print(pd.Series(errors))
