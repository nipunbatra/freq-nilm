import sys
sys.path.append("../code/")
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from tensor_custom_core import stf_4dim, stf_4dim_time
import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(0)
np.random.seed(0)


class CustomRNN(nn.Module):
    def __init__(self, cell_type, hidden_size, num_layers, bidirectional):
        super(CustomRNN, self).__init__()
        torch.manual_seed(0)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        if cell_type == "RNN":
            self.rnn = nn.RNN(input_size=3, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(input_size=3, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=3, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * self.num_directions, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.linear(pred)

        # pred = pred[:, :, 23:24]
        # pred = self.act(pred)
        # pred = torch.clamp(pred, min=0.)
        # pred = self.act(pred)
        # pred = torch.min(pred, x)
        return pred


num_folds = 5

if torch.cuda.is_available():
    cuda_av = True
else:
    cuda_av = False

c = CustomRNN('GRU',100, 1, False)
c

fold_num = 0
num_folds = 5
cell_type="LSTM"
hidden_size = 120
lr = 1
bidirectional = False
appliance = "dr"

hours = np.arange(1, 25, 1)

d=pd.DataFrame([np.sin(2 * np.pi * hours/24), np.cos(2 * np.pi * hours/24)]).T + 1.

torch.manual_seed(0)

appliance_num = APPLIANCE_ORDER.index(appliance)
train, test = get_train_test(num_folds=num_folds, fold_num=fold_num)
train_aggregate = train[:, 0, :, :].reshape(-1, 24, 1)
test_aggregate = test[:, 0, :, :].reshape(-1, 24, 1)

train_appliance = train[:, appliance_num, :, :].reshape(-1, 24, 1)
test_appliance = test[:, appliance_num, :, :].reshape(-1, 24, 1)

train_aggregate_time = np.zeros((train_aggregate.shape[0], 24, 3))
for home in range(train_aggregate.shape[0]):
    temp = d.copy()
    temp['power'] = train_aggregate[home, :, :]
    train_aggregate_time[home, :, :] = temp.values

loss_func = nn.L1Loss()
r = CustomRNN(cell_type, hidden_size, 1, bidirectional)

if cuda_av:
    r = r.cuda()
    loss_func = loss_func.cuda()

optimizer = torch.optim.Adam(r.parameters(), lr=lr)

num_iterations = 1000
for t in range(num_iterations):

    inp = Variable(torch.Tensor(train_aggregate_time), requires_grad=True)
    train_y = Variable(torch.Tensor(train_appliance))
    if cuda_av:
        inp = inp.cuda()
        train_y = train_y.cuda()
    pred = r(inp)
    print(pred.std().data[0], pred.mean().data[0])
    optimizer.zero_grad()
    loss = loss_func(pred, train_y)
    if t % 1 == 0:
        print(t, loss.data[0])
    loss.backward()
    optimizer.step()

