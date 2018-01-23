import sys
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
        if cell_type=="RNN":
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                   num_layers=num_layers, batch_first=True,
                   bidirectional=bidirectional)
        elif cell_type=="GRU":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size*self.num_directions, 1 )
        self.linear2 = nn.Linear(24 * 2, 24)
        self.act = nn.ReLU()

    def forward(self, x, m):
        pred, hidden = self.rnn(x, None)
        #print(pred.size())
        pred = self.linear(pred)
        #print(pred.size())
        m = m.expand_as(pred)
        #print(pred.size(), x.size(), m.size())
        z = torch.cat([pred, m], dim=1).view(-1, 48)
        #print(z.size())
        #print(self.linear2)
        pred = self.linear2(z)
        pred = self.act(pred)
        #print(pred.size(), x.size())
        pred = pred.view(pred.data.shape[0], -1, 1)
        pred = torch.min(pred, x)
        return pred


num_folds = 5

if torch.cuda.is_available():
    cuda_av = True
else:
    cuda_av=False

gts = []
preds = []

def disagg_fold(fold_num, appliance, cell_type, hidden_size,
                num_layers, bidirectional, lr,
                num_iterations):
    torch.manual_seed(0)

    appliance_num = APPLIANCE_ORDER.index(appliance)
    train, test = get_train_test(num_folds=num_folds, fold_num=fold_num)
    train_aggregate = train[:, 0, :, :].reshape(-1, 24, 1)
    test_aggregate = test[:, 0, :, :].reshape(-1, 24, 1)

    train_appliance = train[:, appliance_num, :, :].reshape(-1, 24, 1)
    test_appliance = test[:, appliance_num, :, :].reshape(-1, 24, 1)
    gts.append(test_appliance.reshape(-1, 24))
    loss_func = nn.L1Loss()
    r = CustomRNN(cell_type, hidden_size, num_layers, bidirectional)

    if cuda_av:
        r = r.cuda()
        loss_func = loss_func.cuda()
    #print(r)

    # Setting the params all to be non-negative
        #for param in r.parameters():
        #    param.data = param.data.abs()

    optimizer = torch.optim.Adam(r.parameters(), lr=lr)

    for t in range(num_iterations):

        inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)
        train_y = Variable(torch.Tensor(train_appliance))
        train_y_mean = train_y.mean(dim=0)
        #print(train_y_mean.size())
        if cuda_av:
            inp = inp.cuda()
            train_y = train_y.cuda()
            train_y_mean = train_y_mean.cuda()
        pred = r(inp, train_y_mean)

        optimizer.zero_grad()
        loss = loss_func(pred, train_y)
        if t % 1 == 0:
            print(t, loss.data[0])
        loss.backward()
        optimizer.step()

    test_inp = Variable(torch.Tensor(test_aggregate), requires_grad=False)
    test_y = Variable(torch.Tensor(test_appliance), requires_grad=False)
    if cuda_av:
        test_inp = test_inp.cuda()
    pred = r(test_inp, train_y_mean)
    #pred[pred<0.] = 0.
    pred = torch.clamp(pred, min=0.)
    if cuda_av:
        prediction_fold = pred.cpu().data.numpy()
    else:
        prediction_fold = pred.data.numpy()
    return prediction_fold, test_appliance

def disagg(appliance, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations):
    from sklearn.metrics import mean_absolute_error
    preds = []
    gts = []
    for cur_fold in range(num_folds):
        pred, gt = disagg_fold(cur_fold, appliance, cell_type, hidden_size, num_layers
                               ,bidirectional, lr, num_iterations)

        preds.append(pred)
        gts.append(gt)
    return mean_absolute_error(np.concatenate(gts).flatten(), np.concatenate(preds).flatten())

appliance = "hvac"
cell_type="GRU" # One of GRU, LSTM, RNN
hidden_size=100 # [20, 50, 100, 150]
num_layers=1  # [1, 2, 3, 4]
bidirectional=False # True or False
lr =2 # 1e-3, 1e-2, 1e-1, 1, 2
num_iterations = 20 #200, 400, 600, 800
p = disagg(appliance, cell_type, 180, 1,
                bidirectional, lr, 300)