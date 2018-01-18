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

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
        #pred = self.act(pred)
        #pred = torch.clamp(pred, min=0.)
        pred = torch.min(pred, x)
        return pred


class AppliancesRNN(nn.Module):
    def __init__(self,  cell_type, hidden_size, num_layers,
                 bidirectional, order):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = len(order)
        self.preds = {}
        self.order = order

        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type, hidden_size,
                                                                       num_layers, bidirectional).cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type, hidden_size,
                                                                       num_layers, bidirectional))

    def forward(self, aggregate, out_train, p):
        agg_current = aggregate
        flag = False
        if np.random.random() > p:
            flag = True
            # print("Subtracting prediction")
        else:
            pass
            # print("Subtracting true")
        for appliance_num in range(self.num_appliance):

            self.preds[appliance_num] = getattr(self, "Appliance_" + str(appliance_num))(agg_current)
            if flag:
                agg_current = agg_current - self.preds[appliance_num]

            else:
                agg_current = agg_current - out_train[appliance_num]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])


num_folds = 5

if torch.cuda.is_available():
    cuda_av = True
else:
    cuda_av=False


def disagg_fold(fold_num, cell_type, hidden_size,
                num_layers, bidirectional, lr,
                num_iterations, order, p):
    torch.manual_seed(0)

    train, test = get_train_test(num_folds=num_folds, fold_num=fold_num)
    train_aggregate = train[:, 0, :, :].reshape(-1, 24, 1)
    test_aggregate = test[:, 0, :, :].reshape(-1, 24, 1)

    out_train = [None for temp in range(len(order))]
    for a_num, appliance in enumerate(order):
        out_train[a_num] = Variable(
            torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train_aggregate.shape[0], -1, 1))))
        if cuda_av:
            out_train[a_num] = out_train[a_num].cuda()




    loss_func = nn.L1Loss()
    a = AppliancesRNN(cell_type, hidden_size, num_layers,
                 bidirectional, order)

    if cuda_av:
        a = a.cuda()
        loss_func = loss_func.cuda()



    optimizer = torch.optim.Adam(a.parameters(), lr=lr)

    for t in range(num_iterations):

        inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)
        out = torch.cat([out_train[appliance_num] for appliance_num, appliance in enumerate(order)])
        if cuda_av:
            inp = inp.cuda()
            out = out.cuda()

        pred = a(inp, out_train, p)

        optimizer.zero_grad()
        loss = loss_func(pred, out)
        if t % 5 == 0:
            print(t, loss.data[0])
        loss.backward()
        optimizer.step()

    test_inp = Variable(torch.Tensor(test_aggregate), requires_grad=False)
    if cuda_av:
        test_inp = test_inp.cuda()
    pr = a(test_inp, {appliance:None for appliance in order}, -2)
    pr = torch.clamp(pr, min=0.)
    test_pred = torch.split(pr, test_aggregate.shape[0])
    prediction_fold = [None for x in range(len(order))]
    if cuda_av:
        for appliance_num, appliance in enumerate(order):
            prediction_fold[appliance_num] = test_pred[appliance_num].cpu().data.numpy().reshape(-1, 24)
    else:
        for appliance_num, appliance in enumerate(order):
            prediction_fold[appliance_num] = test_pred[appliance_num].data.numpy().reshape(-1, 24)
    gt_fold = [None for x in range(len(order))]
    for appliance_num, appliance in enumerate(order):
        gt_fold[appliance_num] = test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(test_aggregate.shape[0], -1, 1).reshape(-1, 24)
    return prediction_fold, gt_fold

def disagg(appliance, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations):
    from sklearn.metrics import mean_absolute_error
    preds = []
    gts = []
    for cur_fold in range(num_folds):
        pred, gt = disagg_fold(cur_fold, appliance, cell_type, hidden_size, num_folds
                               ,bidirectional, lr, num_iterations)
        pred[pred<0.] = 0.
        preds.append(pred)
        gts.append(gt)
    return mean_absolute_error(np.concatenate(gts).flatten(), np.concatenate(preds).flatten())

appliance = "hvac"
cell_type="GRU"
hidden_size=100
num_layers=1
bidirectional=True
lr =2
num_iterations = 400
fold_num = 0
order = APPLIANCE_ORDER[1:][::-1]
p = 0.6

pred, gt = disagg_fold(fold_num, cell_type, hidden_size,
                num_layers, bidirectional, lr,
                num_iterations, order, p)