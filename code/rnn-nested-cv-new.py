import sys
from sklearn.metrics import mean_absolute_error
from dataloader import APPLIANCE_ORDER, get_train_test
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

cuda_av = False
if torch.cuda.is_available():
    cuda_av = True

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
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * self.num_directions, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
        # pred = self.act(pred)
        # pred = torch.clamp(pred, min=0.)
        #pred = self.act(pred)
        pred = torch.min(pred, x)
        return pred


class AppliancesRNN(nn.Module):
    def __init__(self, cell_type, hidden_size, num_layers, bidirectional, num_appliance):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type,
                                                                       hidden_size,
                                                                       num_layers,
                                                                       bidirectional).cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type,
                                                                       hidden_size,
                                                                       num_layers,
                                                                       bidirectional))

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
            # print (appliance)
            # print (self.order[appliance])
            # print (args[2+appliance])
            # print(getattr(self, "Appliance_" + str(appliance)))
            self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
            if flag:
                agg_current = agg_current - self.preds[appliance]
            else:
                agg_current = agg_current - args[2 + appliance]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])


def disagg_fold(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p):
    # print (fold_num, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
    print (ORDER)
    torch.manual_seed(0)

    num_folds=5
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    from sklearn.model_selection import train_test_split
    train, valid = train_test_split(train, test_size=0.2, random_state=0)


    train_aggregate = train[:, 0, :, :].reshape(-1, 24, 1)
    valid_aggregate = valid[:, 0, :, :].reshape(-1, 24, 1)
    test_aggregate = test[:, 0, :, :].reshape(-1, 24, 1)


    print (train.shape)
    print (valid.shape)
    print (test.shape)

    out_train = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_train[a_num] = Variable(
            torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train_aggregate.shape[0], -1, 1))))
        if cuda_av:
            out_train[a_num] = out_train[a_num].cuda()

    loss_func = nn.L1Loss()
    a = AppliancesRNN(cell_type, hidden_size, num_layers, bidirectional, len(ORDER))
    # prevent negative
    #for param in a.parameters():
    #    param.data = param.data.abs()
    #print(a)
    if cuda_av:
        a = a.cuda()
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(a.parameters(), lr=lr)

    inp = Variable(torch.Tensor(train_aggregate.reshape((train_aggregate.shape[0], -1, 1))).type(torch.FloatTensor),
                   requires_grad=True)

    valid_pred = {}
    train_pred = {}

    valid_inp = Variable(torch.Tensor(valid_aggregate), requires_grad=False)
    if cuda_av:
        valid_inp = valid_inp.cuda()

    for t in range(1, num_iterations+1):
        inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)
        out = torch.cat([out_train[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
        if cuda_av:
            inp = inp.cuda()
            out = out.cuda()

        params = [inp, p]
        for a_num, appliance in enumerate(ORDER):
            params.append(out_train[a_num])
        # print(params)
        pred = a(*params)

        optimizer.zero_grad()
        loss = loss_func(pred, out)
        if t % 100 == 0:
            print(t, loss.data[0])

        if t%1000 == 0 and t!=0:
            if cuda_av:
                valid_inp = valid_inp.cuda()
            valid_params = [valid_inp, -2]
            for i in range(len(ORDER)):
                valid_params.append(None)
            valid_pr = a(*valid_params)
            valid_pr = torch.clamp(valid_pr, min=0.)
            valid_pred[t] = valid_pr

            train_pr = pred
            train_pr = torch.clamp(train_pr, min=0.)
            train_pred[t] = train_pr


        loss.backward()
        optimizer.step()

    # store training prediction
    train_fold = {}
    for t in range(1000, num_iterations + 1, 1000):
        # train_pred[t] = torch.clamp(train_pred[t], min=0.)
        train_pred[t] = torch.split(train_pred[t], train_aggregate.shape[0])
        train_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                train_fold[t][appliance_num] = train_pred[t][appliance_num].cpu().data.numpy().reshape(-1, 24)
        else:
            for appliance_num, appliance in enumerate(ORDER):
                train_fold[t][appliance_num] = train_pred[t][appliance_num].data.numpy().reshape(-1, 24)


    # test one validation set

    valid_fold = {}
    for t in range(1000, num_iterations+1, 1000):

        valid_pred[t] = torch.split(valid_pred[t], valid_aggregate.shape[0])
        valid_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                valid_fold[t][appliance_num] = valid_pred[t][appliance_num].cpu().data.numpy().reshape(-1, 24)
        else:
            for appliance_num, appliance in enumerate(ORDER):
                valid_fold[t][appliance_num] = valid_pred[t][appliance_num].data.numpy().reshape(-1, 24)
    
    # store gound truth of validation set
    gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        gt_fold[appliance_num] = valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(valid_aggregate.shape[0], -1, 
                                                                                        1).reshape(-1, 24)

    # calcualte the error of validation set
    error = {}
    for t in range(1000, num_iterations+1, 1000):
        error[t] = {}
        for appliance_num, appliance in enumerate(ORDER):
            error[t][appliance] = mean_absolute_error(valid_fold[t][appliance_num], gt_fold[appliance_num])
    
    return train_fold, valid_fold, error


fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p = sys.argv[1:10]
fold_num = int(fold_num)
dataset = int(dataset)
hidden_size = int(hidden_size)
num_layers = int(num_layers)
lr = float(lr)
num_iterations = int(num_iterations)
p = float(p)
ORDER = sys.argv[10:len(sys.argv)]

input_dim = 1
num_folds = 5

train_fold, valid_fold, error = disagg_fold(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
print (error)

# import json
for t in range(1000, num_iterations+1, 1000):
    np.save('./baseline/rnn-nested-cv/valid-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, t, p, ORDER), valid_fold[t])

    np.save('./baseline/rnn-nested-cv/valid-error-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, t, p, ORDER), error[t])

    np.save('./baseline/rnn-nested-cv/train-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, t, p, ORDER), train_fold[t])
