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

weight_appliance = {'mw': 1, 'dw': 1, 'dr': 1, 'fridge': 1, 'hvac': 1}


# num_hidden, num_iterations, num_layers, p, num_directions = sys.argv[1:6]


class CustomRNN(nn.Module):
    def __init__(self):
        super(CustomRNN, self).__init__()
        torch.manual_seed(0)
        self.bn_0 = nn.BatchNorm1d(24)
        self.lin_1 = nn.Linear(24, 50)
        self.d_1 = nn.Dropout(p=0.1)
        self.bn_1 = nn.BatchNorm1d(50)
        self.lin_2 = nn.Linear(50, 100)
        self.d_2 = nn.Dropout(p=0.1)
        self.bn_2 = nn.BatchNorm1d(100)
        self.lin_3 = nn.Linear(100, 24)

        self.bn_3 = nn.BatchNorm1d(24)
        # self.lin_3 = nn.Linear(48, 24)

        self.act_1 = nn.ReLU()
        self.act_2 = nn.ReLU()
        self.act_3 = nn.ReLU()
        # self.act_3 = nn.ReLU()

    def forward(self, x):
        #print(x.size())
        #x = self.bn_0(x)
        pred = self.lin_1(x)
        pred = self.d_1(pred)
        #print(pred.size())
        pred = self.act_1(pred)
        #print(pred.size())
        pred = self.bn_1(pred)
        #print(pred.size())
        pred = self.lin_2(pred)
        pred = self.d_2(pred)
        #print(pred.size())
        pred = self.act_2(pred)
        #print(pred.size())
        pred = self.bn_2(pred)
        #print(pred.size())
        pred = self.lin_3(pred)
        pred = self.act_3(pred)
        #pred = self.bn_3(pred)

        #pred = torch.clamp(pred, min=0.)
        # pred = self.act(pred)
        pred = torch.min(pred, x)
        return pred


class AppliancesRNN(nn.Module):
    def __init__(self, num_appliance):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomRNN().cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomRNN())

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
            # print(getattr(self, "Appliance_" + str(appliance)))
            self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
            if flag:
                agg_current = agg_current - self.preds[appliance]
            else:
                agg_current = agg_current - args[2 + appliance]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])


def disagg_fold(fold_num, dataset, lr, num_iterations, p):

    torch.manual_seed(0)

    num_folds=5
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    from sklearn.model_selection import train_test_split
    train, valid = train_test_split(train, test_size=0.2)


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

    out_valid = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_valid[a_num] = Variable(
            torch.Tensor(valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((valid_aggregate.shape[0], -1))))
        if cuda_av:
            out_valid[a_num] = out_valid[a_num].cuda()


    loss_func = nn.L1Loss()
    a = AppliancesRNN(num_appliance=len(ORDER))
    # for param in a.parameters():
    #    param.data = param.data.abs()
    # print(a)
    if cuda_av:
        a = a.cuda()
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(a.parameters(), lr=lr)
    
    inp = Variable(torch.Tensor(train_aggregate.reshape((train_aggregate.shape[0], -1))).type(torch.FloatTensor),
                   requires_grad=True)
    for t in range(num_iterations):
        inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)
        out = torch.cat([out_train[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
        ot = torch.cat([out_valid[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
        if cuda_av:
            inp = inp.cuda()
            out = out.cuda()
            ot = ot.cuda()

        params = [inp, p]
        for a_num, appliance in enumerate(ORDER):
            params.append(out_train[a_num])
        # print(params)
        pred = a(*params)

        optimizer.zero_grad()
        pred_split = torch.split(pred, pred.size(0) // len(ORDER))

        losses = [loss_func(pred_split[appliance_num], out_train[appliance_num]) * weight_appliance[appliance] for
                  appliance_num, appliance in enumerate(ORDER)]

        loss = sum(losses)/len(ORDER)
        if t % 10 == 0:
            print(t, loss.data[0])

        loss.backward()
        optimizer.step()

    # store training prediction
    train_pred = torch.clamp(pred, min=0.)
    train_pred = torch.split(train_pred, train_aggregate.shape[0])
    train_fold = [None for x in range(len(ORDER))]
    if cuda_av:
        for appliance_num, appliance in enumerate(ORDER):
            train_fold[appliance_num] = train_pred[appliance_num].cpu().data.numpy().reshape(-1, 24)
    else:
        for appliance_num, appliance in enumerate(ORDER):
            train_fold[appliance_num] = train_pred[appliance_num].data.numpy().reshape(-1, 24)


    # test one validation set
    valid_inp = Variable(torch.Tensor(valid_aggregate), requires_grad=False)
    if cuda_av:
        valid_inp = valid_inp.cuda()

    params = [valid_inp, -2]
    for i in range(len(ORDER)):
        params.append(None)
    pr = a(*params)
    pr = torch.clamp(pr, min=0.)
    valid_pred = torch.split(pr, valid_aggregate.shape[0])
    valid_fold = [None for x in range(len(ORDER))]
    if cuda_av:
        for appliance_num, appliance in enumerate(ORDER):
            valid_fold[appliance_num] = valid_pred[appliance_num].cpu().data.numpy().reshape(-1, 24)
    else:
        for appliance_num, appliance in enumerate(ORDER):
            valid_fold[appliance_num] = valid_pred[appliance_num].data.numpy().reshape(-1, 24)
    
    # store gound truth of validation set
    gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        gt_fold[appliance_num] = valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(valid_aggregate.shape[0], -1, 
                                                                                        1).reshape(-1, 24)

    # calcualte the error of validation set
    error = {}
    for appliance_num, appliance in enumerate(ORDER):
        error[appliance] = mean_absolute_error(valid_fold[appliance_num], gt_fold[appliance_num])
    
    return train_fold, valid_fold, error


fold_num, dataset, lr, num_iterations, p = sys.argv[1:6]
fold_num = int(fold_num)
dataset = int(dataset)
lr = float(lr)
num_iterations = int(num_iterations)
p = float(p)
ORDER = sys.argv[6:len(sys.argv)]

input_dim = 1
num_folds = 5

train_fold, valid_fold, error = disagg_fold(fold_num, dataset, lr, num_iterations, p)
np.save('./baseline/dnn-nested-cv/valid-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, lr, num_iterations, p, ORDER), valid_fold)
np.save('./baseline/dnn-nested-cv/valid-error-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, lr, num_iterations, p, ORDER), error) 
np.save('./baseline/dnn-nested-cv/train-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, lr, num_iterations, p, ORDER), train_fold) 

