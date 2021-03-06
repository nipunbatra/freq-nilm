import sys

sys.path.append("../code/")
from sklearn.metrics import mean_absolute_error
from dataloader import APPLIANCE_ORDER, get_train_test
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import KFold

if torch.cuda.is_available():
    cuda_av = True

torch.manual_seed(0)
np.random.seed(0)

weight_appliance = {'mw': 1, 'dw': 1, 'dr': 1, 'fridge': 1, 'hvac': 1}


# num_hidden, num_iterations, num_layers, p, num_directions = sys.argv[1:6]


class CustomRNN(nn.Module):
    def __init__(self, size1, size2, size3):
        super(CustomRNN, self).__init__()
        torch.manual_seed(0)
        
        self.bn_0 = nn.BatchNorm1d(24)

        self.lin_1 = nn.Linear(24, size1)
        self.d_1 = nn.Dropout(p=0.1)
        self.bn_1 = nn.BatchNorm1d(size1)

        self.lin_2 = nn.Linear(size1, size2)
        self.d_2 = nn.Dropout(p=0.1)
        self.bn_2 = nn.BatchNorm1d(size2)

        self.lin_3 = nn.Linear(size2, size3)
        self.d_3 = nn.Dropout(p=0.1)
        self.bn_3 = nn.BatchNorm1d(size3)

        self.lin_4 = nn.Linear(size3, 24)
        self.bn_4 = nn.BatchNorm1d(24)



        # self.lin_3 = nn.Linear(48, 24)

        self.act_1 = nn.ReLU()
        self.act_2 = nn.ReLU()
        self.act_3 = nn.ReLU()
        self.act_4 = nn.ReLU()
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
        pred = self.d_3(pred)
        pred = self.act_3(pred)
        pred = self.bn_3(pred)

        pred = self.lin_4(pred)
        pred = self.act_4(pred)
        #pred = self.bn_3(pred)

        #pred = torch.clamp(pred, min=0.)
        # pred = self.act(pred)
        pred = torch.min(pred, x)
        return pred


class AppliancesRNN(nn.Module):
    def __init__(self, size1, size2, size3, num_appliance):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(size1, size2, size3).cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(size1, size2, size3))

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


# ORDER = APPLIANCE_ORDER[1:][::-1]

dataset, size1, size2, size3, lr, num_iterations = sys.argv[1:7]
dataset = int(dataset)
size1 = int(size1)
size2 = int(size2)
size3 = int(size3)
lr = float(lr)
num_iterations = int(num_iterations)

ORDER = sys.argv[7:]

p = 0
num_folds = 5

torch.manual_seed(0)

#ORDER = ['hvac']

preds = []
gts = []
for fold_num in range(5):
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    train_aggregate = train[:, 0, :, :].reshape(-1, 24)
    test_aggregate = test[:, 0, :, :].reshape(-1, 24)
    #ORDER = APPLIANCE_ORDER[1:][:][::-1]
    out_train = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_train[a_num] = Variable(
            torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train_aggregate.shape[0], -1))))
        if cuda_av:
            out_train[a_num] = out_train[a_num].cuda()

    out_test = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_test[a_num] = Variable(
            torch.Tensor(test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((test_aggregate.shape[0], -1))))
        if cuda_av:
            out_test[a_num] = out_test[a_num].cuda()

    loss_func = nn.L1Loss()
    a = AppliancesRNN(size1, size2, size3, len(ORDER))
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
        ot = torch.cat([out_test[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
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
        #if t % 10 == 0:
        #    print(t, loss.data[0])

        loss.backward()
        optimizer.step()

    test_inp = Variable(torch.Tensor(test_aggregate), requires_grad=False)
    if cuda_av:
        test_inp = test_inp.cuda()

    params = [test_inp, -2]
    for i in range(len(ORDER)):
        params.append(None)
    pr = a(*params)
    pr = torch.clamp(pr, min=0.)
    test_pred = torch.split(pr, test_aggregate.shape[0])
    prediction_fold = [None for x in range(len(ORDER))]

    if cuda_av:
        for appliance_num, appliance in enumerate(ORDER):
            prediction_fold[appliance_num] = test_pred[appliance_num].cpu().data.numpy().reshape(-1, 24)
    else:
        for appliance_num, appliance in enumerate(ORDER):
            prediction_fold[appliance_num] = test_pred[appliance_num].data.numpy().reshape(-1, 24)
    gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        gt_fold[appliance_num] = test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(test_aggregate.shape[0], -1,
                                                                                         1).reshape(-1, 24)

    preds.append(prediction_fold)
    gts.append(gt_fold)

prediction_flatten = {}
gt_flatten = {}
for appliance_num, appliance in enumerate(ORDER):
    prediction_flatten[appliance] = []
    gt_flatten[appliance] = []

for appliance_num, appliance in enumerate(ORDER):
    for fold in range(5):
        prediction_flatten[appliance].append(preds[fold][appliance_num])
        gt_flatten[appliance].append(gts[fold][appliance_num])
    gt_flatten[appliance] = np.concatenate(gt_flatten[appliance])
    prediction_flatten[appliance] = np.concatenate(prediction_flatten[appliance])

err = {}
for appliance in ORDER:
    print(appliance)
    err[appliance] = mean_absolute_error(gt_flatten[appliance], prediction_flatten[appliance])

np.save("./baseline/dnn-set2-result/dnn-{}-{}-{}.npy".format(num_iterations, lr, ORDER), err)


print(pd.Series(err))
