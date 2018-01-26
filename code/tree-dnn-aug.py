import sys
sys.path.append("../code/")
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


weight_appliance = {'mw':1, 'dw':1, 'dr':1,'fridge':1, 'hvac':1}

# num_hidden, num_iterations, num_layers, p, num_directions = sys.argv[1:6]


class CustomRNN(nn.Module):
    def __init__(self):
        super(CustomRNN, self).__init__()
        torch.manual_seed(0)

      
        self.lin_1 = nn.Linear(24, 100)
        self.lin_2 = nn.Linear(100, 24)
        self.bn = nn.BatchNorm1d(100)
        #self.lin_3 = nn.Linear(48, 24)
        
        
        self.act_1 = nn.ReLU()
        self.act_2 = nn.ReLU()
        #self.act_3 = nn.ReLU()

    def forward(self, x):
        
        pred = self.lin_1(x)
        pred = self.act_1(pred)
        pred = self.bn(pred)
        pred = self.lin_2(pred)
        pred = self.act_2(pred)
        
        
        #pred = torch.clamp(pred, min=0.)
        #pred = self.act(pred)
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
            #print(getattr(self, "Appliance_" + str(appliance)))
            self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
            if flag:
                agg_current = agg_current - self.preds[appliance]
            else:
                agg_current = agg_current - args[2 + appliance]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])

#ORDER = APPLIANCE_ORDER[1:][::-1]


num_hidden = 120
num_iterations = 300
num_layers = 1
num_directions = 1

input_dim = 1
hidden_size = num_hidden
num_layers = num_layers
if num_directions == 1:
    bidirectional = False
else:
    bidirectional = True
lr = 0.5
p = 0.1
num_folds = 5
fold_num = 0

torch.manual_seed(0)

ORDER = APPLIANCE_ORDER[1:][:3]
ORDER = ['dw','fridge','dr','hvac']
#ORDER = ['fridge']
#ORDER = ['dr']

for fold_num in range(num_folds):

    train, test = get_train_test(num_folds=num_folds, fold_num=fold_num)
    train_aggregate = train[:, 0, :, :].reshape(-1, 24)
    test_aggregate = test[:, 0, :, :].reshape(-1, 24)



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
    a = AppliancesRNN(num_appliance=len(ORDER))

    if cuda_av:
        a = a.cuda()
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(a.parameters(), lr=lr)

    inp = Variable(torch.Tensor(train_aggregate.reshape((train_aggregate.shape[0], -1))).type(torch.FloatTensor),
                   requires_grad=True)

    for t in range(num_iterations):
        inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)
        out = torch.cat([out_train[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
        ot =  torch.cat([out_test[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
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
        pred_split = torch.split(pred, pred.size(0)//len(ORDER))

        losses= [loss_func(pred_split[appliance_num], out_train[appliance_num])*weight_appliance[appliance] for appliance_num, appliance in enumerate(ORDER)]
        
        loss = sum(losses)
        if t % 100 == 0:
            print(t, loss.data[0])

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

    for i in range(len(ORDER)):
        print (ORDER[i], mean_absolute_error(prediction_fold[i], gt_fold[i]))
