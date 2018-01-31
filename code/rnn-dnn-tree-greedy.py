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

weight_appliance = {'mw': 1, 'dw': 1, 'dr': 1, 'fridge': 1, 'hvac': 1}
appliance_contri = {'hvac':0.83003428, 'fridge':0.0827564, 'dr':0.06381463, 'dw':0.01472098, 'mw':0.00867371}


# num_hidden, num_iterations, num_layers, p, num_directions = sys.argv[1:6]

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
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        pred, hidden = self.rnn(x, None)
        pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
        #pred = self.act(pred)
        #pred = torch.clamp(pred, min=0.)
        pred = self.act(pred)
        pred = torch.min(pred, x).view(pred.size(0), -1)
        return pred


class CustomDNN(nn.Module):
    def __init__(self):
        super(CustomDNN, self).__init__()
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
    def __init__(self, cell_type, hidden_size, num_layers, bidirectional, num_appliance):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if ORDER[appliance] in ['hvac']:
                if cuda_av:
                    setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type, hidden_size,
                                                                           num_layers, bidirectional).cuda())
                else:
                    setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type, hidden_size,
                                                                           num_layers, bidirectional))
            else:
                if cuda_av:
                    setattr(self, "Appliance_" + str(appliance), CustomDNN().cuda())
                else:
                    setattr(self, "Appliance_" + str(appliance), CustomDNN())


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


# lr = 0.1
# p = 0.6
num_folds = 5
# fold_num = 0
#num_iterations = 800

torch.manual_seed(0)


# num_folds_run = 5


# ORDER = sys.argv[6:len(sys.argv)]

def disagg(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p):
    preds = []
    gts = []
    for fold_num in range(num_folds):
        print("-"*40)
        sys.stdout.flush()
        train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
        train_aggregate = train[:, 0, :, :].reshape(-1, 24)
        test_aggregate = test[:, 0, :, :].reshape(-1, 24)
        #ORDER = APPLIANCE_ORDER[1:][:][::-1]
        # ORDER = ['mw','dw','fridge','dr','hvac']
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
        a = AppliancesRNN(cell_type, hidden_size, num_layers, bidirectional, num_appliance=len(ORDER))
        for param in a.parameters():
            param.data = param.data.abs()
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
            if t % 20 == 0:
                print(t, loss.data[0])
                sys.stdout.flush()

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
        for fold in range(num_folds):
            prediction_flatten[appliance].append(preds[fold][appliance_num])
            gt_flatten[appliance].append(gts[fold][appliance_num])
        gt_flatten[appliance] = np.concatenate(gt_flatten[appliance])
        prediction_flatten[appliance] = np.concatenate(prediction_flatten[appliance])

    err = {}
    for appliance in ORDER:
        print(appliance)
        sys.stdout.flush()
        err[appliance] = mean_absolute_error(gt_flatten[appliance], prediction_flatten[appliance])
    return err


dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p = sys.argv[1:]
dataset = int(dataset)
hidden_size = int(hidden_size)
num_layers = int(num_layers)
lr = float(lr)
num_iterations = int(num_iterations)
p = float(p)



order_candidate = {}
for appliance in APPLIANCE_ORDER[1:]:
    print (appliance)
    ORDER = appliance.split()
    error = disagg(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
    print (error)
    order_candidate[appliance] = error[appliance]/appliance_contri[appliance]
    print (order_candidate)

k = 3
top_k = pd.Series(order_candidate).nsmallest(k).to_dict()

for j in range(4):
    order_candidate = {}
    for order, e in top_k.items():
        for appliance in APPLIANCE_ORDER[1:]:
            if appliance in order:
                continue
            
            new_order = order + " " + appliance
            ORDER = new_order.split()
            new_error = disagg(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)

            order_candidate[new_order] = 0
            for a in ORDER:
                order_candidate[new_order] += new_error[a]/appliance_contri[a]
            print (new_order, order_candidate[new_order])
    top_k = pd.Series(order_candidate).nsmallest(k).to_dict()

best_order = pd.Series(order_candidate).idxmin()
ORDER = best_order.split()
best_error = disagg(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
result = {}
result[best_order] = best_error

np.save("./baseline/rnn-dnn-tree-greedy/{}-{}-{}-{}-{}-{}-{}-{}".format(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p), result)
