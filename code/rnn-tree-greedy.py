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
        pred = self.act(pred)
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
    # print (ORDER)
    torch.manual_seed(0)

    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    train_aggregate = train[:, 0, :, :].reshape(-1, 24, 1)
    test_aggregate = test[:, 0, :, :].reshape(-1, 24, 1)

    out_train = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_train[a_num] = Variable(
            torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train_aggregate.shape[0], -1, 1))))
        if cuda_av:
            out_train[a_num] = out_train[a_num].cuda()

    loss_func = nn.L1Loss()
    a = AppliancesRNN(cell_type, hidden_size, num_layers, bidirectional, len(ORDER))
    # prevent negative
    for param in a.parameters():
        param.data = param.data.abs()
    #print(a)
    if cuda_av:
        a = a.cuda()
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(a.parameters(), lr=lr)

    inp = Variable(torch.Tensor(train_aggregate.reshape((train_aggregate.shape[0], -1, 1))).type(torch.FloatTensor),
                   requires_grad=True)
    for t in range(num_iterations):
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
        # if t % 1 == 0:
        #     print(t, loss.data[0])

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

    return prediction_fold, gt_fold


def disagg(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p):
    from sklearn.metrics import mean_absolute_error
    preds = {}
    gts = {}
    error = {}

    for appliance_num, appliance in enumerate(ORDER):
        preds[appliance] = []
        gts[appliance] = []

    for cur_fold in range(num_folds):
        # print (cur_fold, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
        pred, gt = disagg_fold(cur_fold, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
        # pred[pred<0.] = 0.
        for appliance_num, appliance in enumerate(ORDER):
                preds[appliance].append(pred[appliance_num])
                gts[appliance].append(gt[appliance_num])
    for appliance in ORDER:
        print (appliance, mean_absolute_error(np.concatenate(gts[appliance]).flatten(), np.concatenate(preds[appliance]).flatten()))
        error[appliance] = mean_absolute_error(np.concatenate(gts[appliance]).flatten(), np.concatenate(preds[appliance]).flatten())
    return error

dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p = sys.argv[1:9]
dataset = int(dataset)
hidden_size = int(hidden_size)
num_layers = int(num_layers)
lr = float(lr)
num_iterations = int(num_iterations)
p = float(p)
num_folds = 5
# ORDER = sys.argv[8:len(sys.argv)]

# stage one: run 5 iterations:
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

np.save("./baseline/rnn-tree-greedy/{}-{}-{}-{}-{}-{}-{}-{}.npy".format(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p), result)








