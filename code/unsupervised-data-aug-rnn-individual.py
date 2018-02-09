import sys
sys.path.append("../code/")

from sklearn.metrics import mean_absolute_error
from dataloader import APPLIANCE_ORDER, get_train_test
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from unsupervised_aug import augmented_data

cuda_av = False
if torch.cuda.is_available():
    cuda_av = True

torch.manual_seed(0)
np.random.seed(0)

weight_appliance = {'mw': 1, 'dw': 1, 'dr': 1, 'fridge': 1, 'hvac': 1}


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


def disagg_fold(num_aug, case, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p):
    # print (fold_num, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
    print (ORDER)
    torch.manual_seed(0)

    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    print (train.shape[0])
    train = augmented_data(train, num_aug, case, test, True)
    print (train.shape[0])
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

    return prediction_fold, gt_fold


best_params = np.load("../code/baseline/best_param_nested_rnn.npy").item()

num_augs, a = sys.argv[1:]
num_augs = int(num_augs)

# ORDER = ['hvac']
num_folds = 5
dataset = 1
p = 0

case = 1

for num_aug in [num_augs]:
    for appliance in [a]:
        ORDER = [appliance]
        preds = []
        gts = []
        for fold_num in range(5):
            print(appliance, fold_num)

            cell_type = best_params[dataset][fold_num][appliance]['cell_type']
            hidden_size = best_params[dataset][fold_num][appliance]['hidden_size']
            num_layers = best_params[dataset][fold_num][appliance]['num_layers']
            bidirectional = best_params[dataset][fold_num][appliance]['bidirectional']
            lr = best_params[dataset][fold_num][appliance]['lr']
            num_iterations = best_params[dataset][fold_num][appliance]['iters']

            print(cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations)
            pred_fold, gt_fold = disagg_fold(num_aug, case, fold_num, dataset, cell_type, hidden_size, num_layers,
                                             bidirectional, lr, num_iterations, 0)
            preds.append(pred_fold)
            gts.append(gt_fold)

        prediction_flatten = {}
        gt_flatten = {}
        for appliance_num, app in enumerate(ORDER):
            prediction_flatten[app] = []
            gt_flatten[app] = []

        for appliance_num, app in enumerate(ORDER):
            for fold in range(5):
                prediction_flatten[app].append(preds[fold][appliance_num])
                gt_flatten[app].append(gts[fold][appliance_num])
            gt_flatten[app] = np.concatenate(gt_flatten[app])
            prediction_flatten[app] = np.concatenate(prediction_flatten[app])

        err = {}
        for app in ORDER:
            print(app)
            err[app] = mean_absolute_error(gt_flatten[app], prediction_flatten[app])
        print(err)

        np.save("./baseline/rnn-aug-pred-{}-{}-{}.npy".format(dataset, appliance, num_aug), prediction_flatten)
        np.save("./baseline/rnn-aug-error-{}-{}-{}.npy".format(dataset, appliance, num_aug), err)

