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

appliance="hvac"
cell_type = "GRU"
num_hidden = 120
num_iterations = 5000
num_layers = 1
num_directions = 1

input_dim = 1
hidden_size = num_hidden
num_layers = num_layers
if num_directions == 1:
    bidirectional = False
else:
    bidirectional = True
p = 0.5
num_folds = 5
fold_num = 2

torch.manual_seed(0)

train, test = get_train_test(2, num_folds=num_folds, fold_num=fold_num)
train_aggregate = train[:, 0, None, :, :]
test_aggregate = test[:, 0, None, :, :]

train_input = Variable(torch.Tensor(train_aggregate), requires_grad=True)

if cuda_av:
    train_input = train_input.cuda()

t = train_input.view(-1, 1, 24)

train_appliance = train[:, APPLIANCE_ORDER.index(appliance),None, :, :]
test_appliance = test[:, APPLIANCE_ORDER.index(appliance),None, :, :]


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        torch.manual_seed(0)

        self.q1 = nn.Conv1d(1, 20, 10, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(30)
        self.q2 = nn.Conv1d(20, 30, 8, 1)
        self.q3 = nn.Conv1d(30, 30, 6, 1)
        self.q4 = nn.Conv1d(30, 24, 3, 1)
        self.l = nn.Linear(24, 24)
        self.act3 = nn.ReLU()

    def forward(self, x):
        pred = self.q1(x)
        pred = self.act1(pred)
        pred = self.q2(pred)
        pred = self.act2(pred)
        pred = self.q3(pred)
        pred = self.q4(pred).view(-1, 24)
        pred = self.l(pred)
        pred = self.act3(pred)

        return pred

loss_func = nn.L1Loss()
c = CustomCNN()
lr = 0.01
if cuda_av:
    c = c.cuda()
    loss_func = loss_func.cuda()
optimizer = torch.optim.Adam(c.parameters(), lr=lr)

inp = t

test_inp = Variable(torch.Tensor(test_aggregate.reshape(-1, 1, 24)), requires_grad=False)
test_out = Variable(torch.Tensor(test_appliance.reshape(-1, 1, 24)), requires_grad=False)
train_out = Variable(torch.Tensor(train_appliance.reshape(-1, 1, 24)), requires_grad=False)

if cuda_av:
    test_inp = test_inp.cuda()
    test_out = test_out.cuda()


for it in range(num_iterations):
    if cuda_av:
        inp = inp.cuda()
        train_out = train_out.cuda()

    pred = c(inp)

    optimizer.zero_grad()

    loss = loss_func(pred, train_out)
    if it % 10 == 0:
        p = c(test_inp)
        print("Iterations: {}, Train loss: {}, Test loss: {}".format(it, loss.data[0], loss_func(p, test_out).cpu().data.numpy()[0]))

    loss.backward()
    optimizer.step()
