import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets,transforms
import torch.nn.functional as F
from dataloader import APPLIANCE_ORDER, get_train_test
import os
import sys

torch.manual_seed(0)
np.random.seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 16, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.ConvTranspose2d(16, 6, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(6)

        self.conv6 = nn.ConvTranspose2d(6, 1, kernel_size=5, stride=1, padding=2) 
        
        self.act = nn.ReLU()
        
    # forward method
    def forward(self, input):
        
        e1 = self.conv1(input)
        bn1 = self.bn1(self.act(e1))
        e2 = self.bn2(self.conv2(bn1))
        
        e5 = self.bn5(self.conv5(e2))
        e6 = self.conv6(e5)

        return e6



# Input parameters
dataset, fold_num, appliance, lr, iterations = sys.argv[1:]
dataset = int(dataset)
fold_num = int(fold_num)
appliance_index = APPLIANCE_ORDER.index(appliance)
lr = float(lr)
iterations = int(iterations)
num_folds = 5
print(dataset, fold_num, appliance, lr, iterations)

# prepare the data
train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
valid = train[int(0.8*len(train)):].copy()
train = train[:int(0.8 * len(train))].copy()
train_aggregate = train[:, 0, :, :].reshape(train.shape[0], 1, -1, 24)
valid_aggregate = valid[:, 0, :, :].reshape(valid.shape[0], 1, -1, 24)
test_aggregate = test[:, 0, :, :].reshape(test.shape[0], 1, -1, 24)

# Initialize model and loss function.
model = Net()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.5)

# Check if cuda is available
cuda_av = False
if torch.cuda.is_available():
    cuda_av = True
if cuda_av:
    model = model.cuda()
    loss_fn = loss_fn.cuda()
    
# Prepare the input/output for CNN model
inp = Variable(torch.Tensor(train_aggregate))
out = Variable(torch.Tensor(train[:, appliance_index, :, :].reshape(train.shape[0], 1, train.shape[2], -1)))
test_inp = Variable(torch.Tensor(test_aggregate))
test_out = Variable(torch.Tensor(test[:, appliance_index, :, :].reshape(test.shape[0], 1, test.shape[2], -1)))
if cuda_av:
    inp = inp.cuda()
    out = out.cuda()
    test_inp = test_inp.cuda()
    test_out = test_out.cuda()

# Train the CNN model
for epoch in range(iterations):
    
    pred = model(inp)
    loss = loss_fn(pred, out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch %1000 == 0:
        
        test_pr = model(test_inp)
        test_loss = loss_fn(test_pr, test_out)
        
        print(epoch, "Training Error:", loss.data[0], "Test Error:", test_loss.data[0])

# collect the results
test_pr = model(test_inp)

test_loss = loss_fn(test_pr, test_out)
test_pr = torch.clamp(test_pr, min=0.)

pred = torch.clamp(pred, min=0.)
loss = loss_fn(pred, out)

pred = pred.cpu().data.numpy()
test_pr = test_pr.cpu().data.numpy()

directory = "./baseline/cnn-individual/{}/{}/{}/{}".format(dataset, fold_num, lr, iterations)
if not os.path.exists(directory):
    os.makedirs(directory)

np.save("{}/train-pred-{}".format(directory, appliance), pred)
np.save("{}/train-loss-{}".format(directory, appliance), loss)
np.save("{}/test-pred-{}".format(directory, appliance), test_pr)
np.save("{}/test-loss-{}".format(directory, appliance), test_loss)












