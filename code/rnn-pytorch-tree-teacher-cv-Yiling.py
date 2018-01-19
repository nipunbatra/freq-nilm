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
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor

torch.manual_seed(0)
np.random.seed(0)

# num_hidden, num_iterations, num_layers, p, num_directions = sys.argv[1:6]
num_hidden = 150
num_iterations = 100
num_layers = 1
p = 0.6
num_directions = 1
num_hidden = int(num_hidden)
num_layers = int(num_layers)
num_iterations = int(num_iterations)
p = float(p)
num_directions = int(num_directions)
#ORDER = sys.argv[6:len(sys.argv)]

input_dim = 1
hidden_size = num_hidden
num_layers = num_layers
if num_directions == 1:
	bidirectional = False
else:
	bidirectional = True


class CustomRNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(CustomRNN, self).__init__()
		self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
						  num_layers=num_layers, batch_first=True,
						  dropout=0.1, bidirectional=bidirectional)
		self.linear = nn.Linear(hidden_size * num_directions, output_size, )

	def forward(self, x):
		pred, hidden = self.rnn(x, None)
		pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
		#pred = torch.clamp(pred, min=0.)
		pred = torch.min(pred, x)

		return pred


class AppliancesRNN(nn.Module):
	def __init__(self, hidden_size,num_appliance):
		super(AppliancesRNN, self).__init__()
		self.num_appliance = num_appliance
		self.preds = {}
		self.order = ORDER
		for appliance in range(self.num_appliance):
			if cuda_av:
				setattr(self, "Appliance_" + str(appliance), CustomRNN(1, hidden_size, 1).cuda())
			else:
				setattr(self, "Appliance_" + str(appliance), CustomRNN(1, hidden_size, 1))

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
			self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
			if flag:
				agg_current = agg_current - torch.clamp(self.preds[appliance], min=0.)
			else:
				agg_current = agg_current - args[2 + appliance]

		return torch.cat([self.preds[a] for a in range(self.num_appliance)])


def disagg_fold(fold_num, hidden_size, num_layers, bidirectional, lr, num_iterations, p):
	torch.manual_seed(0)

	train, test = get_train_test(num_folds=num_folds, fold_num=fold_num)
	train_aggregate = train[:, 0, :, :].reshape(-1, 24, 1)
	test_aggregate = test[:, 0, :, :].reshape(-1, 24, 1)

	out_train = [None for temp in range(len(ORDER))]
	for a_num, appliance in enumerate(ORDER):
		out_train[a_num] = Variable(
			torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train_aggregate.shape[0], -1, 1))))
		if cuda_av:
			out_train[a_num] = out_train[a_num].cuda()

	loss_func = nn.L1Loss()
	a = AppliancesRNN(hidden_size,len(ORDER))
	
	if cuda_av:
		a = a.cuda()
		loss_func = loss_func.cuda()
	optimizer = torch.optim.Adam(a.parameters(), lr=lr)

	inp = Variable(torch.Tensor(train_aggregate.reshape((train_aggregate.shape[0], -1, 1))).type(torch.FloatTensor), requires_grad=True)
	for t in range(num_iterations):
		inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)
		out = torch.cat([out_train[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
		if cuda_av:
			inp = inp.cuda()
			out = out.cuda()

		params = [inp, p]
		for a_num, appliance in enumerate(ORDER):
			params.append(out_train[a_num])

		pred = a(*params)

		optimizer.zero_grad()
		loss = loss_func(pred, out)
		if t % 1 == 0:
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
		gt_fold[appliance_num] = test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(test_aggregate.shape[0], -1, 1).reshape(-1, 24)

	return prediction_fold, gt_fold

def disagg( hidden_size, num_layers, bidirectional, lr, num_iterations, p):
	from sklearn.metrics import mean_absolute_error
	preds = []
	gts = []
	for cur_fold in range(num_folds):
		pred, gt = disagg_fold(cur_fold, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
		# pred[pred<0.] = 0.
		preds.append(pred)
		gts.append(gt)
	return mean_absolute_error(np.concatenate(gts).flatten(), np.concatenate(preds).flatten())

ORDER = APPLIANCE_ORDER[1:][::-1]
#ORDER = APPLIANCE_ORDER[1:]

lr = 3
p=0.6
num_folds=5
error = disagg(hidden_size,
                num_layers, bidirectional, lr,
                num_iterations, p)
