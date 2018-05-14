import numpy as np
import sys
from pathlib import Path
import itertools

c=0
e=0
dataset = 5
for fold_num in range(5):
	for lr in [0.01]:
		for iters in [20000]:
			for order in list(itertools.permutations(['hvac', 'fridge', 'dr', 'dw', 'mw', 'residual'])):
				for p in [0.2, 0.4, 0.8]:
					o = "\', \'".join(str(x) for x in order)
					directory="../code/baseline/cnn-tree/{}/{}/{}/20000/{}/".format(dataset, fold_num, lr, p)
					filename = "valid-pred-[\'{}\'].npy".format(o)
				
					full_path = directory + filename
					my_file = Path(full_path)

					if not my_file.exists():
						c+=1
					else:
						e+=1

print("Current progress: {} of {}, {}%".format(e, (c+e), 100*e/(c+e)))
