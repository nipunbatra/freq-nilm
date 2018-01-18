import pickle

result = {}

for num_latent in range(1,21):
	result[num_latent] = {}
	for lr in [0.01, 0.1, 1, 2]:
		lr = float(lr)
		result[num_latent][lr] = {}
		for iters in range(100, 2500, 400):
			result[num_latent][lr][iters] = pickle.load(open("./baseline-mtf-{}-{}-{}.pkl".format(num_latent, lr, iters), "rb"))


pickle.dump(result, open("./baseline-mtf.pkl", "wb"))
			




