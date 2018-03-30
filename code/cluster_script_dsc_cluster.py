#from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os

import delegator

# Enter your username on the cluster
username = 'yj9xs'

# Location of .out and .err files
SLURM_OUT = "~/git/slurm_out"

# Create the SLURM out directory if it does not exist
if not os.path.exists(SLURM_OUT):
	os.makedirs(SLURM_OUT)

# Max. num running processes you want. This is to prevent hogging the cluster
MAX_NUM_MY_JOBS = 500
# Delay between jobs when we exceed the max. number of jobs we want on the cluster
DELAY_NUM_JOBS_EXCEEDED = 10
import time
dataset=2
for cluster in [2, 3]:
	for cur_fold in range(5):
		for num_latent in range(1, 51):
			for lr in [0.01]:
				for iters in range(10, 110, 10):
					OFILE = "{}/{}-{}-{}-{}-{}.out".format(SLURM_OUT, dataset, cur_fold, num_latent, lr, iters)
					EFILE = "{}/{}-{}-{}-{}-{}.err".format(SLURM_OUT, dataset, cur_fold, num_latent, lr, iters)
					SLURM_SCRIPT = "{}/dsc-{}-{}-{}-{}-{}.pbs".format(SLURM_OUT, dataset, cur_fold, num_latent, lr, iters)
					#CMD = 'python3 baseline-mtf-nested.py {} {} {} {} {}'.format(dataset, cur_fold, num_latent, lr, iters)
					CMD = 'python3 baseline-sc-with-disc-nested-cluster.py {} {} {} {} {}'.format(dataset, cluster, cur_fold, num_latent, iters)
					lines = []
					lines.append("#!/bin/sh\n")
					lines.append('#SBATCH --time=1-16:0:00\n')
					lines.append('#SBATCH --mem=64\n')
					#lines.append('#SBATCH -c 32\n')
					lines.append('#SBATCH --exclude=artemis[1-5]\n')
					lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
					lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
					lines.append(CMD + '\n')
					with open(SLURM_SCRIPT, 'w') as f:
						f.writelines(lines)
					command = ['sbatch', SLURM_SCRIPT]
					while len(delegator.run('squeue -u %s' % username).out.split("\n")) > MAX_NUM_MY_JOBS + 2:
						time.sleep(DELAY_NUM_JOBS_EXCEEDED)

					delegator.run(command, block=False)
					print (SLURM_SCRIPT)


