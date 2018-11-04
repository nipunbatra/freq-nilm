#from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os
from pathlib import Path

import delegator

c = 0
cmds = {}
for dataset in [6]:
    for cur_fold in range(5):
        for num_latent in range(1, 21):
            for lr in [0.01, 0.1, 1, 2]:
                lr = float(lr)
                for iters in range(100, 2600, 400):
                    filename = "mtf-pred-{}-{}-{}-{}-{}.npy".format(dataset, cur_fold, num_latent, lr, iters)
                    directory = "../code/baseline/mtf/{}/valid/".format(dataset)
                    full_path = directory + filename
                    my_file = Path(full_path)
                    if not my_file.exists():
                        c += 1
                        print("python baseline-mtf-nested-valid.py {} {} {} {} {}".format(dataset, cur_fold, num_latent, lr, iters))
                        cmds[c-1] = "python baseline-mtf-nested-valid.py {} {} {} {} {}".format(dataset, cur_fold, num_latent, lr, iters)
                        

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
for c in range(len(cmds)):
	OFILE = "{}/{}.out".format(SLURM_OUT, c)
	EFILE = "{}/{}.err".format(SLURM_OUT, c)
	SLURM_SCRIPT = "{}/{}.pbs".format(SLURM_OUT, c)
	CMD = cmds[c]
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


