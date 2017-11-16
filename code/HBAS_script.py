from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os
# region = "Austin"
# year = 2014
import delegator

# Enter your username on the cluster
username = 'yj9xs'

# Location of .out and .err files
SLURM_OUT = "~/git/slurm_out"

# Create the SLURM out directory if it does not exist
if not os.path.exists(SLURM_OUT):
	os.makedirs(SLURM_OUT)

# Max. num running processes you want. This is to prevent hogging the cluster
MAX_NUM_MY_JOBS = 20
# Delay between jobs when we exceed the max. number of jobs we want on the cluster
DELAY_NUM_JOBS_EXCEEDED = 10
import time


for setting in ['transfer']:
	for constant_use in ['True']:
		for source in ['Austin']:
			for target in ['SanDiego']:
				for random_seed in range(5):
					for train_percentage in [6,8,9,10,15,20,30,40,50,60,70,80,90,100]:
						OFILE = "{}/{}-{}-{}-{}-{}-{}.out".format(SLURM_OUT, setting, constant_use, source, target, random_seed, train_percentage)
						EFILE = "{}/{}-{}-{}-{}-{}-{}.err".format(SLURM_OUT, setting, constant_use, source, target, random_seed, train_percentage)
						SLURM_SCRIPT = "{}/{}-{}-{}-{}-{}-{}.pbs".format('pbs_files', setting, constant_use, source, target, random_seed, train_percentage)
						CMD = 'python HBAS_parallel.py {} {} {} {} {} {} 1 13 '.format(setting, constant_use, source, target, random_seed, train_percentage)
						lines = []
						lines.append("#!/bin/sh\n")
						lines.append('#SBATCH --time=1-16:0:00\n')
						lines.append('#SBATCH --mem=16\n')
						lines.append('#SBATCH -c 16\n')
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
						print SLURM_SCRIPT
