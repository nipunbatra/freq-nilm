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
MAX_NUM_MY_JOBS = 5
# Delay between jobs when we exceed the max. number of jobs we want on the cluster
DELAY_NUM_JOBS_EXCEEDED = 10
import time
dataset=5

for cell_type in ['GRU', 'LSTM', 'RNN']:
    for hidden_size in [20, 50]:
        for num_layers in [1, 2, 3]:
            for bidirectional in [True, False]:
                for lr in [0.01, 0.1, 1]:
                    for num_iterations in [3000]:
                        for fold_num in range(5):

                            OFILE = "{}/{}-{}-{}-{}-{}-{}-{}.out".format(SLURM_OUT, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, fold_num)
                            EFILE = "{}/{}-{}-{}-{}-{}-{}-{}.err".format(SLURM_OUT, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, fold_num)
                            SLURM_SCRIPT = "{}/{}-{}-{}-{}-{}-{}-{}.pbs".format(SLURM_OUT, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, fold_num)
                            #CMD = 'python3 baseline-mtf-nested.py {} {} {} {} {}'.format(dataset, cur_fold, num_latent, lr, iters)
                            CMD = 'python3 rnn-nested-cv.py'.format(dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, 0, fold_num)
                            lines = []
                            lines.append("#!/bin/sh\n")
                            lines.append('#SBATCH --time=1-16:0:00\n')
                            lines.append('#SBATCH --mem=64\n')
                            #lines.append('#SBATCH -c 32\n')
        # 					lines.append('#SBATCH --exclude=artemis[1-5]\n')
                            lines.append('#SBATCH --nodelist=ai[01-05]\n')
                            lines.append('#SBATCH --nodelist=gpusrv[01-06]\n')
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


