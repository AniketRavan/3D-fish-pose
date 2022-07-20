#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J gen_train_4
#SBATCH -o gen_train_log_4
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_4.m');exit;"
