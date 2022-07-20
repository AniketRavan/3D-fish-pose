#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J gen_train_2
#SBATCH -o gen_train_log_2
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_2.m');exit;"
