#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J gen_train
#SBATCH -o gen_train_log
module load MATLAB/2017b
matlab -nodesktop -r "run('runme.m');exit;"
