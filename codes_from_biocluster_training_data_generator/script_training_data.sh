#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J gen_train
#SBATCH -o gen_train_log
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_generate_training_data.m');exit;"
