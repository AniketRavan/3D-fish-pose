#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J gen_val
#SBATCH -o gen_val_log
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_validation.m');exit;"
