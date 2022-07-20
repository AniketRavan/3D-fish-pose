#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_79
#SBATCH -o dat_79
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_79.m');exit;"
