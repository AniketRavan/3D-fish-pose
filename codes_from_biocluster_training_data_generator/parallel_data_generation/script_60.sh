#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_60
#SBATCH -o dat_60
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_60.m');exit;"
