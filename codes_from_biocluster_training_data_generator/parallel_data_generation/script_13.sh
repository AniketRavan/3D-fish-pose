#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_13
#SBATCH -o dat_13
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_13.m');exit;"
