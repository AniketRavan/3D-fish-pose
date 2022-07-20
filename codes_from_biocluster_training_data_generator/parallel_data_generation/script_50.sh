#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_50
#SBATCH -o dat_50
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_50.m');exit;"
