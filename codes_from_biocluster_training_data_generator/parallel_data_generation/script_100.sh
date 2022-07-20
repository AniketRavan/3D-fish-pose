#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_100
#SBATCH -o dat_100
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_100.m');exit;"
