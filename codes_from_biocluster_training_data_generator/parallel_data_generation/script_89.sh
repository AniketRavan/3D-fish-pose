#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_89
#SBATCH -o dat_89
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_89.m');exit;"
