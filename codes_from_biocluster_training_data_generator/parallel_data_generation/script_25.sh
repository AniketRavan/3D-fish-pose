#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_25
#SBATCH -o dat_25
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_25.m');exit;"
