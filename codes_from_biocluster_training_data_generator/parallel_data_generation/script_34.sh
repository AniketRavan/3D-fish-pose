#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_34
#SBATCH -o dat_34
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_34.m');exit;"
