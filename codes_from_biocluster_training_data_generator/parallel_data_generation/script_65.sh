#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_65
#SBATCH -o dat_65
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_65.m');exit;"
