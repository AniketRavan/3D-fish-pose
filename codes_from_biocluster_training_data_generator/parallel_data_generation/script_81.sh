#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_81
#SBATCH -o dat_81
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_81.m');exit;"
