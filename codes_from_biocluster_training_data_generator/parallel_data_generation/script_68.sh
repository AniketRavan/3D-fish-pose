#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_68
#SBATCH -o dat_68
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_68.m');exit;"
