#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_78
#SBATCH -o dat_78
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_78.m');exit;"
