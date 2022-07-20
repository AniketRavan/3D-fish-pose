#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_45
#SBATCH -o dat_45
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_45.m');exit;"
