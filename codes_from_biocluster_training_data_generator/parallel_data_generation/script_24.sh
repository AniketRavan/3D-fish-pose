#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_24
#SBATCH -o dat_24
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_24.m');exit;"
