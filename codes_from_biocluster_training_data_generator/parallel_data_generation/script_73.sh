#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_73
#SBATCH -o dat_73
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_73.m');exit;"
