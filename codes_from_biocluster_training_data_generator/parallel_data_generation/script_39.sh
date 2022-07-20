#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_39
#SBATCH -o dat_39
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_39.m');exit;"
