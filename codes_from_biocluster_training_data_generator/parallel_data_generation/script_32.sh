#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_32
#SBATCH -o dat_32
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_32.m');exit;"
