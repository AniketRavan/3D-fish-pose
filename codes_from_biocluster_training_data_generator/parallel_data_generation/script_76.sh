#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_76
#SBATCH -o dat_76
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_76.m');exit;"
