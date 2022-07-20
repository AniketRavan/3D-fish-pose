#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dat_41
#SBATCH -o dat_41
module load MATLAB/2017b
matlab -nodesktop -r "run('runme_41.m');exit;"
