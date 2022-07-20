#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J test
#SBATCH -o test_log
module load MATLAB/2017b
matlab -nodesktop -r "run('test.m');exit;"
