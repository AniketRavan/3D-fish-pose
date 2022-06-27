#!/bin/bash
#SBATCH --job-name=1model
#SBATCH --output="log/validation.out"
#SBATCH --error="error/validation.err"
#SBATCH --partition=gpux2
#SBATCH --nodes=1
#SBATCH --time=24

echo Running
module load opence/1.5.1
echo Module loaded
python validate.py -e 1  -o "outputs/validation"
echo Done
