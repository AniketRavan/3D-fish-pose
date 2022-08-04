#!/bin/bash
#SBATCH --job-name=val_1model
#SBATCH --output="log/validation.out"
#SBATCH --error="error/validation.err"
#SBATCH --partition=gpux1
#SBATCH --nodes=1
#SBATCH --time=24

echo Running
module load opence/1.5.1
echo Module loaded
python runme_validate.py -e 1  -o "outputs/val_220731_4"
echo Done
