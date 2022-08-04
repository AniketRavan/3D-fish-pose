#!/bin/bash
#SBATCH --job-name=val_par_1model
#SBATCH --output="log/par_validation.out"
#SBATCH --error="error/par_validation.err"
#SBATCH --partition=gpux1
#SBATCH --nodes=1
#SBATCH --time=24

echo Running
module load opence/1.5.1
echo Module loaded
python runme_validate_parallel.py -e 1  -o "outputs/val_par_220731_4"
echo Done
