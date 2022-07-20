#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J coor_3d
#SBATCH -o coor_3d
module load PyTorch/0.4.0-IGB-gcc-4.9.4-Python-3.6.1 
python convert_mat_to_tensor_coor_3d.py
