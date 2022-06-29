#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J crop_coor
#SBATCH -o crop_coor
module load PyTorch/0.4.0-IGB-gcc-4.9.4-Python-3.6.1 
python convert_mat_to_tensor_crop_coor.py
