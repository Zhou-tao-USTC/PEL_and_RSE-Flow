#!/bin/bash
#SBATCH -J python3_PES
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -p xahcnormal

source activate PES_calculation

python3 main.py
