#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 1

module purge

source /public21/soft/modules/module.sh
module load miniforge/24.11
source activate deepmd-kit

python3 active_flow_control.py
