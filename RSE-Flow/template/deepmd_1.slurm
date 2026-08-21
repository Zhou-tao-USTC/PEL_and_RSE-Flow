#!/bin/bash
#SBATCH -x g0075

module purge
module load miniforge/24.1.2
source activate /HOME/scw6f12/run/deepmd-kit
export OMP_NUM_THREADS=1
module load cuda/11.8
dp train input.json

#冻结模型
dp freeze -o graph.pb
