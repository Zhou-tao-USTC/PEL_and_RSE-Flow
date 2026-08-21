#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 16

module purge
source /public21/soft/modules/module.sh
module load cp2k/2024.3-wuxm
export OMPI_MCA_btl_openib_allow_ib=1

/usr/mpi/gcc/openmpi-4.0.3rc4/bin/mpirun -np 16 cp2k.popt -i cp2k_fp.inp -o output
