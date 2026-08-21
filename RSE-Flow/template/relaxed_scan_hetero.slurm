#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 64

#module purge
#
#source /public21/soft/modules/module.sh
#module load miniforge/24.11
#source activate deepmd-kit

#控制OpenMP线程数

export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1  # 总核心数的1/16
export DP_INFER_BATCH_SIZE=162

echo "使用$SLURM_NTASKS个CPU核心进行计算"

cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/flow_control.py ./flow_control.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/flow_control_cold_start.py ./flow_control_cold_start.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/grid_plan.py ./grid_plan.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/modify_scan.py ./modify_scan.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/opt_constrain_IRC.py ./opt_constrain_IRC.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/opt_constrain_XYZ.py ./opt_constrain_XYZ.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/opt_constrain_XYZ_IRC.py ./opt_constrain_XYZ_IRC.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/read.py ./read.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/renew_grid_coordinate.py ./renew_grid_coordinate.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/scan_plan.py ./scan_plan.py
cp /public23/home/sc61773/zhoutao/soft/relaxed-scan-hetero/V1-try_1/stage-4/goal-2/test-10/parallel_executor.py ./parallel_executor.py

python3 flow_control.py
