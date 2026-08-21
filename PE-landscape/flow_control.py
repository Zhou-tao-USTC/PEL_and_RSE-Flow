from condition_check import *
import pandas as pd
from utils import *
import os
import shutil
from reorder import *

def main_flow_control(params):
    # 初始化变量
    data = pd.read_csv(params['csv_file_name'])
    current_energy_limit = params['initial_energy_limit']
    step_size_current = params['step_size']
    final_step_size_limit = params['final_step_size_limit'] 
    start_point = params['start_point']
    end_point = params['end_point']
    grid_type = params['grid_type']
    init_data_batch_float = params['init_data_batch']
    init_data_batch = int(init_data_batch_float)
    cpu_number_float = params.get('cpu_number', 1) #默认1进程
    cpu_number = int(cpu_number_float)

    last_connected_data = data.copy()
    iteration = 0
    skip_number = 0
    regret_number = 0
    new_data_batch = init_data_batch
    per_point_stage_number = 0

    iteration_status = [] # 格式: [{"iter":1, "status":"T-T"}, ...]
    iteration = 0

    current_condition = 'condition1'  # 初始阶段

    while True:
        current_status = {"iter": iteration}  # 初始化当前迭代状态
        if current_condition == "condition1":
            """如果满足第一种条件判断,即连通"""
            """（1）更新last_connected_data（通过last_connected_data = data.copy()实现）。
            （2）然后更新能量上限1（即current_energy_limit）（更新的逻辑为current_energy_limit -= step_size_current）。
            （3）再然后是以csv格式输出data（通过output_file_name = f'DPMD_PES_spillover_cycle_{iteration}.csv'以及write_csv_file(data, output_file_name)实现）。
            （4）接着用新的能量上限1对data进行过滤（通过filtered_data = filter_data(data, current_energy_limit)实现）。
            （5）再然后进行data的更新（通过data = filtered_data.copy()实现）。
            （6）最后再次进入第一种条件判断与相应决策。
            """
            """如果不满足第一种条件判断，即不连通"""
            """如果不连通，即（if not connect），则进入第二种条件判断与相应决策"""
            iteration += 1
            logging.info(f"\n{'='*30} 第 {iteration} 轮循环开始 {'='*30}")
            meet_condition = first_condition_check(data, grid_type, start_point, end_point, cpu_number)
            if meet_condition:
                current_status["status"] = "F-F"
                iteration_status.append(current_status) 
                logging.info(f"当前能量上限: {current_energy_limit:.5f}，步长: {step_size_current:.5f}")
                logging.info(f"{'☆'*5} 未达到一级收敛 {'☆'*5}")
                output_file_name = f'DPMD_PES_spillover_cycle_{iteration}.csv'
                write_csv_file(data, output_file_name)
                target_dir = './Simplified'
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(output_file_name, target_dir)
                last_connected_data = data.copy()
                current_energy_limit -= step_size_current
                filtered_data = filter_data(data, current_energy_limit)
                data = filtered_data.copy()
                current_condition = "condition1"
            if not meet_condition:
                current_condition = "condition2"

        if current_condition == "condition2":
            """如果不满足第二种条件判断"""
            """（1）先恢复能量上限2（即current_energy_limit_stage2）（恢复的逻辑为current_energy_limit_stage2 = current_energy_limit + step_size_current，）。
            （2）之后调整步长step_size_current（通过step_size_current = max(step_size_current * 0.5, final_step_size_limit)实现）。
            （3）再然后更新能量上限2（通过current_energy_limit_stage2 = current_energy_limit_stage2 - step_size_current实现）。
            （4）再然后是更新能量上限1（通过current_energy_limit = current_energy_limit_stage2实现）。
            （5）接着用能量上限2来对last_connected_data进行过滤（通过filtered_data = filter_data(last_connected_data, current_energy_limit_stage2)实现）。
            （6）接着实现data的更新（通过data = filtered_data.copy()实现）。
            （7）最后再次进入（通过flow_control.py实现）第一种条件判断与相应决策。
            """      
            """如果满足第二种条件判断（即step_size_current <= final_step_size_limit）"""
            """进入第一个不需要条件判断的决策"""
            iteration += 1
            logging.info(f"\n{'='*30} 第 {iteration} 轮循环开始 {'='*30}")
            meet_condition = second_condition_check(step_size_current, final_step_size_limit)
            if meet_condition:
                current_status["status"] = "T-T"
                logging.info(f"当前能量上限: {current_energy_limit:.5f}，步长: {step_size_current:.5f}")
                logging.info(f"{'★'*5}已得到初步简化的势能景观{'★'*5}")
                iteration_status.append(current_status)
                logging.info(f"\n{'='*30} 迭代状态明细 {'='*30}")
                status_desc = {
                    "T-T": "一级收敛和二级收敛均已达成",
                    "T-F": "一级收敛但二级未达成",
                    "F-F": "未达到一级收敛"
                }
                current_condition = "uncondition1"
            if not meet_condition:
                current_status["status"] = "T-F"
                iteration_status.append(current_status) 
                logging.info(f"{'★'*5} 二级收敛没有达成 {'★'*5}")
                logging.info(f"当前能量上限: {current_energy_limit:.5f}，旧的步长: {step_size_current:.5f}")
                current_energy_limit_stage2 = current_energy_limit + step_size_current
                step_size_current = max(step_size_current * 0.5, final_step_size_limit)
                current_energy_limit_stage2 -= step_size_current
                current_energy_limit = current_energy_limit_stage2
                logging.info(f"随后进行的过滤能量上限: {current_energy_limit_stage2:.5f}，新的步长: {step_size_current:.5f}")
                filtered_data = filter_data(last_connected_data, current_energy_limit_stage2)
                data = filtered_data.copy()
                current_condition = "condition1"

        if current_condition == "uncondition1":                 
            """（1）新建立stage3_data_pre数据集和stage3_data_after数据集，并从last_connected_data中把数据复制过来（可以通过stage3_data_pre = last_connected_data.copy()和stage3_data_after = last_connected_data.copy()实现）
            （2）对stage3_data_pre和stage3_data_after进行能量排序（从高到低）。设置skip_number=0
            （3）进入第三个条件判断
            """
            stage3_data_pre = last_connected_data.copy()
            stage3_data_after = last_connected_data.copy()
            # 按能量从高到低排序
            stage3_data_pre = stage3_data_pre.sort_values(by='Energy_DPMD (eV)', ascending=False)
            stage3_data_after = stage3_data_after.sort_values(by='Energy_DPMD (eV)', ascending=False)
            skip_number = 0
            current_condition = "condition3"

        if current_condition == "condition3":
            """如果相等，则整个程序终止"""
            """如果不相等，则进入第二个不需要条件判断的决策"""
            logging.info(f"进入条件3检查，当前跳过点数: {skip_number}，剩余数据量: {len(stage3_data_pre)}")
            meet_condition = third_condition_check(skip_number, stage3_data_pre)
            if meet_condition:
                min_energy_path = find_min_energy_path(stage3_data_pre, start_point, end_point, grid_type)
                output_file_name2 = f'MEP_4D.csv'
                write_csv_file(min_energy_path, output_file_name2)
                target_dir2 = './MEP_4D'
                os.makedirs(target_dir2, exist_ok=True)
                shutil.move(output_file_name2, target_dir2)
                logging.info(f"已找到最小能量路径")
                break
            if not meet_condition:
                current_condition = "uncondition2"

        if current_condition == "uncondition2":
            """（1）并从stage3_data_pre中过滤掉init_data_batch数量的点，需要跳过skip_number的点（例如当skip_number=4时，就从第五个点开始过滤）。过滤完之后更新stage3_data_pre。
            （2）进入第四种条件判断与相应决策
            """
            start_idx = skip_number
            end_idx = skip_number + init_data_batch
            filtered_data = stage3_data_pre.iloc[start_idx:end_idx]
            stage3_data_pre = stage3_data_pre.drop(filtered_data.index)
            current_condition = "condition4"

        if current_condition == "condition4":
            """如果满足第四种条件判断，说明此时过滤点过滤对了，那就进行以下操作"""
            """（1）先更新stage3_data_after，使得其更新为过滤后的数据集（通过stage3_data_after = stage3_data_pre实现）
           （2）回到第三个条件判断与相应决策
            """
            """如果不满足第四种条件判断，说明此时过滤点过滤错了，其中有我们要找的最小能量路径上的点，那就进行以下操作"""
            """（1）regret_number += 1
            （2）还原stage3_data_pre为过滤前的状态（通过stage3_data_pre = stage3_data_after）
            （3）更新new_data_batch（通过new_data_batch = new_data_batch/10）
            （4）进入第五种条件判断与相应决策。
            """
            meet_condition = fourth_condition_check(stage3_data_pre, grid_type, start_point, end_point, cpu_number)
            if meet_condition:
                logging.info(f"无可疑点，继续以初始过滤规模进行")
                stage3_data_after = stage3_data_pre
                new_data_batch = init_data_batch
                current_condition = "condition3"
            if not meet_condition:
                logging.info(f"当前过滤规模: {new_data_batch}，出现可疑点")
                regret_number += 1
                stage3_data_pre = stage3_data_after
                new_data_batch = max(1, new_data_batch // 10)
                current_condition = "condition5"

        if current_condition == "condition5":
            """如果不等于1"""
            """（1）从stage3_data_pre中过滤掉new_data_batch数量的点，同样地，需要跳过skip_number的点（例如当skip_number=4时，就从第五个点开始过滤）。过滤完之后更新stage3_data_pre。
            （2）进入第四种条件判断与相应决策
		    """
            """如果等于1，则进入第三个不需要条件判断的决策"""
            """如果等于1，则进入第四个不需要条件判断的决策"""
            meet_condition = fifth_condition_check(new_data_batch)
            if meet_condition:
                logging.info(f"进入详细检查阶段")
                current_condition = "uncondition3"
            if not meet_condition:
                start_idx = skip_number
                end_idx = skip_number + new_data_batch
                filtered_data = stage3_data_pre.iloc[start_idx:end_idx]
                stage3_data_pre = stage3_data_pre.drop(filtered_data.index)
                current_condition = "condition4"

        if current_condition == "uncondition3":
            """（1）从stage3_data_pre中过滤掉new_data_batch数量的点，同样地，需要跳过skip_number的点（例如当skip_number=4时，就从第五个点开始过滤）。过滤完之后更新stage3_data_pre。
            （2）per_point_stage_number += 1
            （3）进入第六种条件判断与相应决策
            """
            start_idx = skip_number
            end_idx = skip_number + new_data_batch
            filtered_data = stage3_data_pre.iloc[start_idx:end_idx]
            stage3_data_pre = stage3_data_pre.drop(filtered_data.index)
            per_point_stage_number += 1
            logging.info(f"正在详细检查第{per_point_stage_number}点")
            current_condition = "condition6"

        if current_condition == "condition6":
            """如果满足条件，即连通，说明此时过滤点过滤对了，那就进行以下操作"""
            """（1）先更新stage3_data_after，使得其更新为过滤后的数据集（通过stage3_data_after = stage3_data_pre实现）
            （2）进入第七种条件判断与相应决策
            """
            """如果不满足条件，即不连通，说明此时过滤点过滤错了，其中有我们要找的最小能量路径上的点，那就进行以下操作"""
            """（1）per_point_stage_number += 1
            （2）还原stage3_data_pre为过滤前的状态（通过stage3_data_pre = stage3_data_after实现）
            （3）skip_number += 1
            （4）进入第七种条件判断与相应决策
            """
            meet_condition = sixth_condition_check(stage3_data_pre, grid_type, start_point, end_point, cpu_number)
            if meet_condition:
                logging.info(f"该点为干扰点，滤之而后快")
                stage3_data_after = stage3_data_pre
                current_condition = "condition7"
            if not meet_condition:
                logging.info(f"该点为目标点，留之而先安")
                stage3_data_pre = stage3_data_after
                skip_number += 1
                current_condition = "condition7"

        if current_condition == "condition7":
            """如果不等于10，则进入第三个不需要条件判断的决策"""
            """如果等于10"""
            """（1）恢复new_data batch（通过new_data batch = init data batch实现）
            (2)把per_point_stage_number重新设置成0
            （3）进入第三个条件判断与相应决策
            """
            meet_condition = seventh_condition_check(per_point_stage_number)
            if meet_condition:
                logging.info(f"详细检查阶段结束")
                new_data_batch = init_data_batch
                per_point_stage_number = 0
                current_condition = "condition3"
            if not meet_condition:
                current_condition = "uncondition3"