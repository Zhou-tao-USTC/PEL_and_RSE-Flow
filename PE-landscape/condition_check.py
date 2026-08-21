from connect_pure import is_connect_fs_pure
from connect_log import is_connect_fs_log
import logging

#第一种条件判断与相应决策
def first_condition_check(data, grid_type, start_point, end_point, cpu_number):
    """对数据data进行连通性判断（通过connect = is_connect_fs(data, grid_type, start_point, end_point, cpu_number)实现"""
    meet_condition = is_connect_fs_log(data, grid_type, start_point, end_point, cpu_number)
    return meet_condition

"""如果满足第一种条件判断,即连通"""
"""通过self_process.py里的first_condition_T_decision实现相关操作"""

"""如果不满足第一种条件判断，即不连通"""
"""如果不连通，即（if not connect），则进入第二种条件判断与相应决策, 这个逻辑属于flow控制了，在flow_control.py里控制"""

#第二种条件判断与相应决策
def second_condition_check(step_size_current, final_step_size_limit):
    """得知了data不连通后（即if not connect），对step_size_current进行第二种条件判断（step_size_current <= final_step_size_limit时，第二种条件满足）。"""
    meet_condition = (step_size_current <= final_step_size_limit)
    return meet_condition

"""如果不满足第二种条件判断"""
"""通过self_process.py里的second_condition_F_decision实现"""

"""如果满足第二种条件判断（即step_size_current <= final_step_size_limit）"""
"""进入第一个不需要条件判断的决策，这个逻辑也属于flow控制，在flow_control.py里控制"""


#第三种条件判断与相应决策
def third_condition_check(skip_number, stage3_data_pre):
    """对skip_number与 stage3_data_pre中的点数是否相等做个判断"""
    meet_condition = (skip_number >= len(stage3_data_pre))
    return meet_condition

"""如果相等，则整个程序终止，这个逻辑也属于flow控制，在flow_control.py里控制"""

"""如果不相等，则进入第二个不需要条件判断的决策，这个逻辑也属于flow控制，在flow_control.py里控制"""


#第四种条件判断与相应决策
def fourth_condition_check(stage3_data_pre, grid_type, start_point, end_point, cpu_number):
    """对stage3_data_pre进行条件判断（即通过connect = is_connect_fs(stage3_data_pre, grid_type, start_point, end_point, cpu_number)。"""
    meet_condition = is_connect_fs_pure(stage3_data_pre, grid_type, start_point, end_point, cpu_number)
    return meet_condition

"""如果满足第四种条件判断，说明此时过滤点过滤对了，那就进行以下操作"""
"""通过self_process.py里的fourth_condition_F_decision实现"""

"""如果不满足第四种条件判断，说明此时过滤点过滤错了，其中有我们要找的最小能量路径上的点，那就进行以下操作"""
"""通过self_process.py里的fourth_condition_T_decision实现"""


#第五种条件判断与相应决策
def fifth_condition_check(new_data_batch):
    """对new_data_batch进行值是否等于1的判断"""
    return (new_data_batch == 1)

"""如果不等于1"""
"""通过self_process.py里的fifth_condition_F_decision实现"""

"""如果等于1，则进入第三个不需要条件判断的决策，这个逻辑也属于flow控制，在flow_control.py里控制"""


#第六种条件判断与相应决策
def sixth_condition_check(stage3_data_pre, grid_type, start_point, end_point, cpu_number):
    """对stage3_data_pre进行条件判断（即通过connect = is_connect_fs(stage3_data_pre, grid_type, start_point, end_point, cpu_number)。"""
    meet_condition = is_connect_fs_pure(stage3_data_pre, grid_type, start_point, end_point, cpu_number)
    return meet_condition

"""如果不满足条件，说明此时过滤点过滤对了，那就进行以下操作"""
"""通过self_process.py里的sixth_condition_F_decision实现"""

"""如果满足条件，说明此时过滤点过滤错了，其中有我们要找的最小能量路径上的点，那就进行以下操作"""
"""通过self_process.py里的sixth_condition_T_decision实现"""


#第七种条件判断与相应决策
def seventh_condition_check(per_point_stage_number):
    """对per_point_stage_number进行其值是否等于10的判断"""
    return (per_point_stage_number >= 10)

"""如果不等于10，则进入第三个不需要条件判断的决策，这个逻辑也属于flow控制，在flow_control.py里控制"""

"""如果等于10"""
"""通过self_process.py里的seventh_condition_T_decision实现"""
