import os

def read_input_file(file_name):
    """
    读取输入文件，并解析其中的参数。
    参数:
    file_name (str): 输入文件的路径。
    返回:
    dict: 包含输入文件中所有参数的字典。
    """
    params = {}
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            # 忽略空行和以 '%' 开头的注释行
            if line.startswith('%') or not line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # 去除注释部分
            if '%' in value:
                value = value.split('%')[0].strip()
            try:
                # 尝试将值转换为数值或列表
                if '[' in value or ',' in value:
                    value = eval(value)
                else:
                    value = float(value)
            except ValueError:
                # 如果转换失败，保留为字符串
                pass
            params[key] = value
    return params