import pandas as pd

def filter_data(data, energy_limit):
    """根据能量限制过滤数据"""
    return data[data['Energy_DPMD (eV)'] <= energy_limit]

def write_csv_file(data, filename):
    """保存数据到CSV"""
    data.to_csv(filename, index=False)
