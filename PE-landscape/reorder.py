import csv
import numpy as np
import logging
import pandas as pd

def is_adjacent(p1, p2, grid_type):
    """
    判断两个点是否相邻（仅共享面，立方体网格）。

    参数:
        p1 (tuple): 第一个点的坐标 (x, y, z)。
        p2 (tuple): 第二个点的坐标 (x, y, z)。
        grid_type (str): 网格类型，目前仅支持 'cubic'。

    返回:
        bool: 如果两个点相邻则返回 True，否则返回 False。
    """
    if grid_type == 'cubic':
        diff = np.abs(np.array(p1) - np.array(p2))
        # 仅允许在一个轴上相差约0.1（考虑浮点误差），其他轴接近0
        axis_diff = diff > 1e-6
        if np.sum(axis_diff) != 1:
            return False  # 超过一个轴有差异
        # 检查差异轴是否接近0.1
        return np.max(diff) <= 0.1 + 1e-6
    return False


def find_next_point(current_coord, points, grid_type, visited):
    """
    查找下一个相邻点。

    参数:
        current_coord (tuple): 当前点的坐标 (x, y, z)。
        points (list): 所有点的列表，每个点包含 'coords' 和 'energy'。
        grid_type (str): 网格类型，目前仅支持 'cubic'。
        visited (set): 已访问的点的集合。

    返回:
        tuple: 下一个相邻点的坐标。如果未找到则返回 None。
    """
    for point in points:
        coord = tuple(point['coords'])
        if coord in visited:
            continue  # 跳过已访问的点
        if is_adjacent(current_coord, coord, grid_type):
            return coord
    return None


def sort_data_by_energy(data):
    """
    按能量从高到低对数据进行排序。

    参数:
        data (pd.DataFrame): 包含 'Energy_DPMD (eV)' 列的数据集。

    返回:
        pd.DataFrame: 排序后的数据集。
    """
    if 'Energy_DPMD (eV)' not in data.columns:
        raise ValueError("数据集中缺少 'Energy_DPMD (eV)' 列，无法按能量排序。")
    return data.sort_values(by='Energy_DPMD (eV)', ascending=False).reset_index(drop=True)


def find_min_energy_path(data, start_point, end_point, grid_type='cubic'):
    """
    查找从起点到终点的最小能量路径。

    参数:
        data (pd.DataFrame): 包含坐标和能量信息的数据集。
        start_point (tuple): 起点的坐标 (x, y, z)。
        end_point (tuple): 终点的坐标 (x, y, z)。
        grid_type (str): 网格类型，目前仅支持 'cubic'。

    返回:
        list: 最小能量路径上的点列表。
    """
    # 确保数据集中包含坐标信息
    if not all(col in data.columns for col in ['x', 'y', 'z', 'Energy_DPMD (eV)']):
        raise ValueError("数据集中缺少必要的列 ('x', 'y', 'z', 'Energy_DPMD (eV)')。")

    # 提取坐标和能量信息
    points = []
    for _, row in data.iterrows():
        x = round(float(row['x']), 1)
        y = round(float(row['y']), 1)
        z = round(float(row['z']), 1)
        energy = float(row['Energy_DPMD (eV)'])
        points.append({'coords': (x, y, z), 'energy': energy})

    # 构建坐标到点的映射
    coord_map = {tuple(point['coords']): point for point in points}

    # 检查起点和终点是否存在
    start_coord = tuple(start_point)
    end_coord = tuple(end_point)

    if start_coord not in coord_map or end_coord not in coord_map:
        raise ValueError("起点或终点不存在于数据中")

    # 从起点开始，逐步找到路径
    path = []
    visited = set()
    current_coord = start_coord
    point_count = 0
    max_steps = len(points) * 2  # 防止无限循环

    while current_coord != end_coord and point_count < max_steps:
        if current_coord in visited:
            raise RuntimeError(f"检测到循环路径，当前点 {current_coord} 已被访问")
        visited.add(current_coord)
        path.append(current_coord)
        point_count += 1
        logging.info(f"已找到 {point_count} 个点，当前点: {current_coord}")

        # 查找下一个相邻点
        next_coord = find_next_point(current_coord, points, grid_type, visited)
        if not next_coord:
            raise RuntimeError(f"无法找到从 {current_coord} 到终点的路径")
        current_coord = next_coord

    if point_count >= max_steps:
        raise RuntimeError(f"超过最大步数限制 {max_steps}，路径查找终止")

    path.append(end_coord)
    logging.info(f"已找到 {point_count + 1} 个点，当前点: {end_coord}")

    # 将路径中的点映射回原始数据行
    sorted_data = []
    for coord in path:
        for _, row in data.iterrows():
            x = round(float(row['x']), 1)
            y = round(float(row['y']), 1)
            z = round(float(row['z']), 1)
            if (x, y, z) == coord:
                sorted_data.append(row)
                break
        else:
            raise ValueError(f"路径中的点 {coord} 未在原始数据中找到")

    # 返回排序后的数据
    return pd.DataFrame(sorted_data)