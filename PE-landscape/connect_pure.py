from scipy.spatial import KDTree
import numpy as np
import logging
from collections import defaultdict
from union_find import UnionFind
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from functools import partial

def is_adjacent(p1, p2, grid_type):
    """判断两个点是否相邻（仅共享面，立方体网格）"""
    if grid_type == 'cubic':
        diff = np.abs(np.array(p1) - np.array(p2))
        # 仅允许在一个轴上相差约0.1（考虑浮点误差），其他轴接近0
        axis_diff = diff > 1e-6
        if np.sum(axis_diff) != 1:
            return False  # 超过一个轴有差异
        # 检查差异轴是否接近0.1
        return np.max(diff) <= 0.1 + 1e-6
    return False

def process_batch(indices, coords, grid_type, radius):
    """处理数据块的独立函数"""
    tree = KDTree(coords)  # 每个子进程独立构建KDTree
    batch_edges = []
    for i in indices:
        neighbors = tree.query_ball_point(coords[i], radius)
        for j in neighbors:
            if i < j and is_adjacent(coords[i], coords[j], grid_type):
                batch_edges.append((i, j))
    return batch_edges

def build_adjacency_list(coords, grid_type, radius=0.1732, max_workers=None):
    """使用多进程分块构建邻接表（修复pickle问题）"""
    n = len(coords)
    chunks = np.array_split(range(n), max_workers*1 if max_workers else 1)
    total_chunks = len(chunks)

    #提交所有任务
    processor = partial(process_batch, coords=coords, grid_type=grid_type, radius=radius)
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk in chunks:
            futures.append(executor.submit(processor, chunk))

    #收集结果
    results = (future.result() for future in futures)
    edges = list(itertools.chain.from_iterable(results))
    
    #构建邻接表
    adj_list = defaultdict(list)
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)
    return adj_list

def is_connect_fs_pure(data, grid_type, start_point, end_point, cpu_number=1):
    """检查起点和终点是否连通"""
    if data.empty:
        logging.warning("空数据视为已收敛")
        return False

    coords = data[['x','y','z']].values
    tolerance = 1e-6

    # 查找起点和终点索引
    start_mask = (
        (np.abs(data['x'] - start_point[0]) < tolerance) &
        (np.abs(data['y'] - start_point[1]) < tolerance) &
        (np.abs(data['z'] - start_point[2]) < tolerance)
    )
    end_mask = (
        (np.abs(data['x'] - end_point[0]) < tolerance) &
        (np.abs(data['y'] - end_point[1]) < tolerance) &
        (np.abs(data['z'] - end_point[2]) < tolerance)
    )
    
    start_indices = np.where(start_mask)[0]
    end_indices = np.where(end_mask)[0]
    
    if len(start_indices) == 0 or len(end_indices) == 0:
        return False
    
    start_idx = start_indices[0]
    end_idx = end_indices[0]

    # 构建邻接表
    adjacency_list = build_adjacency_list(coords, grid_type, max_workers=cpu_number)

    # 并查集连通性检查
    uf = UnionFind(len(coords))

    total_nodes = len(adjacency_list)
    processed = 0
    update_step = max(1, total_nodes // 1)  # 每100%进度输出一次

    for node in adjacency_list:
        for neighbor in adjacency_list[node]:
            uf.union(node, neighbor)
        processed += 1
        if processed % update_step == 0 or processed == total_nodes:
            progress = 100 * processed / total_nodes

    return uf.find(start_idx) == uf.find(end_idx)
