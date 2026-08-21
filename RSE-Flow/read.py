#!/usr/bin/env python3
"""
read.py - 解析relaxed_scan_explore_mlp输入文件的模块
"""

import json
import os
import logging

# 配置日志
logger = logging.getLogger(__name__)

def parse_input(input_file='relaxed_scan_explore_mlp_input.json'):
    """
    解析输入JSON文件
    
    参数:
    input_file (str): 输入文件路径，默认为'relaxed_scan_explore_mlp_input.json'
    
    返回:
    dict: 包含所有参数的字典
    """
    logger.info(f"开始解析输入文件: {input_file}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件 {input_file} 不存在")
        raise FileNotFoundError(f"输入文件 {input_file} 不存在")
    
    try:
        with open(input_file, 'r') as f:
            params = json.load(f)
        logger.info("输入文件解析成功")
    except json.JSONDecodeError as e:
        logger.error(f"输入文件 {input_file} 不是有效的JSON格式: {e}")
        raise ValueError(f"输入文件 {input_file} 不是有效的JSON格式: {e}")
    
    # 验证必需参数
    required_params = [
        "deepmd_version", "deepmd_slurm_templates",
        "relaxed_scan_input", "relaxed_scan_slurm",
        "model_devi_f_trust_lo", "model_devi_f_trust_hi",
        "convergence", "traj_format", "DFT_type"
    ]
    
    for param in required_params:
        if param not in params:
            logger.error(f"必需参数 '{param}' 在输入文件中缺失")
            raise ValueError(f"必需参数 '{param}' 在输入文件中缺失")
    
    # 验证DeePMD slurm模板存在
    for template_path in params["deepmd_slurm_templates"]:
        if not os.path.exists(template_path):
            logger.warning(f"DeePMD slurm模板 {template_path} 不存在")
    
    # 验证柔性能量扫描输入文件存在
    if not os.path.exists(params["relaxed_scan_input"]):
        logger.warning(f"柔性能量扫描输入文件 {params['relaxed_scan_input']} 不存在")
    
    # 验证柔性能量扫描slurm脚本存在
    if not os.path.exists(params["relaxed_scan_slurm"]):
        logger.warning(f"柔性能量扫描slurm脚本 {params['relaxed_scan_slurm']} 不存在")

    # 验证DFT提交脚本存在
    dft_type = params["DFT_type"]
    dft_sub_key = f"DFT_sub_{dft_type}"
    if dft_sub_key in params and not os.path.exists(params[dft_sub_key]):
        logger.warning(f"DFT提交脚本 {params[dft_sub_key]} 不存在")
    
    # 验证预训练模型存在（如果提供了）
    if params.get("pretrained_model") and not os.path.exists(params["pretrained_model"]):
        logger.warning(f"预训练模型 {params['pretrained_model']} 不存在")
    
    # 验证初始数据集存在（如果提供了）
    initial_dataset = params.get("initial_dataset")
    if initial_dataset:
        # 支持单个路径或路径列表
        if isinstance(initial_dataset, str):
            initial_dataset = [initial_dataset]
            params["initial_dataset"] = initial_dataset
        
        for dataset_path in initial_dataset:
            if not os.path.exists(dataset_path):
                logger.warning(f"初始数据集 {dataset_path} 不存在")

    # 验证DeePMD输入JSON文件存在
    if "deepmd_input_deepmd-kit_v2" in params:
        for i, json_path in enumerate(params["deepmd_input_deepmd-kit_v2"]):
            if not os.path.exists(json_path):
                logger.warning(f"DeePMD输入JSON文件 {json_path} 不存在")
    else:
        # 向后兼容：检查旧的deepmd_input_{deepmd_type}参数
        deepmd_type = params.get("deepmd_type", "deepmd-kit_v2")
        old_param_key = f"deepmd_input_{deepmd_type}"
        if old_param_key in params:
            # 将旧的单个文件转换为四个相同的文件列表
            old_json_path = params[old_param_key][0] if isinstance(params[old_param_key], list) else params[old_param_key]
            params["deepmd_input_deepmd-kit_v2"] = [old_json_path] * 4
            logger.warning(f"使用旧的参数 '{old_param_key}'，已转换为四个相同的输入文件")
            
            # 验证文件存在
            if not os.path.exists(old_json_path):
                logger.warning(f"DeePMD输入JSON文件 {old_json_path} 不存在")
        else:
            logger.error("必须提供DeePMD输入JSON文件，使用'deepmd_input_deepmd-kit_v2'参数")
            raise ValueError("必须提供DeePMD输入JSON文件，使用'deepmd_input_deepmd-kit_v2'参数")

    # 验证刚性扫描相关参数（如果启用）
    if params.get("rigid_scan_init_data_method", False):
        rigid_scan_input = params.get("rigid_scan_input")
        if not rigid_scan_input or not os.path.exists(rigid_scan_input):
            logger.error(f"刚性扫描输入文件 {rigid_scan_input} 不存在")
            raise ValueError(f"刚性扫描输入文件 {rigid_scan_input} 不存在")
        
        rigid_scan_mlp = params.get("rigid_scan_mlp")
        if not rigid_scan_mlp or not os.path.exists(rigid_scan_mlp):
            logger.error(f"刚性扫描MLP模型 {rigid_scan_mlp} 不存在")
            raise ValueError(f"刚性扫描MLP模型 {rigid_scan_mlp} 不存在")

    # 如果启用刚性扫描模式，忽略initial_dataset参数
    if params.get("rigid_scan_init_data_method", False):
        logger.info("刚性扫描模式已启用，将忽略initial_dataset参数")
        params["initial_dataset"] = []

    # 验证pre_scan_structures参数
    pre_scan_structures = params.get("pre_scan_structures", "need")
    if pre_scan_structures not in ["need", "delete"]:
        logger.warning(f"pre_scan_structures参数值 {pre_scan_structures} 无效，应为'need'或'delete'")
        params["pre_scan_structures"] = "need"  # 设置默认值

    # 验证柔性能量扫描轨迹目录
    relaxed_scan_traj_dirs = params.get("relaxed_scan_traj_dirs", ["traj"])
    if not isinstance(relaxed_scan_traj_dirs, list):
        logger.warning("relaxed_scan_traj_dirs 应该是一个列表，已转换为列表")
        relaxed_scan_traj_dirs = [relaxed_scan_traj_dirs]
        params["relaxed_scan_traj_dirs"] = relaxed_scan_traj_dirs

    # 设置默认值（如果某些参数缺失）
    params.setdefault("non_zero_start", False)
    params.setdefault("iter_start", 1)
    params.setdefault("stage_start", 1)
    params.setdefault("dft_code", "cp2k")
    params.setdefault("DFT_batch_size", 10)
    
    # 设置日志默认值
    params.setdefault("logging", {})
    params["logging"].setdefault("level", "INFO")
    params["logging"].setdefault("file", "relaxed_scan_explore_mlp.log")
    
    logger.info("输入文件验证完成")
    return params

def validate_params(params):
    """
    验证参数的有效性
    
    参数:
    params (dict): 参数字典
    
    返回:
    bool: 参数是否有效
    list: 错误消息列表（如果有）
    """
    logger.info("开始验证参数")
    errors = []
    
    # 验证DeePMD版本
    if params["deepmd_version"] not in ["v1", "v2"]:
        errors.append(f"不支持的DeePMD版本: {params['deepmd_version']}，必须是'v1'或'v2'")
    
    # 验证轨迹格式
    if params["traj_format"] not in ["vasp", "xyz"]:
        errors.append(f"不支持的轨迹格式: {params['traj_format']}，必须是'vasp'或'xyz'")
    
    # 验证DFT类型
    if params["DFT_type"] not in ["CP2K", "VASP"]:
        errors.append(f"不支持的DFT类型: {params['DFT_type']}，必须是'CP2K'或'VASP'")
    
    # 验证收敛阈值
    convergence = params["convergence"]
    if "accurate_ratio" not in convergence:
        errors.append("收敛设置中缺少'accurate_ratio'参数")
    elif not (0 <= convergence["accurate_ratio"] <= 1):
        errors.append(f"收敛阈值'accurate_ratio'必须在0到1之间，当前值: {convergence['accurate_ratio']}")
    
    # 验证最大迭代次数
    if "max_iterations" not in convergence:
        errors.append("收敛设置中缺少'max_iterations'参数")
    elif not isinstance(convergence["max_iterations"], int) or convergence["max_iterations"] <= 0:
        errors.append(f"最大迭代次数必须是正整数，当前值: {convergence['max_iterations']}")
    
    # 验证力偏差阈值
    if params["model_devi_f_trust_lo"] >= params["model_devi_f_trust_hi"]:
        errors.append("力偏差下限必须小于上限")
    
    # 验证非零启动设置
    if params["non_zero_start"]:
        if "pretrained_model" not in params or not params["pretrained_model"]:
            errors.append("非零启动需要指定预训练模型")
        if "iter_start" not in params or params["iter_start"] < 1:
            errors.append("非零启动需要指定有效的起始迭代次数")
        if "stage_start" not in params or not (1 <= params["stage_start"] <= 7):
            errors.append("非零启动需要指定有效的起始阶段(1-7)")
    
    if errors:
        logger.warning(f"参数验证发现 {len(errors)} 个问题")
        for error in errors:
            logger.warning(f"参数问题: {error}")
    else:
        logger.info("参数验证通过")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    # 测试代码
    try:
        params = parse_input()
        is_valid, errors = validate_params(params)
        
        if is_valid:
            logger.info("输入文件解析成功，参数有效")
        else:
            logger.error("输入文件参数无效:")
            for error in errors:
                logger.error(f"  - {error}")
                
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"错误: {e}")