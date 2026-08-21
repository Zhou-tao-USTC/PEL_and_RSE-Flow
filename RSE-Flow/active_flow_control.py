import os
import shutil
import subprocess
import time
import numpy as np
import json
import glob
import warnings
import dpdata
import pandas as pd
from collections import Counter
from ase.io import read, write
from deepmd.calculator import DP
from deepmd.infer import DeepPot
import tensorflow as tf
import logging
import re
import getpass
from pathlib import Path
import contextlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('relaxed_scan_explore_mlp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 假设的read.py中的parse_input函数
def parse_input(input_file):
    """解析输入JSON文件"""
    with open(input_file, 'r') as f:
        params = json.load(f)
    return params

class ActiveLearningFlow:
    def __init__(self, input_file='relaxed_scan_explore_mlp_input.json'):
        self.params = parse_input(input_file)
        
        # 添加刚性扫描相关属性
        self.rigid_scan_mode = self.params.get('rigid_scan_init_data_method', False)
        self.pre_scan_structures = self.params.get('pre_scan_structures', 'need')
        
        # 非从零启动设置
        if self.params.get('non_zero_start', False):
            self.iter_count = self.params.get('iter_start', 1)
            stage_map = {
                1: 'uncondition1',
                2: 'uncondition2', 
                3: 'uncondition3',
                4: 'condition4',
                5: 'condition5',
                6: 'uncondition6',
                7: 'uncondition7',
                'rigid_scan_init': 'rigid_scan_init'
            }
            self.current_state = stage_map.get(self.params.get('stage_start', 1), 'uncondition1')
            self.current_iter_dir = f"iter_{self.iter_count}"
        else:
            # 如果是刚性扫描模式，从rigid_scan_init状态开始
            if self.rigid_scan_mode:
                self.iter_count = 0
                self.current_state = 'rigid_scan_init'
                self.current_iter_dir = f"iter_{self.iter_count}"
            else:
                self.iter_count = 0
                self.current_state = 'uncondition1'
                self.current_iter_dir = None

        # 添加目录存在性检查
        if self.current_iter_dir and not os.path.exists(self.current_iter_dir):
            logger.warning(f"警告: 指定的迭代目录 {self.current_iter_dir} 不存在")

        #最大迭代次数设置
        self.max_iterations = self.params['convergence']['max_iterations']

        self.all_dft_data = []
        self.marked_structures = {}
        self.current_stats = {}
        
        # 预绑定常用方法到局部变量（微优化）
        self._makedirs = os.makedirs
        self._join = os.path.join
        self._copy = shutil.copy
        self._run = subprocess.run

        # 存储作业ID
        self.job_ids = {
            'deepmd': [],
            'relaxed-scan': [],
            'dft': []
        }

    def run(self):
        """主运行循环 - 使用状态字典避免频繁条件判断"""
        logger.info("开始执行主动学习流程")
        logger.info(f"初始状态: {self.current_state}")
        logger.info(f"最大迭代次数: {self.max_iterations}")
        logger.info(f"刚性扫描模式: {self.rigid_scan_mode}")
        logger.info(f"预扫描结构处理: {self.pre_scan_structures}")

        state_handlers = {
            'uncondition1': self._handle_uncondition1,
            'uncondition2': self._handle_uncondition2,
            'uncondition3': self._handle_uncondition3,
            'condition4': self._handle_condition4,
            'condition5': self._handle_condition5,
            'uncondition6': self._handle_uncondition6,
            'check_dft': self._handle_check_dft,
            'uncondition7': self._handle_uncondition7,
            'rigid_scan_init': self._handle_rigid_scan_init,
        }

        # 非从零启动时，检查已存在的迭代目录
        if self.params.get('non_zero_start', False):
            if os.path.exists(self.current_iter_dir):
                logger.info(f"从已存在的迭代 {self.iter_count} 开始，阶段: {self.current_state}")
            else:
                logger.warning(f"警告: 指定的迭代目录 {self.current_iter_dir} 不存在，将从初始状态开始")
                self.iter_count = 0
                self.current_state = 'uncondition1'
                self.current_iter_dir = None

        while self.current_state:
            # 检查是否已达到最大迭代次数
            if self.iter_count >= self.max_iterations:
                logger.info(f"已达到最大迭代次数 {self.max_iterations}，终止程序。")
                break

            logger.info(f"进入状态: {self.current_state}")    
            handler = state_handlers.get(self.current_state)
            if handler:
                handler()
                logger.info(f"完成状态: {self.current_state}")
            else:
                logger.error(f"未知状态: {self.current_state}")
                break
    def _handle_rigid_scan_init(self):
        """刚性扫描初始化：使用预训练模型执行刚性扫描并生成初始数据集"""
        logger.info("开始刚性扫描初始化")
        
        # 创建初始迭代目录
        self.iter_count = 0
        iter_dir = "iter_0"
        os.makedirs(iter_dir, exist_ok=True)
        self.current_iter_dir = iter_dir
        logger.info(f"创建初始迭代目录: {iter_dir}")
        
        # 创建 rigid_explore 目录
        rigid_explore_dir = os.path.join(iter_dir, "rigid_explore")
        os.makedirs(rigid_explore_dir, exist_ok=True)
        
        # 创建 rigid_scan 目录
        rigid_dir = os.path.join(rigid_explore_dir, "rigid_scan")
        os.makedirs(rigid_dir, exist_ok=True)
        
        # 复制slurm脚本和输入文件
        slurm_script = self.params['relaxed_scan_slurm']
        input_file = self.params['rigid_scan_input']
        
        shutil.copy(slurm_script, os.path.join(rigid_dir, "rigid_scan.slurm"))
        shutil.copy(input_file, os.path.join(rigid_dir, "input.txt"))
        
        # 复制预训练模型
        model_path = self.params['rigid_scan_mlp']
        if os.path.exists(model_path):
            shutil.copy(model_path, os.path.join(rigid_dir, "graph.pb"))
            logger.info(f"复制预训练模型: {model_path} -> {rigid_dir}/graph.pb")
        else:
            logger.error(f"找不到预训练模型文件 {model_path}")
            self.current_state = None
            return
        
        # 提交作业
        os.chdir(rigid_dir)
        result = subprocess.run(["sbatch", "rigid_scan.slurm"], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"提交刚性扫描作业失败: {result.stderr}")
        else:
            # 提取作业ID
            job_id = self.get_slurm_job_id(result.stdout)
            if job_id:
                self.job_ids['rigid-scan'] = [job_id]
                logger.info(f"提交刚性扫描作业成功，作业ID: {job_id}")
        os.chdir("../../..")
        
        # 等待作业完成
        self.wait_for_job_completion("rigid-scan")
        
        # 创建 structure 文件夹并复制结构文件
        structure_dir = os.path.join(rigid_explore_dir, "structures")
        os.makedirs(structure_dir, exist_ok=True)
        
        # 获取轨迹文件
        traj_path = os.path.join(rigid_dir, "traj")
        if os.path.exists(traj_path):
            # 根据输入参数确定文件格式
            file_format = self.params.get('traj_format', 'vasp')
            ext = '.vasp' if file_format == 'vasp' else '.xyz'

            # 只复制pre_point_*文件（刚性扫描只需要这些）
            copied_files = 0
            for f in os.listdir(traj_path):
                if f.startswith('pre_point_') and f.endswith(ext):
                    idx = f.split('_')[-1].split('.')[0]
                    new_name = f"pre_{idx}{ext}"
                    shutil.copy(os.path.join(traj_path, f), 
                               os.path.join(structure_dir, new_name))
                    copied_files += 1
            
            logger.info(f"复制了 {copied_files} 个结构文件到 {structure_dir}")
        else:
            logger.error(f"轨迹路径不存在: {traj_path}")
            return
        
        # 将所有结构标记为candidate
        candidate_data = []
        for f in os.listdir(structure_dir):
            if f.startswith('pre_') and f.endswith(ext):
                candidate_data.append({'structure': f, 'F_devi': 0.0})  # 刚性扫描没有力偏差，设为0
        
        # 保存candidate.csv
        candidate_csv = os.path.join(rigid_explore_dir, "candidate.csv")
        pd.DataFrame(candidate_data).to_csv(candidate_csv, index=False)
        logger.info(f"已将所有 {len(candidate_data)} 个结构标记为候选，保存到 {candidate_csv}")
        
        # 设置标记结构
        self.marked_structures = {
            f: os.path.join(structure_dir, f) for f in os.listdir(structure_dir) 
            if f.startswith('pre_') and f.endswith(ext)
        }

        # 直接进入DFT计算
        self.current_state = 'uncondition6'
   
    def _handle_uncondition1(self):
        """操作2.1：创建迭代目录并提交DPMD训练"""
        # 检查是否已达到最大迭代次数
        if self.iter_count >= self.max_iterations:
            logger.info(f"已达到最大迭代次数 {self.max_iterations}，终止程序。")
            self.current_state = None
            return

        # 如果是刚性扫描模式且是第一轮，跳过DPMD训练
        if self.rigid_scan_mode and self.iter_count == 0:
            logger.info("刚性扫描模式: 跳过iter_0的DPMD训练")
            self.iter_count += 1
            self.current_state = 'uncondition2'
            return

        self.iter_count += 1
        logger.info(f"开始迭代 {self.iter_count}")

        iter_dir = f"iter_{self.iter_count}"
        os.makedirs(iter_dir, exist_ok=True)
        self.current_iter_dir = iter_dir
        logger.info(f"创建迭代目录: {iter_dir}")
        
        # 创建deepmd目录
        deepmd_parent = os.path.join(iter_dir, "deepmd")
        os.makedirs(deepmd_parent, exist_ok=True)

        # 获取DeePMD输入JSON文件列表
        deepmd_input_jsons = self.params.get('deepmd_input_deepmd-kit_v2', [])
        if len(deepmd_input_jsons) != 4:
            logger.warning(f"需要4个DeePMD输入JSON文件，但提供了{len(deepmd_input_jsons)}个")

        # 提交4个DPMD训练任务
        self.job_ids['deepmd'] = []  # 重置作业ID列表
        for i in range(4):
            # 为每个任务创建独立目录
            task_dir = os.path.join(deepmd_parent, f"deepmd_{i}")
            os.makedirs(task_dir, exist_ok=True)
            
            # 复制slurm脚本到当前目录
            slurm_script = f"deepmd_{i}.slurm"
            if not os.path.exists(os.path.join(task_dir, slurm_script)):
                shutil.copy(self.params['deepmd_slurm_templates'][i], 
                           os.path.join(task_dir, slurm_script))

            # 复制对应的deepmd输入文件
            if i < len(deepmd_input_jsons):
                input_file = deepmd_input_jsons[i]
                if os.path.exists(input_file):
                    shutil.copy(input_file, os.path.join(task_dir, "input.json"))
                else:
                    logger.warning(f"找不到DeePMD输入文件 {input_file}")

            # 根据模式决定是否更新训练数据路径
            if self.rigid_scan_mode:
                # 刚性扫描模式：从第一轮迭代开始更新
                if self.iter_count > 0:
                    self.update_deepmd_training_data(task_dir)
            else:
                # 用户指定初始数据集模式：从第二轮迭代开始更新
                if self.iter_count > 1:
                    self.update_deepmd_training_data(task_dir)

            # 提交作业
            os.chdir(task_dir)
            result = subprocess.run(["sbatch", slurm_script], 
                                   capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"提交DPMD训练作业失败: {result.stderr}")
            else:
                # 提取作业ID
                job_id = self.get_slurm_job_id(result.stdout)
                if job_id:
                    self.job_ids['deepmd'].append(job_id)
                    logger.info(f"提交DPMD训练作业成功，作业ID: {job_id}")
            os.chdir("../../..")
        
        # 等待训练完成
        self.wait_for_job_completion("deepmd")
        
        self.current_state = 'uncondition2'
    
    def _handle_uncondition2(self):
        """操作2.2：执行柔性能量扫描"""
        logger.info("开始执行柔性能量扫描")
    
        relaxed_dir = os.path.join(self.current_iter_dir, "relaxed_explore")
        os.makedirs(relaxed_dir, exist_ok=True)
        
        # 创建relaxed-scan-hetero子目录
        scan_dir = os.path.join(relaxed_dir, "relaxed-scan-hetero")
        os.makedirs(scan_dir, exist_ok=True)
    
        # 复制slurm脚本和输入文件
        slurm_script = self.params['relaxed_scan_slurm']
        input_file = self.params['relaxed_scan_input']
    
        shutil.copy(slurm_script, os.path.join(scan_dir, "relaxed_scan_hetero.slurm"))
        shutil.copy(input_file, os.path.join(scan_dir, "input.txt"))
    
        # 复制第一个训练好的模型
        model_path = os.path.join(self.current_iter_dir, "deepmd", "deepmd_0", "graph.pb")
        if os.path.exists(model_path):
            shutil.copy(model_path, os.path.join(scan_dir, "graph.pb"))
            logger.info(f"复制模型文件: {model_path} -> {scan_dir}/graph.pb")
        else:
            logger.warning(f"找不到模型文件 {model_path}")
            self.current_state = None
            return
    
        # 提交作业
        os.chdir(scan_dir)
        result = subprocess.run(["sbatch", "relaxed_scan_hetero.slurm"], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"提交柔性能量扫描作业失败: {result.stderr}")
        else:
            # 提取作业ID
            job_id = self.get_slurm_job_id(result.stdout)
            if job_id:
                self.job_ids['relaxed-scan'] = [job_id]
                logger.info(f"提交柔性能量扫描作业成功，作业ID: {job_id}")
    
        os.chdir("../../../")
        
        # 等待作业完成
        self.wait_for_job_completion("relaxed-scan")
        
        # 创建structure文件夹并复制结构文件
        structure_dir = os.path.join(relaxed_dir, "structures")
        os.makedirs(structure_dir, exist_ok=True)
        logger.info(f"创建结构目录: {structure_dir}")
    
        # 获取轨迹文件夹列表
        traj_dirs = self.params.get('relaxed_scan_traj_dirs', ['traj'])
        logger.info(f"轨迹文件夹列表: {traj_dirs}")
        
        # 根据输入参数确定文件格式
        file_format = self.params.get('traj_format', 'vasp')
        ext = '.vasp' if file_format == 'vasp' else '.xyz'
    
        # 复制并重命名文件
        copied_files = 0
        for traj_dir_name in traj_dirs:
            traj_path = os.path.join(scan_dir, traj_dir_name)
            logger.info(f"检查轨迹路径: {traj_path}")
    
            if os.path.exists(traj_path):
                logger.info(f"找到轨迹目录 {traj_dir_name}，包含文件: {os.listdir(traj_path)}")
                
                # 为每个系统创建标识符
                sys_id = traj_dir_name.replace('traj_', 'sys')  # 例如: traj_X0_Y0 -> sysX0_Y0
                
                # 复制所有文件
                for f in os.listdir(traj_path):
                    logger.info(f"处理文件: {f}")
    
                    # 根据pre_scan_structures设置决定复制哪些文件
                    if self.pre_scan_structures == "need":
                        # 复制所有文件
                        if f.startswith('pre_point_') and f.endswith(ext):
                            idx = f.split('_')[-1].split('.')[0]
                            new_name = f"{sys_id}_pre_{idx}{ext}"
                            shutil.copy(os.path.join(traj_path, f), 
                                       os.path.join(structure_dir, new_name))
                            copied_files += 1
                            logger.info(f"复制前扫描结构: {f} -> {new_name}")
                        elif f.startswith('point_') and f.endswith(ext):
                            idx = f.split('_')[-1].split('.')[0]
                            new_name = f"{sys_id}_after_{idx}{ext}"
                            shutil.copy(os.path.join(traj_path, f), 
                                       os.path.join(structure_dir, new_name))
                            copied_files += 1
                            logger.info(f"复制后扫描结构: {f} -> {new_name}")
                    else:  # pre_scan_structures == "delete"
                        # 只复制point_*文件
                        if f.startswith('point_') and f.endswith(ext):
                            idx = f.split('_')[-1].split('.')[0]
                            new_name = f"{sys_id}_after_{idx}{ext}"
                            shutil.copy(os.path.join(traj_path, f), 
                                       os.path.join(structure_dir, new_name))
                            copied_files += 1
                            logger.info(f"复制扫描结构: {f} -> {new_name}")
            else:
                logger.warning(f"轨迹路径不存在: {traj_path}")
    
        logger.info(f"总共复制了 {copied_files} 个结构文件到 {structure_dir}")
    
        self.current_state = 'uncondition3'

    def _handle_uncondition3(self):
        """操作2.3：使用四个势函数推断和标记 - 优化版本"""
        logger.info("开始模型推断和标记")
    
        # 获取柔性能量扫描生成的结构
        structure_dir = os.path.join(self.current_iter_dir, "relaxed_explore", "structures")
    
        # 检查结构目录是否存在
        if not os.path.exists(structure_dir):
            logger.error(f"结构目录不存在: {structure_dir}")
            os.makedirs(structure_dir, exist_ok=True)
    
        structure_files = self.get_structure_files_from_dir(structure_dir)
        logger.info(f"找到 {len(structure_files)} 个结构文件")
    
        if len(structure_files) == 0:
            logger.warning("没有找到任何结构文件，跳过模型推断")
            # 创建空的CSV文件
            csv_path = os.path.join(self.current_iter_dir, "relaxed_explore", "model_predictions.csv")
            pd.DataFrame().to_csv(csv_path, index=False)
            
            marking_csv_path = os.path.join(self.current_iter_dir, "relaxed_explore", "model_devi_marking.csv")
            pd.DataFrame(columns=['structure', 'accurate', 'candidate', 'failed', 'F_devi']).to_csv(marking_csv_path, index=False)
    
            self.current_state = 'condition4'
            return
    
        # 准备四个模型的计算器
        calculators = []
        model_paths = []
        for i in range(4):
            model_path = os.path.join(self.current_iter_dir, "deepmd", f"deepmd_{i}", "graph.pb")
            model_paths.append(model_path)
            
            if not os.path.exists(model_path):
                logger.warning(f"找不到模型文件 {model_path}")
                calculators.append(None)
                continue
    
            try:
                calc_manager = self.GlobalCalculatorManager(model_path)
                calculators.append(calc_manager)
                logger.info(f"成功创建计算器 {i}")
            except Exception as e:
                logger.error(f"创建计算器失败: {e}")
                calculators.append(None)
    
        # 为每个结构使用四个模型进行推断
        all_results = []
        csv_data = []
    
        for struct_file in structure_files:
            struct_name = os.path.basename(struct_file)
            struct_results = {'structure_name': struct_name}
            
            try:
                # 读取结构
                atoms = read(struct_file)
                n_atoms = len(atoms)
                logger.info(f"成功读取结构: {struct_name}, 原子数: {n_atoms}")
                
                # 使用四个模型进行推断
                energies = []
                forces_list = []
                
                for i, calc_manager in enumerate(calculators):
                    if calc_manager is None:
                        energies.append(np.nan)
                        forces_list.append(None)
                        struct_results[f'DP_{i+1}_energy'] = np.nan
                        continue
                        
                    try:
                        # 获取计算器并计算
                        calc_atoms = calc_manager.get_calculator(atoms)
                        energy = calc_atoms.get_potential_energy()
                        forces = calc_atoms.get_forces()
                        
                        energies.append(energy)
                        forces_list.append(forces)
                        
                        # 添加到CSV数据
                        struct_results[f'DP_{i+1}_energy'] = energy
                        logger.info(f"模型{i}推断结构{struct_name}成功，能量: {energy:.6f}")
                        
                    except Exception as e:
                        logger.error(f"模型{i}推断结构{struct_name}失败: {e}")
                        energies.append(np.nan)
                        forces_list.append(None)
                        struct_results[f'DP_{i+1}_energy'] = np.nan
                
                # 记录结果
                all_results.append({
                    'structure_name': struct_name,
                    'structure_file': struct_file,
                    'energies': energies,
                    'forces_list': forces_list 
                })
    
                csv_data.append(struct_results)
                
            except Exception as e:
                logger.error(f"读取结构{struct_file}失败: {e}")
                continue
        
        # 保存CSV文件
        csv_path = os.path.join(self.current_iter_dir, "relaxed_explore", "model_predictions.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"已保存模型预测结果到: {csv_path}")
        
        # 计算模型偏差并标记结构（使用修正的方法）
        self.marked_structures, marking_df = self.calculate_model_deviations_and_mark(all_results)
        
        # 保存标记结果的CSV
        marking_csv_path = os.path.join(self.current_iter_dir, "relaxed_explore", "model_devi_marking.csv")
        marking_df.to_csv(marking_csv_path, index=False)
        logger.info(f"已保存模型偏差标记结果到: {marking_csv_path}")
        
        self.current_state = 'condition4'

    def _handle_condition4(self):
        """操作2.4：输出标记结构和统计信息"""
        logger.info("开始处理标记结构和统计信息")

        dft_dir = os.path.join(self.current_iter_dir, "DFT", "structures")
        os.makedirs(dft_dir, exist_ok=True)
        logger.info(f"创建DFT结构目录: {dft_dir}")
        
        # 复制之前的所有结构
        if self.iter_count > 1:
            prev_dft_dir = os.path.join(f"iter_{self.iter_count-1}", "DFT", "structures")
            if os.path.exists(prev_dft_dir):
                copied_count = 0
                for f in os.listdir(prev_dft_dir):
                    shutil.copy(os.path.join(prev_dft_dir, f), dft_dir)
                    copied_count += 1
                logger.info(f"从上一迭代复制了 {copied_count} 个结构文件")
        
        # 保存新标记的结构
        candidate_csv = os.path.join(self.current_iter_dir, "relaxed_explore", "candidate.csv")

        # 检查candidate.csv文件是否存在
        if os.path.exists(candidate_csv):
            try:
                candidate_df = pd.read_csv(candidate_csv)
                copied_count = 0
                for _, row in candidate_df.iterrows():
                    struct_name = row['structure']
                    struct_file = os.path.join(self.current_iter_dir, "relaxed_explore", "structures", struct_name)
                    if os.path.exists(struct_file):
                        shutil.copy(struct_file, os.path.join(dft_dir, struct_name))
                        copied_count += 1
                logger.info(f"复制了 {copied_count} 个候选结构文件")

            except Exception as e:
                logger.error(f"读取candidate.csv文件失败: {e}")
                # 创建空的DataFrame避免后续错误
                candidate_df = pd.DataFrame(columns=['structure', 'F_devi'])

        else:
            logger.warning(f"candidate.csv文件不存在: {candidate_csv}")
            candidate_df = pd.DataFrame(columns=['structure', 'F_devi'])

        # 计算统计信息
        self.current_stats = self.calculate_statistics()
        logger.info(f"迭代 {self.iter_count} 统计:")
        logger.info(f"Accurate比例: {self.current_stats['accurate_ratio']:.2f}")
        logger.info(f"Candidate比例: {self.current_stats['candidate_ratio']:.2f}")
        logger.info(f"Failed比例: {self.current_stats['failed_ratio']:.2f}")
        
        self.current_state = 'condition5'
    
    def _handle_condition5(self):
        """判断2.5：检查收敛条件"""
        logger.info("检查收敛条件")
        
        # 检查是否已达到最大迭代次数
        if self.iter_count >= self.max_iterations:
            logger.info(f"已达到最大迭代次数 {self.max_iterations}，终止程序。")
            self.current_state = None
            return
            
        accurate_ratio = self.current_stats['accurate_ratio']
        convergence_threshold = self.params['convergence']['accurate_ratio']
        
        if accurate_ratio >= convergence_threshold:
            logger.info('已达到收敛，终止程序')
            self.current_state = None  # 终止程序
        else:
            logger.info('未达到收敛，继续执行DFT计算')
            self.current_state = 'uncondition6'

    def _handle_uncondition6(self):
        """操作2.6：提交DFT计算"""
        logger.info("开始提交DFT计算")
    
        # 如果是刚性扫描模式且是第一轮，使用rigid_scan目录
        if self.rigid_scan_mode and self.iter_count == 0:
            dft_parent = os.path.join(self.current_iter_dir, "DFT")
        else:
            dft_parent = os.path.join(self.current_iter_dir, "DFT")
    
        calculate_dir = os.path.join(dft_parent, "calculate")
        os.makedirs(calculate_dir, exist_ok=True)
        logger.info(f"创建DFT计算目录: {calculate_dir}")
    
        # 获取candidate结构
        if self.rigid_scan_mode and self.iter_count == 0:
            candidate_csv = os.path.join(self.current_iter_dir, "rigid_explore", "candidate.csv")
        else:
            candidate_csv = os.path.join(self.current_iter_dir, "relaxed_explore", "candidate.csv")
    
        # 检查 candidate.csv 是否存在     
        if not os.path.exists(candidate_csv):
            logger.warning("没有找到candidate结构，跳过DFT计算")
            self.current_state = None  # 直接终止程序
            return
    
        # 检查 candidate.csv 是否为空
        if os.path.getsize(candidate_csv) == 0:
            logger.warning("candidate.csv 文件为空，跳过DFT计算")
            self.current_state = None  # 直接终止程序
            return
    
        # 检查 candidate.csv 有多少侯选结构
        try:
            candidate_df = pd.read_csv(candidate_csv)
        except Exception as e:
            logger.error(f"读取 candidate.csv 失败: {e}，跳过DFT计算")
            self.current_state = None  # 直接终止程序
            return
    
        logger.info(f"找到 {len(candidate_df)} 个候选结构")
    
        # 获取批量大小
        batch_size = self.params.get('DFT_batch_size', 15)
        logger.info(f"DFT批量大小: {batch_size}")
    
        # 为每个结构创建独立目录（修复目录结构问题）
        self.job_ids['dft'] = []  # 重置DFT作业ID列表
        all_directories = []
        
        for _, row in candidate_df.iterrows():
            struct_name = row['structure']
            struct_base_name = os.path.splitext(struct_name)[0]  # 去掉扩展名
            
            # 直接为每个结构创建独立的计算目录
            struct_dir = os.path.join(calculate_dir, struct_base_name)
            os.makedirs(struct_dir, exist_ok=True)
            
            # 保存结构文件
            structure_path = os.path.join(struct_dir, struct_name)
            
            # 确定源结构文件路径
            if self.rigid_scan_mode and self.iter_count == 0:
                src_structure_path = os.path.join(self.current_iter_dir, "rigid_explore", "structures", struct_name)
            else:
                src_structure_path = os.path.join(self.current_iter_dir, "relaxed_explore", "structures", struct_name)
    
            if os.path.exists(src_structure_path):
                shutil.copy(src_structure_path, structure_path)
                logger.info(f"复制结构文件: {src_structure_path} -> {structure_path}")
                
                # 如果是CP2K计算，生成coord.xyz文件
                if self.params.get('DFT_type') == 'CP2K':
                    self.generate_coord_xyz(struct_dir, struct_name)
            else:
                logger.warning(f"找不到结构文件 {src_structure_path}")
                continue
            
            # 复制DFT输入文件和slurm脚本
            dft_type = self.params.get('DFT_type', 'CP2K')
            dft_inputs = self.params.get(f'DFT_input_{dft_type}', [])
            dft_sub = self.params.get(f'DFT_sub_{dft_type}')
    
            for input_file in dft_inputs:
                if os.path.exists(input_file):
                    shutil.copy(input_file, struct_dir)
                    logger.info(f"复制DFT输入文件: {input_file} -> {struct_dir}")
                else:
                    logger.warning(f"找不到DFT输入文件 {input_file}")
    
            if dft_sub and os.path.exists(dft_sub):
                slurm_script = os.path.basename(dft_sub)
                shutil.copy(dft_sub, os.path.join(struct_dir, slurm_script))
                logger.info(f"复制DFT提交脚本: {dft_sub} -> {struct_dir}")
                
                all_directories.append(struct_dir)
            else:
                logger.warning(f"找不到提交脚本 {dft_sub}")
               
        # 分批提交作业
        for i in range(0, len(all_directories), batch_size):
            batch = all_directories[i:i+batch_size]
            logger.info(f"提交第 {i//batch_size + 1} 批 DFT 计算作业，共 {len(batch)} 个")
            
            batch_job_ids = []
            for struct_dir in batch:
                try:
                    # 提交作业
                    # 使用上下文管理器安全地切换目录
                    with self.change_directory(struct_dir):
                        logger.info(f"在处理目录: {os.getcwd()}")
                        
                        # 提交作业
                        result = subprocess.run(
                            ["sbatch", slurm_script], 
                            capture_output=True, 
                            text=True,
                            timeout=30  # 添加超时防止挂起
                        )
                        
    
                        # 检查作业是否提交成功
                        if result.returncode == 0:
                            job_id = self.get_slurm_job_id(result.stdout)
                            if job_id:
                                batch_job_ids.append(job_id)
                                logger.info(f"提交DFT计算作业成功，作业ID: {job_id}")
                            else:
                                # 尝试从stderr提取作业ID
                                job_id = self.get_slurm_job_id(result.stderr)
                                if job_id:
                                    batch_job_ids.append(job_id)
                                    logger.info(f"提交DFT计算作业成功，作业ID: {job_id} (从stderr提取)")
                                else:
                                    logger.error(f"无法提取作业ID，stdout: {result.stdout}, stderr: {result.stderr}")
                        else:
                            logger.error(f"提交作业失败，返回码: {result.returncode}, stderr: {result.stderr}")
                            
                except Exception as e:
                    logger.error(f"处理目录 {struct_dir} 时发生异常: {e}")
            
            # 等待当前批次完成
            if batch_job_ids:
                self.job_ids['dft'].extend(batch_job_ids)
                logger.info(f"等待第 {i//batch_size + 1} 批 DFT 计算作业完成")
                self.wait_for_job_completion("dft", specific_jobs=batch_job_ids)
            else:
                logger.warning(f"第 {i//batch_size + 1} 批没有提交任何DFT计算作业")
        
        self.current_state = 'check_dft'
    
    def _handle_check_dft(self):
        """检查DFT计算结果是否正常收敛"""
        logger.info("检查DFT计算结果收敛性")
        
        dft_parent = os.path.join(self.current_iter_dir, "DFT")
        
        # 根据输入参数选择检查脚本
        dft_code = self.params.get('dft_code', 'cp2k')
        if dft_code == 'cp2k':
            self.cp2k_check_fp(dft_parent)
        elif dft_code == 'vasp':
            self.vasp_check_fp(dft_parent)
        else:
            logger.error(f"未知的DFT代码: {dft_code}")
            self.current_state = None
            return
        
        # 如果有收敛问题，终止程序
        if hasattr(self, 'dft_convergence_issues') and self.dft_convergence_issues:
            logger.error("DFT计算存在收敛问题，终止工作流")
            self.current_state = None
        else:
            logger.info("DFT计算全部收敛，继续处理结果")
            self.current_state = 'uncondition7'
    
    def _handle_uncondition7(self):
        """操作2.7：处理DFT结果并更新数据集"""
        logger.info("开始处理DFT结果并更新数据集")
        
        # 转换DFT结果为npy格式
        self.convert_dft_to_npy()
        
        # 创建database目录
        database_dir = os.path.join(self.current_iter_dir, "database")
        os.makedirs(database_dir, exist_ok=True)
        logger.info(f"创建数据库目录: {database_dir}")
    
        # 如果是刚性扫描模式且是第一轮，创建初始数据集
        if self.rigid_scan_mode and self.iter_count == 0:
            # 将当前迭代的数据集复制到初始数据集位置
            iter_data_dir = os.path.join(self.current_iter_dir, "database", f"iter_{self.iter_count}_data")
            
            if os.path.exists(iter_data_dir):
                logger.info(f"刚性扫描模式: 已将数据集保存到 {iter_data_dir}")
            
            # 重置迭代计数并进入正常流程
            self.iter_count = 0
            self.rigid_scan_mode = False  # 关闭刚性扫描模式
            self.current_state = 'uncondition1'
            return
    
        # 复制初始数据集（根据模式不同处理）
        if self.rigid_scan_mode:
            # 刚性扫描模式：复制iter_0_data
            src_path = os.path.join("iter_0", "database", "iter_0_data")
            dst_path = os.path.join(database_dir, "iter_0_data")
            if os.path.exists(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                logger.info(f"复制刚性扫描初始数据集: {src_path} -> {dst_path}")
        else:
            # 用户指定初始数据集模式：复制用户指定的数据集
            initial_datasets = self.params.get('initial_dataset', [])
            copied_count = 0
            for dataset_path in initial_datasets:
                dataset_name = os.path.basename(dataset_path.rstrip('/'))
                dst_path = os.path.join(database_dir, dataset_name)
                
                if os.path.exists(dataset_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(dataset_path, dst_path)
                    copied_count += 1
                    logger.info(f"复制用户初始数据集: {dataset_path} -> {dst_path}")
                else:
                    logger.warning(f"初始数据集路径不存在: {dataset_path}")
            logger.info(f"复制了 {copied_count} 个初始数据集")
        
        # 复制之前迭代的所有数据集
        if self.iter_count > 1:
            prev_database_dir = os.path.join(f"iter_{self.iter_count-1}", "database")
            if os.path.exists(prev_database_dir):
                copied_count = 0
                for item in os.listdir(prev_database_dir):
                    # 复制所有数据集目录（iter_x_data）
                    if item.startswith("iter_") and item.endswith("_data"):
                        src_path = os.path.join(prev_database_dir, item)
                        dst_path = os.path.join(database_dir, item)
                        if os.path.isdir(src_path):
                            if os.path.exists(dst_path):
                                shutil.rmtree(dst_path)
                            shutil.copytree(src_path, dst_path)
                            copied_count += 1
                logger.info(f"从上一迭代复制了 {copied_count} 个数据集")
        
        # 复制当前迭代的DFT结果数据集（如果有）
        current_iter_data_dir = os.path.join(self.current_iter_dir, "database", f"iter_{self.iter_count}_data")
        if os.path.exists(current_iter_data_dir):
            logger.info(f"当前迭代的数据集已存在: {current_iter_data_dir}")
        else:
            # 检查是否有临时生成的数据集需要移动
            temp_data_dirs = glob.glob(os.path.join(self.current_iter_dir, "temp_npy", "*"))
            if temp_data_dirs:
                for temp_dir in temp_data_dirs:
                    if os.path.isdir(temp_dir):
                        shutil.move(temp_dir, current_iter_data_dir)
                        logger.info(f"移动临时数据集: {temp_dir} -> {current_iter_data_dir}")
        
        # 回到下一轮迭代
        logger.info(f"完成迭代 {self.iter_count}，准备下一轮迭代")
        self.current_state = 'uncondition1'
    
    # 以下是辅助方法的实现
    
    def wait_for_job_completion(self, job_type, specific_jobs=None, max_retries=3):
        """等待作业完成 - 使用实际的作业状态检查"""
        logger.info(f"等待 {job_type} 作业完成")
        
        if specific_jobs is None:
            if job_type not in self.job_ids or not self.job_ids[job_type]:
                logger.warning(f"没有找到 {job_type} 作业ID，跳过等待")
                return
                
            job_ids = self.job_ids[job_type]
        else:
            job_ids = specific_jobs
        
        username = getpass.getuser()
        check_interval = 300
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 获取当前运行中的作业
                running_jobs = self.get_running_jobs(username)
            
                # 检查当前作业是否还在运行
                remaining_jobs = [job_id for job_id in job_ids if job_id in running_jobs]
            
                if not remaining_jobs:
                    logger.info(f"所有 {job_type} 作业已完成")
                    break
                
                logger.info(f"{job_type} 作业剩余 {len(remaining_jobs)}/{len(job_ids)} 仍在运行，{check_interval//60} 分钟后重新检查...")
                time.sleep(check_interval)
                retry_count = 0  # 重置重试计数，因为这次查询成功了

            except Exception as e:
                retry_count += 1
                logger.warning(f"作业状态查询失败 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    logger.error(f"作业状态查询连续失败 {max_retries} 次，尝试基于文件的验证")
                    break

    def get_slurm_job_id(self, output):
        """从sbatch输出中提取作业ID"""
        match = re.search(r'\d+', output)
        return match.group() if match else None
    
    def get_running_jobs(self, username):
        """获取当前用户所有运行中的作业ID列表"""
        try:
            result = subprocess.run(
                ['squeue', '-u', username, '--noheader', '--format=%i'],
                capture_output=True, text=True, timeout=30  # 添加超时
            )
            
            if result.returncode != 0:
                logger.error(f"squeue命令执行失败: {result.stderr}")
                raise Exception(f"squeue error: {result.stderr}")
                
            output = result.stdout.strip()
            return output.split() if output else []
            
        except subprocess.TimeoutExpired:
            logger.warning("squeue命令执行超时")
            raise Exception("squeue command timeout")
        except subprocess.CalledProcessError as e:
            logger.error(f"squeue命令执行错误: {e.stderr if e.stderr else e.output}")
            raise Exception(f"squeue command failed: {e.stderr if e.stderr else e.output}")
        except Exception as e:
            logger.error(f"获取作业列表异常: {e}")
            raise

    def update_deepmd_training_data(self, task_dir):
        """更新deepmd训练数据的路径"""
        input_json_path = os.path.join(task_dir, "input.json")
        if not os.path.exists(input_json_path):
            return
        
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        # 构建所有之前迭代的数据集路径
        systems = []
        
        # 获取当前工作目录的绝对路径
        current_work_dir = os.getcwd()

        # 添加初始数据集（根据模式不同处理）
        if self.rigid_scan_mode:
            # 刚性扫描模式：添加iter_0_data
            iter_0_data_path = os.path.join(current_work_dir, "iter_0", "database", "iter_0_data")
            if os.path.exists(iter_0_data_path):
                # 计算相对于task_dir的路径
                rel_path = os.path.relpath(iter_0_data_path, task_dir)
                systems.append(rel_path)
                logger.info(f"添加刚性扫描初始数据集路径: {rel_path}")
            else:
                logger.warning(f"刚性扫描初始数据集路径不存在: {iter_0_data_path}")
        else:
            # 用户指定初始数据集模式：添加用户指定的数据集
            initial_datasets = self.params.get('initial_dataset', [])
            for dataset_path in initial_datasets:
                # 构建完整路径
                full_dataset_path = os.path.join(current_work_dir, dataset_path)
                if os.path.exists(full_dataset_path):
                    # 计算相对于task_dir的路径
                    rel_path = os.path.relpath(full_dataset_path, task_dir)
                    systems.append(rel_path)
                    logger.info(f"添加用户初始数据集路径: {rel_path}")
                else:
                    logger.warning(f"用户初始数据集路径不存在: {full_dataset_path}")

        # 添加后续迭代的数据集（从1到当前迭代的前一个迭代）
        for i in range(1, self.iter_count):
            iter_data_path = os.path.join(current_work_dir, f"iter_{i}", "database", f"iter_{i}_data")
            if os.path.exists(iter_data_path):
                # 计算相对于task_dir的路径
                rel_path = os.path.relpath(iter_data_path, task_dir)
                systems.append(rel_path)
                logger.info(f"添加迭代 {i} 数据集路径: {rel_path}")
            else:
                logger.warning(f"迭代 {i} 数据集路径不存在: {iter_data_path}")
        
        # 更新训练数据路径
        if "training" in data and "training_data" in data["training"]:
            data["training"]["training_data"]["systems"] = systems
            logger.info(f"更新训练数据路径为: {systems}")

            # 更新batch_size，使其与systems列表长度匹配
            num_systems = len(systems)
            if "batch_size" in data["training"]["training_data"]:
                # 如果batch_size已经存在，确保其长度与systems匹配
                current_batch_size = data["training"]["training_data"]["batch_size"]
                if len(current_batch_size) != num_systems:
                    # 如果长度不匹配，创建一个新的batch_size列表
                    data["training"]["training_data"]["batch_size"] = [1] * num_systems
                    logger.info(f"更新batch_size为: {[1] * num_systems}")
            else:
                # 如果batch_size不存在，创建一个新的
                data["training"]["training_data"]["batch_size"] = [1] * num_systems
                logger.info(f"创建batch_size: {[1] * num_systems}")
            
            logger.info(f"更新训练数据路径为: {systems}")

        with open(input_json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_structure_files_from_dir(self, structure_dir):
        """从目录中获取结构文件列表"""
        file_format = self.params.get('traj_format', 'vasp')
        ext = '.vasp' if file_format == 'vasp' else '.xyz'
        
        struct_files = glob.glob(os.path.join(structure_dir, f"*{ext}"))
        return struct_files

    def calculate_model_deviations_and_mark(self, all_results):
        """计算模型偏差（仅力）并标记结构 - 使用RMSD方法"""
        accurate_data = []
        candidate_data = []
        failed_data = []
        
        marked_structures = {}
    
        for result in all_results:
            struct_name = result['structure_name']
            struct_file = result['structure_file']
            forces_list = result['forces_list']
            
            # 检查是否有有效的力数据
            valid_forces = [f for f in forces_list if f is not None]
            if len(valid_forces) < 2:  # 至少需要2个有效力值来计算偏差
                logger.warning(f"结构{struct_name}的有效力数据不足，跳过")
                failed_data.append({'structure': struct_name, 'F_devi': float('inf')})
                marked_structures[struct_name] = struct_file
                continue
                
            # 将有效的力数据转换为numpy数组，形状为 (n_models, n_atoms, 3)
            forces_array = np.array(valid_forces)
            n_models, n_atoms, _ = forces_array.shape
            
            # 新的力偏差计算逻辑：使用RMSD方法
            atom_deviations = []  # 存储每个原子的RMSD
            
            # 对每个原子计算力偏差（RMSD）
            for atom_idx in range(n_atoms):
                # 获取该原子在M个模型中的力矢量 (n_models, 3)
                atom_forces = forces_array[:, atom_idx, :]
                
                # 计算平均力矢量 (3,)
                mean_force = np.mean(atom_forces, axis=0)
                
                # 计算每个模型力矢量与平均力矢量的差的平方模长
                squared_deviations = []
                for model_idx in range(n_models):
                    # 力矢量差
                    force_diff = atom_forces[model_idx] - mean_force
                    # 计算模长的平方
                    squared_deviation = np.dot(force_diff, force_diff)
                    squared_deviations.append(squared_deviation)
                
                # 计算RMSD：sqrt(mean(squared_deviations))
                rmsd = np.sqrt(np.mean(squared_deviations))
                atom_deviations.append(rmsd)
            
            # 取所有原子的最大力偏差（最大RMSD）
            max_atom_force_devi = max(atom_deviations) if atom_deviations else 0.0
            
            # 记录每个原子的偏差信息（用于调试）
            logger.debug(f"结构 {struct_name} 的原子力偏差统计:")
            logger.debug(f"  最小RMSD: {min(atom_deviations):.6f}")
            logger.debug(f"  平均RMSD: {np.mean(atom_deviations):.6f}")
            logger.debug(f"  最大RMSD: {max_atom_force_devi:.6f}")
            
            # 根据力的阈值标记结构
            f_trust_lo = self.params['model_devi_f_trust_lo']
            f_trust_hi = self.params['model_devi_f_trust_hi']
    
            if max_atom_force_devi < f_trust_lo:
                # Accurate结构
                accurate_data.append({'structure': struct_name, 'F_devi': max_atom_force_devi})
                logger.info(f"结构 {struct_name} 标记为 Accurate, 力偏差(RMSD): {max_atom_force_devi:.6f}")
            elif max_atom_force_devi < f_trust_hi:
                # Candidate结构
                candidate_data.append({'structure': struct_name, 'F_devi': max_atom_force_devi})
                marked_structures[struct_name] = struct_file
                logger.info(f"结构 {struct_name} 标记为 Candidate, 力偏差(RMSD): {max_atom_force_devi:.6f}")
            else:
                # Failed结构
                failed_data.append({'structure': struct_name, 'F_devi': max_atom_force_devi})
                marked_structures[struct_name] = struct_file
                logger.info(f"结构 {struct_name} 标记为 Failed, 力偏差(RMSD): {max_atom_force_devi:.6f}")
    
        # 保存为三个CSV文件
        relaxed_dir = os.path.join(self.current_iter_dir, "relaxed_explore")
        
        # 确保目录存在
        os.makedirs(relaxed_dir, exist_ok=True)
        
        # 保存CSV文件
        if accurate_data:
            pd.DataFrame(accurate_data).to_csv(os.path.join(relaxed_dir, "accurate.csv"), index=False)
        else:
            # 创建空的accurate.csv
            pd.DataFrame(columns=['structure', 'F_devi']).to_csv(os.path.join(relaxed_dir, "accurate.csv"), index=False)
        
        if candidate_data:
            pd.DataFrame(candidate_data).to_csv(os.path.join(relaxed_dir, "candidate.csv"), index=False)
        else:
            pd.DataFrame(columns=['structure', 'F_devi']).to_csv(os.path.join(relaxed_dir, "candidate.csv"), index=False)
        
        if failed_data:
            pd.DataFrame(failed_data).to_csv(os.path.join(relaxed_dir, "failed.csv"), index=False)
        else:
            pd.DataFrame(columns=['structure', 'F_devi']).to_csv(os.path.join(relaxed_dir, "failed.csv"), index=False)
    
        # 创建标记数据框
        marking_data = []
        for item in accurate_data:
            marking_data.append({'structure': item['structure'], 'accurate': '✓', 'candidate': '', 'failed': '', 'F_devi': item['F_devi']})
        for item in candidate_data:
            marking_data.append({'structure': item['structure'], 'accurate': '', 'candidate': '✓', 'failed': '', 'F_devi': item['F_devi']})
        for item in failed_data:
            marking_data.append({'structure': item['structure'], 'accurate': '', 'candidate': '', 'failed': '✓', 'F_devi': item['F_devi']})
        
        marking_df = pd.DataFrame(marking_data)
        
        logger.info(f"标记结果: Accurate={len(accurate_data)}, Candidate={len(candidate_data)}, Failed={len(failed_data)}")
        
        return marked_structures, marking_df

    def calculate_statistics(self):
        """计算统计信息"""
        # 从标记结果中获取统计信息
        marking_csv_path = os.path.join(self.current_iter_dir, "relaxed_explore", "model_devi_marking.csv")
        if not os.path.exists(marking_csv_path):
            return {
                'accurate_ratio': 0.0,
                'candidate_ratio': 0.0,
                'failed_ratio': 0.0
            }
        
        df = pd.read_csv(marking_csv_path)
        
        # 计算各类别的数量
        accurate_count = len(df[df['accurate'].notna() & (df['accurate'] != '')])
        candidate_count = len(df[df['candidate'].notna() & (df['candidate'] != '')])
        failed_count = len(df[df['failed'].notna() & (df['failed'] != '')])
        
        total_count = accurate_count + candidate_count + failed_count
        
        if total_count == 0:
            return {
                'accurate_ratio': 0.0,
                'candidate_ratio': 0.0,
                'failed_ratio': 0.0
            }
        
        return {
            'accurate_ratio': accurate_count / total_count,
            'candidate_ratio': candidate_count / total_count,
            'failed_ratio': failed_count / total_count
        }

    def generate_coord_xyz(self, struct_dir, struct_filename):
        '''生成coord.xyz文件，包含绝对坐标，没有前两行'''
        struct_path = os.path.join(struct_dir, struct_filename)
        
        # 读取结构文件
        try:
            if self.params.get('traj_format') == 'vasp':
                atoms = read(struct_path, format='vasp')
            else:
                atoms = read(struct_path)
        except Exception as e:
            logger.error(f"读取结构文件{struct_path}失败: {e}")
            return
        
        # 构建coord.xyz文件路径
        coord_path = os.path.join(struct_dir, 'coord.xyz')
        
        # 写入坐标，不包含前两行
        with open(coord_path, 'w') as f:
            for atom in atoms:
                symbol = atom.symbol
                pos = atom.position
                # 格式化坐标，保留6位小数，足够精度
                line = f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
                f.write(line)
        
        logger.info(f"已生成coord.xyz文件: {coord_path}")

    def collect_all_dft_data(self):
        """收集所有DFT数据"""
        # 收集当前迭代的DFT结果
        dft_dir = os.path.join(self.current_iter_dir, "DFT")
        dft_results = []
        
        # 假设DFT结果是以某种格式存储的
        result_files = glob.glob(os.path.join(dft_dir, "*.json"))
        for result_file in result_files:
            with open(result_file, 'r') as f:
                dft_data = json.load(f)
                dft_results.append(dft_data)
                
        # 合并之前的数据
        all_dft_data = self.all_dft_data + dft_results
        return all_dft_data
        
    def convert_dft_to_npy(self):
        """将DFT计算结果转换为npy格式"""
        dft_type = self.params.get('DFT_type', 'CP2K')
        calculate_dir = os.path.join(self.current_iter_dir, "DFT", "calculate")
        
        if dft_type == 'CP2K':
            self.convert_cp2k_to_npy(calculate_dir)
        elif dft_type == 'VASP':
            self.convert_vasp_to_npy(calculate_dir)
        else:
            logger.error(f"未知的DFT类型: {dft_type}")

    def convert_cp2k_to_npy(self, calculate_dir):
        """转换CP2K结果为npy格式"""
        from dpdata import MultiSystems
        
        ms = MultiSystems()
        
        # 修改：递归搜索所有子目录中的output文件
        cp2k_output_files = glob.glob(os.path.join(calculate_dir, "**", "output"), recursive=True)
        logger.info(f"找到 {len(cp2k_output_files)} 个CP2K输出文件")
        
        # 处理每个输出文件
        processed_count = 0
        for output_file in cp2k_output_files:
            try:
                # 跳过空文件或过小的文件
                if os.path.getsize(output_file) < 100:  # 100字节以下认为是空文件
                    logger.warning(f"跳过空文件: {output_file}")
                    continue
                    
                ls = dpdata.LabeledSystem(output_file, fmt='cp2k/output')
                if len(ls) > 0:
                    ms.append(ls)
                    processed_count += 1
                    logger.info(f"成功处理CP2K输出文件: {output_file}")
                else:
                    logger.warning(f"CP2K输出文件 {output_file} 中没有有效数据")
            except Exception as e:
                logger.error(f"处理CP2K输出文件 {output_file} 失败: {e}")
        
        logger.info(f"成功处理了 {processed_count}/{len(cp2k_output_files)} 个CP2K输出文件")
        
        # 保存为npy格式
        if len(ms) > 0:
            # 创建临时目录保存数据
            temp_dir = os.path.join(self.current_iter_dir, "temp_npy")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # 保存到临时目录
            ms.to('deepmd/npy', temp_dir)
            
            # 重命名目录
            iter_data_dir = os.path.join(self.current_iter_dir, "database", f"iter_{self.iter_count}_data")
            if os.path.exists(iter_data_dir):
                shutil.rmtree(iter_data_dir)
            
            # 找到自动生成的目录（可能有多个系统）
            auto_dirs = [d for d in glob.glob(os.path.join(temp_dir, "*")) if os.path.isdir(d)]
            
            if auto_dirs:
                if len(auto_dirs) == 1:
                    # 只有一个系统，直接移动
                    shutil.move(auto_dirs[0], iter_data_dir)
                    logger.info(f"转换CP2K结果到npy格式: {auto_dirs[0]} -> {iter_data_dir}")
                else:
                    # 多个系统，合并到一个目录
                    os.makedirs(iter_data_dir)
                    for auto_dir in auto_dirs:
                        system_name = os.path.basename(auto_dir)
                        dest_dir = os.path.join(iter_data_dir, system_name)
                        shutil.move(auto_dir, dest_dir)
                        logger.info(f"移动系统数据: {auto_dir} -> {dest_dir}")
            else:
                logger.warning("在临时目录中没有找到自动生成的数据目录")
            
            # 删除临时目录
            shutil.rmtree(temp_dir)
            
            # 验证转换结果
            if os.path.exists(iter_data_dir):
                npy_files = glob.glob(os.path.join(iter_data_dir, "**", "*.npy"), recursive=True)
                logger.info(f"转换完成，生成 {len(npy_files)} 个npy文件在 {iter_data_dir}")
            else:
                logger.error("npy数据目录创建失败")
        else:
            logger.warning("没有找到有效的CP2K结果可以转换")

    def convert_vasp_to_npy(self, calculate_dir):
        """转换VASP结果为npy格式"""
        from dpdata import MultiSystems
        
        ms = MultiSystems()
        
        # 修改：递归搜索所有子目录中的OUTCAR文件
        vasp_output_files = glob.glob(os.path.join(calculate_dir, "**", "OUTCAR"), recursive=True)
        logger.info(f"找到 {len(vasp_output_files)} 个VASP输出文件")
        
        # 处理每个输出文件
        processed_count = 0
        for output_file in vasp_output_files:
            try:
                # 跳过空文件或过小的文件
                if os.path.getsize(output_file) < 100:  # 100字节以下认为是空文件
                    logger.warning(f"跳过空文件: {output_file}")
                    continue
                    
                ls = dpdata.LabeledSystem(output_file, fmt='vasp/OUTCAR')
                if len(ls) > 0:
                    ms.append(ls)
                    processed_count += 1
                    logger.info(f"成功处理VASP输出文件: {output_file}")
                else:
                    logger.warning(f"VASP输出文件 {output_file} 中没有有效数据")
            except Exception as e:
                logger.error(f"处理VASP输出文件 {output_file} 失败: {e}")
        
        logger.info(f"成功处理了 {processed_count}/{len(vasp_output_files)} 个VASP输出文件")
        
        # 保存为npy格式（逻辑与CP2K相同）
        if len(ms) > 0:
            temp_dir = os.path.join(self.current_iter_dir, "temp_npy")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            ms.to('deepmd/npy', temp_dir)
            
            iter_data_dir = os.path.join(self.current_iter_dir, "database", f"iter_{self.iter_count}_data")
            if os.path.exists(iter_data_dir):
                shutil.rmtree(iter_data_dir)
            
            auto_dirs = [d for d in glob.glob(os.path.join(temp_dir, "*")) if os.path.isdir(d)]
            
            if auto_dirs:
                if len(auto_dirs) == 1:
                    shutil.move(auto_dirs[0], iter_data_dir)
                    logger.info(f"转换VASP结果到npy格式: {auto_dirs[0]} -> {iter_data_dir}")
                else:
                    os.makedirs(iter_data_dir)
                    for auto_dir in auto_dirs:
                        system_name = os.path.basename(auto_dir)
                        dest_dir = os.path.join(iter_data_dir, system_name)
                        shutil.move(auto_dir, dest_dir)
                        logger.info(f"移动系统数据: {auto_dir} -> {dest_dir}")
            else:
                logger.warning("在临时目录中没有找到自动生成的数据目录")
            
            shutil.rmtree(temp_dir)
            
            if os.path.exists(iter_data_dir):
                npy_files = glob.glob(os.path.join(iter_data_dir, "**", "*.npy"), recursive=True)
                logger.info(f"转换完成，生成 {len(npy_files)} 个npy文件在 {iter_data_dir}")
            else:
                logger.error("npy数据目录创建失败")
        else:
            logger.warning("没有找到有效的VASP结果可以转换")
    
    def cp2k_check_fp(self, dft_dir):
        """检查CP2K计算结果是否正常收敛（适配新的目录结构）"""
        calculate_dir = os.path.join(dft_dir, "calculate")
        if not os.path.exists(calculate_dir):
            logger.error(f"DFT计算目录不存在: {calculate_dir}")
            self.dft_convergence_issues = True
            return
        
        folders_without_out = []
        folders_with_convergence_issue = set()
        folders_with_local_log = []
    
        # 修改：直接检查每个结构目录
        for struct_dir in os.listdir(calculate_dir):
            full_struct_dir = os.path.join(calculate_dir, struct_dir)
            
            # 确保是目录
            if not os.path.isdir(full_struct_dir):
                continue
                
            # 检查output文件
            output_file = os.path.join(full_struct_dir, 'output')
            
            if not os.path.exists(output_file):
                folders_without_out.append(struct_dir)
                continue
    
            # 检查输出文件
            try:
                with open(output_file, 'r') as file:
                    content = file.read()
                    # 检查收敛状态
                    has_not_converged = "SCF run NOT converged" in content
                    has_converged = "SCF run converged in" in content
                
                    if has_not_converged or not has_converged:
                        folders_with_convergence_issue.add(struct_dir)
            except Exception as e:
                logger.error(f"读取文件失败: {output_file}, 错误: {e}")
    
            # 检查是否存在 input_localLog_p*.out 文件
            local_log_files = glob.glob(os.path.join(full_struct_dir, "input_localLog_p*.out"))
            if local_log_files:
                folders_with_local_log.append(struct_dir)
    
        # 记录结果
        if folders_without_out:
            logger.warning("以下文件夹不包含输出文件:")
            for folder in folders_without_out:
                logger.warning(folder)
        else:
            logger.info("所有文件夹都包含输出文件。")
    
        if folders_with_convergence_issue:
            logger.warning("以下文件夹的输出文件包含收敛问题:")
            for folder in folders_with_convergence_issue:
                logger.warning(folder)
            self.dft_convergence_issues = True
        else:
            logger.info("没有任何文件夹的输出文件包含收敛问题。")
            self.dft_convergence_issues = False
    
        if folders_with_local_log:
            logger.warning("以下文件夹包含 input_localLog_p*.out 文件:")
            for folder in folders_with_local_log:
                logger.warning(folder)
        else:
            logger.info("所有文件夹都不包含 input_localLog_p*.out 文件。")
    
    def vasp_check_fp(self, dft_dir):
        """检查VASP计算结果是否正常收敛"""
        # 实现VASP检查逻辑
        logger.warning("VASP检查功能尚未实现")
        self.dft_convergence_issues = False

    def check_existing_iter_dirs(self):
        """检查已存在的迭代目录，用于非从零启动"""
        existing_iters = []
        for item in os.listdir('.'):
            if item.startswith('iter_') and os.path.isdir(item):
                try:
                    iter_num = int(item.split('_')[1])
                    existing_iters.append(iter_num)
                except ValueError:
                    continue
        return sorted(existing_iters)

    @staticmethod
    @contextlib.contextmanager
    def change_directory(path):
        """安全地切换目录的上下文管理器"""
        original_cwd = os.getcwd()
        abs_path = os.path.abspath(path)
        
        try:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"目录不存在: {abs_path}")
            os.chdir(abs_path)
            logger.info(f"成功切换到目录: {abs_path}")
            yield

        except Exception as e:
            logger.error(f"切换目录失败: {e}")
            # 记录目录内容以便调试
            parent_dir = os.path.dirname(abs_path)
            if os.path.exists(parent_dir):
                logger.error(f"父目录 '{parent_dir}' 内容: {os.listdir(parent_dir)}")
            raise

        finally:
            try:
                os.chdir(original_cwd)
                logger.info(f"切换回原目录: {original_cwd}")
            except Exception as e:
                logger.error(f"切换回原目录失败: {e}")

    class GlobalCalculatorManager:
        """全局计算器管理类"""
        _instances = {}
        
        def __new__(cls, model_path):
            if model_path not in cls._instances:
                instance = super().__new__(cls)
                instance.model_path = model_path
                instance.calculator = None
                instance.supported_types = None
                instance.initialize()
                cls._instances[model_path] = instance
            return cls._instances[model_path]
        
        def initialize(self):
            """初始化全局计算器"""
            # 清除之前的TensorFlow会话
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            
            # 加载模型获取支持的原子类型
            model = DeepPot(self.model_path)
            self.supported_types = model.get_type_map()
            logger.info(f"模型 {self.model_path} 支持的原子类型: {self.supported_types}")
            
            # 创建计算器
            self.calculator = DP(model=self.model_path, type_map=self.supported_types)

        def get_calculator(self, atoms):
            """为原子结构获取计算器"""
            # 确保使用最新的原子结构
            atoms.calc = self.calculator
            return atoms

        def calculate_forces(self, atoms):
            """计算力"""
            atoms_with_calc = self.get_calculator(atoms)
            return atoms_with_calc.get_forces()    

if __name__ == "__main__":
    flow = ActiveLearningFlow()
    flow.run()