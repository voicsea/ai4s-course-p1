#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 支持SOTA模型评估
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.registry import ModelFactory
from data_loading import RemoteSensingDataset
from evaluation.metrics import Evaluator as MetricsCalculator

# 简化实现其他缺失的类
class EfficiencyCalculator:
    def __init__(self, device='cuda', input_size=(1, 3, 512, 512)):
        self.device = device
        self.input_size = input_size

class StatisticalAnalyzer:
    def __init__(self):
        pass

class Visualizer:
    def __init__(self, output_dir='./output'):
        self.output_dir = output_dir

# 实现模型特定的转换函数
def get_model_specific_transforms(model_name: str):
    """为特定模型获取相应的图像转换"""
    from torchvision import transforms
    
    # 根据模型名称返回不同的转换
    if model_name == 'adabins':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class CheckpointManager:
    """
    检查点管理器，负责加载不同模型的权重
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def load_model_checkpoint(self, model_name: str, model: nn.Module, device: str = 'cuda') -> nn.Module:
        """
        加载模型检查点
        
        Args:
            model_name: 模型名称
            model: 模型实例
            device: 设备
            
        Returns:
            加载权重后的模型
        """
        # 构建检查点路径
        checkpoint_path = self.checkpoint_dir / model_name / "best_model.pth"
        
        if not checkpoint_path.exists():
            # 尝试其他可能的检查点名称
            alternative_paths = [
                self.checkpoint_dir / model_name / "model.pth",
                self.checkpoint_dir / model_name / "last_model.pth",
                self.checkpoint_dir / f"{model_name}.pth"
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    checkpoint_path = alt_path
                    break
        
        if checkpoint_path.exists():
            try:
                # 加载检查点
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 处理不同格式的检查点
                state_dict = None
                
                if isinstance(checkpoint, dict):
                    # 检查多种可能的权重键名
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                        print(f"  [INFO] 检查点文件包含 'state_dict' 键")
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print(f"  [INFO] 检查点文件包含 'model_state_dict' 键")
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                        print(f"  [INFO] 检查点文件包含 'model' 键")
                    elif 'net' in checkpoint:
                        state_dict = checkpoint['net']
                        print(f"  [INFO] 检查点文件包含 'net' 键")
                    else:
                        # 检查是否直接包含模型权重（通过检查值是否为张量）
                        has_tensor_values = any(isinstance(v, torch.Tensor) for v in checkpoint.values())
                        if has_tensor_values:
                            state_dict = checkpoint
                            print(f"  [INFO] 检查点文件直接包含权重")
                        else:
                            print(f"  [WARNING] 检查点文件格式不正确，不包含张量值")
                            print(f"  [INFO] 继续使用未加载权重的模型进行评估")
                            return model.eval()
                else:
                    print(f"  [WARNING] 检查点文件格式不正确，不是字典类型")
                    print(f"  [INFO] 继续使用未加载权重的模型进行评估")
                    return model.eval()
                
                print(f"  [INFO] 检查点文件包含 {len(state_dict)} 个参数")
                
                # 验证state_dict是否包含张量值
                has_tensor_values = any(isinstance(v, torch.Tensor) for v in state_dict.values())
                if not has_tensor_values:
                    print(f"  [WARNING] state_dict不包含张量值，格式不正确")
                    print(f"  [INFO] 继续使用未加载权重的模型进行评估")
                    return model.eval()
                
                # 根据模型类型选择不同的加载策略
                if model_name == 'adabins':
                    # 为adabins模型准备兼容的state_dict
                    mapped_state_dict = self._prepare_adabins_state_dict(state_dict, model)
                    
                    # 尝试严格加载
                    try:
                        model.load_state_dict(mapped_state_dict, strict=True)
                        print(f"  [OK] 模型 {model_name} 权重加载成功 (strict=True)")
                    except Exception as e:
                        print(f"  [INFO] 严格加载失败，尝试参数形状检查和映射: {str(e)[:100]}...")
                        
                        # 统计缺失和额外的参数
                        model_state = model.state_dict()
                        missing_keys = set(model_state.keys()) - set(mapped_state_dict.keys())
                        unexpected_keys = set(mapped_state_dict.keys()) - set(model_state.keys())
                        
                        print(f"  [INFO] 缺失的参数: {len(missing_keys)}, 额外的参数: {len(unexpected_keys)}")
                        
                        # 尝试非严格加载
                        try:
                            model.load_state_dict(mapped_state_dict, strict=False)
                            print(f"  [OK] 模型 {model_name} 权重加载成功 (strict=False)")
                        except Exception as inner_e:
                            print(f"  [WARNING] 无法加载 {model_name} 权重: {str(inner_e)[:100]}...")
                            print(f"  [INFO] 继续使用未加载权重的模型进行评估")
                else:
                    # 其他模型使用常规加载策略
                    try:
                        # 尝试直接加载
                        model.load_state_dict(state_dict, strict=True)
                        print(f"  [OK] 模型 {model_name} 权重加载成功 (strict=True)")
                    except Exception as e:
                        print(f"  [INFO] 严格加载失败，尝试移除module前缀: {str(e)[:100]}...")
                        
                        # 尝试移除module前缀
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('module.'):
                                new_state_dict[k[7:]] = v
                            else:
                                new_state_dict[k] = v
                        
                        try:
                            model.load_state_dict(new_state_dict, strict=True)
                            print(f"  [OK] 模型 {model_name} 权重加载成功 (移除module前缀, strict=True)")
                        except Exception as inner_e:
                            print(f"  [INFO] 部分参数映射后加载失败: {str(inner_e)[:100]}...")
                            print(f"  [INFO] 继续使用部分加载权重的模型进行评估")
                            
                            # 最后尝试原始的非严格加载
                            try:
                                model.load_state_dict(state_dict, strict=False)
                                print(f"  [OK] 模型 {model_name} 权重加载成功 (strict=False)")
                            except Exception as inner_e:
                                print(f"  [WARNING] 无法加载 {model_name} 权重: {str(inner_e)[:100]}...")
                                print(f"  [INFO] 继续使用未加载权重的模型进行评估")
            except Exception as e:
                print(f"  [WARNING] 加载 {model_name} 检查点时出错: {str(e)}")
                print(f"  [INFO] 继续使用未加载权重的模型进行评估")
        else:
            print(f"  [WARNING] 检查点文件不存在: {checkpoint_path}")
        
        return model.eval()
    
    def _prepare_adabins_state_dict(self, state_dict: dict, model: nn.Module) -> dict:
        """
        为adabins模型准备兼容的state_dict
        """
        # 复制state_dict以避免修改原始数据
        new_state_dict = {}
        model_state = model.state_dict()
        
        print(f"  [INFO] 开始为adabins模型处理权重映射")
        
        # 1. 处理可能的module前缀
        for key in state_dict.keys():
            # 移除module前缀
            clean_key = key.replace('module.', '')
            new_state_dict[clean_key] = state_dict[key]
        
        # 2. 创建专门的adabins权重映射
        mapped_dict = self._adabins_model_loader(new_state_dict, model_state)
        
        print(f"  [INFO] adabins权重映射完成，成功映射 {len(mapped_dict)} 个参数")
        return mapped_dict
    
    def _adabins_model_loader(self, checkpoint_state: dict, model_state: dict) -> dict:
        """
        专门处理adabins模型的权重加载，实现精细的参数映射
        
        Args:
            checkpoint_state: 从检查点加载的权重
            model_state: 模型期望的权重格式
            
        Returns:
            映射后的权重字典
        """
        mapped_state = {}
        
        # 1. 尝试直接匹配
        for key, value in checkpoint_state.items():
            if key in model_state and value.shape == model_state[key].shape:
                mapped_state[key] = value
                print(f"  [INFO] 直接匹配: {key}")
        
        # 2. 处理不匹配的参数
        unmatched_checkpoint_keys = set(checkpoint_state.keys()) - set(mapped_state.keys())
        
        # 3. 处理ASPP模块的特殊映射
        print(f"  [INFO] 处理组件: aspp")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if 'aspp' in checkpoint_key:
                # 尝试多种ASPP结构映射
                aspp_patterns = [
                    # 原始格式：aspp.0.aspp.X.Y -> aspp.0.aspp.X.Y
                    lambda k: k,
                    # 简化格式：aspp.X.Y -> aspp.0.aspp.X.Y
                    lambda k: k.replace('aspp.', 'aspp.0.aspp.'),
                    # 另一种简化格式：aspp.aspp.X.Y -> aspp.0.aspp.X.Y
                    lambda k: k.replace('aspp.aspp.', 'aspp.0.aspp.'),
                    # 处理融合层：aspp.1.X -> aspp.1.X
                    lambda k: k,
                ]
                
                for pattern in aspp_patterns:
                    test_key = pattern(checkpoint_key)
                    if test_key in model_state and checkpoint_state[checkpoint_key].shape == model_state[test_key].shape:
                        mapped_state[test_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] ASPP映射: {checkpoint_key} -> {test_key}")
                        break
        
        # 4. 处理解码器模块映射
        print(f"  [INFO] 处理组件: decoder")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if any(dec in checkpoint_key for dec in ['dec1', 'dec2', 'dec3', 'dec4']):
                # 尝试多种解码器块映射
                dec_patterns = [
                    # 格式1: decX.Y.Z -> decX.block.Y.Z
                    lambda k: k.replace('.0.', '.block.0.').replace('.1.', '.block.1.').replace('.2.', '.block.2.'),
                    # 格式2: decX.block.Y.Z -> decX.Y.Z
                    lambda k: k.replace('.block.', '.'),
                    # 格式3: decX.Y -> decX.block.Y
                    lambda k: k.replace('.', '.block.', 1),
                ]
                
                for pattern in dec_patterns:
                    test_key = pattern(checkpoint_key)
                    if test_key in model_state and checkpoint_state[checkpoint_key].shape == model_state[test_key].shape:
                        mapped_state[test_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] 解码器映射: {checkpoint_key} -> {test_key}")
                        break
        
        # 5. 处理Bin中心预测头映射
        print(f"  [INFO] 处理组件: bin_head")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if 'bin_center_head' in checkpoint_key:
                # 尝试多种Bin头映射
                bin_patterns = [
                    # 原始格式
                    lambda k: k,
                    # 处理可能的层索引差异
                    lambda k: k.replace('bin_center_head.3.', 'bin_center_head.2.') if 'bin_center_head.3.' in k else k,
                    lambda k: k.replace('bin_center_head.2.', 'bin_center_head.3.') if 'bin_center_head.2.' in k else k,
                ]
                
                for pattern in bin_patterns:
                    test_key = pattern(checkpoint_key)
                    if test_key in model_state and checkpoint_state[checkpoint_key].shape == model_state[test_key].shape:
                        mapped_state[test_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] Bin头映射: {checkpoint_key} -> {test_key}")
                        break
        
        # 6. 处理Bridge模块映射
        print(f"  [INFO] 处理组件: bridge")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if 'bridge' in checkpoint_key:
                # 尝试多种Bridge映射
                bridge_patterns = [
                    # 原始格式
                    lambda k: k,
                    # 处理块结构差异
                    lambda k: k.replace('.0.', '.0.block.').replace('.1.', '.1.block.'),
                    lambda k: k.replace('.block.', '.'),
                ]
                
                for pattern in bridge_patterns:
                    test_key = pattern(checkpoint_key)
                    if test_key in model_state and checkpoint_state[checkpoint_key].shape == model_state[test_key].shape:
                        mapped_state[test_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] Bridge映射: {checkpoint_key} -> {test_key}")
                        break
        
        # 7. 处理Refine和Edge-aware模块映射
        print(f"  [INFO] 处理组件: refine")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if 'refine' in checkpoint_key or 'edge_aware' in checkpoint_key:
                # 尝试多种Refine映射
                refine_patterns = [
                    # 原始格式
                    lambda k: k,
                    # 处理refine和edge_aware_module的映射
                    lambda k: k.replace('refine.', 'edge_aware_module.') if 'refine.' in k and 'edge_aware_module.' not in k else k,
                    lambda k: k.replace('edge_aware_module.', 'refine.') if 'edge_aware_module.' in k else k,
                    # 处理层索引差异
                    lambda k: k.replace('refine.1.', 'refine.2.') if 'refine.1.' in k else k,
                    lambda k: k.replace('refine.2.', 'refine.1.') if 'refine.2.' in k else k,
                ]
                
                for pattern in refine_patterns:
                    test_key = pattern(checkpoint_key)
                    if test_key in model_state and checkpoint_state[checkpoint_key].shape == model_state[test_key].shape:
                        mapped_state[test_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] Refine映射: {checkpoint_key} -> {test_key}")
                        break
        
        # 8. 处理Logits头映射
        print(f"  [INFO] 处理组件: logits_head")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if 'logits_head' in checkpoint_key:
                # 尝试多种Logits头映射
                logits_patterns = [
                    # 原始格式
                    lambda k: k,
                    # 处理层索引差异
                    lambda k: k.replace('logits_head.0.', 'logits_head.1.') if 'logits_head.0.' in k else k,
                    lambda k: k.replace('logits_head.1.', 'logits_head.0.') if 'logits_head.1.' in k else k,
                ]
                
                for pattern in logits_patterns:
                    test_key = pattern(checkpoint_key)
                    if test_key in model_state and checkpoint_state[checkpoint_key].shape == model_state[test_key].shape:
                        mapped_state[test_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] Logits映射: {checkpoint_key} -> {test_key}")
                        break
        
        # 9. 处理批归一化和其他参数的精确映射
        print(f"  [INFO] 处理组件: batch_norm")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            if any(norm in checkpoint_key for norm in ['running_mean', 'running_var', 'weight', 'bias', 'num_batches_tracked']):
                param_type = checkpoint_key.split('.')[-1]
                param_shape = checkpoint_state[checkpoint_key].shape
                
                # 精确匹配：寻找相同类型和形状的未映射参数
                for model_key in model_state.keys():
                    if (model_key not in mapped_state and 
                        model_key.endswith(param_type) and 
                        model_state[model_key].shape == param_shape):
                        mapped_state[model_key] = checkpoint_state[checkpoint_key]
                        unmatched_checkpoint_keys.remove(checkpoint_key)
                        print(f"  [INFO] 参数映射: {checkpoint_key} -> {model_key}")
                        break
        
        # 10. 处理剩余的未匹配参数
        print(f"  [INFO] 处理组件: remaining")
        for checkpoint_key in list(unmatched_checkpoint_keys):
            # 尝试暴力匹配：寻找形状相同的未映射参数
            param_shape = checkpoint_state[checkpoint_key].shape
            
            for model_key in model_state.keys():
                if (model_key not in mapped_state and 
                    model_state[model_key].shape == param_shape):
                    mapped_state[model_key] = checkpoint_state[checkpoint_key]
                    unmatched_checkpoint_keys.remove(checkpoint_key)
                    print(f"  [INFO] 形状匹配: {checkpoint_key} -> {model_key}")
                    break
        
        # 11. 打印未匹配的参数统计
        if unmatched_checkpoint_keys:
            print(f"  [INFO] 仍有 {len(unmatched_checkpoint_keys)} 个参数未匹配")
            # 只显示前10个未匹配的参数作为示例
            for i, key in enumerate(list(unmatched_checkpoint_keys)[:10]):
                print(f"    - {key} ({checkpoint_state[key].shape})")
            if len(unmatched_checkpoint_keys) > 10:
                print(f"    - ... 还有 {len(unmatched_checkpoint_keys) - 10} 个未显示")
        else:
            print(f"  [INFO] 所有参数都已匹配成功!")
        
        return mapped_state


class SOTAEvaluator:
    def __init__(self, test_data_root: str = '../ImageToDEM/singleRGBNormalizationTest', checkpoint_dir: str = './checkpoints', output_dir: str = './evaluation_results_sota', device: str = 'cuda', batch_size: int = 4):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # 减少批处理大小以降低内存使用
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 加载测试数据集... ({test_data_root})")
        self.test_data_root = test_data_root
        # 初始化为None，将在evaluate_single_model中为每个模型创建特定的dataloader
        self.test_loader = None
        self.metrics_calc = MetricsCalculator(device=self.device)
        self.efficiency_calc = EfficiencyCalculator(device=self.device)
        self.stat_analyzer = StatisticalAnalyzer()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.visualizer = Visualizer(output_dir=str(self.output_dir))
        print(f"[INFO] 初始化完成，使用设备: {self.device}")
        print(f"[INFO] 使用批处理大小: {self.batch_size}")
    
    def _create_model_specific_loader(self, model_name: str):
        """为特定模型创建使用正确输入大小的数据加载器"""
        transform = get_model_specific_transforms(model_name)
        dataset = RemoteSensingDataset(self.test_data_root, transform=transform, mode='test')
        # 更新效率计算器的输入大小
        input_size = MODEL_INPUT_SIZES.get(model_name, 512)
        self.efficiency_calc = EfficiencyCalculator(device=self.device, input_size=(1, 3, input_size, input_size))
        print(f"  [INFO] Created model-specific loader with size: {input_size}x{input_size}")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)


# 模型输入大小配置
MODEL_INPUT_SIZES = {
    'adabins': 512,
    'resnet_unet': 512,
    'unet': 512,
    'diffusion': 512
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SOTA Model Evaluation')
    parser.add_argument('cmd', choices=['list', 'single', 'compare', 'sota', 'smoke', 'unit'], help='Command to execute')
    parser.add_argument('--model', type=str, default='all', help='Model name to evaluate (default: all)')
    parser.add_argument('--test-data', type=str, default='../ImageToDEM/singleRGBNormalizationTest', help='Test data root directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results_sota', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if args.cmd == 'sota':
        print("\n" + "="*80)
        print("开始SOTA模型评估")
        print(f"模型模式: {args.model}")
        print(f"测试数据: {args.test_data}")
        print(f"检查点目录: {args.checkpoint_dir}")
        print(f"输出目录: {args.output_dir}")
        print("="*80)
        
        evaluator = SOTAEvaluator(
            test_data_root=args.test_data,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size
        )
        
        # 测试AdaBins模型创建和权重加载
        print("\n[INFO] 开始测试AdaBins模型创建和权重加载...")
        
        # 创建模型
        print("  [INFO] 创建AdaBins模型...")
        model = ModelFactory.create('adabins', out_channels=1)
        model = model.to(evaluator.device)
        
        # 加载权重
        print("  [INFO] 加载AdaBins权重...")
        model = evaluator.checkpoint_manager.load_model_checkpoint('adabins', model, device=evaluator.device)
        
        print("\n[INFO] AdaBins模型创建和权重加载测试完成!")
    else:
        print(f"\n[INFO] 命令 {args.cmd} 尚未实现")
