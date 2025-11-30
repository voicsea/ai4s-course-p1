#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将测试集 RGB 图像通过已训练模型推理为 DEM，并保存到本项目的 `output/` 目录下。

需求实现:
1. 完全一致性: 输出文件的格式、尺寸、数据类型（uint8 vs float32）、命名规则与 Label 完全一致
2. 逆变换: 处理训练时的 Normalization，在保存前进行逆变换
3. 批处理: 使用 DataLoader 进行批量推理，但保存时拆分为单独的文件
4. 验证: 保存后对比输出文件和 Label 文件的 Shape 和 Data Type

用法示例:
    python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth
    python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth --batch-size 16

特点:
- 支持 DataLoader 批量推理（默认 batch_size=8）
- 自动进行逆变换还原 DEM 到原始空间 [0, 255]
- 输出与标签文件完全一致（GeoTIFF, uint8, 512x512）
- 推理后打印验证日志，对比输出和标签
"""

import os
import sys
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 确保使用 Hugging Face 镜像（如果模块存在）
try:
    import hf_mirror_config  # sets HF_ENDPOINT on import
except Exception:
    # 非必须，仅用于确保 HF 镜像环境变量在可用时被设置
    pass
from models.registry import ModelFactory
from data_loading import RemoteSensingDataset, get_val_transforms

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception as e:
    rasterio = None
    print(f"警告: rasterio 未安装: {e}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 数据归一化和逆变换配置 ==========
# 这些参数必须与 data_loading.py 中的训练配置完全一致
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])  # RGB 通道均值
IMAGENET_STD = np.array([0.229, 0.224, 0.225])   # RGB 通道标准差
DEM_MIN = 0.0      # DEM 的最小值（训练时归一化的下界）
DEM_MAX = 255.0    # DEM 的最大值（训练时归一化的上界）
TARGET_SIZE = 512  # 训练时的目标尺寸


def denormalize_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """
    逆变换 RGB 图像从 ImageNet 归一化回 [0, 255] 空间
    
    Args:
        image_tensor: 形状 (C, H, W) 或 (B, C, H, W) 的张量，已被归一化
        
    Returns:
        numpy 数组，形状 (H, W, C) 或 (B, H, W, C)，值域 [0, 255]
    """
    if len(image_tensor.shape) == 3:  # (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    # 转换为 numpy (B, C, H, W)
    image_np = image_tensor.cpu().numpy()
    
    # 逆变换
    mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = IMAGENET_STD.reshape(1, 3, 1, 1)
    image_np = (image_np * std + mean) * 255.0
    
    # 裁剪到 [0, 255]
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    
    # 转换格式 (B, C, H, W) -> (B, H, W, C)
    image_np = np.transpose(image_np, (0, 2, 3, 1))
    
    if squeeze_batch:
        image_np = image_np[0]  # (H, W, C)
    
    return image_np


def denormalize_dem(dem_tensor: torch.Tensor) -> np.ndarray:
    """
    逆变换 DEM 图像从 [0, 1] 归一化回原始空间 [DEM_MIN, DEM_MAX]
    
    Args:
        dem_tensor: 形状 (1, H, W) 或 (B, 1, H, W) 的张量，值域 [0, 1]
        
    Returns:
        numpy 数组，形状 (H, W) 或 (B, H, W)，值域 [DEM_MIN, DEM_MAX]，dtype=uint8
    """
    if len(dem_tensor.shape) == 3:  # (1, H, W)
        dem_tensor = dem_tensor.unsqueeze(0)  # (1, 1, H, W)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    # 转换为 numpy (B, 1, H, W)
    dem_np = dem_tensor.cpu().numpy()  # 值域 [0, 1]
    
    # 逆变换：从 [0, 1] 回到 [DEM_MIN, DEM_MAX]
    dem_np = dem_np * (DEM_MAX - DEM_MIN) + DEM_MIN
    dem_np = np.clip(dem_np, DEM_MIN, DEM_MAX)
    
    # 转换为 uint8（以匹配标签文件格式）
    dem_np = dem_np.astype(np.uint8)
    
    # 移除通道维度 (B, 1, H, W) -> (B, H, W)
    dem_np = dem_np.squeeze(1)
    
    if squeeze_batch:
        dem_np = dem_np[0]  # (H, W)
    
    return dem_np


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    加载模型权重，处理各种检查点格式
    
    Args:
        model: 要加载的模型
        ckpt_path: 检查点文件路径
        device: 设备
    """
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"[ERROR] 模型文件不存在: {ckpt_path}")
    
    logger.info(f"加载模型权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 处理不同格式的检查点，支持常见键名
    state = None
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'model', 'net', 'state'):
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            # 整个文件就是 state dict
            state = ckpt
    else:
        state = ckpt
    
    # 尝试直接加载（严格匹配）
    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
            logger.info("✓ 权重加载成功 (strict=True)")
            return
        except Exception as e_strict:
            logger.warning(f"严格加载失败: {e_strict}")

        # 再尝试宽松加载 (strict=False) —— 允许缺失/多余键
        try:
            res = model.load_state_dict(state, strict=False)
            logger.info(f"✓ 权重加载成功 (strict=False) - missing_keys={len(res.missing_keys)} unexpected_keys={len(res.unexpected_keys)}")
            return
        except Exception as e_nonstrict:
            logger.warning(f"宽松加载 (strict=False) 仍失败: {e_nonstrict}")

        # 尝试修复常见的前缀问题：'module.', 'model.', 'net.'
        prefixes = ('module.', 'model.', 'net.')
        def strip_prefixes(state_dict, prefixes):
            new = {}
            for k, v in state_dict.items():
                new_k = k
                for p in prefixes:
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                new[new_k] = v
            return new

        remapped = strip_prefixes(state, prefixes)
        try:
            res = model.load_state_dict(remapped, strict=False)
            logger.info(f"✓ 权重加载成功 (移除前缀后, strict=False) - missing_keys={len(res.missing_keys)} unexpected_keys={len(res.unexpected_keys)}")
            return
        except Exception as e_remap:
            logger.error(f"尝试移除前缀后仍无法加载: {e_remap}")

        # 如果所有策略都失败，抛出最后的失败信息
        raise RuntimeError(f"无法加载权重: 多种加载尝试均失败（最后一次异常: {e_nonstrict}）")
    else:
        raise RuntimeError("无法解析检查点文件为 state dict")


def save_dem_tiff(save_path: Path, array: np.ndarray, label_path: Path = None) -> None:
    """
    保存 DEM 为 GeoTIFF 格式，与标签文件格式一致
    
    Args:
        save_path: 保存路径
        array: DEM 数据 (H, W)，dtype=uint8，值域 [0, 255]
        label_path: （可选）对应的标签文件路径，用于对比验证
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if rasterio is None:
        # fallback: 保存为 numpy .npy
        np.save(str(save_path.with_suffix('.npy')), array)
        logger.warning(f"[WARN] rasterio 未安装，已保存为 NPY: {save_path.with_suffix('.npy')}")
        return
    
    # 写入单波段 uint8 GeoTIFF（与标签格式完全一致）
    h, w = array.shape
    transform = from_origin(0, 0, 1, 1)  # 默认变换
    
    try:
        with rasterio.open(
            str(save_path),
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=1,  # 单波段
            dtype=rasterio.uint8,
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(array, 1)
        
        logger.info(f"✓ 已保存: {save_path} (shape={array.shape}, dtype={array.dtype})")
        
        # 验证对比
        if label_path and label_path.exists():
            verify_output(save_path, label_path)
    
    except Exception as e:
        logger.error(f"[ERROR] 保存 TIF 失败: {e}")
        raise


def verify_output(output_path: Path, label_path: Path) -> None:
    """
    对比输出文件和标签文件的形状、数据类型，打印验证日志
    
    Args:
        output_path: 输出文件路径
        label_path: 标签文件路径
    """
    if not rasterio:
        return
    
    try:
        with rasterio.open(str(output_path)) as src_out:
            out_shape = (src_out.height, src_out.width)
            out_dtype = src_out.dtypes[0]
        
        with rasterio.open(str(label_path)) as src_label:
            label_shape = (src_label.height, src_label.width)
            label_dtype = src_label.dtypes[0]
        
        match = "✓" if (out_shape == label_shape and out_dtype == label_dtype) else "✗"
        logger.info(f"{match} 验证 | Output: {out_shape} {out_dtype} | Label: {label_shape} {label_dtype}")
    
    except Exception as e:
        logger.warning(f"[WARN] 验证失败: {e}")


def run_batch_inference(
    model_path: str,
    model_name: str = 'resnet_unet',
    test_data_root: str = '../ImageToDEM/singleRGBNormalizationTest',
    output_dir: str = './output',
    batch_size: int = 8,
    device: str = None,
    num_workers: int = 4
) -> None:
    """
    批量推理推荐的入口函数
    
    Args:
        model_path: 模型权重路径
        model_name: 模型名称
        test_data_root: 测试数据根目录（包含 RGB/ 和 DEM/）
        output_dir: 输出目录（实际输出将存储在 {output_dir}/{model_name}/ 下）
        batch_size: 批大小
        device: 设备 ('cuda' 或 'cpu')
        num_workers: DataLoader 工作线程数
    """
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"设备: {device}")
    logger.info(f"批大小: {batch_size}")
    
    # 创建模型
    logger.info(f"创建模型: {model_name}")
    # For vit_unet, disable pretrained weights to use only trained weights
    if model_name == 'vit_unet':
        model = ModelFactory.create(model_name, out_channels=1, pretrained=False)
    elif model_name == 'adabins':
        # AdaBins模型：尝试自动检测配置，优先使用与训练时一致的参数
        try:
            # 首先尝试从权重文件推断配置
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint if isinstance(checkpoint, dict) else {}
            
            # 从state_dict推断n_bins
            n_bins = 128  # 默认值改为128，与训练配置一致
            for key in state_dict.keys():
                if 'bin_center_head' in key and 'bias' in key:
                    n_bins = state_dict[key].shape[0]
                    logger.info(f"从权重文件检测到n_bins={n_bins}")
                    break
                elif 'logits_head' in key and 'weight' in key and len(state_dict[key].shape) == 4:
                    # logits_head最后的conv层输出通道数就是n_bins (shape: [n_bins, 128, 1, 1])
                    n_bins = state_dict[key].shape[0]
                    logger.info(f"从权重文件检测到n_bins={n_bins}")
                    break
            
            logger.info(f"创建AdaBins模型: backbone=resnet50, n_bins={n_bins}")
            model = ModelFactory.create(model_name, out_channels=1, backbone='resnet50', n_bins=n_bins)
            logger.info(f"模型创建成功: ResNet50 backbone, n_bins={n_bins}")
        except Exception as e:
            logger.warning(f"自动检测配置失败，使用默认配置: {e}")
            model = ModelFactory.create(model_name, out_channels=1, backbone='resnet50', n_bins=128)
    else:
        model = ModelFactory.create(model_name, out_channels=1)
    model = model.to(device)
    
    # 加载权重
    load_checkpoint(model, model_path, device)
    model.eval()
    
    # 创建数据集和数据加载器
    test_data_root = Path(test_data_root).resolve()
    if not test_data_root.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {test_data_root}")
    
    logger.info(f"创建测试数据集: {test_data_root}")
    test_dataset = RemoteSensingDataset(
        root_dir=str(test_data_root),
        transform=get_val_transforms(),
        mode='test'
    )
    
    if len(test_dataset) == 0:
        logger.error("[ERROR] 测试数据集为空")
        return
    
    logger.info(f"✓ 数据集加载: {len(test_dataset)} 个样本")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"✓ DataLoader 创建: {len(test_loader)} 个批次")
    
    # 创建输出目录（按模型名称分类）
    output_dir = Path(output_dir)
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {model_output_dir}")
    
    # 获取标签目录
    label_dir = test_data_root / 'DEM'
    
    # 批量推理
    logger.info("=" * 70)
    logger.info("开始批量推理...")
    logger.info("=" * 70)
    
    total_processed = 0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            batch_size_actual = images.shape[0]
            
            # 前向推理
            outputs = model(images)  # (B, 1, H, W)
            
            # 逆变换 DEM
            outputs_denorm = denormalize_dem(outputs)  # (B, H, W) uint8
            
            # 提取文件名（从原始数据集获取）
            filenames = []
            for i in range(batch_size_actual):
                idx = batch_idx * batch_size + i
                if idx < len(test_dataset.filenames):
                    rgb_fname = test_dataset.filenames[idx]
                    dem_fname = rgb_fname.replace('.png', '.tif')
                    filenames.append(dem_fname)
            
            # 逐个保存
            for i, dem_array in enumerate(outputs_denorm):
                if i >= len(filenames):
                    break
                
                dem_fname = filenames[i]
                output_path = model_output_dir / dem_fname
                label_path = label_dir / dem_fname
                
                save_dem_tiff(output_path, dem_array, label_path)
                total_processed += 1
            
            logger.info(f"[批次 {batch_idx + 1}/{len(test_loader)}] 已处理 {total_processed}/{len(test_dataset)} 个样本")
    
    logger.info("=" * 70)
    logger.info(f"✓ 批量推理完成！共处理 {total_processed} 个样本")
    logger.info(f"✓ 输出保存到: {model_output_dir}")
    logger.info("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description='将测试集 RGB 转换为 DEM 并保存到输出目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth
  python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth --batch-size 16
  python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth --device cuda --num-workers 8
        """
    )
    parser.add_argument('--model-path', type=str, required=True, 
                        help='模型权重路径 (e.g., ./checkpoints/resnet_unet/best_model.pth)')
    parser.add_argument('--model-name', type=str, default='resnet_unet', 
                        help='模型名称 (默认: resnet_unet)')
    parser.add_argument('--test-data-root', type=str, 
                        default='../ImageToDEM/singleRGBNormalizationTest',
                        help='测试数据根目录 (默认: ../ImageToDEM/singleRGBNormalizationTest)')
    parser.add_argument('--output-dir', type=str, default='./output', 
                        help='输出目录 (默认: ./output)')
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='批处理大小 (默认: 8)')
    parser.add_argument('--device', type=str, default=None, 
                        help='设备 (cuda 或 cpu，默认自动选择)')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='DataLoader 工作线程数 (默认: 4)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_batch_inference(
        model_path=args.model_path,
        model_name=args.model_name,
        test_data_root=args.test_data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers
    )
