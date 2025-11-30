# DEM预测框架：从RGB图像到数字高程模型的深度学习解决方案

## 1. 项目概述

本项目是一个用于从RGB遥感图像预测数字高程模型(DEM)的深度学习框架。该框架支持多种模型架构，提供完整的训练、评估和推理功能，旨在为遥感图像处理提供高效、准确的DEM生成解决方案。

### 核心功能

- **RGB到DEM转换**：将RGB遥感图像转换为高精度DEM
- **多模型架构支持**：包括ResNet-UNet、ViT-UNet、LeRes、DPT和扩散模型等
- **完整训练流程**：数据加载、模型训练、早停机制、检查点保存
- **全面评估功能**：MAE、MSE、RMSE、R²、SSIM、PSNR等多种指标
- **批量推理与GeoTIFF输出**：支持批量处理并输出标准格式的DEM文件
- **高级统计分析**：包含显著性检验、最差案例分析等功能

## 2. 系统架构

### 2.1 支持的模型架构

| 模型 | 特点 | 参数量 | 推理速度 | 适用场景 |
|------|------|--------|---------|----------|
| **ResNet UNet** | 均衡型，ResNet50骨干 | ~26M | 250ms | 通用生产环境 |
| **ResNet UNet Large** | 高精度，ResNet101骨干 | ~44M | 350ms | 高精度需求 |
| **ViT UNet** | 全局建模，Transformer架构 | ~86M | 800ms | 充足数据与资源 |
| **LeRes** | 轻量型，残差密集连接 | ~10M | 150ms | 实时应用，边缘设备 |
| **DPT** | 最优设计，专用Transformer | ~130M | 900ms | 科研，精度第一 |
| **Diffusion DEM** | 生成模型，多样性 | 100M+ | 30-60s | 不确定性估计，研究 |

### 2.2 核心模块

1. **模型注册表** (`models/registry.py`): 统一管理和创建不同模型
2. **数据加载** (`data_loading.py`): 处理RGB和DEM数据的加载与预处理
3. **训练模块** (`train.py`): 实现模型训练、早停、检查点保存等功能
4. **评估系统** (`test.py`, `evaluation/metrics.py`): 计算多种评估指标并生成报告
5. **高级统计分析** (`advanced_statistics.py`): 提供显著性检验和分析功能
6. **批量推理** (`run_test_to_output.py`): 支持批量处理并输出GeoTIFF格式

## 3. 快速开始

### 3.1 系统要求

- Python 3.8+
- PyTorch 1.10+
- CUDA支持（推荐用于训练）
- 其他依赖见`../requirements.txt`

### 3.2 安装与环境设置

```bash
# 克隆项目
cd d:\rs_image\Unet

# 安装依赖
pip install -r ../requirements.txt
```

### 3.3 验证安装

```bash
# 运行集成测试
python integration_test.py
```

### 3.4 快速使用示例

#### 模型评估

```bash
# 评估ResNet UNet（推荐）
python evaluate.py --model-name resnet_unet

# 评估ViT UNet
python evaluate.py --model-name vit_unet
```

#### 批量推理与输出

```bash
# 使用ResNet UNet进行批量推理
python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth

# 自定义输出目录
python run_test_to_output.py --model-path ./checkpoints/resnet_unet/best_model.pth --output-dir ./custom_output
```

## 4. 模型训练

### 4.1 配置文件

训练配置可以通过JSON或YAML文件指定，主要参数包括：

```python
{
    "model_name": "resnet_unet",      # 模型名称
    "batch_size": 16,                  # 批处理大小
    "learning_rate": 1e-4,             # 学习率
    "num_epochs": 50,                  # 训练轮数
    "pixel_loss": "l1",               # 像素级损失函数
    "gradient_loss": "gradient",      # 梯度损失函数
    "pixel_weight": 1.0,               # 像素损失权重
    "gradient_weight": 0.5             # 梯度损失权重
}
```

### 4.2 损失函数选择指南

| 模型 | 推荐损失函数 | 权重配置 |
|------|------------|---------|
| ResNet UNet | Charbonnier + 一阶梯度 | 1.0 + 0.5 |
| ViT UNet/LeRes/DPT | Log-Cosh + 二阶梯度 + 自适应 | 1.0 + 0.6 + 0.2 |
| Diffusion DEM | DDPMNoiseLoss | - |

### 4.3 训练命令

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置文件
python train.py --config ./configs/resnet50_mid.json
```

## 5. 评估与分析

### 5.1 评估指标

框架支持以下评估指标：

- **MAE** (平均绝对误差): 测量预测值与真实值的平均绝对差异
- **MSE/RMSE** (均方误差/均方根误差): 更强调大误差
- **R²** (决定系数): 衡量模型解释目标变量变异的能力
- **SSIM** (结构相似性指数): 评估感知质量
- **PSNR** (峰值信噪比): 衡量重建质量

### 5.2 执行SOTA评估

```bash
# 运行多模型SOTA评估
python test.py sota

# 自定义输出目录
python test.py sota --output-dir ./my_sota_results
```

### 5.3 高级统计分析

高级统计分析模块提供：

- 独立样本t检验：比较不同模型性能差异的显著性
- Bonferroni校正：多重比较时的显著性校正
- 最差案例分析：识别模型性能最差的样本
- 结果表格生成：生成带显著性标记的LaTeX表格

## 6. 故障排除

### 6.1 常见问题

#### 模型加载错误
```bash
# 确保使用正确的pretrained参数
# 对于ViT UNet，使用pretrained=False
model = ModelFactory.create("vit_unet", out_channels=1, pretrained=False)
```

#### CUDA内存不足
```bash
# 减小批量大小
batch_size = 4  # 原来8

# 使用梯度累积
for i in range(num_accumulation_steps):
    loss = model(x) / num_accumulation_steps
    loss.backward()
optimizer.step()

# 混合精度训练
from torch.cuda.amp import autocast
with autocast():
    loss = model(x)
```

#### HuggingFace镜像配置
```python
# 使用国内镜像源
import hf_mirror_config
# 自动设置为最稳定的镜像源
hf_mirror_config.set_hf_mirror('hf-mirror')
```

## 7. 模型选择指南

### 7.1 实时应用（推理<500ms）
- **首选**: LeRes（最快，150ms）
- **备选**: ResNet UNet（250ms）

### 7.2 精度优先（推理<2s）
- **首选**: ViT UNet（精度与速度均衡，800ms）
- **备选**: DPT（最高精度，900ms）

### 7.3 计算资源有限
- **首选**: LeRes或ResNet UNet
- **备选**: 减小输入分辨率

### 7.4 生成多样性需求
- **首选**: Diffusion DEM（支持多样本生成）

## 8. 部署与优化

### 8.1 模型优化建议

1. **模型量化**: 使用INT8量化减少模型大小和计算量
2. **ONNX导出**: 导出为ONNX格式以提高推理速度
3. **模型剪枝**: 移除不重要的权重以减小模型大小
4. **知识蒸馏**: 从大模型向小模型转移知识

### 8.2 部署流程

```bash
# ONNX导出示例
python -c "
import torch
from models.registry import ModelFactory

model = ModelFactory.create('resnet_unet', out_channels=1)
model.load_state_dict(torch.load('./checkpoints/resnet_unet/best_model.pth'))
model.eval()

# 导出为ONNX
torch.onnx.export(model, torch.randn(1, 3, 512, 512), 'resnet_unet.onnx', 
                 input_names=['input'], output_names=['output'])
"
```

## 9. 未来改进方向

1. 增加更多模型架构支持
2. 实现模型集成（Ensemble）以提高性能
3. 优化推理速度（ONNX/TensorRT）
4. 增加更多评估指标和可视化工具
5. 支持更多数据格式和预处理选项

---

*文档生成时间: 2025年11月*  
*项目维护者: 遥感图像研究团队*