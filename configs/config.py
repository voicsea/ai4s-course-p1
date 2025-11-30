"""
配置管理模块
"""
from dataclasses import dataclass
from typing import Dict, Any
import json
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    """训练配置 - 支持AdaBins原论文的所有参数"""
    # 模型配置
    model_name: str = "resnet_unet"
    model_kwargs: Dict[str, Any] = None
    
    # 训练超参数
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 优化器配置
    optimizer_name: str = "adamw"
    scheduler_name: str = "reduce_on_plateau"
    scheduler_kwargs: Dict[str, Any] = None
    
    # 损失函数配置 - 支持AdaBins原论文的损失函数
    pixel_loss: str = "l1"
    gradient_loss: str = "gradient"
    scale_invariant_loss: bool = False  # AdaBins论文中的尺度不变损失
    bin_regularization: bool = False    # AdaBins论文中的bins正则化
    pixel_weight: float = 1.0
    gradient_weight: float = 0.5
    scale_invariant_weight: float = 0.5  # 尺度不变损失权重
    
    # 数据配置
    data_root: str = "../../ImageToDEM/singleRGBNormalization"
    test_root: str = "../../ImageToDEM/singleRGBNormalizationTest"
    train_val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # 输入尺寸
    input_size: int = 512
    
    # 数据增强配置 - 支持AdaBins论文中使用的数据增强
    random_crop: bool = True
    flip_augmentation: bool = True
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    
    # 检查点配置
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 5  # 每N个epoch保存
    
    # 设备配置
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.scheduler_kwargs is None:
            self.scheduler_kwargs = {
                "mode": "min",
                "factor": 0.5,
                "patience": 5,
                "verbose": True
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """转换为字典 - 包含AdaBins论文所需的所有参数"""
        return {
            'model_name': self.model_name,
            'model_kwargs': self.model_kwargs,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer_name': self.optimizer_name,
            'scheduler_name': self.scheduler_name,
            'scheduler_kwargs': self.scheduler_kwargs,
            'pixel_loss': self.pixel_loss,
            'gradient_loss': self.gradient_loss,
            'scale_invariant_loss': self.scale_invariant_loss,  # AdaBins论文中的尺度不变损失
            'bin_regularization': self.bin_regularization,    # AdaBins论文中的bins正则化
            'pixel_weight': self.pixel_weight,
            'gradient_weight': self.gradient_weight,
            'scale_invariant_weight': self.scale_invariant_weight,  # 尺度不变损失权重
            'data_root': self.data_root,
            'test_root': self.test_root,
            'train_val_split': self.train_val_split,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'input_size': self.input_size,
            'random_crop': self.random_crop,
            'flip_augmentation': self.flip_augmentation,
            'brightness_jitter': self.brightness_jitter,
            'contrast_jitter': self.contrast_jitter,
            'checkpoint_dir': self.checkpoint_dir,
            'save_frequency': self.save_frequency,
            'device': self.device,
        }
    
    def save_json(self, json_path: str):
        """保存为JSON文件"""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def save_yaml(self, yaml_path: str):
        """保存为YAML文件"""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class EvalConfig:
    """评估配置"""
    model_path: str
    test_data_root: str
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    
    # 评估指标
    metrics: list = None
    
    # 输出配置
    output_dir: str = "./evaluation_results"
    save_visualizations: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['mae', 'rmse', 'ssim', 'psnr']


# 默认配置
DEFAULT_CONFIG = TrainConfig(
    model_name="resnet_unet",
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
)


if __name__ == '__main__':
    # 测试配置
    config = TrainConfig(num_epochs=100)
    print(config)
    
    # 保存配置
    config.save_json('test_config.json')
    print("配置已保存")
