"""
模型注册表和工厂类
支持多个模型架构，便于扩展和对比
"""
import torch
import torch.nn as nn
from typing import Dict, Type, Callable
from functools import wraps


class ModelRegistry:
    """模型注册表"""
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册模型"""
        def wrapper(model_class):
            cls._registry[name] = model_class
            model_class.model_name = name
            return model_class
        return wrapper
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """获取模型实例"""
        if name not in cls._registry:
            raise ValueError(
                f"模型 '{name}' 未注册。"
                f"可用模型: {list(cls._registry.keys())}"
            )
        model_class = cls._registry[name]
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls):
        """列出所有可用模型"""
        return list(cls._registry.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type):
        """手动注册模型"""
        cls._registry[name] = model_class


# 导入模型
from .unet import ResNetUNet, ResNetUNet_Large, DepthAnythingV2, DepthAnythingV2Lite, LeRes
from .adabins import AdaBins
from .diffusion import DiffusionDEM
from .unet import ViTUNet


# 注册模型
ModelRegistry.register_model("resnet_unet", ResNetUNet)
ModelRegistry.register_model("resnet_unet_large", ResNetUNet_Large)
ModelRegistry.register_model("diffusion_dem", DiffusionDEM)
ModelRegistry.register_model("vit_unet", ViTUNet)

# 注册 SOTA 深度预测模型
ModelRegistry.register_model("depth_anything_v2", DepthAnythingV2)
ModelRegistry.register_model("depth_anything_v2_lite", DepthAnythingV2Lite)
ModelRegistry.register_model("leres", LeRes)

# Register AdaBins
ModelRegistry.register_model("adabins", AdaBins)


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create(model_name: str, **kwargs) -> nn.Module:
        """创建模型"""
        return ModelRegistry.get(model_name, **kwargs)
    
    @staticmethod
    def list_available_models():
        """列出可用模型"""
        return ModelRegistry.list_models()
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """获取模型信息"""
        model = ModelFactory.create(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "name": model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_class": model.__class__.__name__,
        }


if __name__ == '__main__':
    # 测试模型注册表
    print("可用模型:", ModelRegistry.list_models())
    
    # 创建模型
    model = ModelFactory.create("resnet_unet", out_channels=1)
    print(f"模型创建成功: {model.__class__.__name__}")
    
    # 获取模型信息
    info = ModelFactory.get_model_info("resnet_unet")
    print(f"\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
