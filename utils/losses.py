"""
损失函数定义
用于DEM预测的像素级损失、梯度损失以及基于模型弹性的自适应损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np


class GradientLoss(nn.Module):
    """
    梯度损失 (Gradient Loss)
    用于保留深度图的边界信息和细节
    计算预测深度图和真实深度图的梯度差异
    """
    def __init__(self, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测的深度图 (B, 1, H, W)
            target: 真实的深度图 (B, 1, H, W)
        
        Returns:
            梯度损失值
        """
        # 计算梯度 (使用Sobel算子)
        # Sobel X
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        # Sobel Y
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # 梯度差异
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        # 合并梯度差异
        grad_loss = grad_diff_x + grad_diff_y
        
        if self.reduction == 'mean':
            return grad_loss.mean()
        elif self.reduction == 'sum':
            return grad_loss.sum()
        else:
            return grad_loss


class SmoothL1Loss(nn.Module):
    """
    平滑L1损失
    对异常值更加鲁棒
    """
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, reduction=self.reduction, beta=self.beta)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合像素级损失和梯度损失
    """
    def __init__(self, pixel_weight=1.0, gradient_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        
        self.pixel_loss = nn.L1Loss()
        self.gradient_loss = GradientLoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测的深度图 (B, 1, H, W)
            target: 真实的深度图 (B, 1, H, W)
        
        Returns:
            组合损失值
        """
        loss_p = self.pixel_loss(pred, target)
        loss_g = self.gradient_loss(pred, target)
        
        return self.pixel_weight * loss_p + self.gradient_weight * loss_g


class PSNRLoss(nn.Module):
    """
    PSNR损失 (Peak Signal-to-Noise Ratio)
    用于图像质量评估
    """
    def __init__(self, reduction='mean', max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.reduction = reduction
        self.max_val = max_val
        
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none')
        psnr = 20 * torch.log10(self.max_val / (torch.sqrt(mse) + 1e-8))
        
        if self.reduction == 'mean':
            return -psnr.mean()  # 负值使其可以作为损失最小化
        elif self.reduction == 'sum':
            return -psnr.sum()
        else:
            return -psnr


class DDPMNoiseLoss(nn.Module):
    """
    DDPM 噪声预测损失
    标准 DDPM 训练目标：预测添加到原始样本的噪声
    
    L_simple = E_t [ ||eps - eps_theta(x_t, t)|| ^ 2 ]
    """
    def __init__(self, reduction='mean', loss_type='l2'):
        super(DDPMNoiseLoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type  # 'l2', 'l1', 或 'smooth_l1'
        
    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """
        计算 DDPM 噪声预测损失
        
        Args:
            noise_pred: (batch, channels, H, W) - 模型预测的噪声
            noise_target: (batch, channels, H, W) - 真实的高斯噪声
        
        Returns:
            标量损失
        """
        if self.loss_type == 'l2':
            loss = F.mse_loss(noise_pred, noise_target, reduction=self.reduction)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(noise_pred, noise_target, reduction=self.reduction)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(noise_pred, noise_target, reduction=self.reduction, beta=1.0)
        else:
            raise ValueError(f"未知的损失类型: {self.loss_type}")
        
        return loss


class DDPMVLBLoss(nn.Module):
    """
    DDPM 变分下界 (Variational Lower Bound) 损失
    更接近原始 DDPM 论文的目标函数
    
    支持多个损失项：
    1. 均值预测损失 (预测去噪后的均值)
    2. 方差预测损失 (可选)
    3. KL 散度 (可选)
    """
    def __init__(self, reduction='mean', predict_variance=False):
        super(DDPMVLBLoss, self).__init__()
        self.reduction = reduction
        self.predict_variance = predict_variance
        
    def forward(self, model_output: torch.Tensor, target: torch.Tensor, 
                t: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """
        计算 VLB 损失 (简化版本，主要用于均值预测)
        
        Args:
            model_output: (batch, channels, H, W) - 模型输出 (均值预测)
            target: (batch, channels, H, W) - 目标 (原始样本或噪声)
            t: (batch,) - 时间步长
            alphas_cumprod: (num_timesteps,) - 累积 alpha 值
        
        Returns:
            标量损失
        """
        # 简化实现: 直接使用 L2 损失作为 VLB 的主要项
        loss = F.mse_loss(model_output, target, reduction=self.reduction)
        
        return loss


# ============================================================================
# 基于弹性的自适应损失函数族
# ============================================================================

class HuberLoss(nn.Module):
    """
    Huber损失 (鲁棒于异常值的损失)
    - 在小误差上类似 L2 (smooth)
    - 在大误差上类似 L1 (robust to outliers)
    
    适用场景: 中等弹性的模型 (resnet_unet, 一般性模型)
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            Huber 损失
        """
        loss = F.huber_loss(pred, target, delta=self.delta, reduction=self.reduction)
        return loss


class LogCoshLoss(nn.Module):
    """
    Log-Cosh 损失
    L(x) = log(cosh(x)) ≈ (x²/2) for small x, |x| - log(2) for large x
    
    优势:
    - 对异常值鲁棒 (类似 L1 的长尾)
    - 更平滑的梯度 (类似 L2)
    - 二阶可微
    
    适用场景: 高弹性模型 (多尺度模型、复杂架构)
    """
    def __init__(self, reduction: str = 'mean'):
        super(LogCoshLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 log-cosh 损失"""
        diff = pred - target
        loss = torch.log(torch.cosh(diff))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"未知的 reduction: {self.reduction}")


class CharbonnierLoss(nn.Module):
    """
    Charbonnier 损失 (平滑的 L1 变体)
    L(x) = sqrt(x² + eps²) - eps
    
    优势:
    - 比 L1 更平滑
    - 对异常值更鲁棒
    - 在零处可微
    
    适用场景: 高弹性、需要细节保留的模型 (DPT-like)
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Charbonnier 损失"""
        diff = pred - target
        loss = torch.sqrt(diff**2 + self.eps**2) - self.eps
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"未知的 reduction: {self.reduction}")


class SobelGradientLoss(nn.Module):
    """
    改进的梯度损失 (使用 Sobel 算子)
    
    关键特性:
    - 检测边界和纹理
    - 支持多阶导数
    - 可配置权重
    
    适用场景: 所有模型 (边界细节保留)
    """
    def __init__(self, order: int = 1, weight: float = 1.0, reduction: str = 'mean'):
        super(SobelGradientLoss, self).__init__()
        self.order = order  # 1 = 一阶导数, 2 = 二阶导数
        self.weight = weight
        self.reduction = reduction
        
        # 预定义 Sobel 算子
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算梯度损失"""
        # 确保 Sobel 算子与输入在同一设备上
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        # 一阶梯度
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)
        
        grad_loss = torch.abs(grad_pred_x - grad_target_x) + torch.abs(grad_pred_y - grad_target_y)
        
        # 二阶梯度
        if self.order == 2:
            grad2_pred_x = F.conv2d(grad_pred_x, sobel_x, padding=1)
            grad2_pred_y = F.conv2d(grad_pred_y, sobel_y, padding=1)
            
            grad2_target_x = F.conv2d(grad_target_x, sobel_x, padding=1)
            grad2_target_y = F.conv2d(grad_target_y, sobel_y, padding=1)
            
            grad_loss = grad_loss + torch.abs(grad2_pred_x - grad2_target_x) + torch.abs(grad2_pred_y - grad2_target_y)
        
        if self.reduction == 'mean':
            return self.weight * grad_loss.mean()
        elif self.reduction == 'sum':
            return self.weight * grad_loss.sum()
        elif self.reduction == 'none':
            return self.weight * grad_loss
        else:
            raise ValueError(f"未知的 reduction: {self.reduction}")


class PerceptualSimilarityLoss(nn.Module):
    """
    感知相似度损失 (使用预训练的特征提取器)
    可通过特征空间的相似度来优化高级语义一致性
    
    适用场景: 高弹性模型、需要感知一致性的任务
    """
    def __init__(self, feature_extractor: Optional[nn.Module] = None, layers: List[str] = None, 
                 weights: Optional[Dict[str, float]] = None, reduction: str = 'mean'):
        super(PerceptualSimilarityLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers or []
        self.weights = weights or {}
        self.reduction = reduction
        self.use_features = feature_extractor is not None and len(layers) > 0
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算感知相似度损失"""
        if not self.use_features:
            # 退回到 L1 损失
            return F.l1_loss(pred, target, reduction=self.reduction)
        
        # 提取特征
        with torch.no_grad():
            target_features = self.feature_extractor(target)
        pred_features = self.feature_extractor(pred)
        
        loss = 0.0
        for layer_name in self.layers:
            if layer_name in target_features and layer_name in pred_features:
                weight = self.weights.get(layer_name, 1.0)
                layer_loss = F.l1_loss(pred_features[layer_name], target_features[layer_name])
                loss = loss + weight * layer_loss
        
        return loss if loss > 0 else torch.tensor(0.0, device=pred.device)


class AdaptiveWeightedLoss(nn.Module):
    """
    自适应加权损失函数
    根据像素错误的大小动态调整权重
    - 小错误 → 高权重 (精细拟合)
    - 大错误 → 低权重 (鲁棒性)
    
    适用场景: 所有模型类型，特别是需要平衡精度和鲁棒性的场景
    """
    def __init__(self, base_loss: str = 'l1', alpha: float = 1.0, reduction: str = 'mean'):
        super(AdaptiveWeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha  # 控制权重衰减速度
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算自适应加权损失"""
        # 计算基础误差
        if self.base_loss == 'l1':
            errors = torch.abs(pred - target)
        elif self.base_loss == 'l2':
            errors = (pred - target) ** 2
        else:
            errors = torch.abs(pred - target)
        
        # 计算自适应权重 (使用指数衰减)
        # 对于小误差：weight ≈ 1.0
        # 对于大误差：weight → 0
        weights = torch.exp(-self.alpha * errors)
        
        # 应用权重
        weighted_loss = errors * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        elif self.reduction == 'none':
            return weighted_loss
        else:
            raise ValueError(f"未知的 reduction: {self.reduction}")


class ElasticLossFunctionRegistry:
    """
    弹性感知的损失函数注册表
    根据模型特性自动选择最优的损失组合
    """
    
    # 模型弹性分类
    MODEL_ELASTICITY = {
        # 低弹性 (简单、快速的模型)
        'low': {
            'pixel_loss': 'huber',      # Huber: 鲁棒于异常值
            'grad_loss': 'sobel_1st',   # 一阶梯度
            'weights': {'pixel': 1.0, 'grad': 0.3}
        },
        # 中等弹性 (标准 CNN 模型)
        'medium': {
            'pixel_loss': 'charbonnier',    # Charbonnier: 平滑 + 鲁棒
            'grad_loss': 'sobel_1st',       # 一阶梯度
            'weights': {'pixel': 1.0, 'grad': 0.5}
        },
        # 高弹性 (复杂模型、多尺度)
        'high': {
            'pixel_loss': 'log_cosh',           # Log-Cosh: 平滑 + 鲁棒 + 二阶可微
            'grad_loss': 'sobel_2nd',           # 二阶梯度 (细节)
            'adaptive': True,                   # 启用自适应加权
            'weights': {'pixel': 1.0, 'grad': 0.6, 'adaptive': 0.2}
        },
        # AdaBins专用配置
        'adabins_special': {
            'pixel_loss': 'log_cosh',
            'grad_loss': 'sobel_2nd',
            'adaptive': True,
            'bin_regularization': True,
            'edge_aware': True,
            'weights': {
                'pixel': 1.0, 
                'grad': 0.5, 
                'adaptive': 0.2,
                'bin_reg': 1e-5,
                'edge_aware': 0.1
            }
        }
    }
    
    @staticmethod
    def get_loss_config(model_name: str) -> Dict:
        """
        根据模型名称获取推荐的损失函数配置
        
        Args:
            model_name: 模型名称 (resnet_unet, vit_unet, leres, dpt, diffusion_dem)
        
        Returns:
            损失函数配置字典
        """
        # 模型弹性分类
        elasticity_mapping = {
            'resnet_unet': 'medium',      # 标准 ResNet-UNet
            'vit_unet': 'high',            # Vision Transformer (高弹性)
            'leres': 'high',               # LeRes (多尺度, 高弹性)
            'dpt': 'high',                 # DPT (Vision Transformer, 高弹性)
            'diffusion_dem': 'high',       # 扩散模型 (高弹性)
            'adabins': 'high',             # AdaBins (自适应分桶, 高弹性)
            'default': 'medium'
        }
        
        # 特殊处理adabins
        if model_name == 'adabins':
            return ElasticLossFunctionRegistry.MODEL_ELASTICITY['adabins_special']
        
        elasticity = elasticity_mapping.get(model_name, 'medium')
        return ElasticLossFunctionRegistry.MODEL_ELASTICITY[elasticity]
    
    @staticmethod
    def build_loss_function(model_name: str, device: torch.device = None) -> 'CombinedElasticLoss':
        """
        根据模型名称构建完整的损失函数
        
        Args:
            model_name: 模型名称
            device: 计算设备
        
        Returns:
            CombinedElasticLoss 实例
        """
        config = ElasticLossFunctionRegistry.get_loss_config(model_name)
        return CombinedElasticLoss(config=config, device=device)


class CombinedElasticLoss(nn.Module):
    """
    组合的弹性感知损失函数
    根据模型特性选择和组合多个损失项
    
    支持的像素级损失:
    - 'l1': L1 损失
    - 'l2': L2 损失
    - 'smooth_l1': SmoothL1
    - 'huber': Huber 损失
    - 'charbonnier': Charbonnier 损失
    - 'log_cosh': Log-Cosh 损失
    
    支持的梯度损失:
    - 'sobel_1st': 一阶 Sobel 梯度
    - 'sobel_2nd': 二阶 Sobel 梯度
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        super(CombinedElasticLoss, self).__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # 默认配置 (中等弹性)
        if config is None:
            config = ElasticLossFunctionRegistry.MODEL_ELASTICITY['medium']
        
        self.config = config
        
        # 构建像素级损失
        pixel_loss_type = config.get('pixel_loss', 'charbonnier')
        self.pixel_loss = self._build_pixel_loss(pixel_loss_type).to(device)
        
        # 构建梯度损失
        grad_loss_type = config.get('grad_loss', 'sobel_1st')
        self.grad_loss = self._build_grad_loss(grad_loss_type).to(device)
        
        # 自适应加权损失 (可选)
        self.adaptive_loss = None
        if config.get('adaptive', False):
            self.adaptive_loss = AdaptiveWeightedLoss(base_loss='l1').to(device)
        
        # Bin正则化损失 (用于AdaBins)
        self.use_bin_regularization = config.get('bin_regularization', False)
        
        # 边缘感知损失 (用于AdaBins)
        self.use_edge_aware = config.get('edge_aware', False)
        if self.use_edge_aware:
            self.edge_aware_loss = SobelGradientLoss(order=1).to(device)
        
        # 损失权重
        self.weights = config.get('weights', {'pixel': 1.0, 'grad': 0.5})
    
    def _build_pixel_loss(self, loss_type: str) -> nn.Module:
        """构建像素级损失函数"""
        loss_type = loss_type.lower()
        
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'l2':
            return nn.MSELoss()
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        elif loss_type == 'huber':
            return HuberLoss()
        elif loss_type == 'charbonnier':
            return CharbonnierLoss()
        elif loss_type == 'log_cosh':
            return LogCoshLoss()
        else:
            print(f"警告: 未知的像素损失类型 {loss_type}，使用默认 Charbonnier")
            return CharbonnierLoss()
    
    def _build_grad_loss(self, loss_type: str) -> nn.Module:
        """构建梯度损失函数"""
        loss_type = loss_type.lower()
        
        if loss_type == 'sobel_1st':
            return SobelGradientLoss(order=1)
        elif loss_type == 'sobel_2nd':
            return SobelGradientLoss(order=2)
        elif loss_type == 'none':
            return nn.Identity()  # 虚拟损失，返回 0
        else:
            print(f"警告: 未知的梯度损失类型 {loss_type}，使用默认一阶 Sobel")
            return SobelGradientLoss(order=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                bins=None, bin_centers=None, return_breakdown: bool = False) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测值
            target: 目标值
            bins: 预测的分桶分布 (B, n_bins, H, W) - 用于AdaBins
            bin_centers: 预测的分桶中心 (B, n_bins) - 用于AdaBins
            return_breakdown: 是否返回损失分解信息
        
        Returns:
            总损失 (或包含分解信息的元组)
        """
        total_loss = 0.0
        loss_breakdown = {}
        
        # 像素级损失
        loss_pixel = self.pixel_loss(pred, target)
        total_loss += self.weights.get('pixel', 1.0) * loss_pixel
        loss_breakdown['pixel'] = loss_pixel.item()
        
        # 梯度损失
        loss_grad = self.grad_loss(pred, target)
        total_loss += self.weights.get('grad', 0.5) * loss_grad
        loss_breakdown['grad'] = loss_grad.item()
        
        # 自适应损失 (如果启用)
        if self.adaptive_loss is not None:
            loss_adaptive = self.adaptive_loss(pred, target)
            total_loss += self.weights.get('adaptive', 0.2) * loss_adaptive
            loss_breakdown['adaptive'] = loss_adaptive.item()
        
        # Bin正则化损失 (用于AdaBins)
        if self.use_bin_regularization and bin_centers is not None:
            # 确保分桶中心是有序的且间隔合理
            # 计算分桶中心之间的差值
            bin_diffs = bin_centers[:, 1:] - bin_centers[:, :-1]
            # 正则化损失：惩罚过小的间隔
            loss_bin_reg = torch.mean(torch.exp(-bin_diffs)) * 0.1
            total_loss += self.weights.get('bin_reg', 1e-5) * loss_bin_reg
            loss_breakdown['bin_reg'] = loss_bin_reg.item()
        
        # 边缘感知损失 (用于AdaBins)
        if self.use_edge_aware:
            # 在边缘区域加强梯度损失
            # 计算边缘掩码
            target_grad = self.edge_aware_loss(target, target * 0)  # 计算目标的梯度幅度
            edge_mask = (target_grad > torch.mean(target_grad) + 0.5 * torch.std(target_grad)).float()
            
            # 在边缘区域计算梯度损失
            pred_grad = self.edge_aware_loss(pred, pred * 0)
            target_grad = self.edge_aware_loss(target, target * 0)
            loss_edge = torch.mean(edge_mask * torch.abs(pred_grad - target_grad))
            total_loss += self.weights.get('edge_aware', 0.1) * loss_edge
            loss_breakdown['edge_aware'] = loss_edge.item()
        
        loss_breakdown['total'] = total_loss.item()
        
        if return_breakdown:
            return total_loss, loss_breakdown
        
        return total_loss


if __name__ == '__main__':
    # 测试损失函数
    batch_size, channels, height, width = 4, 1, 128, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)
    
    print("=" * 60)
    print("测试基础损失函数")
    print("=" * 60)
    
    # 测试各个损失函数
    losses_to_test = [
        ('Huber Loss', HuberLoss()),
        ('Log-Cosh Loss', LogCoshLoss()),
        ('Charbonnier Loss', CharbonnierLoss()),
        ('Sobel Gradient Loss (1st)', SobelGradientLoss(order=1)),
        ('Sobel Gradient Loss (2nd)', SobelGradientLoss(order=2)),
        ('Adaptive Weighted Loss', AdaptiveWeightedLoss()),
    ]
    
    for name, loss_fn in losses_to_test:
        loss_fn = loss_fn.to(device)
        value = loss_fn(pred, target)
        print(f"{name:.<40} {value.item():.6f}")
    
    print("\n" + "=" * 60)
    print("测试弹性感知的组合损失")
    print("=" * 60)
    
    # 测试不同弹性级别的模型
    models_to_test = [
        'resnet_unet',
        'vit_unet', 
        'leres',
        'dpt',
    ]
    
    for model_name in models_to_test:
        print(f"\n模型: {model_name}")
        config = ElasticLossFunctionRegistry.get_loss_config(model_name)
        print(f"  弹性级别配置: {config}")
        
        loss_fn = CombinedElasticLoss(config=config, device=device)
        loss_value, breakdown = loss_fn(pred, target, return_breakdown=True)
        
        print(f"  总损失: {loss_value.item():.6f}")
        print(f"  损失分解:")
        for key, val in breakdown.items():
            print(f"    {key:.<15} {val:.6f}")

