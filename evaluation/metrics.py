"""
评估模块 - 计算各种性能指标
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


class Evaluator:
    """模型评估类"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metrics = {}
    
    def mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """平均绝对误差 (Mean Absolute Error)"""
        return torch.mean(torch.abs(pred - target)).item()
    
    def mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """平均平方误差 (Mean Square Error)"""
        return torch.mean((pred - target) ** 2).item()
    
    def rmse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """均方根误差 (Root Mean Square Error)"""
        mse_val = self.mse(pred, target)
        return np.sqrt(mse_val)
    
    def ssim(self, pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
        """结构相似性指数 (Structural Similarity Index)"""
        if len(pred.shape) == 4:  # Batch
            pred = pred[0, 0]
            target = target[0, 0]
        elif len(pred.shape) == 3:  # Channel dimension
            pred = pred[0]
            target = target[0]
        
        return ssim(pred, target, data_range=data_range)
    
    def psnr(self, pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
        """峰值信噪比 (Peak Signal to Noise Ratio)"""
        if len(pred.shape) == 4:
            pred = pred[0, 0]
            target = target[0, 0]
        elif len(pred.shape) == 3:
            pred = pred[0]
            target = target[0]
        
        return psnr(target, pred, data_range=data_range)
    
    def r2_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """R² 得分 (R-squared)"""
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        return 1 - (ss_res / ss_tot).item()
    
    def absrel(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """平均相对绝对误差 (Average Relative Absolute Error)"""
        # 使用torch.clamp确保分母不会过小，并取绝对值以避免符号问题
        abs_diff = torch.abs(pred - target)
        safe_target = torch.clamp(torch.abs(target), min=1e-6)
        return torch.mean(abs_diff / safe_target).item()
    
    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """评估一个批次"""
        # 确保在CPU上进行numpy操作
        pred_cpu = pred.cpu()
        target_cpu = target.cpu()
        
        pred_np = pred_cpu.numpy()
        target_np = target_cpu.numpy()
        
        metrics = {
            'mae': self.mae(pred_cpu, target_cpu),
            'mse': self.mse(pred_cpu, target_cpu),
            'rmse': self.rmse(pred_cpu, target_cpu),
            'absrel': self.absrel(pred_cpu, target_cpu),
            'r2': self.r2_score(pred_cpu, target_cpu),
            'ssim': self.ssim(pred_np, target_np),
            'psnr': self.psnr(pred_np, target_np),
        }
        
        return metrics
    
    def evaluate_dataset(self, model: nn.Module, data_loader, device: str = None) -> Dict[str, float]:
        """评估整个数据集"""
        if device is None:
            device = self.device
        
        model.eval()
        all_metrics = {
            'mae': [],
            'mse': [],
            'rmse': [],
            'r2': [],
            'ssim': [],
            'psnr': [],
        }
        
        with torch.no_grad():
            for imgs, dems in data_loader:
                imgs = imgs.to(device)
                dems = dems.to(device)
                
                pred = model(imgs)
                batch_metrics = self.evaluate_batch(pred, dems)
                
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
        
        # 计算平均值
        avg_metrics = {
            key: np.mean(values) for key, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """打印指标"""
        print("\n" + "="*60)
        print("评估指标")
        print("="*60)
        for key, value in metrics.items():
            print(f"  {key.upper():6s}: {value:.6f}")
        print("="*60 + "\n")


if __name__ == '__main__':
    # 测试评估器
    evaluator = Evaluator()
    
    pred = torch.randn(4, 1, 512, 512)
    target = torch.randn(4, 1, 512, 512)
    
    metrics = evaluator.evaluate_batch(pred, target)
    evaluator.print_metrics(metrics)
