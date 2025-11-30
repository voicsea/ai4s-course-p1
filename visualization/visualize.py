"""
可视化模块 - 生成评估结果的可视化
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Visualizer:
    """可视化类"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def normalize_for_display(x: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
        """归一化用于显示"""
        if vmin is None:
            vmin = x.min()
        if vmax is None:
            vmax = x.max()
        
        x_norm = (x - vmin) / (vmax - vmin + 1e-8)
        return np.clip(x_norm, 0, 1)
    
    def visualize_predictions(self, 
                             images: torch.Tensor,
                             predictions: torch.Tensor,
                             targets: torch.Tensor,
                             save_path: str = None,
                             num_samples: int = 4):
        """可视化预测结果"""
        num_samples = min(num_samples, images.shape[0])
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        images_np = images.cpu().numpy()
        preds_np = predictions.cpu().detach().numpy()
        targets_np = targets.cpu().numpy()
        
        for i in range(num_samples):
            # 原始图像
            img = images_np[i]
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
                img = self.normalize_for_display(img)
                axes[i, 0].imshow(img)
            else:
                img = img[0]
                axes[i, 0].imshow(img, cmap='viridis')
            axes[i, 0].set_title(f'输入图像 #{i+1}')
            axes[i, 0].axis('off')
            
            # 预测深度
            pred = preds_np[i, 0]
            pred_norm = self.normalize_for_display(pred)
            im1 = axes[i, 1].imshow(pred_norm, cmap='viridis')
            axes[i, 1].set_title(f'预测深度 #{i+1}')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # 真实深度
            target = targets_np[i, 0]
            target_norm = self.normalize_for_display(target)
            im2 = axes[i, 2].imshow(target_norm, cmap='viridis')
            axes[i, 2].set_title(f'真实深度 #{i+1}')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'predictions.png'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Prediction visualization saved to: {save_path}")
    
    def plot_error_map(self,
                      predictions: torch.Tensor,
                      targets: torch.Tensor,
                      save_path: str = None,
                      num_samples: int = 4):
        """可视化误差分布"""
        num_samples = min(num_samples, predictions.shape[0])
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        preds_np = predictions.cpu().detach().numpy()
        targets_np = targets.cpu().numpy()
        
        for i in range(num_samples):
            pred = preds_np[i, 0]
            target = targets_np[i, 0]
            error = np.abs(pred - target)
            
            # 误差分布
            im1 = axes[i, 0].imshow(error, cmap='hot')
            axes[i, 0].set_title(f'绝对误差分布 #{i+1}')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # 误差柱状图
            error_flat = error.flatten()
            axes[i, 1].hist(error_flat, bins=50, color='steelblue', edgecolor='black')
            axes[i, 1].set_title(f'误差分布直方图 #{i+1}')
            axes[i, 1].set_xlabel('绝对误差')
            axes[i, 1].set_ylabel('频数')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'error_maps.png'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Error map saved to: {save_path}")
    
    def plot_metrics_comparison(self,
                               results: Dict,
                               save_path: str = None):
        """绘制指标对比图"""
        metrics_names = ['mae', 'rmse', 'ssim', 'psnr', 'r2']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric_name in enumerate(metrics_names):
            ax = axes[idx]
            values = [results[model]['metrics'][metric_name] for model in model_names]
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)]
            bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_title(f'{metric_name.upper()} 对比', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name.upper())
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=10)
        
        # 隐藏最后一个子图
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'metrics_comparison.png'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Metrics comparison saved to: {save_path}")
    
    def plot_scatter_analysis(self,
                             predictions: torch.Tensor,
                             targets: torch.Tensor,
                             save_path: str = None):
        """绘制散点分析图"""
        preds_flat = predictions.cpu().detach().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 散点图
        axes[0].scatter(targets_flat, preds_flat, alpha=0.5, s=10)
        axes[0].plot([targets_flat.min(), targets_flat.max()],
                    [targets_flat.min(), targets_flat.max()],
                    'r--', lw=2, label='完美预测')
        axes[0].set_xlabel('真实值')
        axes[0].set_ylabel('预测值')
        axes[0].set_title('预测值 vs 真实值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 残差图
        residuals = preds_flat - targets_flat
        axes[1].scatter(targets_flat, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('真实值')
        axes[1].set_ylabel('残差 (预测 - 真实)')
        axes[1].set_title('残差分析')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'scatter_analysis.png'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Scatter plot saved to: {save_path}")


if __name__ == '__main__':
    # 测试可视化
    visualizer = Visualizer(output_dir='./test_visualizations')
    
    # 创建示例数据
    images = torch.randn(4, 3, 512, 512)
    predictions = torch.randn(4, 1, 512, 512)
    targets = torch.randn(4, 1, 512, 512)
    
    # 可视化
    visualizer.visualize_predictions(images, predictions, targets, num_samples=2)
    visualizer.plot_error_map(predictions, targets, num_samples=2)
    visualizer.plot_scatter_analysis(predictions, targets)
    
    print("[OK] Visualization test completed")
