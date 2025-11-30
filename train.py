"""
改进的训练脚本 - 支持配置和多个模型
"""
# 首先设置 Hugging Face 镜像（必须在导入 HF 相关模块前）
import hf_mirror_config  # 自动设置 HF_ENDPOINT 环境变量

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import logging
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.registry import ModelFactory
from losses import (
    GradientLoss, DDPMNoiseLoss, CombinedLoss,
    CombinedElasticLoss, ElasticLossFunctionRegistry
)
from data_loading import create_dataloaders
from configs.config import TrainConfig

# 数据加载器将在 AdvancedTrainer 初始化时基于配置创建，避免使用全局固定 batch_size

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class AdvancedTrainer:
    """改进的训练器类"""
    
    def __init__(self, config: TrainConfig, use_amp: bool = False):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and 'cuda' in str(self.device).lower()
        
        # 创建模型
        logger.info(f"创建模型: {config.model_name}")
        # 支持Dropout参数
        model_kwargs = dict(config.model_kwargs) if config.model_kwargs else {}
        if config.model_name.startswith('resnet_unet'):
            model_kwargs.setdefault('dropout', getattr(config, 'dropout', 0.0))
        self.model = ModelFactory.create(config.model_name, **model_kwargs)
        self.model = self.model.to(self.device)
        
        # ========== 损失函数选择 ==========
        # 支持两种模式:
        # 1. 自动模式: 根据模型名称自动选择弹性感知的损失组合 (推荐)
        # 2. 传统模式: 使用原始的简单损失组合 (向后兼容)
        
        use_elastic_loss = getattr(config, 'use_elastic_loss', True)  # 默认启用
        
        if use_elastic_loss:
            # ========== 自动模式: 弹性感知的损失函数 ==========
            logger.info(f"使用弹性感知的损失函数 (模型: {config.model_name})")
            
            # 获取模型对应的损失函数配置
            loss_config = ElasticLossFunctionRegistry.get_loss_config(config.model_name)
            logger.info(f"  损失函数配置: {loss_config}")
            
            # 构建弹性感知的组合损失
            self.criterion_pixel = CombinedElasticLoss(config=loss_config, device=self.device)
            self.criterion_grad = None  # 已包含在 combined loss 中
            self.criterion_ddpm = DDPMNoiseLoss(loss_type='l2').to(self.device)
            self.criterion_combined = self.criterion_pixel  # 使用弹性损失作为主损失
            
            self.loss_mode = 'elastic'
        else:
            # ========== 传统模式: 原始简单损失 ==========
            logger.info(f"使用传统的损失函数")
            
            self.criterion_pixel = nn.L1Loss()
            self.criterion_grad = GradientLoss().to(self.device)
            self.criterion_ddpm = DDPMNoiseLoss(loss_type='l2').to(self.device)
            self.criterion_combined = CombinedLoss(
                pixel_weight=getattr(config, 'pixel_weight', 1.0),
                gradient_weight=getattr(config, 'gradient_weight', 0.5)
            ).to(self.device)
            
            self.loss_mode = 'traditional'
        
        logger.info(f"损失函数模式: {self.loss_mode}")
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练 (AMP)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("启用混合精度 (AMP)")
        
        # 检查点目录（按模型分离）
        model_checkpoint_dir = config.checkpoint_dir.rstrip('/\\')
        self.checkpoint_dir = Path(model_checkpoint_dir) / config.model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint 目录: {self.checkpoint_dir}")
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []

        # 早停机制
        patience = getattr(config, 'early_stopping_patience', 10)
        min_delta = getattr(config, 'early_stopping_min_delta', 1e-4)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        # gradient accumulation steps (default 1)
        self.grad_accum_steps = getattr(self.config, 'grad_accum', 1)
        
        # 创建数据加载器（根据配置的 batch_size），并在 diffusion 情况下自动降低以避免 OOM
        data_root = os.path.join(os.path.dirname(__file__), "../ImageToDEM/singleRGBNormalization")
        self._init_dataloaders(data_root)
        
        # 保存配置
        self._save_config()
        
        # 打印模型信息
        self._print_model_info()
    
    def _init_dataloaders(self, data_root: str, initial_bs: int = None):
        """初始化 DataLoaders，支持 OOM 回退"""
        if initial_bs is None:
            initial_bs = self.config.batch_size
        
        bs = initial_bs
        retry_count = 0
        max_retries = 3
        
        while retry_count <= max_retries:
            try:
                logger.info(f"尝试创建 DataLoaders (batch_size={bs})...")
                _loaders = create_dataloaders(data_root, batch_size=bs)
                self.train_loader, self.val_loader = _loaders['train'], _loaders['val']
                self.current_batch_size = bs  # 记录当前 batch size
                logger.info(f"✓ DataLoaders 创建成功 (batch_size={bs})")
                return
            except Exception as e:
                if retry_count < max_retries and 'cuda' in str(self.device).lower():
                    bs = max(1, bs // 2)
                    retry_count += 1
                    logger.warning(f"DataLoader 创建失败: {str(e)[:100]}... 降低 batch_size 到 {bs} 后重试 ({retry_count}/{max_retries})...")
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"DataLoader 创建失败 (所有重试已用尽): {e}")
                    raise
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config.optimizer_name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"未知的优化器: {self.config.optimizer_name}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler_name.lower() == 'reduce_on_plateau':
            # 复制一份参数，避免修改到原始配置
            scheduler_kwargs = dict(self.config.scheduler_kwargs) if getattr(self.config, 'scheduler_kwargs', None) else {}
            # 一些 PyTorch 版本的 ReduceLROnPlateau 不支持 'verbose' 参数，移除以保证兼容性
            scheduler_kwargs.pop('verbose', None)
            # Ensure a sensible default for mode if not provided
            scheduler_kwargs.setdefault('mode', 'min')
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_kwargs
            )
        elif self.config.scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        else:
            return None
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"模型信息")
        logger.info(f"{'='*60}")
        logger.info(f"模型名称: {self.config.model_name}")
        logger.info(f"总参数数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"设备: {self.device}")
        logger.info(f"{'='*60}\n")
    
    def _save_config(self):
        """保存配置"""
        config_path = self.checkpoint_dir / 'config.json'
        self.config.save_json(str(config_path))
        logger.info(f"配置已保存到: {config_path}")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        
        loop = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Train]')
        for imgs, dems in loop:
            imgs = imgs.to(self.device)
            dems = dems.to(self.device)
            
            # zero grads only at optimizer step time when using accumulation
            if getattr(self, 'grad_accum_steps', 1) <= 1:
                self.optimizer.zero_grad()
            
            try:
                if self.config.model_name == 'diffusion_dem':
                    # DDPM 训练：使用标准 DDPM 目标 (预测噪声 eps)
                    t = torch.randint(0, self.model.num_timesteps, (imgs.size(0),), device=self.device, dtype=torch.long)
                    x_t, eps_target = self.model.ddpm_forward(dems, t)
                    
                    # 通过混合精度运行 forward/backward
                    if self.use_amp:
                        with autocast():
                            eps_pred = self.model._denoise_step(x_t, t, condition_rgb=imgs)
                            # 使用 DDPM 专用损失函数
                            loss = self.criterion_ddpm(eps_pred, eps_target)
                        # scale and backward with accumulation
                        self.scaler.scale(loss / self.grad_accum_steps).backward()
                        if (loop.n + 1) % self.grad_accum_steps == 0:
                            # 梯度裁剪（可选但推荐用于 DDPM）
                            if hasattr(self.config, 'grad_clip_norm'):
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                            eps_pred = self.model._denoise_step(x_t, t, condition_rgb=imgs)
                            loss = self.criterion_ddpm(eps_pred, eps_target)
                            # divide loss for accumulation
                            (loss / self.grad_accum_steps).backward()
                            if (loop.n + 1) % self.grad_accum_steps == 0:
                                # 梯度裁剪
                                if hasattr(self.config, 'grad_clip_norm'):
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                else:
                    # 标准 UNet 训练
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(imgs)
                            
                            # 根据损失模式选择相应的损失计算
                            if self.loss_mode == 'elastic':
                                # 弹性损失模式
                                if self.config.model_name == 'adabins' and isinstance(outputs, tuple) and len(outputs) == 3:
                                    # AdaBins返回 (depth_map, bins, bin_centers)
                                    depth_map, bins, bin_centers = outputs
                                    loss, breakdown = self.criterion_combined(depth_map, dems, bins=bins, bin_centers=bin_centers, return_breakdown=True)
                                else:
                                    loss, breakdown = self.criterion_combined(outputs, dems, return_breakdown=True)
                            else:
                                # 传统损失模式
                                if self.config.model_name == 'adabins' and isinstance(outputs, tuple) and len(outputs) == 3:
                                    depth_map, _, _ = outputs
                                    loss_p = self.criterion_pixel(depth_map, dems)
                                    loss_g = self.criterion_grad(depth_map, dems)
                                    loss = self.config.pixel_weight * loss_p + self.config.gradient_weight * loss_g
                                else:
                                    loss_p = self.criterion_pixel(outputs, dems)
                                    loss_g = self.criterion_grad(outputs, dems)
                                    loss = self.config.pixel_weight * loss_p + self.config.gradient_weight * loss_g

                        # scale loss and backward with accumulation handling
                        self.scaler.scale(loss / self.grad_accum_steps).backward()
                        if (loop.n + 1) % self.grad_accum_steps == 0:
                            if hasattr(self.config, 'grad_clip_norm'):
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        outputs = self.model(imgs)
                        
                        # 根据损失模式选择相应的损失计算
                        if self.loss_mode == 'elastic':
                            # 弹性损失模式
                            if self.config.model_name == 'adabins' and isinstance(outputs, dict):
                                # AdaBins返回字典，使用模型自带的损失函数
                                loss_dict = self.model.get_loss_components(outputs, dems, config=self.config)
                                loss = loss_dict['total_loss']
                                breakdown = {k: v.item() for k, v in loss_dict.items()}
                            else:
                                loss, breakdown = self.criterion_combined(outputs, dems, return_breakdown=True)
                        else:
                            # 传统损失模式
                            if self.config.model_name == 'adabins' and isinstance(outputs, dict):
                                # AdaBins返回字典，提取深度图进行计算
                                depth_map = outputs['depth']
                                loss_p = self.criterion_pixel(depth_map, dems)
                                loss_g = self.criterion_grad(depth_map, dems)
                                loss = self.config.pixel_weight * loss_p + self.config.gradient_weight * loss_g
                            else:
                                loss_p = self.criterion_pixel(outputs, dems)
                                loss_g = self.criterion_grad(outputs, dems)
                                loss = self.config.pixel_weight * loss_p + self.config.gradient_weight * loss_g

                        (loss / self.grad_accum_steps).backward()
                        if (loop.n + 1) % self.grad_accum_steps == 0:
                            if hasattr(self.config, 'grad_clip_norm'):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'out of memory' in error_msg and 'cuda' in str(self.device).lower():
                    logger.warning(f"⚠️ GPU OOM 发生 (Batch size: {self.current_batch_size})，尝试恢复...")
                    torch.cuda.empty_cache()
                    
                    # 尝试以更小的 batch size 重新加载数据
                    if hasattr(self, '_oom_retry_count'):
                        self._oom_retry_count += 1
                    else:
                        self._oom_retry_count = 1
                    
                    if self._oom_retry_count < 5:
                        # 暂时降低 batch size 并重新加载
                        new_bs = max(1, self.current_batch_size // 2)
                        logger.warning(f"降低 batch size 到 {new_bs} 并重新初始化 DataLoaders...")
                        try:
                            data_root = os.path.join(os.path.dirname(__file__), "../ImageToDEM/singleRGBNormalization")
                            self._init_dataloaders(data_root, initial_bs=new_bs)
                            logger.info(f"✓ DataLoaders 重新初始化成功")
                        except Exception as reload_error:
                            logger.error(f"DataLoaders 重新初始化失败: {reload_error}")
                        loop.set_postfix(loss='OOM-retry')
                    else:
                        logger.error(f"OOM 重试超过上限 ({self._oom_retry_count})，跳过此批次")
                        loop.set_postfix(loss='OOM-failed')
                    continue
                else:
                    # 其他运行时错误
                    logger.error(f"训练出错: {e}")
                    raise
        
        avg_train_loss = train_loss / len(self.train_loader)
        self.train_losses.append(avg_train_loss)
        
        return avg_train_loss
    
    def validate(self, epoch: int) -> float:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Val]')
            for imgs, dems in loop:
                imgs = imgs.to(self.device)
                dems = dems.to(self.device)
                
                try:
                    if self.config.model_name == 'diffusion_dem':
                        # DDPM 验证
                        t = torch.randint(0, self.model.num_timesteps, (imgs.size(0),), device=self.device, dtype=torch.long)
                        x_t, eps_target = self.model.ddpm_forward(dems, t)
                        eps_pred = self.model._denoise_step(x_t, t, condition_rgb=imgs)
                        loss = self.criterion_ddpm(eps_pred, eps_target)
                    else:
                        outputs = self.model(imgs)
                        
                        # 根据损失模式选择相应的损失计算
                        if self.loss_mode == 'elastic':
                            # 弹性损失模式
                            if self.config.model_name == 'adabins' and isinstance(outputs, dict):
                                # AdaBins返回字典，使用模型自带的损失函数
                                loss_dict = self.model.get_loss_components(outputs, dems, config=self.config)
                                loss = loss_dict['total_loss']
                            else:
                                loss, _ = self.criterion_combined(outputs, dems, return_breakdown=True)
                        else:
                            # 传统损失模式
                            if self.config.model_name == 'adabins' and isinstance(outputs, dict):
                                # AdaBins返回字典，提取深度图进行计算
                                depth_map = outputs['depth']
                                loss_p = self.criterion_pixel(depth_map, dems)
                                loss_g = self.criterion_grad(depth_map, dems)
                                loss = self.config.pixel_weight * loss_p + self.config.gradient_weight * loss_g
                            else:
                                loss_p = self.criterion_pixel(outputs, dems)
                                loss_g = self.criterion_grad(outputs, dems)
                                loss = self.config.pixel_weight * loss_p + self.config.gradient_weight * loss_g
                    
                    val_loss += loss.item()
                    val_count += 1
                    loop.set_postfix(loss=loss.item())
                
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if 'out of memory' in error_msg and 'cuda' in str(self.device).lower():
                        logger.warning(f"⚠️ 验证时 GPU OOM，清空缓存后继续...")
                        torch.cuda.empty_cache()
                        loop.set_postfix(loss='OOM-skip')
                        continue
                    else:
                        logger.error(f"验证出错: {e}")
                        raise
        
        avg_val_loss = val_loss / max(val_count, 1)
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict(),
        }
        
        # 最新检查点
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ 最佳模型已保存: {best_path}")
        
        # 定期保存
        if (epoch + 1) % self.config.save_frequency == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_path)
    
    def train(self):
        """完整训练流程"""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练")
        logger.info(f"总Epoch: {self.config.num_epochs}")
        logger.info(f"批次大小: {self.config.batch_size}")
        logger.info(f"初始学习率: {self.config.learning_rate}")
        logger.info(f"{'='*60}\n")
        
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots()
        train_loss_list = []
        val_loss_list = []
        line1, = ax.plot([], [], label='Train Loss')
        line2, = ax.plot([], [], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training/Validation Loss')
        ax.legend()

        for epoch in range(self.config.num_epochs):
            avg_train_loss = self.train_epoch(epoch)
            avg_val_loss = self.validate(epoch)
            logger.info(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                        f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
                logger.info(f"  ✓ 新的最佳验证损失: {self.best_val_loss:.6f}")
            self.save_checkpoint(epoch, avg_val_loss, is_best=is_best)
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"  当前学习率: {current_lr:.2e}")

            # 动态loss曲线
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)
            line1.set_data(range(1, len(train_loss_list)+1), train_loss_list)
            line2.set_data(range(1, len(val_loss_list)+1), val_loss_list)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

            # 早停判断
            self.early_stopping.step(avg_val_loss)
            if self.early_stopping.early_stop:
                logger.info(f"早停触发: {self.early_stopping.patience} 个epoch未提升，提前终止训练。")
                break
        plt.ioff()
        plt.show()
        self._save_training_log()
        logger.info(f"\n{'='*60}")
        logger.info(f"训练完成！")
        logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        logger.info(f"{'='*60}\n")
    
    def _save_training_log(self):
        """保存训练日志"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'num_epochs': self.config.num_epochs,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        log_path = self.checkpoint_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=4)
        
        logger.info(f"训练日志已保存到: {log_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练DEM预测模型')
    parser.add_argument('--config', type=str,
                       help='配置文件路径 (JSON或YAML)')
    parser.add_argument('--model', '--model-name', dest='model', type=str, default='resnet_unet',
                       help='模型名称 (alias: --model-name)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--amp', action='store_true',
                       help='启用混合精度 (Automatic Mixed Precision)')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps to simulate larger batch sizes')
    parser.add_argument('--dropout', type=float, default=None, help='模型内 dropout 比例（覆盖配置）')
    parser.add_argument('--save-frequency', type=int, default=None, help='保存检查点频率（覆盖配置）')
    
    args = parser.parse_args()
    
    # 从配置文件或命令行参数创建配置
    if args.config:
        if args.config.endswith('.json'):
            config = TrainConfig.from_json(args.config)
        elif args.config.endswith('.yaml'):
            config = TrainConfig.from_yaml(args.config)
        else:
            raise ValueError("配置文件必须是JSON或YAML格式")
        # 允许命令行覆盖配置中的部分字段
        if args.model:
            config.model_name = args.model
        if args.epochs is not None:
            config.num_epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.learning_rate = args.lr
        if args.device is not None:
            config.device = args.device
        if args.dropout is not None:
            # 将 dropout 放入 model_kwargs
            if config.model_kwargs is None:
                config.model_kwargs = {}
            config.model_kwargs['dropout'] = args.dropout
        if args.save_frequency is not None:
            config.save_frequency = args.save_frequency
        if args.grad_accum is not None:
            # store grad accumulation setting directly on config
            setattr(config, 'grad_accum', int(args.grad_accum))
    else:
        config = TrainConfig(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )
        # apply optional CLI overrides
        if args.dropout is not None:
            config.model_kwargs = config.model_kwargs or {}
            config.model_kwargs['dropout'] = args.dropout
        if args.save_frequency is not None:
            config.save_frequency = args.save_frequency
        if args.grad_accum is not None:
            setattr(config, 'grad_accum', int(args.grad_accum))
    
    # 创建训练器并开始训练
    trainer = AdvancedTrainer(config, use_amp=args.amp)
    trainer.train()


if __name__ == '__main__':
    main()
