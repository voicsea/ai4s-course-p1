"""
Diffusion 模型用于 DEM 深度预测
基于 DDPM (Denoising Diffusion Probabilistic Models) 思想的简化实现
支持从噪声逐步去噪生成 DEM 深度图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """时间步长的位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) 时间步长张量
        Returns:
            (batch, dim) 编码后的时间步长
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResnetBlock(nn.Module):
    """带时间步长条件的 ResNet 块"""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_ch),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_ch, H, W)
            t_emb: (batch, time_dim)
        Returns:
            (batch, out_ch, H, W)
        """
        h = self.norm1(self.conv1(x))
        t_out = self.time_mlp(t_emb)
        h = h + t_out[:, :, None, None]  # 加入时间条件
        h = F.relu(h)
        h = self.norm2(self.conv2(h))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """轻量级的自注意力块 (仅在低分辨率使用)"""
    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        if not use_attention:
            self.norm = nn.GroupNorm(8, channels)
            return
        
        self.norm = nn.GroupNorm(8, channels)
        # 使用较少的查询点来减少内存消耗
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 降采样
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            (batch, channels, H, W)
        """
        if not self.use_attention:
            return x
        
        batch, channels, H, W = x.shape
        h = self.norm(x)
        
        # 在低分辨率计算注意力
        h_pool = self.pool(h)  # (batch, channels, 4, 4)
        b, c, ph, pw = h_pool.shape
        
        # 获取 Q, K, V
        qkv = self.qkv(h_pool).view(b, c * 3, -1)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 计算注意力
        scale = (c ** -0.5)
        attn = torch.bmm(q.transpose(1, 2), k)  # (batch, 16, 16)
        attn = F.softmax(attn * scale, dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)
        out = out.view(b, c, ph, pw)
        
        # 上采样回原分辨率 (H, W) 而不是固定大小
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return x + self.proj_out(out)


class UNet(nn.Module):
    """基础 UNet 编码-解码网络 (用于 Diffusion) - 轻量化版本"""
    def __init__(self, in_channels: int = 4, out_channels: int = 1, time_dim: int = 128, 
                 channel_mult: tuple = (1, 2, 4)):
        super().__init__()
        self.time_dim = time_dim
        self.channel_mult = channel_mult
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            PositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # 编码器
        base_ch = 64
        self.down_convs = nn.ModuleList()
        
        ch = base_ch
        for i, mult in enumerate(channel_mult):
            out_ch = base_ch * mult
            use_attn = (i > 0)  # 仅在 downsample 后使用注意力
            self.down_convs.append(nn.Sequential(
                ResnetBlock(ch, out_ch, time_dim),
                AttentionBlock(out_ch, use_attention=use_attn),
                ResnetBlock(out_ch, out_ch, time_dim),
                nn.Conv2d(out_ch, out_ch, 4, 2, 1),  # 下采样
            ))
            ch = out_ch
        
        # 桥接
        self.mid = nn.Sequential(
            ResnetBlock(ch, ch, time_dim),
            ResnetBlock(ch, ch, time_dim),
        )
        
        # 解码器
        self.up_convs = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_ch * mult
            use_attn = (i < len(channel_mult) - 1)  # 仅在 upsample 前使用注意力
            self.up_convs.append(nn.Sequential(
                ResnetBlock(ch + out_ch, out_ch, time_dim),
                AttentionBlock(out_ch, use_attention=use_attn),
                ResnetBlock(out_ch, out_ch, time_dim),
                nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1),  # 上采样
            ))
            ch = out_ch
        
        # 输出
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.ReLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W) - 噪声或部分去噪的 DEM
            t: (batch,) - 时间步长 (0-1000)
        Returns:
            (batch, out_channels, H, W)
        """
        # 时间编码
        t_emb = self.time_mlp(t)
        
        # 初始卷积
        h = self.conv_in(x)
        
        # 编码
        hs = [h]
        for down_conv in self.down_convs:
            h_list = []
            # 遍历模块
            for module in down_conv:
                if isinstance(module, ResnetBlock):
                    h = module(h, t_emb)
                elif isinstance(module, AttentionBlock):
                    h = module(h)
                else:
                    h = module(h)
            hs.append(h)
        
        # 桥接
        for module in self.mid:
            if isinstance(module, ResnetBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 解码
        for up_conv in self.up_convs:
            h_skip = hs.pop()
            h = torch.cat([h, h_skip], dim=1)
            
            for module in up_conv:
                if isinstance(module, ResnetBlock):
                    h = module(h, t_emb)
                elif isinstance(module, AttentionBlock):
                    h = module(h)
                else:
                    h = module(h)
        
        # 输出
        h = self.out(h)
        return h


class DiffusionDEM(nn.Module):
    """
    Diffusion 模型用于 DEM 生成
    支持从 RGB 图像条件生成 DEM 深度图
    """
    def __init__(self, out_channels: int = 1, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.out_channels = out_channels
        
        # Diffusion UNet (DEM 噪声 1 + RGB 条件 3 = 4 通道输入)
        self.unet = UNet(
            in_channels=4,  # DEM (1) + RGB (3)
            out_channels=out_channels,
            time_dim=128,
            channel_mult=(1, 2, 4)  # 减少深度以降低内存使用
        )
        
        # 注册 beta schedule (DDPM 线性 schedule)
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # 注册为 buffer（不是参数，不参与优化，但会跟随模型移动到 device）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 计算预处理常数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))
    
    def ddpm_forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        DDPM 前向扩散：从 x0 和时间步 t 生成噪声样本 x_t
        
        Args:
            x0: (batch, 1, H, W) - 原始 DEM
            t: (batch,) - 时间步长 [0, num_timesteps)
        
        Returns:
            x_t: (batch, 1, H, W) - 在时间步 t 的噪声样本
            eps: (batch, 1, H, W) - 添加的高斯噪声 (用于训练目标)
        """
        # 生成高斯噪声
        eps = torch.randn_like(x0)
        
        # 获取时间步对应的 sqrt_alphas_cumprod 和 sqrt_one_minus_alphas_cumprod
        # t 的形状是 (batch,)，需要扩展为 (batch, 1, 1, 1) 以广播
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * eps
        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * eps
        
        return x_t, eps

    
    def forward(self, x: torch.Tensor, t: torch.Tensor = None, 
                condition_rgb: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 可以是两种情况:
               1) DEM 噪声: (batch, 1, H, W) - 则需要 t 和可选的 condition_rgb
               2) RGB 图像: (batch, 3, H, W) - 直接推理模式
            t: (batch,) - 时间步长 (仅在 x 是 DEM 时使用)
            condition_rgb: (batch, 3, H, W) - RGB 条件图像 (可选)
        
        Returns:
            (batch, 1, H, W) - 去噪预测或生成的 DEM
        """
        # 检测是否为 RGB 推理模式 (x 有 3 通道)
        if x.shape[1] == 3:
            # RGB 直接推理模式，忽略 t
            return self.inference(x, num_steps=50)
        
        # DEM + 时间步长模式
        if t is None:
            raise ValueError("时间步长 t 不能为空，DEM 输入需要时间步长")
        
        return self._denoise_step(x, t, condition_rgb)
    
    def _denoise_step(self, x: torch.Tensor, t: torch.Tensor, 
                      condition_rgb: torch.Tensor = None) -> torch.Tensor:
        """单步去噪 (内部方法，避免递归)"""
        batch_size, _, h, w = x.shape
        
        # 构建输入张量 (4 通道: x (1通道) + RGB (3通道))
        if condition_rgb is not None:
            # 确保 RGB 是 3 通道
            assert condition_rgb.shape[1] == 3, f"RGB should have 3 channels, got {condition_rgb.shape[1]}"
            x_input = torch.cat([x, condition_rgb], dim=1)  # (batch, 4, H, W)
        else:
            # 无条件: 用零填充
            padding = torch.zeros(batch_size, 3, h, w, device=x.device, dtype=x.dtype)
            x_input = torch.cat([x, padding], dim=1)  # (batch, 4, H, W)
        
        # 通过 UNet 进行去噪
        pred = self.unet(x_input, t)
        
        return pred
    
    def inference(self, rgb: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """
        推理: 从随机噪声逐步去噪生成 DEM
        
        Args:
            rgb: (batch, 3, H, W) - 输入 RGB 图像
            num_steps: 去噪步数
        
        Returns:
            (batch, out_channels, H, W) - 生成的 DEM
        """
        batch, _, H, W = rgb.shape
        device = rgb.device
        
        # 初始化为高斯噪声
        x = torch.randn(batch, self.out_channels, H, W, device=device)
        
        # 逐步去噪
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        
        for t_idx in timesteps:
            t = torch.full((batch,), t_idx, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # 预测噪声 (使用内部方法避免递归)
                noise_pred = self._denoise_step(x, t, condition_rgb=rgb)
                
                # 简单的去噪步骤
                alpha = 1.0 - (t_idx.float() / self.num_timesteps)
                x = alpha * x + (1 - alpha) * noise_pred
        
        return torch.clamp(x, 0, 1)


if __name__ == '__main__':
    print("Testing DiffusionDEM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = DiffusionDEM(out_channels=1, num_timesteps=1000).to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # 前向传播测试
    rgb = torch.randn(2, 3, 256, 256).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    with torch.no_grad():
        pred = model(x, t, condition_rgb=rgb)
        dem = model.inference(rgb, num_steps=10)
    
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input noise shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Generated DEM shape: {dem.shape}")
    print("Model forward OK.")
