import os
import torch
import rasterio
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RemoteSensingDataset(Dataset):
    """
    遥感图像数据集
    用于加载RGB图像和DEM深度图
    """
    def __init__(self, root_dir, transform=None, mode='train', indices=None):
        """
        Args:
            root_dir (str): 数据集根目录
            transform: albumentations 的数据增强
            mode (str): 'train' 或 'test'
            indices (list): 要使用的样本索引（用于train/val划分）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # RGB和DEM文件目录
        self.rgb_dir = os.path.join(root_dir, 'RGB')
        self.dem_dir = os.path.join(root_dir, 'DEM')
        
        # 获取所有RGB文件名
        if os.path.exists(self.rgb_dir):
            self.filenames = [f for f in os.listdir(self.rgb_dir) if f.endswith('.png')]
        else:
            self.filenames = []
            print(f"警告: RGB目录不存在: {self.rgb_dir}")
        
        # 如果指定了indices，则只使用这些索引
        if indices is not None:
            self.filenames = [self.filenames[i] for i in indices]
        
        # DEM归一化参数（需要根据实际数据调整）
        self.dem_max = 255.0  # 假设最大海拔
        self.dem_min = 0.0    # 假设最小海拔
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        rgb_path = os.path.join(self.rgb_dir, fname)
        
        # 对应的DEM文件（后缀从.png改为.tif）
        dem_fname = fname.replace('.png', '.tif')
        dem_path = os.path.join(self.dem_dir, dem_fname)
        
        # 1. 读取RGB图像（使用OpenCV读取为BGR，转换为RGB）
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB图像不存在: {rgb_path}")
        
        image = cv2.imread(rgb_path)
        if image is None:
            raise ValueError(f"无法读取RGB图像: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 读取DEM图像（使用rasterio读取地理信息数据）
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM图像不存在: {dem_path}")
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # 读取第一通道
            # 处理可能存在的无效值
            dem = np.nan_to_num(dem, nan=0.0)
        
        # 3. DEM数据归一化 -> [0, 1]
        dem = (dem - self.dem_min) / (self.dem_max - self.dem_min + 1e-8)
        dem = np.clip(dem, 0, 1)  # 截断超出范围的值
        
        # 4. 数据增强（Augmentation）
        if self.transform:
            augmented = self.transform(image=image, mask=dem)
            image = augmented['image']
            dem = augmented['mask']
        
        # 5. 格式调整
        # 如果transform中包含ToTensorV2，image已经是Tensor
        # DEM可能是Tensor或numpy数组
        if isinstance(dem, torch.Tensor):
            dem = dem.unsqueeze(0).float()  # (1, H, W)
        else:
            dem = torch.from_numpy(dem).unsqueeze(0).float()
        
        # 如果image不是Tensor（当transform为None时），转换为Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, dem


def get_transforms():
    """
    训练集数据增强
    包含随机旋转、翻转等增强操作
    """
    return A.Compose([
        A.Resize(512, 512),  # 统一尺寸
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet标准
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_val_transforms():
    """
    验证/测试集数据处理
    不进行数据增强，只做归一化和resize
    """
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


if __name__ == '__main__':
    # 测试数据集和数据加载
    print("数据加载模块测试...")
    
    # 创建数据集
    dataset = RemoteSensingDataset(
        root_dir="../../ImageToDEM/singleRGBNormalization",
        transform=get_transforms(),
        mode='train'
    )
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        image, dem = dataset[0]
        print(f"图像形状: {image.shape}, DEM形状: {dem.shape}")
        print("数据加载成功！")
    
    # 数据划分示例
    train_root = "../ImageToDEM/singleRGBNormalization"
    if os.path.exists(train_root):
        base_dataset = RemoteSensingDataset(train_root, transform=None, mode='train')
        indices = list(range(len(base_dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
        
        train_dataset = RemoteSensingDataset(train_root, transform=get_transforms(), mode='train', indices=train_idx)
        val_dataset = RemoteSensingDataset(train_root, transform=get_val_transforms(), mode='train', indices=val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
        
        print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
        print(f"DataLoader - 训练: {len(train_loader)}, 验证: {len(val_loader)}")

# ============== 快速创建数据加载器的工厂函数 ==============
def create_dataloaders(train_root, test_root=None, batch_size=16, num_workers=4):
    """快速创建训练/验证/测试数据加载器"""
    base_dataset = RemoteSensingDataset(train_root, transform=None, mode='train')
    indices = list(range(len(base_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    
    train_dataset = RemoteSensingDataset(train_root, transform=get_transforms(), mode='train', indices=train_idx)
    val_dataset = RemoteSensingDataset(train_root, transform=get_val_transforms(), mode='train', indices=val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    result = {'train': train_loader, 'val': val_loader}
    
    if test_root and os.path.exists(test_root):
        test_dataset = RemoteSensingDataset(test_root, transform=get_val_transforms(), mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        result['test'] = test_loader
    
    return result