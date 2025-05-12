import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

# 数据集归一化参数 (计算得到的ADE20K特定参数)
ADE20K_MEAN = [0.489, 0.465, 0.429]  # RGB通道均值
ADE20K_STD = [0.254, 0.251, 0.270]   # RGB通道标准差

class ADE20KDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=256):
        """
        ADE20K 数据集加载器 - 多标签分类版本
        Args:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'val'
            transform (callable, optional): 图像转换函数
            img_size (int): 图像大小
        """
        self.root_dir = root_dir
        self.split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.img_size = img_size
        
        # 获取图像路径
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'images', self.split, '*.jpg')))
        
        # 获取对应的标注路径
        self.annotation_paths = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            anno_path = os.path.join(root_dir, 'annotations', self.split, 
                                    filename.replace('.jpg', '.png'))
            self.annotation_paths.append(anno_path)
        
        # 读取对象信息（150个类别）
        self.object_info = {}
        self.idx_to_name = {0: "background"}  # 背景类
        # with open(os.path.join(root_dir, 'objectInfo150.txt'), 'r') as f:
        #     next(f)  # 跳过标题行
        #     for line in f:
        #         parts = line.strip().split('\t')
        #         if len(parts) >= 5:
        #             obj_id = int(parts[0])
        #             obj_name = parts[4]
        #             self.object_info[obj_id] = obj_name
        #             self.idx_to_name[obj_id] = obj_name
        import yaml
        with open(os.path.join(root_dir, 'ade_class.yaml'), 'r') as f:
            self.idx_to_name = yaml.safe_load(f)
        # 类别数量（包括背景类）
        self.num_classes = len(self.idx_to_name)
                    
        # 默认转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=ADE20K_MEAN, std=ADE20K_STD)
            ])
        
        print(f"找到 {len(self.image_paths)} 张 {split} 图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_class_names(self, label):
        """
        根据one-hot标签获取类名列表
        Args:
            label (torch.Tensor): one-hot标签
        Returns:
            list: 类名列表
        """
        class_ids = torch.nonzero(label, as_tuple=True)[0].tolist()
        return [self.idx_to_name.get(idx, f"未知类别-{idx}") for idx in class_ids]
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        Args:
            idx (int): 索引
        Returns:
            tuple: (image, label) 其中label是一个one-hot向量，表示图像中出现的所有类别
        """
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 加载标注掩码
        anno_path = self.annotation_paths[idx]
        mask = np.array(Image.open(anno_path))
        
        # 在ADE20K数据集中，掩码像素值直接表示类别ID (0-150)
        
        # 创建one-hot向量，表示图像中出现的所有类别
        # 0是背景类
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # 获取图像中出现的所有唯一类别，并设置对应的标签位
        unique_classes = np.unique(mask)
        
        for cls_id in unique_classes:
            if 0 <= cls_id < self.num_classes:
                label[cls_id] = 1.0
        
        # 应用图像变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloader(root_dir, split='train', batch_size=32, img_size=256, num_workers=4, shuffle=None):
    """
    创建数据加载器
    Args:
        root_dir (str): 数据集根目录
        split (str): 'train' 或 'val'
        batch_size (int): 批次大小
        img_size (int): 图像大小
        num_workers (int): 数据加载线程数
        shuffle (bool): 是否打乱数据顺序，默认训练集打乱，验证集不打乱
    Returns:
        tuple: (dataloader, dataset)
    """
    # 设置默认的shuffle行为
    if shuffle is None:
        shuffle = (split == 'train')
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE20K_MEAN, std=ADE20K_STD)
    ])
    
    # 创建数据集
    dataset = ADE20KDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        img_size=img_size
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset


def calculate_normalization_stats(root_dir, img_size=256, batch_size=64, num_workers=8):
    """
    计算数据集的归一化参数（均值和标准差）
    Args:
        root_dir (str): 数据集根目录
        img_size (int): 图像大小
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
    Returns:
        tuple: (mean, std) 归一化参数，每个都是包含3个通道值的列表
    """
    # 不使用归一化的转换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()  # 将图像转换为[0,1]范围的张量
    ])
    
    # 创建训练和验证数据集
    train_dataset = ADE20KDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        img_size=img_size
    )
    
    val_dataset = ADE20KDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
        img_size=img_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # 用于累积的变量
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    total_images = 0
    
    print("计算训练集和验证集的均值...")
    
    # 第一遍：计算均值
    # 处理训练集
    for images, _ in tqdm(train_loader, desc="训练集均值"):
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean_sum += torch.mean(images, dim=(0, 2)) * batch_size
        total_images += batch_size
    
    # 处理验证集
    for images, _ in tqdm(val_loader, desc="验证集均值"):
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean_sum += torch.mean(images, dim=(0, 2)) * batch_size
        total_images += batch_size
    
    # 计算总体均值
    mean = mean_sum / total_images
    
    print("计算训练集和验证集的标准差...")
    
    # 第二遍：计算标准差
    # 处理训练集
    for images, _ in tqdm(train_loader, desc="训练集标准差"):
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        std_sum += torch.sum(((images - mean.view(1, 3, 1)) ** 2), dim=(0, 2))
    
    # 处理验证集
    for images, _ in tqdm(val_loader, desc="验证集标准差"):
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        std_sum += torch.sum(((images - mean.view(1, 3, 1)) ** 2), dim=(0, 2))
    
    # 计算总体标准差
    std = torch.sqrt(std_sum / (total_images * img_size * img_size))
    
    # 转换为Python列表
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print(f"数据集归一化参数:")
    print(f"均值 (RGB): {mean_list}")
    print(f"标准差 (RGB): {std_list}")
    
    return mean_list, std_list


def test_dataloader():
    """测试训练数据加载器功能"""
    # 设置数据集路径
    dataset_root = 'ADEChallengeData2016'
    
    # 获取训练数据加载器
    train_loader, train_dataset = get_dataloader(
        root_dir=dataset_root,
        split='train',
        batch_size=32
    )
    
    # 只查看第一个样本，不使用批处理
    image, label = train_dataset[0]
    print(f"单个图像形状: {image.shape}")
    print(f"单个标签形状: {label.shape}")
    print(f"图像中出现的类别数量: {torch.sum(label).item()}")
    
    # 查看出现的类别
    class_names = train_dataset.get_class_names(label)
    print(f"出现的类别: {class_names}")
    
    # 获取一个批次
    for images, labels in train_loader:
        # 打印批次信息
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        
        # 计算每个样本中出现的类别数量
        class_counts = torch.sum(labels, dim=1)
        avg_classes_per_image = torch.mean(class_counts).item()
        
        print(f"每张图像平均类别数量: {avg_classes_per_image:.2f}")
        print(f"标签中的最小值: {labels.min()}")
        print(f"标签中的最大值: {labels.max()}")
        
        # 找出最常见的5个类别
        class_freq = torch.sum(labels, dim=0)
        top_classes = torch.argsort(class_freq, descending=True)[:5]
        print(f"批次中最常见的5个类别ID: {top_classes.tolist()}")
        
        # 打印最常见类别的名称
        top_class_names = [train_dataset.idx_to_name.get(idx.item(), "未知") for idx in top_classes]
        print(f"批次中最常见的5个类别名称: {top_class_names}")
        
        # 只打印一个批次信息
        break


def test_eval_dataloader():
    """测试评估数据加载器功能"""
    print("\n" + "="*50)
    print("测试评估数据加载器")
    print("="*50)
    
    # 设置数据集路径
    dataset_root = 'ADEChallengeData2016'
    
    # 获取验证数据加载器
    val_loader, val_dataset = get_dataloader(
        root_dir=dataset_root,
        split='val',
        batch_size=32
    )
    
    # 获取一个批次
    for images, labels in val_loader:
        # 打印批次信息
        print(f"评估图像批次形状: {images.shape}")
        print(f"评估标签批次形状: {labels.shape}")
        
        # 计算每个样本中出现的类别数量
        class_counts = torch.sum(labels, dim=1)
        avg_classes_per_image = torch.mean(class_counts).item()
        
        print(f"评估每张图像平均类别数量: {avg_classes_per_image:.2f}")
        
        # 统计每个类别在批次中出现的频率
        class_freq = torch.sum(labels, dim=0)
        
        # 找出在批次中出现的非零类别数量
        non_zero_classes = torch.sum(class_freq > 0).item()
        print(f"批次中出现的非零类别数量: {non_zero_classes}")
        
        # 只打印一个批次信息
        break


def test_normalization_stats():
    """测试计算归一化参数功能"""
    print("\n" + "="*50)
    print("计算数据集归一化参数")
    print("="*50)
    
    # 设置数据集路径
    dataset_root = 'ADEChallengeData2016'
    
    # 计算归一化参数
    # 注意：这个过程可能需要较长时间
    mean, std = calculate_normalization_stats(
        root_dir=dataset_root,
        img_size=256,
        batch_size=64
    )
    
    print("\n计算结果:")
    print(f"均值 (RGB): {mean}")
    print(f"标准差 (RGB): {std}")


if __name__ == "__main__":
    test_dataloader()
    test_eval_dataloader()
    # 如果要计算归一化参数，取消下面一行的注释
    # test_normalization_stats()
