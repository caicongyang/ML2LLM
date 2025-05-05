#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块
提供数据集加载、预处理和数据加载器创建功能
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_datasets(data_dir=None, val_split=0.2, augment=True):
    """
    获取CIFAR10数据集
    
    Args:
        data_dir (str): 数据存储目录
        val_split (float): 验证集比例
        augment (bool): 是否进行数据增强
        
    Returns:
        tuple: (训练集, 验证集, 测试集)
    """
    # 如果未指定数据目录，则使用auto-vision/data
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义数据预处理
    if augment:
        # 带数据增强的训练集变换
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else:
        # 不带数据增强的训练集变换
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    # 验证集和测试集变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载完整训练集
    full_train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试集
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 分割训练集和验证集
    val_size = int(len(full_train_set) * val_split)
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子，确保结果可复现
    )
    
    # 为验证集重新设置变换（不使用数据增强）
    val_set.dataset.transform = test_transform
    
    return train_set, val_set, test_set


def get_dataloaders(train_set, val_set, test_set, batch_size=128, num_workers=4):
    """
    创建数据加载器
    
    Args:
        train_set: 训练数据集
        val_set: 验证数据集
        test_set: 测试数据集
        batch_size (int): 批次大小
        num_workers (int): 加载数据的工作线程数
        
    Returns:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info():
    """
    获取CIFAR10数据集的信息
    
    Returns:
        dict: 包含数据集信息的字典
    """
    return {
        'name': 'CIFAR10',
        'num_classes': 10,
        'input_channels': 3,
        'input_size': (32, 32),
        'classes': [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    }


def get_cifar10_loaders(batch_size=128, augment=True, val_split=0.2, num_workers=4, data_dir=None):
    """
    便捷函数：一站式获取CIFAR10数据加载器
    
    Args:
        batch_size (int): 批次大小
        augment (bool): 是否进行数据增强
        val_split (float): 验证集比例
        num_workers (int): 加载数据的工作线程数
        data_dir (str): 数据存储目录
        
    Returns:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    # 如果未指定数据目录，则使用auto-vision/data
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
    
    # 获取数据集
    train_set, val_set, test_set = get_cifar10_datasets(
        data_dir=data_dir,
        val_split=val_split,
        augment=augment
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(
        train_set, val_set, test_set,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载模块
    print("测试数据加载模块...")
    
    # 获取数据集信息
    dataset_info = get_dataset_info()
    print(f"数据集: {dataset_info['name']}")
    print(f"类别数量: {dataset_info['num_classes']}")
    print(f"输入通道数: {dataset_info['input_channels']}")
    print(f"输入尺寸: {dataset_info['input_size']}")
    print(f"类别列表: {dataset_info['classes']}")
    
    # 获取默认的数据目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # 加载数据集
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=2, data_dir=data_dir)
    
    # 打印数据集大小
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 获取一批数据样本
    images, labels = next(iter(train_loader))
    print(f"图像批次形状: {images.shape}")
    print(f"标签批次形状: {labels.shape}")
    
    # 打印第一批的类别统计
    label_counts = {}
    for label in labels:
        label = label.item()
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    
    print("第一批中的类别分布:")
    for label, count in label_counts.items():
        print(f"  类别 {label} ({dataset_info['classes'][label]}): {count}个")
    
    print("数据加载模块测试完成!") 