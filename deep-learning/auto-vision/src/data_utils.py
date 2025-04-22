#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理模块
提供CIFAR-10数据集加载和预处理功能
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def load_cifar10(data_dir='./data', val_size=0.1, apply_augmentation=True):
    """
    加载CIFAR-10数据集，并分割为训练集、验证集和测试集
    
    Args:
        data_dir (str): 数据存储目录
        val_size (float): 验证集比例
        apply_augmentation (bool): 是否应用数据增强
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names)
    """
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义数据变换
    # 标准化参数 - CIFAR-10的均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # 训练集变换
    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # 测试集变换 - 不包含增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 加载训练集
    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 将部分训练集数据分割为验证集
    val_size = int(len(train_full) * val_size)
    train_size = len(train_full) - val_size
    
    train_dataset, val_dataset = random_split(
        train_full, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子以确保可重复性
    )
    
    # 获取类别名称
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, class_names


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=4):
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size (int): 批量大小
        num_workers (int): 数据加载线程数
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 创建训练集加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 创建验证集加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 创建测试集加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_sample_data(data_loader, num_samples=1, device='cpu'):
    """
    从数据加载器中获取样本数据
    
    Args:
        data_loader: 数据加载器
        num_samples (int): 样本数量
        device (str): 设备类型 ('cpu' 或 'cuda')
    
    Returns:
        tuple: (data, targets)
    """
    data_iter = iter(data_loader)
    data, targets = next(data_iter)
    
    # 仅保留指定数量的样本
    if num_samples > 0 and num_samples < data.shape[0]:
        data = data[:num_samples]
        targets = targets[:num_samples]
    
    # 将数据转移到指定设备
    data = data.to(device)
    targets = targets.to(device)
    
    return data, targets


def create_noisy_data(data, noise_factor=0.2):
    """
    为自编码器训练创建带噪声数据
    
    Args:
        data (torch.Tensor): 输入数据
        noise_factor (float): 噪声系数
    
    Returns:
        torch.Tensor: 带噪声的数据
    """
    # 创建噪声
    noise = torch.randn_like(data) * noise_factor
    
    # 添加噪声到数据
    noisy_data = data + noise
    
    # 确保数据值在[0,1]范围内
    noisy_data = torch.clamp(noisy_data, 0., 1.)
    
    return noisy_data


if __name__ == "__main__":
    # 测试数据加载功能
    print("测试数据加载功能...")
    train_dataset, val_dataset, test_dataset, class_names = load_cifar10()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # 获取样本数据
    data, targets = get_sample_data(train_loader, num_samples=5)
    
    print(f"样本数据形状: {data.shape}")
    print(f"样本标签: {targets}")
    print(f"标签类别: {[class_names[t] for t in targets]}")
    
    # 创建带噪声数据，用于自编码器测试
    noisy_data = create_noisy_data(data)
    print(f"带噪声数据形状: {noisy_data.shape}")
    
    print("数据加载测试完成!") 