#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自编码器异常检测示例脚本
展示如何使用自编码器进行异常检测，检测与训练数据分布不同的异常样本
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import random

from src.models import get_model
from src.data_utils import get_data_loaders


def create_anomaly_dataset(dataset, normal_classes, anomaly_ratio=0.05, seed=42):
    """
    创建包含正常样本和异常样本的数据集
    
    Args:
        dataset: 原始数据集
        normal_classes: 正常类别的列表
        anomaly_ratio: 异常样本在测试集中的比例
        seed: 随机种子
    
    Returns:
        tuple: (训练数据加载器, 测试数据加载器, 测试集标签)
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 将数据集分为正常类别和异常类别
    normal_indices = []
    anomaly_indices = []
    
    for idx, (_, label) in enumerate(dataset):
        if label in normal_classes:
            normal_indices.append(idx)
        else:
            anomaly_indices.append(idx)
    
    # 打乱索引顺序
    random.shuffle(normal_indices)
    random.shuffle(anomaly_indices)
    
    # 划分训练集和测试集
    train_size = int(len(normal_indices) * 0.7)
    train_indices = normal_indices[:train_size]
    
    # 确定测试集的正常样本数量和异常样本数量
    test_normal_size = len(normal_indices) - train_size
    test_anomaly_size = int(test_normal_size * anomaly_ratio / (1 - anomaly_ratio))
    test_anomaly_size = min(test_anomaly_size, len(anomaly_indices))
    
    test_indices = normal_indices[train_size:train_size + test_normal_size]
    test_indices.extend(anomaly_indices[:test_anomaly_size])
    
    # 创建测试集标签（0表示正常，1表示异常）
    test_labels = np.zeros(len(test_indices), dtype=np.int)
    test_labels[test_normal_size:] = 1
    
    # 打乱测试集顺序
    combined = list(zip(test_indices, test_labels))
    random.shuffle(combined)
    test_indices, test_labels = zip(*combined)
    
    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"训练集大小: {len(train_dataset)} (全部正常样本)")
    print(f"测试集大小: {len(test_dataset)} (正常: {test_normal_size}, 异常: {test_anomaly_size})")
    
    return train_dataset, test_dataset, np.array(test_labels)


def compute_reconstruction_errors(model, data_loader, device):
    """
    计算重构误差
    
    Args:
        model: 自编码器模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        numpy.ndarray: 重构误差数组
    """
    model.eval()
    reconstruction_errors = []
    original_images = []
    reconstructed_images = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="计算重构误差"):
            inputs = inputs.to(device)
            
            # 对于VAE模型，只取重构结果
            if hasattr(model, 'reparameterize'):
                outputs, _, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            # 计算每个样本的重构误差（均方误差）
            batch_errors = ((outputs - inputs) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            
            reconstruction_errors.extend(batch_errors)
            
            # 保存一些图像用于可视化
            if len(original_images) < 20:
                original_images.extend(inputs.cpu().numpy()[:20 - len(original_images)])
                reconstructed_images.extend(outputs.cpu().numpy()[:20 - len(reconstructed_images)])
    
    return np.array(reconstruction_errors), original_images, reconstructed_images


def evaluate_anomaly_detection(reconstruction_errors, true_labels, save_dir=None):
    """
    评估异常检测性能
    
    Args:
        reconstruction_errors: 重构误差数组
        true_labels: 真实标签数组 (0=正常, 1=异常)
        save_dir: 保存结果的目录
    
    Returns:
        dict: 包含评估指标的字典
    """
    # 计算ROC和PR曲线
    fpr, tpr, thresholds_roc = roc_curve(true_labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds_pr = precision_recall_curve(true_labels, reconstruction_errors)
    pr_auc = average_precision_score(true_labels, reconstruction_errors)
    
    # 寻找最佳阈值（使用Youden指数）
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[optimal_idx]
    
    # 使用最佳阈值进行预测
    predictions = (reconstruction_errors >= optimal_threshold).astype(int)
    
    # 计算各项指标
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    accuracy = (tp + tn) / len(true_labels)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    # 绘制ROC曲线
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    
    if save_dir:
        plt.save***REMOVED***g(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # 绘制PR曲线
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR 曲线 (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线 (PR)')
    plt.legend(loc="lower left")
    
    if save_dir:
        plt.save***REMOVED***g(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()
    
    # 绘制重构误差分布
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    plt.hist(reconstruction_errors[true_labels == 0], bins=50, alpha=0.5, label='正常样本')
    plt.hist(reconstruction_errors[true_labels == 1], bins=50, alpha=0.5, label='异常样本')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'阈值 = {optimal_threshold:.3f}')
    plt.xlabel('重构误差')
    plt.ylabel('样本数量')
    plt.title('正常样本和异常样本的重构误差分布')
    plt.legend()
    
    if save_dir:
        plt.save***REMOVED***g(os.path.join(save_dir, 'error_distribution.png'))
    plt.close()
    
    # 返回评估指标
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    return metrics


def visualize_samples(original_images, reconstructed_images, errors, save_path=None):
    """
    可视化原始图像和重构图像
    
    Args:
        original_images: 原始图像列表
        reconstructed_images: 重构图像列表
        errors: 重构误差列表
        save_path: 保存路径
    """
    n = min(10, len(original_images))
    plt.***REMOVED***gure(***REMOVED***gsize=(20, 4))
    
    for i in range(n):
        # 原始图像
        plt.subplot(2, n, i + 1)
        if original_images[0].shape[0] == 1:  # 灰度图
            plt.imshow(original_images[i][0], cmap='gray')
        else:  # 彩色图
            plt.imshow(np.transpose(original_images[i], (1, 2, 0)))
        plt.title("原始图像")
        plt.axis('off')
        
        # 重构图像
        plt.subplot(2, n, i + n + 1)
        if reconstructed_images[0].shape[0] == 1:  # 灰度图
            plt.imshow(reconstructed_images[i][0], cmap='gray')
        else:  # 彩色图
            plt.imshow(np.transpose(reconstructed_images[i], (1, 2, 0)))
        plt.title(f"重构图像\n误差: {errors[i]:.4f}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.save***REMOVED***g(save_path)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='自编码器异常检测示例')
    parser.add_argument('--model-type', type=str, default='vanilla_ae', 
                        choices=['vanilla_ae', 'conv_ae', 'vae'], 
                        help='自编码器模型类型')
    parser.add_argument('--latent-dim', type=int, default=128, 
                        help='潜在空间维度（仅用于vanilla_ae和vae）')
    parser.add_argument('--normal-classes', type=int, nargs='+', default=[0, 1, 2, 3, 4], 
                        help='视为正常的类别（剩余类别视为异常）')
    parser.add_argument('--anomaly-ratio', type=float, default=0.1, 
                        help='测试集中异常样本的比例')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./output', 
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备 (cuda|cpu)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    save_dir = os.path.join(args.output_dir, 'results', f"{args.model_type}_anomaly")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载CIFAR-10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    cifar10_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    
    # 创建异常检测数据集
    print(f"正常类别: {args.normal_classes}")
    print(f"异常样本比例: {args.anomaly_ratio}")
    train_dataset, test_dataset, test_labels = create_anomaly_dataset(
        cifar10_dataset, 
        normal_classes=args.normal_classes,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    print(f"创建{args.model_type}模型...")
    if args.model_type in ['vanilla_ae', 'vae']:
        model = get_model(args.model_type, in_channels=3, latent_dim=args.latent_dim)
    else:
        model = get_model(args.model_type, in_channels=3)
    model = model.to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 训练模型
    print(f"开始训练{args.model_type}模型...")
    model.train()
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            data = data.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 对于VAE模型，处理多个返回值
            if args.model_type == 'vae':
                outputs, mu, logvar = model(data)
                # VAE损失 = 重构损失 + KL散度
                recon_loss = criterion(outputs, data)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss
            else:
                outputs = model(data)
                loss = criterion(outputs, data)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 打印训练信息
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, 平均损失: {avg_loss:.6f}")
    
    # 保存模型
    model_path = os.path.join(args.output_dir, 'models', f"{args.model_type}_anomaly.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'loss': avg_loss
    }, model_path)
    print(f"模型已保存到 {model_path}")
    
    # 计算重构误差
    print("计算测试集的重构误差...")
    test_errors, original_images, reconstructed_images = compute_reconstruction_errors(model, test_loader, device)
    
    # 评估异常检测性能
    print("评估异常检测性能...")
    metrics = evaluate_anomaly_detection(test_errors, test_labels, save_dir)
    
    # 打印评估指标
    print("\n异常检测评估指标:")
    print(f"最佳阈值: {metrics['threshold']:.6f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # 可视化样本
    visualize_samples(
        original_images, 
        reconstructed_images, 
        test_errors[:len(original_images)],
        save_path=os.path.join(save_dir, 'sample_reconstructions.png')
    )
    
    print(f"实验结果已保存到 {save_dir}")


if __name__ == "__main__":
    main() 