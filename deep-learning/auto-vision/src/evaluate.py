"""
模型评估模块
包含以下功能：
1. 分类模型评估（准确率、精确率、召回率、F1得分）
2. 自编码器模型评估（重建误差、异常检测）
3. 混淆矩阵计算和可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classi***REMOVED***cation_report, roc_curve, auc
import seaborn as sns
from tqdm import tqdm


def evaluate_classi***REMOVED***er(model, test_loader, device):
    """
    评估分类器模型
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        包含各种评估指标的字典
    """
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0
    
    # 在不计算梯度的情况下进行前向传播
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # 累加损失
            test_loss += loss.item() * inputs.size(0)
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            # 收集所有预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算总损失
    test_loss = test_loss / len(test_loader.dataset)
    
    # 计算各种评估指标
    metrics = {
        'loss': test_loss,
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='macro'),
        'recall': recall_score(all_targets, all_preds, average='macro'),
        'f1': f1_score(all_targets, all_preds, average='macro'),
        'all_preds': all_preds,
        'all_targets': all_targets
    }
    
    return metrics


def evaluate_autoencoder(model, test_loader, device, threshold=None):
    """
    评估自编码器模型
    
    Args:
        model: 待评估的自编码器模型
        test_loader: 测试数据加载器
        device: 计算设备
        threshold: 异常检测阈值，若为None则自动计算
        
    Returns:
        包含重建误差和异常检测结果的字典
    """
    model.eval()
    reconstruction_errors = []
    original_images = []
    reconstructed_images = []
    
    # 在不计算梯度的情况下进行前向传播
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Evaluating Autoencoder"):
            inputs = inputs.to(device)
            
            # 前向传播获取重建图像
            outputs = model(inputs)
            
            # 计算每个样本的重建误差
            batch_errors = F.mse_loss(outputs, inputs, reduction='none').mean([1, 2, 3])
            
            # 收集重建误差
            reconstruction_errors.extend(batch_errors.cpu().numpy())
            
            # 收集原始图像和重建图像
            original_images.extend(inputs.cpu())
            reconstructed_images.extend(outputs.cpu())
    
    # 转换为NumPy数组
    reconstruction_errors = np.array(reconstruction_errors)
    
    # 如果没有提供阈值，则使用重建误差的百分位数作为阈值
    if threshold is None:
        threshold = np.percentile(reconstruction_errors, 95)
    
    # 根据阈值识别异常
    anomalies = reconstruction_errors > threshold
    
    # 整理结果
    results = {
        'reconstruction_errors': reconstruction_errors,
        'threshold': threshold,
        'anomalies': anomalies,
        'anomaly_ratio': np.mean(anomalies),
        'mean_error': np.mean(reconstruction_errors),
        'max_error': np.max(reconstruction_errors),
        'min_error': np.min(reconstruction_errors),
        'original_images': original_images[:20],  # 仅保存部分图像用于可视化
        'reconstructed_images': reconstructed_images[:20]
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图像
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # 保存或显示图像
    if save_path:
        plt.tight_layout()
        plt.save***REMOVED***g(save_path)
        plt.close()
        print(f"混淆矩阵已保存到 {save_path}")
    else:
        plt.show()


def plot_autoencoder_results(original_images, reconstructed_images, reconstruction_errors, save_dir=None):
    """
    绘制自编码器结果
    
    Args:
        original_images: 原始图像列表
        reconstructed_images: 重建图像列表
        reconstruction_errors: 重建误差列表
        save_dir: 保存目录，若为None则显示图像而不保存
    """
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 绘制原始图像和重建图像对比
    n = min(10, len(original_images))
    plt.***REMOVED***gure(***REMOVED***gsize=(20, 4))
    
    for i in range(n):
        # 原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].squeeze().numpy(), cmap='gray')
        plt.title(f"Original")
        plt.axis('off')
        
        # 重建图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
        plt.title(f"Reconstructed\nError: {reconstruction_errors[i]:.4f}")
        plt.axis('off')
    
    # 保存或显示图像
    if save_dir:
        plt.tight_layout()
        plt.save***REMOVED***g(os.path.join(save_dir, 'reconstruction_comparison.png'))
        plt.close()
        print(f"重建结果对比图已保存到 {save_dir}")
    else:
        plt.tight_layout()
        plt.show()
    
    # 绘制重建误差分布
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
    sns.histplot(reconstruction_errors, kde=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    
    # 保存或显示图像
    if save_dir:
        plt.tight_layout()
        plt.save***REMOVED***g(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()
        print(f"重建误差分布图已保存到 {save_dir}")
    else:
        plt.tight_layout()
        plt.show()


def print_classi***REMOVED***cation_report(y_true, y_pred, class_names):
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    """
    report = classi***REMOVED***cation_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n分类报告:")
    print(report)


def get_evaluation_function(model_name):
    """
    根据模型名称获取对应的评估函数
    
    Args:
        model_name: 模型名称
        
    Returns:
        对应的评估函数
    """
    if model_name == 'autoencoder':
        return evaluate_autoencoder
    else:
        return evaluate_classi***REMOVED***er 