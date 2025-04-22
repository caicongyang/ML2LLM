#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自编码器特征提取示例脚本
展示如何使用自编码器作为特征提取器，并用提取的特征训练简单分类器
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassi***REMOVED***er
from sklearn.metrics import accuracy_score, classi***REMOVED***cation_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from src.models import get_model
from src.data_utils import get_data_loaders


def extract_features(model, data_loader, device, latent_dim):
    """
    使用自编码器提取特征
    
    Args:
        model: 训练好的自编码器模型
        data_loader: 数据加载器
        device: 计算设备
        latent_dim: 潜在空间维度
        
    Returns:
        tuple: (特征, 标签)
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="提取特征"):
            inputs = inputs.to(device)
            
            # 根据模型类型提取特征
            if hasattr(model, 'encode'):
                # VAE或标准自编码器
                if isinstance(model.encode, torch.nn.Module):
                    z = model.encode(inputs)
                    # 处理VAE的情况，它返回均值和对数方差
                    if isinstance(z, tuple):
                        z = z[0]  # 使用均值作为特征
                else:
                    # 使用encode方法
                    z = model.encode(inputs)
            else:
                # 卷积自编码器，使用编码器部分
                z = model.encoder(inputs)
                z = z.view(z.size(0), -1)  # 将特征展平
            
            features.append(z.cpu().numpy())
            labels.append(targets.numpy())
    
    # 合并所有批次的特征和标签
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels


def train_classi***REMOVED***er(X_train, y_train, X_val, y_val, classi***REMOVED***er_type='svm'):
    """
    训练分类器
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        classi***REMOVED***er_type: 分类器类型，'svm'或'rf'
        
    Returns:
        训练好的分类器
    """
    print(f"训练{classi***REMOVED***er_type}分类器...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.***REMOVED***t_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 选择并训练分类器
    if classi***REMOVED***er_type == 'svm':
        classi***REMOVED***er = SVC(kernel='rbf', C=10, gamma='scale')
        classi***REMOVED***er.***REMOVED***t(X_train_scaled, y_train)
    else:  # Random Forest
        classi***REMOVED***er = RandomForestClassi***REMOVED***er(n_estimators=100, random_state=42)
        classi***REMOVED***er.***REMOVED***t(X_train_scaled, y_train)
    
    # 评估分类器
    train_pred = classi***REMOVED***er.predict(X_train_scaled)
    val_pred = classi***REMOVED***er.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"训练准确率: {train_acc:.4f}")
    print(f"验证准确率: {val_acc:.4f}")
    
    # 返回训练好的分类器和scaler
    return classi***REMOVED***er, scaler


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.tight_layout()
        plt.save***REMOVED***g(save_path)
        plt.close()
        print(f"混淆矩阵已保存到 {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='自编码器特征提取示例')
    parser.add_argument('--model-type', type=str, default='vanilla_ae', 
                        choices=['vanilla_ae', 'conv_ae', 'vae'], 
                        help='自编码器模型类型')
    parser.add_argument('--latent-dim', type=int, default=128, 
                        help='潜在空间维度')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='预训练模型路径')
    parser.add_argument('--classi***REMOVED***er', type=str, default='svm', 
                        choices=['svm', 'rf'], 
                        help='分类器类型')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./output', 
                        help='输出目录')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备 (cuda|cpu)')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_cifar=True
    )
    
    # 创建模型
    print(f"创建{args.model_type}模型...")
    if args.model_type in ['vanilla_ae', 'vae']:
        model = get_model(args.model_type, in_channels=3, latent_dim=args.latent_dim)
    else:
        model = get_model(args.model_type, in_channels=3)
    model = model.to(device)
    
    # 加载预训练模型
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.output_dir, 'models', f"{args.model_type}_best.pth")
    
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"未找到预训练模型: {model_path}")
        print("请先使用 run_autoencoder.py 训练自编码器模型")
        return
    
    # 提取特征
    print("从训练集提取特征...")
    X_train, y_train = extract_features(model, train_loader, device, args.latent_dim)
    
    print("从验证集提取特征...")
    X_val, y_val = extract_features(model, val_loader, device, args.latent_dim)
    
    print("从测试集提取特征...")
    X_test, y_test = extract_features(model, test_loader, device, args.latent_dim)
    
    print(f"特征形状: {X_train.shape}")
    
    # 训练分类器
    classi***REMOVED***er, scaler = train_classi***REMOVED***er(X_train, y_train, X_val, y_val, classi***REMOVED***er_type=args.classi***REMOVED***er)
    
    # 评估测试集
    X_test_scaled = scaler.transform(X_test)
    test_pred = classi***REMOVED***er.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n测试准确率: {test_acc:.4f}")
    print("\n分类报告:")
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    print(classi***REMOVED***cation_report(y_test, test_pred, target_names=class_names))
    
    # 绘制混淆矩阵
    save_dir = os.path.join(args.output_dir, 'results', f"{args.model_type}_{args.classi***REMOVED***er}")
    os.makedirs(save_dir, exist_ok=True)
    
    plot_confusion_matrix(
        y_test, test_pred, class_names, 
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # 比较与原始数据的分类效果
    print("\n比较与原始数据的分类效果...")
    
    # 创建一个简单的MLP分类器
    class SimpleClassi***REMOVED***er(nn.Module):
        def __init__(self, input_dim):
            super(SimpleClassi***REMOVED***er, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # 将提取的特征转换为PyTorch数据集
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader_features = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader_features = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader_features = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 训练MLP分类器
    print("使用自编码器特征训练MLP分类器...")
    mlp = SimpleClassi***REMOVED***er(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    
    # 训练5个轮次
    for epoch in range(5):
        mlp.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader_features:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 验证
        mlp.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader_features:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = mlp(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        print(f"Epoch {epoch+1}/5 - 训练损失: {train_loss/len(train_loader_features):.4f}, "
              f"训练准确率: {100.*correct/total:.2f}%, "
              f"验证损失: {val_loss/len(val_loader_features):.4f}, "
              f"验证准确率: {100.*val_correct/val_total:.2f}%")
    
    # 评估MLP分类器
    mlp.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader_features:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = mlp(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    print(f"\nMLP分类器测试准确率: {100.*test_correct/test_total:.2f}%")
    print(f"对比: 使用{args.classi***REMOVED***er}分类器的测试准确率为 {test_acc*100:.2f}%")
    
    print(f"特征提取和分类实验完成! 结果已保存到 {save_dir}")


if __name__ == "__main__":
    main() 