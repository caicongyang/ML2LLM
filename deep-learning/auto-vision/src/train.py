#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练模块
提供分类器和自编码器模型的训练功能
"""

import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid


def train_classi***REMOVED***er(model, train_loader, val_loader, args):
    """
    训练分类模型
    
    Args:
        model (nn.Module): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        args (dict): 训练参数，包括:
            - epochs (int): 训练轮数
            - lr (float): 学习率
            - weight_decay (float): 权重衰减
            - device (torch.device): 训练设备
            - log_dir (str): 日志目录
            - save_dir (str): 模型保存目录
            - model_name (str): 模型名称
            
    Returns:
        dict: 包含训练历史记录的字典
    """
    device = args["device"]
    epochs = args["epochs"]
    log_dir = args["log_dir"]
    save_dir = args["save_dir"]
    model_name = args["model_name"]
    
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 日志记录器
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0
    }
    
    # 开始训练
    print(f"开始训练 {model_name}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {train_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*train_correct/train_total:.2f}%")
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印结果
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f"模型已保存: {model_name}_best.pth")
    
    # 训练完成，保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(save_dir, f"{model_name}_***REMOVED***nal.pth"))
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总用时: {total_time:.2f}秒")
    print(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
    
    return history


def train_autoencoder(model, train_loader, val_loader, args):
    """
    训练自编码器模型
    
    Args:
        model (nn.Module): 待训练的自编码器模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        args (dict): 训练参数，包括:
            - epochs (int): 训练轮数
            - lr (float): 学习率
            - weight_decay (float): 权重衰减
            - device (torch.device): 训练设备
            - log_dir (str): 日志目录
            - save_dir (str): 模型保存目录
            - model_name (str): 模型名称
            - is_vae (bool): 是否为变分自编码器
            
    Returns:
        dict: 包含训练历史记录的字典
    """
    device = args["device"]
    epochs = args["epochs"]
    log_dir = args["log_dir"]
    save_dir = args["save_dir"]
    model_name = args["model_name"]
    is_vae = args.get("is_vae", False)
    
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 日志记录器
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    
    # 损失函数和优化器
    reconstruction_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }
    
    # 开始训练
    print(f"开始训练 {model_name}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kld_loss = 0.0
        
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            if is_vae:
                # 变分自编码器
                reconstructed, mu, logvar = model(inputs)
                
                # 重建损失
                recon_loss = reconstruction_criterion(reconstructed, inputs)
                
                # KL散度损失
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_loss = kld_loss / inputs.size(0)  # 归一化
                
                # 总损失 = 重建损失 + β * KL散度损失
                beta = 0.1  # 可调整的超参数
                loss = recon_loss + beta * kld_loss
                
                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()
            else:
                # 普通自编码器
                reconstructed = model(inputs)
                loss = reconstruction_criterion(reconstructed, inputs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                if is_vae:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {train_loss/(batch_idx+1):.4f} "
                          f"Recon: {train_recon_loss/(batch_idx+1):.4f} "
                          f"KLD: {train_kld_loss/(batch_idx+1):.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {train_loss/(batch_idx+1):.4f}")
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kld_loss = 0.0
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                
                # 前向传播
                if is_vae:
                    # 变分自编码器
                    reconstructed, mu, logvar = model(inputs)
                    
                    # 重建损失
                    recon_loss = reconstruction_criterion(reconstructed, inputs)
                    
                    # KL散度损失
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kld_loss = kld_loss / inputs.size(0)  # 归一化
                    
                    # 总损失 = 重建损失 + β * KL散度损失
                    beta = 0.1  # 可调整的超参数
                    loss = recon_loss + beta * kld_loss
                    
                    val_recon_loss += recon_loss.item()
                    val_kld_loss += kld_loss.item()
                else:
                    # 普通自编码器
                    reconstructed = model(inputs)
                    loss = reconstruction_criterion(reconstructed, inputs)
                
                # 统计
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        if is_vae:
            writer.add_scalar('Loss/train_reconstruction', train_recon_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/train_kld', train_kld_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/val_reconstruction', val_recon_loss / len(val_loader), epoch)
            writer.add_scalar('Loss/val_kld', val_kld_loss / len(val_loader), epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 添加重建图像样本到TensorBoard
        if epoch % 5 == 0:
            with torch.no_grad():
                # 获取一批数据
                test_inputs, _ = next(iter(val_loader))
                test_inputs = test_inputs.to(device)
                
                # 获取重建图像
                if is_vae:
                    reconstructed, _, _ = model(test_inputs)
                else:
                    reconstructed = model(test_inputs)
                
                # 创建对比图
                comparison = torch.cat([test_inputs[:8], reconstructed[:8]])
                grid = make_grid(comparison, nrow=8, normalize=True, scale_each=True)
                writer.add_image('Reconstruction', grid, epoch)
        
        # 打印结果
        if is_vae:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} "
                  f"(Recon: {train_recon_loss/len(train_loader):.4f}, KLD: {train_kld_loss/len(train_loader):.4f}) "
                  f"Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f"模型已保存: {model_name}_best.pth")
    
    # 训练完成，保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(save_dir, f"{model_name}_***REMOVED***nal.pth"))
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总用时: {total_time:.2f}秒")
    print(f"最佳验证损失: {history['best_val_loss']:.4f}")
    
    return history


# 工具函数：生成图像网格
def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """
    创建图像网格
    简化版的torchvision.utils.make_grid函数
    """
    from torchvision.utils import make_grid as tv_make_grid
    return tv_make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, 
                        range=range, scale_each=scale_each, pad_value=pad_value)


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from models import get_model
    
    print("测试训练模块...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建演示数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 仅使用少量CIFAR10数据进行测试
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 训练参数
    args = {
        "epochs": 2,  # 演示用少量轮次
        "lr": 0.001,
        "weight_decay": 1e-5,
        "device": device,
        "log_dir": "./logs",
        "save_dir": "./models",
    }
    
    # 测试分类器训练
    print("\n测试分类器训练:")
    model = get_model('basic_cnn', device)
    args["model_name"] = model.name
    train_classi***REMOVED***er(model, train_loader, val_loader, args)
    
    # 测试自编码器训练
    print("\n测试自编码器训练:")
    model = get_model('autoencoder', device)
    args["model_name"] = model.name
    train_autoencoder(model, train_loader, val_loader, args)
    
    # 测试变分自编码器训练
    print("\n测试变分自编码器训练:")
    model = get_model('vae', device)
    args["model_name"] = model.name
    args["is_vae"] = True
    train_autoencoder(model, train_loader, val_loader, args)
    
    print("训练模块测试完成!") 