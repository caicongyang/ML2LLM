#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练模块
提供分类器和自编码器模型的训练功能
"""

# 导入操作系统相关功能，用于文件和目录操作
import os
# 导入时间相关功能，用于计算训练时间
import time
# 导入日期时间功能，用于生成唯一的日志目录名
import datetime
# 导入NumPy库，用于数值计算
import numpy as np
# 导入PyTorch库，深度学习框架的核心
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch优化器模块
import torch.optim as optim
# 导入学习率调度器，用于动态调整学习率
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 导入TensorBoard支持，用于可视化训练过程
from torch.utils.tensorboard import SummaryWriter
# 导入进度条库，用于显示训练进度
from tqdm import tqdm
# 导入图像网格生成工具，用于可视化
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
    # 从训练参数中获取设备信息（GPU或CPU）
    device = args["device"]
    # 从训练参数中获取训练轮数
    epochs = args["epochs"]
    # 从训练参数中获取日志保存目录
    log_dir = args["log_dir"]
    # 从训练参数中获取模型保存目录
    save_dir = args["save_dir"]
    # 从训练参数中获取模型名称
    model_name = args["model_name"]
    
    # 创建日志目录，exist_ok=True表示如果目录已存在则不报错
    os.makedirs(log_dir, exist_ok=True)
    # 创建模型保存目录，exist_ok=True表示如果目录已存在则不报错
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard日志记录器，使用模型名称和当前时间生成唯一的日志目录
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    
    # 定义损失函数为交叉熵损失，适用于分类任务
    criterion = nn.CrossEntropyLoss()
    # 定义优化器为Adam，设置学习率和权重衰减参数
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    # 定义学习率调度器，在验证损失不再下降时减小学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 初始化训练历史记录字典，用于保存训练过程中的损失和准确率
    history = {
        'train_loss': [],  # 存储每个epoch的训练损失
        'train_acc': [],   # 存储每个epoch的训练准确率
        'val_loss': [],    # 存储每个epoch的验证损失
        'val_acc': [],     # 存储每个epoch的验证准确率
        'best_val_acc': 0.0  # 记录最佳验证准确率
    }
    
    # 打印开始训练的消息
    print(f"开始训练 {model_name}...")
    # 记录训练开始时间
    start_time = time.time()
    
    # 循环训练指定的轮数
    for epoch in range(epochs):
        # 将模型设置为训练模式，启用批量归一化和dropout
        model.train()
        # 初始化此轮的训练损失
        train_loss = 0.0
        # 初始化此轮的正确预测数量
        train_correct = 0
        # 初始化此轮的总样本数
        train_total = 0
        
        # 遍历训练数据加载器中的每一批数据
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将输入数据移到指定设备（GPU或CPU）
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清空梯度，防止梯度累积
            optimizer.zero_grad()
            # 前向传播，获取模型输出
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            
            # 累加批次损失
            train_loss += loss.item()
            # 获取预测结果（最大概率的类别索引）
            _, predicted = outputs.max(1)
            # 累加本批次的样本总数
            train_total += targets.size(0)
            # 累加本批次中预测正确的样本数
            train_correct += predicted.eq(targets).sum().item()
            
            # 每处理50个批次或处理完最后一个批次时，打印进度
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {train_loss/(batch_idx+1):.4f} "  # 计算平均损失
                      f"Acc: {100.*train_correct/train_total:.2f}%")  # 计算准确率
        
        # 计算整个epoch的平均训练损失
        train_loss = train_loss / len(train_loader)
        # 计算整个epoch的训练准确率
        train_acc = 100. * train_correct / train_total
        
        # 进入验证阶段，将模型设置为评估模式，禁用批量归一化和dropout
        model.eval()
        # 初始化验证损失
        val_loss = 0.0
        # 初始化验证中预测正确的样本数
        val_correct = 0
        # 初始化验证中的总样本数
        val_total = 0
        
        # 在不计算梯度的情况下进行验证
        with torch.no_grad():
            # 遍历验证数据加载器中的每一批数据
            for inputs, targets in val_loader:
                # 将输入数据移到指定设备（GPU或CPU）
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播，获取模型输出
                outputs = model(inputs)
                # 计算验证损失
                loss = criterion(outputs, targets)
                
                # 累加验证损失
                val_loss += loss.item()
                # 获取预测结果（最大概率的类别索引）
                _, predicted = outputs.max(1)
                # 累加验证样本总数
                val_total += targets.size(0)
                # 累加验证中预测正确的样本数
                val_correct += predicted.eq(targets).sum().item()
        
        # 计算整个验证集的平均损失
        val_loss = val_loss / len(val_loader)
        # 计算整个验证集的准确率
        val_acc = 100. * val_correct / val_total
        
        # 根据验证损失更新学习率（如果验证损失不再下降，则减小学习率）
        scheduler.step(val_loss)
        
        # 将当前epoch的训练损失添加到历史记录
        history['train_loss'].append(train_loss)
        # 将当前epoch的训练准确率添加到历史记录
        history['train_acc'].append(train_acc)
        # 将当前epoch的验证损失添加到历史记录
        history['val_loss'].append(val_loss)
        # 将当前epoch的验证准确率添加到历史记录
        history['val_acc'].append(val_acc)
        
        # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        # 记录训练准确率到TensorBoard
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        # 记录验证损失到TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        # 记录验证准确率到TensorBoard
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        # 记录当前学习率到TensorBoard
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印当前epoch的训练和验证结果
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # 如果当前验证准确率高于历史最佳，则保存模型
        if val_acc > history['best_val_acc']:
            # 更新最佳验证准确率记录
            history['best_val_acc'] = val_acc
            # 保存模型检查点，包含epoch、模型参数、优化器状态、验证准确率和损失
            torch.save({
                'epoch': epoch + 1,  # 当前epoch
                'model_state_dict': model.state_dict(),  # 模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
                'val_acc': val_acc,  # 验证准确率
                'val_loss': val_loss,  # 验证损失
            }, os.path.join(save_dir, f"{model_name}_best.pth"))  # 保存路径
            # 打印保存模型的消息
            print(f"模型已保存: {model_name}_best.pth")
    
    # 训练完所有epoch后，保存最终模型
    torch.save({
        'epoch': epochs,  # 总训练轮数
        'model_state_dict': model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'val_acc': val_acc,  # 最后一轮的验证准确率
        'val_loss': val_loss,  # 最后一轮的验证损失
    }, os.path.join(save_dir, f"{model_name}_***REMOVED***nal.pth"))  # 保存路径
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    # 打印训练完成消息和总用时
    print(f"训练完成! 总用时: {total_time:.2f}秒")
    # 打印最佳验证准确率
    print(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
    
    # 返回训练历史记录
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
    # 从训练参数中获取设备信息（GPU或CPU）
    device = args["device"]
    # 从训练参数中获取训练轮数
    epochs = args["epochs"]
    # 从训练参数中获取日志保存目录
    log_dir = args["log_dir"]
    # 从训练参数中获取模型保存目录 
    save_dir = args["save_dir"]
    # 从训练参数中获取模型名称
    model_name = args["model_name"]
    # 从训练参数中获取是否为变分自编码器的标志，若没有则默认为False
    is_vae = args.get("is_vae", False)
    
    # 创建日志目录，exist_ok=True表示如果目录已存在则不报错
    os.makedirs(log_dir, exist_ok=True)
    # 创建模型保存目录，exist_ok=True表示如果目录已存在则不报错
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard日志记录器，使用模型名称和当前时间生成唯一的日志目录
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    
    # 定义重建损失函数为均方误差损失，适用于图像重建任务
    reconstruction_criterion = nn.MSELoss()
    # 定义优化器为Adam，设置学习率和权重衰减参数
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    # 定义学习率调度器，在验证损失不再下降时减小学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 初始化训练历史记录字典，用于保存训练过程中的损失
    history = {
        'train_loss': [],  # 存储每个epoch的训练损失
        'val_loss': [],    # 存储每个epoch的验证损失
        'best_val_loss': float('inf')  # 记录最佳验证损失，初始化为无穷大
    }
    
    # 打印开始训练的消息
    print(f"开始训练 {model_name}...")
    # 记录训练开始时间
    start_time = time.time()
    
    # 循环训练指定的轮数
    for epoch in range(epochs):
        # 将模型设置为训练模式，启用批量归一化和dropout
        model.train()
        # 初始化此轮的训练总损失
        train_loss = 0.0
        # 初始化此轮的训练重建损失（仅用于VAE）
        train_recon_loss = 0.0
        # 初始化此轮的训练KL散度损失（仅用于VAE）
        train_kld_loss = 0.0
        
        # 遍历训练数据加载器中的每一批数据
        for batch_idx, (inputs, _) in enumerate(train_loader):
            # 自编码器只使用输入数据，不需要标签，因此使用_忽略标签
            # 将输入数据移到指定设备（GPU或CPU）
            inputs = inputs.to(device)
            
            # 清空梯度，防止梯度累积
            optimizer.zero_grad()
            
            # 根据模型类型进行不同的处理
            if is_vae:
                # 变分自编码器的前向传播，返回重建图像、均值和对数方差
                reconstructed, mu, logvar = model(inputs)
                
                # 计算重建损失（输入与重建图像之间的均方误差）
                recon_loss = reconstruction_criterion(reconstructed, inputs)
                
                # 计算KL散度损失，这是VAE的正则化项
                # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # 对KL散度进行归一化，除以批次大小
                kld_loss = kld_loss / inputs.size(0)  # 归一化
                
                # 计算总损失 = 重建损失 + β * KL散度损失
                # β是一个超参数，用于平衡重建质量和潜变量分布的正则化
                beta = 0.1  # 可调整的超参数
                loss = recon_loss + beta * kld_loss
                
                # 累加重建损失
                train_recon_loss += recon_loss.item()
                # 累加KL散度损失
                train_kld_loss += kld_loss.item()
            else:
                # 普通自编码器的前向传播，返回重建图像
                reconstructed = model(inputs)
                # 计算重建损失（输入与重建图像之间的均方误差）
                loss = reconstruction_criterion(reconstructed, inputs)
            
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            
            # 累加批次损失
            train_loss += loss.item()
            
            # 每处理50个批次或处理完最后一个批次时，打印进度
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                # 根据模型类型打印不同的信息
                if is_vae:
                    # 对于VAE，打印总损失、重建损失和KL散度损失
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {train_loss/(batch_idx+1):.4f} "  # 平均总损失
                          f"Recon: {train_recon_loss/(batch_idx+1):.4f} "  # 平均重建损失
                          f"KLD: {train_kld_loss/(batch_idx+1):.4f}")  # 平均KL散度损失
                else:
                    # 对于普通自编码器，只打印总损失
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {train_loss/(batch_idx+1):.4f}")  # 平均损失
        
        # 计算整个epoch的平均训练损失
        train_loss = train_loss / len(train_loader)
        
        # 进入验证阶段，将模型设置为评估模式，禁用批量归一化和dropout
        model.eval()
        # 初始化验证总损失
        val_loss = 0.0
        # 初始化验证重建损失（仅用于VAE）
        val_recon_loss = 0.0
        # 初始化验证KL散度损失（仅用于VAE）
        val_kld_loss = 0.0
        
        # 在不计算梯度的情况下进行验证
        with torch.no_grad():
            # 遍历验证数据加载器中的每一批数据
            for inputs, _ in val_loader:
                # 自编码器只使用输入数据，不需要标签，因此使用_忽略标签
                # 将输入数据移到指定设备（GPU或CPU）
                inputs = inputs.to(device)
                
                # 根据模型类型进行不同的处理
                if is_vae:
                    # 变分自编码器的前向传播，返回重建图像、均值和对数方差
                    reconstructed, mu, logvar = model(inputs)
                    
                    # 计算重建损失（输入与重建图像之间的均方误差）
                    recon_loss = reconstruction_criterion(reconstructed, inputs)
                    
                    # 计算KL散度损失，这是VAE的正则化项
                    # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    # 对KL散度进行归一化，除以批次大小
                    kld_loss = kld_loss / inputs.size(0)  # 归一化
                    
                    # 计算总损失 = 重建损失 + β * KL散度损失
                    # β是一个超参数，用于平衡重建质量和潜变量分布的正则化
                    beta = 0.1  # 可调整的超参数，与训练阶段一致
                    loss = recon_loss + beta * kld_loss
                    
                    # 累加验证重建损失
                    val_recon_loss += recon_loss.item()
                    # 累加验证KL散度损失
                    val_kld_loss += kld_loss.item()
                else:
                    # 普通自编码器的前向传播，返回重建图像
                    reconstructed = model(inputs)
                    # 计算重建损失（输入与重建图像之间的均方误差）
                    loss = reconstruction_criterion(reconstructed, inputs)
                
                # 累加验证损失
                val_loss += loss.item()
        
        # 计算整个验证集的平均损失
        val_loss = val_loss / len(val_loader)
        
        # 根据验证损失更新学习率（如果验证损失不再下降，则减小学习率）
        scheduler.step(val_loss)
        
        # 将当前epoch的训练损失添加到历史记录
        history['train_loss'].append(train_loss)
        # 将当前epoch的验证损失添加到历史记录
        history['val_loss'].append(val_loss)
        
        # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        # 记录验证损失到TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        # 对于VAE，额外记录重建损失和KL散度损失
        if is_vae:
            # 记录训练重建损失
            writer.add_scalar('Loss/train_reconstruction', train_recon_loss / len(train_loader), epoch)
            # 记录训练KL散度损失
            writer.add_scalar('Loss/train_kld', train_kld_loss / len(train_loader), epoch)
            # 记录验证重建损失
            writer.add_scalar('Loss/val_reconstruction', val_recon_loss / len(val_loader), epoch)
            # 记录验证KL散度损失
            writer.add_scalar('Loss/val_kld', val_kld_loss / len(val_loader), epoch)
        # 记录当前学习率到TensorBoard
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 每5个epoch在TensorBoard中添加重建图像样本
        if epoch % 5 == 0:
            # 在不计算梯度的情况下进行图像重建
            with torch.no_grad():
                # 获取一批验证数据，用于可视化
                test_inputs, _ = next(iter(val_loader))
                # 将输入数据移到指定设备（GPU或CPU）
                test_inputs = test_inputs.to(device)
                
                # 根据模型类型获取重建图像
                if is_vae:
                    # 变分自编码器的前向传播，获取重建图像
                    reconstructed, _, _ = model(test_inputs)
                else:
                    # 普通自编码器的前向传播，获取重建图像
                    reconstructed = model(test_inputs)
                
                # 创建输入图像和重建图像的对比图
                # 选取前8张图像，并将原始和重建图像连接起来
                comparison = torch.cat([test_inputs[:8], reconstructed[:8]])
                # 创建图像网格，便于可视化
                grid = make_grid(comparison, nrow=8, normalize=True, scale_each=True)
                # 将图像网格添加到TensorBoard
                writer.add_image('Reconstruction', grid, epoch)
        
        # 打印当前epoch的训练和验证结果
        if is_vae:
            # 对于VAE，打印总损失、重建损失和KL散度损失
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} "
                  f"(Recon: {train_recon_loss/len(train_loader):.4f}, KLD: {train_kld_loss/len(train_loader):.4f}) "
                  f"Val Loss: {val_loss:.4f}")
        else:
            # 对于普通自编码器，只打印总损失
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        # 如果当前验证损失低于历史最低损失，则保存模型
        if val_loss < history['best_val_loss']:
            # 更新最佳验证损失记录
            history['best_val_loss'] = val_loss
            # 保存模型检查点，包含epoch、模型参数、优化器状态和验证损失
            torch.save({
                'epoch': epoch + 1,  # 当前epoch
                'model_state_dict': model.state_dict(),  # 模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
                'val_loss': val_loss,  # 验证损失
            }, os.path.join(save_dir, f"{model_name}_best.pth"))  # 保存路径
            # 打印保存模型的消息
            print(f"模型已保存: {model_name}_best.pth")
    
    # 训练完所有epoch后，保存最终模型
    torch.save({
        'epoch': epochs,  # 总训练轮数
        'model_state_dict': model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'val_loss': val_loss,  # 最后一轮的验证损失
    }, os.path.join(save_dir, f"{model_name}_***REMOVED***nal.pth"))  # 保存路径
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    # 打印训练完成消息和总用时
    print(f"训练完成! 总用时: {total_time:.2f}秒")
    # 打印最佳验证损失
    print(f"最佳验证损失: {history['best_val_loss']:.4f}")
    
    # 返回训练历史记录
    return history


# 工具函数：生成图像网格
def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """
    创建图像网格
    简化版的torchvision.utils.make_grid函数
    """
    # 从torchvision工具包导入原始的make_grid函数
    from torchvision.utils import make_grid as tv_make_grid
    # 调用原始函数并返回结果
    return tv_make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, 
                        range=range, scale_each=scale_each, pad_value=pad_value)


# 如果此脚本被直接运行（而不是被导入），则执行以下代码
if __name__ == "__main__":
    # 导入PyTorch库
    import torch
    # 导入数据加载器和数据集分割工具
    from torch.utils.data import DataLoader, random_split
    # 导入torchvision数据集和数据变换工具
    from torchvision import datasets, transforms
    # 从本地模块导入获取模型的函数
    from models import get_model
    
    # 打印测试开始消息
    print("测试训练模块...")
    
    # 设置计算设备，如果CUDA可用则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印使用的设备信息
    print(f"使用设备: {device}")
    
    # 创建演示数据集
    # 定义数据变换：转换为Tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将像素值归一化到[-1,1]
    ])
    
    # 下载并加载CIFAR10训练数据集
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    # 下载并加载CIFAR10测试数据集
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # 将训练数据集分割为训练集和验证集
    # 计算训练集大小（总数据集的80%）
    train_size = int(0.8 * len(dataset))
    # 计算验证集大小（总数据集的20%）
    val_size = len(dataset) - train_size
    # 随机分割数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    # 训练数据加载器，启用随机打乱和多进程加载
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    # 验证数据加载器，不需要随机打乱
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 设置训练参数
    args = {
        "epochs": 2,  # 训练轮数，演示用少量轮次
        "lr": 0.001,  # 学习率
        "weight_decay": 1e-5,  # 权重衰减（L2正则化）
        "device": device,  # 计算设备
        "log_dir": "./logs",  # 日志目录
        "save_dir": "./models",  # 模型保存目录
    }
    
    # 测试分类器训练
    print("\n测试分类器训练:")
    # 创建基础CNN模型
    model = get_model('basic_cnn', device)
    # 设置模型名称
    args["model_name"] = model.name
    # 调用分类器训练函数
    train_classi***REMOVED***er(model, train_loader, val_loader, args)
    
    # 测试自编码器训练
    print("\n测试自编码器训练:")
    # 创建自编码器模型
    model = get_model('autoencoder', device)
    # 设置模型名称
    args["model_name"] = model.name
    # 调用自编码器训练函数
    train_autoencoder(model, train_loader, val_loader, args)
    
    # 测试变分自编码器训练
    print("\n测试变分自编码器训练:")
    # 创建变分自编码器模型
    model = get_model('vae', device)
    # 设置模型名称
    args["model_name"] = model.name
    # 设置is_vae标志为True
    args["is_vae"] = True
    # 调用自编码器训练函数
    train_autoencoder(model, train_loader, val_loader, args)
    
    # 打印测试完成消息
    print("训练模块测试完成!") 