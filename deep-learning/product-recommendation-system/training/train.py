"""
训练推荐系统模型
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__***REMOVED***le__))))
from models.recommender import NCFModel, Simpli***REMOVED***edNCF
from utils.data_utils import load_data, split_data, create_data_loaders, negative_sampling
from con***REMOVED***g import DATA_CONFIG, MODEL_CONFIG, MISC_CONFIG


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch
    
    参数:
        model: 推荐模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备(CPU/GPU)
        
    返回:
        float: 平均训练损失
    """
    model.train()
    total_loss = 0
    
    # 使用tqdm显示进度条
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        user_id = batch['user_id'].to(device)
        item_id = batch['item_id'].to(device)
        rating = batch['rating'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        prediction = model(user_id, item_id)
        
        # 计算损失
        loss = criterion(prediction, rating)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印训练日志
        if batch_idx % MISC_CONFIG['log_interval'] == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 返回平均损失
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型
    
    参数:
        model: 推荐模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备(CPU/GPU)
        
    返回:
        tuple: (average_loss, rmse, mae) 评估指标
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_ratings = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_id = batch['user_id'].to(device)
            item_id = batch['item_id'].to(device)
            rating = batch['rating'].to(device)
            
            # 前向传播
            prediction = model(user_id, item_id)
            loss = criterion(prediction, rating)
            
            total_loss += loss.item()
            
            # 收集预测结果
            all_predictions.append(prediction.cpu().numpy())
            all_ratings.append(rating.cpu().numpy())
    
    # 合并预测结果
    all_predictions = np.concatenate(all_predictions)
    all_ratings = np.concatenate(all_ratings)
    
    # 计算评估指标
    rmse = np.sqrt(np.mean((all_predictions - all_ratings) ** 2))
    mae = np.mean(np.abs(all_predictions - all_ratings))
    
    return total_loss / len(test_loader), rmse, mae


def save_model(model, save_path):
    """保存模型到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")


def load_model(model, load_path, device):
    """从文件加载模型"""
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model


def plot_training_history(train_losses, val_losses, metrics, save_dir='./models'):
    """绘制训练历史"""
    epochs = range(1, len(train_losses) + 1)
    
    # 创建两个子图
    ***REMOVED***g, (ax1, ax2) = plt.subplots(1, 2, ***REMOVED***gsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制评估指标曲线
    ax2.plot(epochs, [m[0] for m in metrics], 'g-', label='RMSE')
    ax2.plot(epochs, [m[1] for m in metrics], 'y-', label='MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('评估指标')
    ax2.legend()
    ax2.grid(True)
    
    # 保存图表
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.save***REMOVED***g(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def train_model():
    """主训练函数"""
    # 设置设备
    device = torch.device(MISC_CONFIG['device'] if torch.cuda.is_available() and MISC_CONFIG['device'] == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    ratings_df, users_df, items_df, id_maps = load_data()
    
    # 获取用户和商品数量
    num_users = len(users_df)
    num_items = len(items_df)
    print(f"用户数量: {num_users}, 商品数量: {num_items}, 交互数量: {len(ratings_df)}")
    
    # 拆分数据为训练集和测试集
    train_df, test_df = split_data(ratings_df)
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    # 负采样 (对训练集)
    print("执行负采样...")
    train_df_with_neg = negative_sampling(train_df, num_items)
    print(f"负采样后训练集大小: {len(train_df_with_neg)}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_df_with_neg, test_df)
    
    # 初始化模型
    print("初始化模型...")
    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_layers=MODEL_CONFIG['hidden_layers'],
        dropout=MODEL_CONFIG['dropout_rate']
    ).to(device)
    
    # 打印模型信息
    print(model)
    
    # 设置损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    
    # 训练循环
    print(f"开始训练，共 {MODEL_CONFIG['epochs']} 轮...")
    train_losses = []
    val_losses = []
    metrics_history = []
    
    for epoch in range(1, MODEL_CONFIG['epochs'] + 1):
        # 训练一个epoch
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 评估模型
        val_loss, rmse, mae = evaluate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        metrics_history.append((rmse, mae))
        
        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{MODEL_CONFIG['epochs']}, "
              f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
              f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
              f"用时: {epoch_time:.2f}s")
    
    # 保存最终模型
    save_model(model, MODEL_CONFIG['model_save_path'])
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, metrics_history)
    
    return model, (train_df, test_df), id_maps


if __name__ == "__main__":
    train_model() 