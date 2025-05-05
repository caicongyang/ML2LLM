"""
训练推荐系统模型 - Neural Collaborative Filtering (NCF) 模型训练脚本
该脚本实现了基于神经协同过滤的推荐系统模型训练、评估和保存流程
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

# 添加上级目录到系统路径，以便导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入自定义的推荐系统模型
from models.recommender import NCFModel, SimplifiedNCF
# 导入数据处理工具函数
from utils.data_utils import load_data, split_data, create_data_loaders, negative_sampling
# 导入配置文件中的参数设置
from config import DATA_CONFIG, MODEL_CONFIG, MISC_CONFIG


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练模型一个完整的epoch（即对整个训练集迭代一次）
    
    参数:
        model: 推荐模型 - NCF神经网络模型实例
        train_loader: 训练数据加载器 - PyTorch DataLoader对象，包含批量训练数据
        optimizer: 优化器 - 如Adam，用于更新模型参数
        criterion: 损失函数 - 如BCELoss（二元交叉熵损失）
        device: 计算设备 - 'cuda'用于GPU训练，'cpu'用于CPU训练
        
    返回:
        float: 整个epoch的平均训练损失值
    """
    # 将模型设置为训练模式（启用dropout等训练特定操作）
    model.train()
    total_loss = 0
    
    # 使用tqdm显示进度条，增强用户体验
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # 将数据移动到指定设备(GPU/CPU)
        user_id = batch['user_id'].to(device)  # 用户ID张量
        item_id = batch['item_id'].to(device)  # 商品ID张量
        rating = batch['rating'].to(device)    # 评分张量
        
        # 将评分归一化到[0,1]区间，适应sigmoid激活函数的输出范围
        normalized_rating = rating / 5.0
        
        # 前向传播 - 计算模型预测值
        optimizer.zero_grad()  # 清除之前的梯度
        prediction = model(user_id, item_id)  # 获取模型对当前batch的预测结果
        
        # 计算预测值与真实值之间的损失
        loss = criterion(prediction, normalized_rating)
        
        # 反向传播 - 计算梯度并更新模型参数
        loss.backward()  # 计算梯度
        optimizer.step() # 更新参数
        
        # 累加batch损失值
        total_loss += loss.item()
        
        # 按配置的间隔打印训练日志，监控训练过程
        if batch_idx % MISC_CONFIG['log_interval'] == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 返回平均损失 = 总损失 / 批次数量
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型性能
    
    参数:
        model: 推荐模型 - 训练好的NCF模型
        test_loader: 测试数据加载器 - 包含测试数据的DataLoader
        criterion: 损失函数 - 与训练时相同的损失函数
        device: 计算设备 - GPU或CPU
        
    返回:
        tuple: (average_loss, rmse, mae) 三个评估指标
            - average_loss: 平均损失值
            - rmse: 均方根误差 (Root Mean Square Error)
            - mae: 平均绝对误差 (Mean Absolute Error)
    """
    # 将模型设置为评估模式（禁用dropout等只在训练时使用的操作）
    model.eval()
    total_loss = 0
    all_predictions = []  # 存储所有预测值
    all_ratings = []      # 存储所有真实评分
    
    # 使用torch.no_grad()禁用梯度计算，提高推理速度和减少内存使用
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 将数据移动到指定设备
            user_id = batch['user_id'].to(device)
            item_id = batch['item_id'].to(device)
            rating = batch['rating'].to(device)
            
            # 归一化评分到[0,1]区间
            normalized_rating = rating / 5.0
            
            # 前向传播 - 获取模型预测
            prediction = model(user_id, item_id)
            # 计算损失
            loss = criterion(prediction, normalized_rating)
            
            # 累加batch损失
            total_loss += loss.item()
            
            # 将预测值缩放回原始评分范围[0,5]，用于计算RMSE和MAE
            rescaled_prediction = prediction * 5.0
            
            # 收集所有批次的预测结果和真实评分
            all_predictions.append(rescaled_prediction.cpu().numpy())
            all_ratings.append(rating.cpu().numpy())
    
    # 合并所有批次的结果为一个大数组
    all_predictions = np.concatenate(all_predictions)
    all_ratings = np.concatenate(all_ratings)
    
    # 计算评估指标
    # RMSE: 预测值与真实值差异的平方的均值的平方根，对较大误差更敏感
    rmse = np.sqrt(np.mean((all_predictions - all_ratings) ** 2))
    # MAE: 预测值与真实值绝对差异的均值，表示平均误差大小
    mae = np.mean(np.abs(all_predictions - all_ratings))
    
    # 返回平均损失和评估指标
    return total_loss / len(test_loader), rmse, mae


def save_model(model, save_path):
    """
    保存模型到指定文件路径
    
    参数:
        model: 要保存的模型实例
        save_path: 模型保存的文件路径
    """
    # 确保目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存模型参数（不保存整个模型结构，只保存权重）
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")


def load_model(model, load_path, device):
    """
    从文件加载预训练模型参数
    
    参数:
        model: 模型实例结构（应与保存时的结构匹配）
        load_path: 模型文件路径
        device: 加载模型的目标设备
        
    返回:
        加载了权重的模型实例
    """
    # 加载模型参数到指定设备
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model


def plot_training_history(train_losses, val_losses, metrics, save_dir='./models'):
    """
    绘制并保存训练历史图表，包括损失曲线和评估指标曲线
    
    参数:
        train_losses: 训练损失历史 - 每个epoch的训练损失值列表
        val_losses: 验证损失历史 - 每个epoch的验证损失值列表
        metrics: 评估指标历史 - 每个epoch的(rmse, mae)元组列表
        save_dir: 图表保存目录
    """
    # 导入字体管理模块用于处理中文显示
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    
    # 尝试设置中文字体，如果系统没有中文字体，则使用默认字体并改为英文标签
    try:
        # 对于MacOS系统的中文字体设置
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        # 对于Linux和Windows系统，检查中文字体是否可用
        if not any([font in fm.findSystemFonts() for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']]):
            raise ValueError("No suitable Chinese font found")
        # 解决保存图像时负号'-'显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 如果无法找到中文字体，使用英文标签
        use_chinese = False
    else:
        use_chinese = True
    
    # 创建epoch序列，从1开始
    epochs = range(1, len(train_losses) + 1)
    
    # 创建包含两个子图的图表，用于展示损失和评估指标
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线子图
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss' if not use_chinese else '训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss' if not use_chinese else '验证损失')
    ax1.set_xlabel('Epochs')  # X轴标签
    ax1.set_ylabel('Loss')    # Y轴标签
    ax1.set_title('Training and Validation Loss' if not use_chinese else '训练和验证损失')  # 图表标题
    ax1.legend()  # 显示图例
    ax1.grid(True)  # 显示网格
    
    # 绘制评估指标曲线子图
    ax2.plot(epochs, [m[0] for m in metrics], 'g-', label='RMSE')  # 绘制RMSE曲线
    ax2.plot(epochs, [m[1] for m in metrics], 'y-', label='MAE')   # 绘制MAE曲线
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Evaluation Metrics' if not use_chinese else '评估指标')
    ax2.legend()
    ax2.grid(True)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 调整布局确保图表美观
    plt.tight_layout()
    # 保存图表到文件
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    # 关闭图表释放资源
    plt.close()


def train_model():
    """
    主训练函数 - 执行完整的模型训练流程
    
    包括数据加载、模型初始化、训练循环、评估、保存模型和绘制训练历史
    
    返回:
        tuple: (trained_model, data_frames, id_maps)
        - trained_model: 训练好的模型
        - data_frames: 包含训练和测试数据的DataFrame元组
        - id_maps: ID映射字典，用于将内部ID映射回原始ID
    """
    # 设置计算设备 - 优先使用GPU（如果可用且配置了使用GPU）
    device = torch.device(MISC_CONFIG['device'] if torch.cuda.is_available() and MISC_CONFIG['device'] == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置模型和图表保存路径 - 基于脚本位置动态确定
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    model_save_path = os.path.join(save_dir, 'recommender_model.pth')
    
    # 加载用户-商品评分数据和元数据
    print("加载数据...")
    ratings_df, users_df, items_df, id_maps = load_data()
    
    # 获取用户和商品数量，用于初始化模型的嵌入层大小
    num_users = len(users_df)  # 用户总数
    num_items = len(items_df)  # 商品总数
    print(f"用户数量: {num_users}, 商品数量: {num_items}, 交互数量: {len(ratings_df)}")
    
    # 将数据分割为训练集和测试集（通常是随机分割或按时间分割）
    train_df, test_df = split_data(ratings_df)
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    # 对训练集执行负采样，生成用户未交互过的商品作为负样本
    # 这对于隐式反馈任务很重要，帮助模型学习区分用户可能喜欢和不喜欢的商品
    print("执行负采样...")
    train_df_with_neg = negative_sampling(train_df, num_items)
    print(f"负采样后训练集大小: {len(train_df_with_neg)}")
    
    # 创建PyTorch数据加载器，用于批量加载训练和测试数据
    train_loader, test_loader = create_data_loaders(train_df_with_neg, test_df)
    
    # 初始化NCF模型
    print("初始化模型...")
    model = NCFModel(
        num_users=num_users,             # 用户数量决定用户嵌入矩阵大小
        num_items=num_items,             # 商品数量决定商品嵌入矩阵大小
        embedding_dim=MODEL_CONFIG['embedding_dim'],  # 嵌入向量维度
        hidden_layers=MODEL_CONFIG['hidden_layers'],  # MLP部分的隐藏层配置
        dropout=MODEL_CONFIG['dropout_rate']          # Dropout比率用于防止过拟合
    ).to(device)  # 将模型移动到指定设备
    
    # 打印模型架构，方便调试和了解模型结构
    print(model)
    
    # 设置损失函数 - 二元交叉熵适用于评分归一化后的[0,1]预测
    criterion = nn.BCELoss()
    # 设置优化器 - Adam优化器具有自适应学习率，通常表现较好
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    
    # 训练循环 - 迭代多个epoch直到模型收敛或达到最大epoch数
    print(f"开始训练，共 {MODEL_CONFIG['epochs']} 轮...")
    train_losses = []  # 记录每个epoch的训练损失
    val_losses = []    # 记录每个epoch的验证损失
    metrics_history = []  # 记录每个epoch的评估指标
    
    # 主训练循环，迭代每个epoch
    for epoch in range(1, MODEL_CONFIG['epochs'] + 1):
        # 记录训练开始时间，用于计算每个epoch的耗时
        start_time = time.time()
        
        # 训练一个完整epoch，并获取平均训练损失
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)  # 记录训练损失
        
        # 在测试集上评估模型，获取验证损失和评估指标
        val_loss, rmse, mae = evaluate(model, test_loader, criterion, device)
        val_losses.append(val_loss)  # 记录验证损失
        metrics_history.append((rmse, mae))  # 记录评估指标
        
        # 计算本epoch耗时
        epoch_time = time.time() - start_time
        
        # 打印当前epoch的训练结果和评估指标
        print(f"Epoch {epoch}/{MODEL_CONFIG['epochs']}, "
              f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
              f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
              f"用时: {epoch_time:.2f}s")
    
    # 训练完成后保存最终模型
    save_model(model, model_save_path)
    
    # 绘制并保存训练历史图表
    plot_training_history(train_losses, val_losses, metrics_history, save_dir)
    
    # 返回训练好的模型、数据和映射，可用于后续分析或部署
    return model, (train_df, test_df), id_maps


# 脚本入口点 - 直接运行此脚本时执行train_model()函数
if __name__ == "__main__":
    train_model() 