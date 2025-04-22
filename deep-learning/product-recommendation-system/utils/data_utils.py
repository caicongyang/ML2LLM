"""
数据处理工具模块
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from con***REMOVED***g import DATA_CONFIG, MODEL_CONFIG


class RatingDataset(Dataset):
    """
    评分数据集类，用于加载用户-商品交互数据
    """
    def __init__(self, user_item_rating_df):
        """
        初始化数据集
        
        参数:
            user_item_rating_df (pd.DataFrame): 包含[user_id, item_id, rating]的数据框
        """
        self.user_item_rating_df = user_item_rating_df
        
        # 转换为PyTorch张量
        self.users = torch.tensor(self.user_item_rating_df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(self.user_item_rating_df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.user_item_rating_df['rating'].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.user_item_rating_df)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'rating': self.ratings[idx]
        }


def load_data(data_path=None):
    """
    加载数据集或生成模拟数据（如果数据文件不存在）
    
    参数:
        data_path (str, optional): 评分数据路径. 如果为None, 使用配置文件中的路径.
        
    返回:
        tuple: (ratings_df, users_df, items_df) 数据框元组
    """
    if data_path and os.path.exists(data_path):
        ratings_df = pd.read_csv(data_path)
    else:
        # 如果数据不存在，生成模拟数据
        print("未找到数据文件，生成模拟数据...")
        ratings_df = generate_mock_data()
    
    # 确保数据格式正确
    if 'rating' not in ratings_df.columns:
        if 'rating_value' in ratings_df.columns:
            ratings_df.rename(columns={'rating_value': 'rating'}, inplace=True)
        else:
            # 假设用户交互为隐式反馈（例如点击、购买）
            ratings_df['rating'] = 1.0
    
    # 创建用户和商品ID映射
    unique_user_ids = ratings_df['user_id'].unique()
    unique_item_ids = ratings_df['item_id'].unique()
    
    users_df = pd.DataFrame({
        'user_id': np.arange(len(unique_user_ids)),
        'original_id': unique_user_ids
    })
    
    items_df = pd.DataFrame({
        'item_id': np.arange(len(unique_item_ids)),
        'original_id': unique_item_ids
    })
    
    # 将原始ID映射到连续的整数ID
    id_maps = {
        'user': dict(zip(unique_user_ids, users_df['user_id'])),
        'item': dict(zip(unique_item_ids, items_df['item_id']))
    }
    
    # 替换ID
    ratings_df['user_id'] = ratings_df['user_id'].map(id_maps['user'])
    ratings_df['item_id'] = ratings_df['item_id'].map(id_maps['item'])
    
    return ratings_df, users_df, items_df, id_maps


def generate_mock_data(num_users=1000, num_items=2000, sparsity=0.01):
    """
    生成模拟评分数据
    
    参数:
        num_users (int): 用户数量
        num_items (int): 商品数量
        sparsity (float): 稀疏度（数据占总可能交互的比例）
        
    返回:
        pd.DataFrame: 包含(user_id, item_id, rating)的评分数据框
    """
    # 计算交互数
    num_interactions = int(num_users * num_items * sparsity)
    
    # 随机生成用户-商品对
    np.random.seed(DATA_CONFIG['random_seed'])
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    
    # 生成随机评分（1-5分，可以改为二元值表示点击/购买）
    ratings = np.random.randint(1, 6, num_interactions)
    
    # 创建数据框
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # 确保每个用户-商品对只出现一次（去重）
    ratings_df = ratings_df.drop_duplicates(['user_id', 'item_id'])
    
    return ratings_df


def split_data(ratings_df, test_size=None):
    """
    拆分数据为训练集和测试集
    
    参数:
        ratings_df (pd.DataFrame): 评分数据框
        test_size (float, optional): 测试集比例. 默认使用配置中的值.
        
    返回:
        tuple: (train_df, test_df) 训练集和测试集
    """
    if test_size is None:
        test_size = DATA_CONFIG['test_size']
    
    # 确保每个用户都有足够数据
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        stratify=ratings_df['user_id'],
        random_state=DATA_CONFIG['random_seed']
    )
    
    return train_df, test_df


def create_data_loaders(train_df, test_df, batch_size=None):
    """
    创建PyTorch数据加载器
    
    参数:
        train_df (pd.DataFrame): 训练数据
        test_df (pd.DataFrame): 测试数据
        batch_size (int, optional): 批次大小. 默认使用配置中的值.
        
    返回:
        tuple: (train_loader, test_loader) 训练和测试数据加载器
    """
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    # 创建数据集
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def negative_sampling(ratings_df, num_items, num_neg_samples=None):
    """
    为每个正样本生成负样本
    
    参数:
        ratings_df (pd.DataFrame): 正样本评分数据
        num_items (int): 商品总数
        num_neg_samples (int, optional): 每个正样本对应的负样本数量
        
    返回:
        pd.DataFrame: 包含正负样本的数据框
    """
    if num_neg_samples is None:
        num_neg_samples = MODEL_CONFIG['num_negative_samples']
    
    # 用户消费过的商品集合
    user_consumed = ratings_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    neg_samples = []
    for _, row in ratings_df.iterrows():
        user = row['user_id']
        # 随机选择用户未消费过的商品作为负样本
        consumed = user_consumed.get(user, set())
        
        # 负采样
        neg_items = []
        while len(neg_items) < num_neg_samples:
            neg_item = np.random.randint(0, num_items)
            if neg_item not in consumed and neg_item not in neg_items:
                neg_items.append(neg_item)
                neg_samples.append({
                    'user_id': user,
                    'item_id': neg_item,
                    'rating': 0.0  # 负样本评分为0
                })
    
    # 合并正负样本
    neg_df = pd.DataFrame(neg_samples)
    combined_df = pd.concat([ratings_df, neg_df], ignore_index=True)
    
    return combined_df 