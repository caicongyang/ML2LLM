"""
生成模拟数据用于推荐系统训练和测试
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__***REMOVED***le__))))
from con***REMOVED***g import DATA_CONFIG


def generate_user_data(num_users=1000):
    """
    生成用户数据
    
    参数:
        num_users (int): 用户数量
        
    返回:
        pd.DataFrame: 用户数据框
    """
    np.random.seed(DATA_CONFIG['random_seed'])
    
    # 生成用户ID
    user_ids = np.arange(num_users)
    
    # 生成用户特征
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    genders = ['男', '女', '其他']
    regions = ['北京', '上海', '广州', '深圳', '成都', '杭州', '武汉', '西安', '南京', '重庆']
    
    # 随机生成用户属性
    ages = np.random.choice(age_groups, size=num_users)
    gender = np.random.choice(genders, size=num_users)
    region = np.random.choice(regions, size=num_users)
    
    # 创建用户数据框
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'age_group': ages,
        'gender': gender,
        'region': region,
        'register_days': np.random.randint(1, 1000, size=num_users)
    })
    
    return users_df


def generate_product_data(num_items=2000):
    """
    生成商品数据
    
    参数:
        num_items (int): 商品数量
        
    返回:
        pd.DataFrame: 商品数据框
    """
    np.random.seed(DATA_CONFIG['random_seed'] + 1)
    
    # 生成商品ID
    item_ids = np.arange(num_items)
    
    # 生成商品特征
    categories = ['电子产品', '服装', '食品', '图书', '家居', '美妆', '运动', '玩具', '健康', '其他']
    brands = [f'品牌{i}' for i in range(1, 51)]  # 50个品牌
    
    # 随机生成商品属性
    category = np.random.choice(categories, size=num_items)
    brand = np.random.choice(brands, size=num_items)
    price = np.random.uniform(10, 1000, size=num_items).round(2)  # 价格范围10-1000
    
    # 创建商品数据框
    items_df = pd.DataFrame({
        'item_id': item_ids,
        'category': category,
        'brand': brand,
        'price': price,
        'popularity': np.random.uniform(1, 10, size=num_items).round(1)  # 商品流行度1-10
    })
    
    return items_df


def generate_ratings_data(num_users=1000, num_items=2000, sparsity=0.01, implicit=False):
    """
    生成用户-商品交互数据
    
    参数:
        num_users (int): 用户数量
        num_items (int): 商品数量
        sparsity (float): 稀疏度（数据占总可能交互的比例）
        implicit (bool): 是否生成隐式反馈数据
        
    返回:
        pd.DataFrame: 交互数据框
    """
    np.random.seed(DATA_CONFIG['random_seed'] + 2)
    
    # 计算交互数
    num_interactions = int(num_users * num_items * sparsity)
    
    # 生成用户的活跃度分布（有些用户比其他用户更活跃）
    # 使用正态分布来模拟
    user_activity = np.abs(norm.rvs(size=num_users))
    user_activity = user_activity / user_activity.sum()
    
    # 生成商品的流行度分布（有些商品比其他商品更受欢迎）
    # 使用幂律分布来模拟
    item_popularity = np.random.power(0.5, size=num_items)
    item_popularity = item_popularity / item_popularity.sum()
    
    # 根据活跃度和流行度随机生成用户-商品对
    user_ids = np.random.choice(
        np.arange(num_users), 
        size=num_interactions, 
        p=user_activity
    )
    
    item_ids = np.random.choice(
        np.arange(num_items), 
        size=num_interactions, 
        p=item_popularity
    )
    
    # 生成时间戳（最近30天内）
    timestamps = np.random.randint(
        int(pd.Timestamp.now().timestamp()) - 30*24*60*60,
        int(pd.Timestamp.now().timestamp()),
        size=num_interactions
    )
    
    if implicit:
        # 隐式反馈：1表示交互（点击、购买等）
        ratings = np.ones(num_interactions)
    else:
        # 显式反馈：评分1-5
        # 生成评分分布（评分倾向于高于中间值）
        ratings = np.random.choice(
            [1, 2, 3, 4, 5], 
            size=num_interactions, 
            p=[0.05, 0.1, 0.2, 0.3, 0.35]
        )
    
    # 创建交互数据框
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # 确保每个用户-商品对只出现一次
    interactions_df = interactions_df.drop_duplicates(['user_id', 'item_id'])
    
    # 转换时间戳为可读格式
    interactions_df['date'] = pd.to_datetime(interactions_df['timestamp'], unit='s')
    
    return interactions_df


def generate_and_save_data(output_dir='./'):
    """
    生成并保存所有模拟数据
    
    参数:
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置常量
    NUM_USERS = 1000
    NUM_ITEMS = 2000
    SPARSITY = 0.01  # 平均每个用户有NUM_ITEMS*SPARSITY个交互
    
    # 生成数据
    print("生成用户数据...")
    users_df = generate_user_data(NUM_USERS)
    
    print("生成商品数据...")
    items_df = generate_product_data(NUM_ITEMS)
    
    print("生成交互数据...")
    ratings_df = generate_ratings_data(NUM_USERS, NUM_ITEMS, SPARSITY)
    
    # 拆分训练集和测试集
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_seed']
    )
    
    # 保存数据
    print("保存数据...")
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    items_df.to_csv(os.path.join(output_dir, 'products.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'ratings_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'ratings_test.csv'), index=False)
    
    print(f"用户数据: {len(users_df)} 行")
    print(f"商品数据: {len(items_df)} 行")
    print(f"训练数据: {len(train_df)} 行")
    print(f"测试数据: {len(test_df)} 行")
    print(f"数据已保存到：{output_dir}")


if __name__ == "__main__":
    # 保存到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__***REMOVED***le__))
    generate_and_save_data(script_dir) 