"""
商品推荐系统配置文件
"""
import os

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置数据目录路径
DATA_DIR = os.path.join(CURRENT_DIR, 'data')

# 数据相关配置
DATA_CONFIG = {
    'train_path': os.path.join(DATA_DIR, 'ratings_train.csv'),
    'test_path': os.path.join(DATA_DIR, 'ratings_test.csv'),
    'product_path': os.path.join(DATA_DIR, 'products.csv'),
    'user_path': os.path.join(DATA_DIR, 'users.csv'),
    'random_seed': 42,
    'test_size': 0.2,
}

# 模型相关配置
MODEL_CONFIG = {
    'embedding_dim': 64,      # 嵌入维度
    'hidden_layers': [128, 64, 32],  # 隐藏层神经元数量
    'dropout_rate': 0.2,      # Dropout比例
    'learning_rate': 0.001,   # 学习率
    'batch_size': 256,        # 批次大小
    'epochs': 20,             # 训练轮数
    'num_negative_samples': 4,  # 每个正样本对应的负样本数量
    'model_save_path': os.path.join(DATA_DIR, 'recommender_model.pth'),  # 模型保存路径
}

# 推荐相关配置
RECOMMEND_CONFIG = {
    'top_k': 10,  # 推荐商品数量
    'min_interactions': 5,  # 用户最小交互数量要求
    'similarity_threshold': 0.5,  # 相似度阈值
}

# 其他配置
MISC_CONFIG = {
    'log_interval': 100,  # 日志打印间隔
    'device': 'cpu',  # 运行设备 'cuda' 或 'cpu'
} 