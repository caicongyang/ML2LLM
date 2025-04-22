"""
推荐系统评估指标工具
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差 (Root Mean Squared Error)
    
    参数:
        y_true (array-like): 真实评分
        y_pred (array-like): 预测评分
        
    返回:
        float: RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    参数:
        y_true (array-like): 真实评分
        y_pred (array-like): 预测评分
        
    返回:
        float: MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_precision_at_k(y_true, y_pred, k=10, threshold=3.5):
    """
    计算Top-K准确率
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度
        threshold (float): 认为是正例的评分阈值
        
    返回:
        float: Precision@K值
    """
    # 确保数据格式一致
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 计算每个用户的Precision@K
    precision_at_k_list = []
    
    for user_id in y_true['user_id'].unique():
        # 获取用户的真实评分
        user_true = y_true[y_true['user_id'] == user_id]
        
        # 获取用户的预测评分
        user_pred = y_pred[y_pred['user_id'] == user_id]
        
        # 获取用户的Top-K推荐
        user_top_k = user_pred.sort_values('score', ascending=False).head(k)
        
        # 获取用户的Top-K推荐中的真实正例数
        relevant_items = set(user_true[user_true['rating'] >= threshold]['item_id'])
        recommended_items = set(user_top_k['item_id'])
        
        # 计算Precision@K
        if not recommended_items:
            precision = 0
        else:
            precision = len(relevant_items.intersection(recommended_items)) / len(recommended_items)
        
        precision_at_k_list.append(precision)
    
    # 返回平均Precision@K
    return np.mean(precision_at_k_list)


def calculate_recall_at_k(y_true, y_pred, k=10, threshold=3.5):
    """
    计算Top-K召回率
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度
        threshold (float): 认为是正例的评分阈值
        
    返回:
        float: Recall@K值
    """
    # 确保数据格式一致
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 计算每个用户的Recall@K
    recall_at_k_list = []
    
    for user_id in y_true['user_id'].unique():
        # 获取用户的真实评分
        user_true = y_true[y_true['user_id'] == user_id]
        
        # 获取用户的预测评分
        user_pred = y_pred[y_pred['user_id'] == user_id]
        
        # 获取用户的Top-K推荐
        user_top_k = user_pred.sort_values('score', ascending=False).head(k)
        
        # 获取用户的所有真实正例
        relevant_items = set(user_true[user_true['rating'] >= threshold]['item_id'])
        recommended_items = set(user_top_k['item_id'])
        
        # 计算Recall@K
        if not relevant_items:
            recall = 0
        else:
            recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items)
        
        recall_at_k_list.append(recall)
    
    # 返回平均Recall@K
    return np.mean(recall_at_k_list)


def calculate_ndcg_at_k(y_true, y_pred, k=10):
    """
    计算归一化折损累积增益 (Normalized Discounted Cumulative Gain)
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度
        
    返回:
        float: NDCG@K值
    """
    # 确保数据格式一致
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 计算每个用户的NDCG@K
    ndcg_at_k_list = []
    
    for user_id in y_true['user_id'].unique():
        # 获取用户的真实评分
        user_true = y_true[y_true['user_id'] == user_id]
        
        # 获取用户的预测评分
        user_pred = y_pred[y_pred['user_id'] == user_id]
        
        # 获取用户的Top-K推荐
        user_top_k = user_pred.sort_values('score', ascending=False).head(k)
        
        # 创建一个字典来存储每个商品的真实评分
        item_ratings = {}
        for _, row in user_true.iterrows():
            item_ratings[row['item_id']] = row['rating']
        
        # 计算DCG@K
        dcg = 0
        for i, (_, row) in enumerate(user_top_k.iterrows(), 1):
            item_id = row['item_id']
            if item_id in item_ratings:
                # 使用2^rel - 1作为相关性分数
                rel = 2 ** item_ratings[item_id] - 1
                dcg += rel / np.log2(i + 1)
        
        # 计算IDCG@K (理想情况下的DCG)
        # 按照真实评分降序排列的商品
        ideal_items = sorted(
            [(item_id, rating) for item_id, rating in item_ratings.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        idcg = 0
        for i, (_, rating) in enumerate(ideal_items, 1):
            rel = 2 ** rating - 1
            idcg += rel / np.log2(i + 1)
        
        # 计算NDCG@K
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0
        
        ndcg_at_k_list.append(ndcg)
    
    # 返回平均NDCG@K
    return np.mean(ndcg_at_k_list)


def calculate_auc(y_true, y_pred):
    """
    计算ROC曲线下面积 (Area Under the Curve)
    
    参数:
        y_true (array-like): 真实标签 (0或1)
        y_pred (array-like): 预测分数
        
    返回:
        float: AUC值
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # 处理只有一个类别的情况
        return 0.5


def evaluate_recommender(y_true, y_pred, k=10, threshold=3.5):
    """
    全面评估推荐系统性能
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度
        threshold (float): 认为是正例的评分阈值
        
    返回:
        dict: 包含各种评估指标的字典
    """
    # 确保数据格式一致
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 合并真实数据和预测数据
    merged_df = pd.merge(
        y_true[['user_id', 'item_id', 'rating']],
        y_pred[['user_id', 'item_id', 'score']],
        on=['user_id', 'item_id']
    )
    
    # 计算评分预测指标
    rmse = calculate_rmse(merged_df['rating'], merged_df['score'])
    mae = calculate_mae(merged_df['rating'], merged_df['score'])
    
    # 计算二分类指标
    binary_true = (merged_df['rating'] >= threshold).astype(int)
    binary_pred = (merged_df['score'] >= threshold).astype(int)
    precision = precision_score(binary_true, binary_pred, zero_division=0)
    recall = recall_score(binary_true, binary_pred, zero_division=0)
    f1 = f1_score(binary_true, binary_pred, zero_division=0)
    
    # 计算排序指标
    precision_at_k = calculate_precision_at_k(y_true, y_pred, k, threshold)
    recall_at_k = calculate_recall_at_k(y_true, y_pred, k, threshold)
    ndcg_at_k = calculate_ndcg_at_k(y_true, y_pred, k)
    
    # 计算AUC
    auc = calculate_auc(binary_true, merged_df['score'])
    
    # 返回评估结果
    return {
        'rmse': rmse,
        'mae': mae,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        f'precision@{k}': precision_at_k,
        f'recall@{k}': recall_at_k,
        f'ndcg@{k}': ndcg_at_k,
        'auc': auc
    }
 