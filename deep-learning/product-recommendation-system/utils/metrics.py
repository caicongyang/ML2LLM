"""
推荐系统评估指标工具 - 各种评估指标的实现
本模块提供了全面的推荐系统评估指标计算函数，包括:
1. 评分预测指标: RMSE, MAE
2. 排序评估指标: Precision@K, Recall@K, NDCG@K
3. 分类评估指标: Precision, Recall, F1, AUC
4. 综合评估函数
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差 (Root Mean Squared Error)
    RMSE衡量预测评分与实际评分的偏差，对较大误差的惩罚更重
    
    计算公式: RMSE = sqrt(sum((y_true - y_pred)^2) / n)
    
    参数:
        y_true (array-like): 真实评分值数组
        y_pred (array-like): 模型预测的评分值数组
        
    返回:
        float: RMSE值，越小表示预测越准确
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差 (Mean Absolute Error)
    MAE衡量预测评分与实际评分的平均绝对偏差，对所有误差的权重相同
    
    计算公式: MAE = sum(|y_true - y_pred|) / n
    
    参数:
        y_true (array-like): 真实评分值数组
        y_pred (array-like): 模型预测的评分值数组
        
    返回:
        float: MAE值，越小表示预测越准确
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_precision_at_k(y_true, y_pred, k=10, threshold=3.5):
    """
    计算Top-K准确率 (Precision@K)
    衡量推荐的Top-K个商品中有多少比例是用户真正喜欢的
    
    计算公式: Precision@K = |推荐的相关商品| / |推荐的商品|
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度，即只考虑预测分数最高的K个商品
        threshold (float): 认为用户喜欢商品的评分阈值，高于此值视为相关/正例
        
    返回:
        float: Precision@K值，范围[0,1]，越高表示推荐越准确
    """
    # 创建数据拷贝避免修改原始数据
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    # 统一列名，确保数据格式一致性
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 用于存储每个用户的Precision@K值
    precision_at_k_list = []
    
    # 对每个用户分别计算Precision@K
    for user_id in y_true['user_id'].unique():
        # 获取该用户的真实评分数据
        user_true = y_true[y_true['user_id'] == user_id]
        
        # 获取该用户的预测评分数据
        user_pred = y_pred[y_pred['user_id'] == user_id]
        
        # 获取预测分数最高的K个商品（Top-K推荐）
        user_top_k = user_pred.sort_values('score', ascending=False).head(k)
        
        # 获取用户实际喜欢的商品集合（评分>=阈值的商品）
        relevant_items = set(user_true[user_true['rating'] >= threshold]['item_id'])
        # 获取推荐给用户的商品集合
        recommended_items = set(user_top_k['item_id'])
        
        # 计算该用户的Precision@K: 推荐的相关商品数 / 推荐的商品总数
        if not recommended_items:
            precision = 0  # 如果没有推荐任何商品，准确率为0
        else:
            # 计算推荐商品中有多少是用户喜欢的（交集大小/推荐集合大小）
            precision = len(relevant_items.intersection(recommended_items)) / len(recommended_items)
        
        precision_at_k_list.append(precision)
    
    # 返回所有用户Precision@K的平均值
    return np.mean(precision_at_k_list)


def calculate_recall_at_k(y_true, y_pred, k=10, threshold=3.5):
    """
    计算Top-K召回率 (Recall@K)
    衡量在用户所有喜欢的商品中，有多少比例被成功推荐在Top-K列表中
    
    计算公式: Recall@K = |推荐的相关商品| / |所有相关商品|
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度，即只考虑预测分数最高的K个商品
        threshold (float): 认为用户喜欢商品的评分阈值，高于此值视为相关/正例
        
    返回:
        float: Recall@K值，范围[0,1]，越高表示推荐的覆盖率越高
    """
    # 创建数据拷贝避免修改原始数据
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    # 统一列名，确保数据格式一致性
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 用于存储每个用户的Recall@K值
    recall_at_k_list = []
    
    # 对每个用户分别计算Recall@K
    for user_id in y_true['user_id'].unique():
        # 获取该用户的真实评分数据
        user_true = y_true[y_true['user_id'] == user_id]
        
        # 获取该用户的预测评分数据
        user_pred = y_pred[y_pred['user_id'] == user_id]
        
        # 获取预测分数最高的K个商品（Top-K推荐）
        user_top_k = user_pred.sort_values('score', ascending=False).head(k)
        
        # 获取用户实际喜欢的所有商品集合（评分>=阈值的商品）
        relevant_items = set(user_true[user_true['rating'] >= threshold]['item_id'])
        # 获取推荐给用户的商品集合
        recommended_items = set(user_top_k['item_id'])
        
        # 计算该用户的Recall@K: 推荐的相关商品数 / 所有相关商品数
        if not relevant_items:
            recall = 0  # 如果用户没有喜欢的商品，召回率为0
        else:
            # 计算用户喜欢的商品中有多少被成功推荐（交集大小/相关集合大小）
            recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items)
        
        recall_at_k_list.append(recall)
    
    # 返回所有用户Recall@K的平均值
    return np.mean(recall_at_k_list)


def calculate_ndcg_at_k(y_true, y_pred, k=10):
    """
    计算归一化折损累积增益 (Normalized Discounted Cumulative Gain, NDCG@K)
    NDCG不仅考虑推荐商品的相关性，还考虑了它们在推荐列表中的排序位置
    排名越靠前的相关商品对最终分数的贡献越大
    
    DCG@k = sum_(i=1)^k (2^rel_i - 1) / log_2(i+1)
    IDCG@k是理想情况下（按相关性降序排列）的DCG@k
    NDCG@k = DCG@k / IDCG@k
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度，即只考虑预测分数最高的K个商品
        
    返回:
        float: NDCG@K值，范围[0,1]，越高表示推荐质量越好
    """
    # 创建数据拷贝避免修改原始数据
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    # 统一列名，确保数据格式一致性
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 用于存储每个用户的NDCG@K值
    ndcg_at_k_list = []
    
    # 对每个用户分别计算NDCG@K
    for user_id in y_true['user_id'].unique():
        # 获取该用户的真实评分数据
        user_true = y_true[y_true['user_id'] == user_id]
        
        # 获取该用户的预测评分数据
        user_pred = y_pred[y_pred['user_id'] == user_id]
        
        # 获取预测分数最高的K个商品（Top-K推荐）
        user_top_k = user_pred.sort_values('score', ascending=False).head(k)
        
        # 创建商品ID到真实评分的映射字典，便于快速查找
        item_ratings = {}
        for _, row in user_true.iterrows():
            item_ratings[row['item_id']] = row['rating']
        
        # 计算折损累积增益(DCG@K)
        # DCG衡量推荐列表的质量，考虑商品的相关性和排序位置
        dcg = 0
        for i, (_, row) in enumerate(user_top_k.iterrows(), 1):
            item_id = row['item_id']
            if item_id in item_ratings:
                # 使用2^rel - 1作为相关性增益，rel为商品评分
                # 这种增益计算方式更强调高评分的商品
                rel = 2 ** item_ratings[item_id] - 1
                # 按位置折损：排序靠前的商品权重更大
                dcg += rel / np.log2(i + 1)
        
        # 计算理想情况下的DCG(IDCG@K)
        # 理想情况是按照真实评分从高到低排序推荐商品
        # 首先获取按评分降序排列的商品列表
        ideal_items = sorted(
            [(item_id, rating) for item_id, rating in item_ratings.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]  # 只取前k个
        
        # 计算理想DCG
        idcg = 0
        for i, (_, rating) in enumerate(ideal_items, 1):
            rel = 2 ** rating - 1
            idcg += rel / np.log2(i + 1)
        
        # 计算NDCG@K = DCG@K / IDCG@K，将DCG归一化到[0,1]区间
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0  # 如果没有相关商品，NDCG为0
        
        ndcg_at_k_list.append(ndcg)
    
    # 返回所有用户NDCG@K的平均值
    return np.mean(ndcg_at_k_list)


def calculate_auc(y_true, y_pred):
    """
    计算ROC曲线下面积 (Area Under the Curve, AUC)
    AUC衡量推荐系统区分相关和不相关商品的能力
    AUC = 0.5表示随机猜测，AUC = 1表示完美区分
    
    计算方法: AUC是ROC曲线（真正例率vs假正例率）下的面积
    
    参数:
        y_true (array-like): 真实标签数组 (0表示不相关，1表示相关)
        y_pred (array-like): 预测分数数组 (连续值，越高表示越可能相关)
        
    返回:
        float: AUC值，范围[0,1]，越高表示推荐系统性能越好
    """
    try:
        # 使用sklearn提供的AUC计算函数
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # 当数据中只有一个类别时，AUC计算会出错
        # 此时返回0.5，表示系统性能等同于随机猜测
        return 0.5


def evaluate_recommender(y_true, y_pred, k=10, threshold=3.5):
    """
    全面评估推荐系统性能 - 集成多种评估指标
    同时计算评分预测指标、分类指标和排序指标，提供完整评估
    
    参数:
        y_true (pd.DataFrame): 包含[user_id, item_id, rating]的真实评分数据
        y_pred (pd.DataFrame): 包含[user_id, item_id, score]的预测评分数据
        k (int): 推荐列表长度，用于计算Precision@K等指标
        threshold (float): 认为用户喜欢商品的评分阈值，用于将评分转化为二分类
        
    返回:
        dict: 包含各种评估指标的字典，包括:
            - rmse: 均方根误差
            - mae: 平均绝对误差
            - precision: 整体准确率
            - recall: 整体召回率
            - f1: F1分数，准确率和召回率的调和平均
            - precision@k: Top-K准确率
            - recall@k: Top-K召回率
            - ndcg@k: 归一化折损累积增益
            - auc: ROC曲线下面积
    """
    # 创建数据拷贝避免修改原始数据
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    
    # 统一列名，确保数据格式一致性
    if 'rating' not in y_true.columns and 'score' in y_true.columns:
        y_true.rename(columns={'score': 'rating'}, inplace=True)
    
    if 'score' not in y_pred.columns and 'rating' in y_pred.columns:
        y_pred.rename(columns={'rating': 'score'}, inplace=True)
    
    # 合并真实数据和预测数据，只保留两者都有的用户-商品对
    # 这样可以在相同的数据上计算所有指标
    merged_df = pd.merge(
        y_true[['user_id', 'item_id', 'rating']],  # 真实评分数据
        y_pred[['user_id', 'item_id', 'score']],   # 预测评分数据
        on=['user_id', 'item_id']  # 按用户ID和商品ID匹配
    )
    
    # 1. 计算评分预测指标 - 衡量评分预测的准确性
    rmse = calculate_rmse(merged_df['rating'], merged_df['score'])
    mae = calculate_mae(merged_df['rating'], merged_df['score'])
    
    # 2. 计算二分类指标 - 将评分问题转化为二分类问题
    # 将评分转换为二分类标签：高于阈值为1(喜欢)，否则为0(不喜欢)
    binary_true = (merged_df['rating'] >= threshold).astype(int)
    binary_pred = (merged_df['score'] >= threshold).astype(int)
    
    # 计算整体的准确率、召回率和F1分数
    precision = precision_score(binary_true, binary_pred, zero_division=0)  # 避免除零错误
    recall = recall_score(binary_true, binary_pred, zero_division=0)
    f1 = f1_score(binary_true, binary_pred, zero_division=0)
    
    # 3. 计算排序指标 - 考虑推荐的排序质量
    # 使用原始DataFrame计算，因为这些指标需要考虑每个用户的所有商品
    precision_at_k = calculate_precision_at_k(y_true, y_pred, k, threshold)
    recall_at_k = calculate_recall_at_k(y_true, y_pred, k, threshold)
    ndcg_at_k = calculate_ndcg_at_k(y_true, y_pred, k)
    
    # 4. 计算AUC - 衡量区分相关/不相关商品的能力
    # 使用预测分数而不是二分类预测，因为AUC需要连续值来绘制ROC曲线
    auc = calculate_auc(binary_true, merged_df['score'])
    
    # 返回包含所有评估指标的字典
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
 