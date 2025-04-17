#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征工程模块：对预处理后的信用卡审批数据集进行特征选择、转换和创建
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassi***REMOVED***er
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib

def load_processed_data(***REMOVED***le_path='data/processed_data.csv'):
    """
    加载预处理后的数据集
    
    参数:
    ***REMOVED***le_path (str): 数据文件路径
    
    返回:
    pandas.DataFrame: 加载的数据集
    """
    try:
        data = pd.read_csv(***REMOVED***le_path)
        print(f"成功加载预处理后的数据，形状为: {data.shape}")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def analyze_feature_importance(data):
    """
    分析特征重要性
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 特征重要性数据框
    """
    if data is None:
        return None
    
    print("\n开始分析特征重要性...")
    
    # 分离特征和目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 使用随机森林估计特征重要性
    print("使用随机森林估计特征重要性...")
    rf = RandomForestClassi***REMOVED***er(n_estimators=100, random_state=42)
    rf.***REMOVED***t(X, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    
    # 创建特征重要性数据框
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    print("特征重要性 (随机森林):")
    print(feature_importance_df)
    
    # 可视化特征重要性
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('特征重要性 (随机森林)')
    plt.tight_layout()
    plt.save***REMOVED***g('data/feature_importance_rf.png')
    print("特征重要性图已保存到 data/feature_importance_rf.png")
    
    # 使用ANOVA F-value估计特征重要性
    print("\n使用ANOVA F-value估计特征重要性...")
    selector = SelectKBest(f_classif, k='all')
    selector.***REMOVED***t(X, y)
    
    # 获取F值和p值
    anova_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    })
    
    # 按F值排序
    anova_scores = anova_scores.sort_values('F_Score', ascending=False)
    
    print("特征重要性 (ANOVA F-value):")
    print(anova_scores)
    
    # 可视化ANOVA F-value
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='F_Score', y='Feature', data=anova_scores)
    plt.title('特征重要性 (ANOVA F-value)')
    plt.tight_layout()
    plt.save***REMOVED***g('data/feature_importance_anova.png')
    print("ANOVA F-value特征重要性图已保存到 data/feature_importance_anova.png")
    
    # 使用互信息估计特征重要性
    print("\n使用互信息估计特征重要性...")
    selector = SelectKBest(mutual_info_classif, k='all')
    selector.***REMOVED***t(X, y)
    
    # 获取互信息分数
    mi_scores = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': selector.scores_
    })
    
    # 按互信息分数排序
    mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
    
    print("特征重要性 (互信息):")
    print(mi_scores)
    
    # 可视化互信息
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=mi_scores)
    plt.title('特征重要性 (互信息)')
    plt.tight_layout()
    plt.save***REMOVED***g('data/feature_importance_mi.png')
    print("互信息特征重要性图已保存到 data/feature_importance_mi.png")
    
    # 返回包含所有重要性分数的字典
    return {
        'random_forest': feature_importance_df,
        'anova': anova_scores,
        'mutual_info': mi_scores
    }

def select_features(data, importance_scores, method='random_forest', threshold=0.01, k=None):
    """
    基于特征重要性选择特征
    
    参数:
    data (pandas.DataFrame): 输入数据集
    importance_scores (dict): 特征重要性分数
    method (str): 特征选择方法 ('random_forest', 'anova', 'mutual_info', 'rfe')
    threshold (float): 特征重要性阈值 (对于基于阈值的选择)
    k (int): 要选择的特征数量 (对于基于数量的选择)
    
    返回:
    pandas.DataFrame: 包含选定特征的数据集
    list: 选定的特征列表
    """
    if data is None or importance_scores is None:
        return None, None
    
    print(f"\n使用 {method} 方法选择特征...")
    
    # 分离特征和目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    
    selected_features = []
    
    if method == 'random_forest':
        if k is not None:
            # 基于数量选择
            selected_features = importance_scores['random_forest']['Feature'].tolist()[:k]
        else:
            # 基于阈值选择
            selected_features = importance_scores['random_forest'].loc[
                importance_scores['random_forest']['Importance'] >= threshold, 'Feature'
            ].tolist()
    
    elif method == 'anova':
        if k is not None:
            # 基于数量选择
            selected_features = importance_scores['anova']['Feature'].tolist()[:k]
        else:
            # 基于p值选择
            selected_features = importance_scores['anova'].loc[
                importance_scores['anova']['P_Value'] <= 0.05, 'Feature'
            ].tolist()
    
    elif method == 'mutual_info':
        if k is not None:
            # 基于数量选择
            selected_features = importance_scores['mutual_info']['Feature'].tolist()[:k]
        else:
            # 基于阈值选择
            selected_features = importance_scores['mutual_info'].loc[
                importance_scores['mutual_info']['MI_Score'] >= threshold, 'Feature'
            ].tolist()
    
    elif method == 'rfe':
        # 使用递归特征消除
        print("使用递归特征消除 (RFE)...")
        estimator = RandomForestClassi***REMOVED***er(n_estimators=100, random_state=42)
        
        if k is None:
            k = X.shape[1] // 2  # 如果未指定，选择一半的特征
        
        rfe = RFE(estimator=estimator, n_features_to_select=k)
        rfe.***REMOVED***t(X, y)
        
        # 获取选定的特征
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask].tolist()
    
    print(f"选定的特征 ({len(selected_features)}):")
    print(selected_features)
    
    # 创建只包含选定特征的数据集
    selected_data = data[selected_features + ['target']]
    
    return selected_data, selected_features

def create_polynomial_features(data, selected_features, degree=2):
    """
    创建多项式特征
    
    参数:
    data (pandas.DataFrame): 输入数据集
    selected_features (list): 用于创建多项式特征的特征列表
    degree (int): 多项式阶数
    
    返回:
    pandas.DataFrame: 增加了多项式特征的数据集
    sklearn.preprocessing.PolynomialFeatures: 多项式转换器
    """
    if data is None or not selected_features:
        return None, None
    
    print(f"\n创建 {degree} 阶多项式特征...")
    
    # 复制数据，避免修改原始数据
    enhanced_data = data.copy()
    
    # 选择要增强的特征
    X_subset = enhanced_data[selected_features]
    
    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.***REMOVED***t_transform(X_subset)
    
    # 创建多项式特征的列名
    poly_feature_names = []
    powers = poly.powers_[1:]  # 跳过原始特征
    
    for i, power in enumerate(powers):
        name = "poly_"
        for j, feat in enumerate(selected_features):
            if power[j] > 0:
                name += f"{feat}^{power[j]}_"
        poly_feature_names.append(name[:-1])  # 移除最后一个下划线
    
    # 创建包含多项式特征的数据框
    poly_df = pd.DataFrame(poly_features[:, len(selected_features):], columns=poly_feature_names)
    
    # 将多项式特征添加到原始数据集
    enhanced_data = pd.concat([enhanced_data, poly_df], axis=1)
    
    print(f"创建了 {poly_df.shape[1]} 个多项式特征")
    print(f"增强后的数据形状: {enhanced_data.shape}")
    
    return enhanced_data, poly

def apply_pca(data, n_components=None, variance_threshold=0.95):
    """
    应用主成分分析 (PCA) 进行降维
    
    参数:
    data (pandas.DataFrame): 输入数据集
    n_components (int): 要保留的主成分数量
    variance_threshold (float): 要保留的方差比例
    
    返回:
    pandas.DataFrame: PCA转换后的数据集
    sklearn.decomposition.PCA: PCA转换器
    """
    if data is None:
        return None, None
    
    print("\n应用主成分分析 (PCA)...")
    
    # 分离特征和目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 确定主成分数量
    if n_components is None:
        # 使用方差阈值自动确定主成分数量
        pca = PCA()
        pca.***REMOVED***t(X)
        
        # 计算解释的方差比例
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 找到满足方差阈值的最小主成分数量
        n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        
        print(f"基于 {variance_threshold*100:.1f}% 的方差阈值，选择 {n_components} 个主成分")
    else:
        pca = PCA(n_components=n_components)
        pca.***REMOVED***t(X)
    
    # 应用PCA变换
    X_pca = pca.transform(X)
    
    # 创建主成分列名
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    
    # 创建PCA转换后的数据框
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    # 将目标变量添加回数据集
    pca_df['target'] = y.values
    
    # 可视化解释的方差比例
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'r-')
    plt.xlabel('主成分')
    plt.ylabel('解释的方差比例')
    plt.title('PCA: 解释的方差比例')
    plt.axhline(y=variance_threshold, color='g', linestyle='--')
    plt.text(1, variance_threshold+0.02, f'{variance_threshold*100:.1f}% 方差阈值', color='g')
    plt.grid(True)
    plt.save***REMOVED***g('data/pca_explained_variance.png')
    print("PCA解释的方差比例图已保存到 data/pca_explained_variance.png")
    
    print(f"PCA转换后的数据形状: {pca_df.shape}")
    print(f"数据降维: {X.shape[1]} -> {n_components} 特征")
    
    return pca_df, pca

def engineer_features(data, output_***REMOVED***le='data/engineered_data.csv'):
    """
    特征工程主函数
    
    参数:
    data (pandas.DataFrame): 输入数据集
    output_***REMOVED***le (str): 输出数据文件路径
    
    返回:
    tuple: (工程处理后的数据集, 选择的特征, 特征转换器)
    """
    if data is None:
        data = load_processed_data()
        if data is None:
            return None, None, None
    
    print("开始特征工程过程...")
    
    # 分析特征重要性
    importance_scores = analyze_feature_importance(data)
    
    # 选择最重要的特征
    # 这里我们使用随机森林方法选择前10个特征
    _, selected_features = select_features(data, importance_scores, method='random_forest', k=10)
    
    # 首先创建多项式特征
    enhanced_data, poly_transformer = create_polynomial_features(data, selected_features, degree=2)
    
    # 然后应用PCA降维（保留95%的方差）
    ***REMOVED***nal_data, pca_transformer = apply_pca(enhanced_data, variance_threshold=0.95)
    
    # 保存处理后的数据
    if ***REMOVED***nal_data is not None:
        ***REMOVED***nal_data.to_csv(output_***REMOVED***le, index=False)
        print(f"特征工程处理后的数据已保存到 {output_***REMOVED***le}")
        
        # 保存特征转换器
        transformers = {
            'selected_features': selected_features,
            'poly_transformer': poly_transformer,
            'pca_transformer': pca_transformer
        }
        joblib.dump(transformers, 'data/feature_transformers.pkl')
        print("特征转换器已保存到 data/feature_transformers.pkl")
    
    print("特征工程完成！")
    return ***REMOVED***nal_data, selected_features, (poly_transformer, pca_transformer)

def main():
    """主函数"""
    data = load_processed_data()
    if data is not None:
        engineer_features(data)

if __name__ == "__main__":
    main() 