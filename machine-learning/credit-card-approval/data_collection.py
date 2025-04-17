#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据收集模块：从UCI机器学习存储库获取信用卡审批数据集
"""

import os
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

def download_credit_card_data(save_path='data/'):
    """
    从UCI机器学习存储库下载信用卡审批数据集
    
    参数:
    save_path (str): 保存数据的路径
    
    返回:
    pandas.DataFrame: 下载的数据集
    """
    print("正在从UCI下载信用卡审批数据集...")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # UCI数据集URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    
    try:
        # 下载数据
        response = requests.get(url)
        response.raise_for_status()  # 检查是否成功下载
        
        # 为数据集定义列名
        column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                         'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
        
        # 将数据加载到pandas DataFrame
        data = pd.read_csv(StringIO(response.text), names=column_names, 
                           na_values='?', header=None)
        
        # 将最后一列作为目标变量（是否批准）
        # 在这个数据集中，'+'表示批准，'-'表示拒绝
        data.rename(columns={'A16': 'target'}, inplace=True)
        
        # 将目标变量转换为二进制(1表示批准，0表示拒绝)
        data['target'] = data['target'].map({'+': 1, '-': 0})
        
        # 保存原始数据
        data.to_csv(os.path.join(save_path, 'credit_card_applications.csv'), index=False)
        print(f"数据已保存到 {os.path.join(save_path, 'credit_card_applications.csv')}")
        
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"下载数据时发生错误: {e}")
        return None

def analyze_data(data):
    """
    对数据集进行基本分析
    
    参数:
    data (pandas.DataFrame): 要分析的数据集
    """
    if data is None:
        print("没有数据可供分析")
        return
    
    print("\n数据基本分析:")
    print(f"数据集形状: {data.shape}")
    print(f"特征数量: {data.shape[1] - 1}")  # 减去目标变量
    print(f"样本数量: {data.shape[0]}")
    
    # 检查数据类型
    print("\n数据类型:")
    print(data.dtypes)
    
    # 检查缺失值
    print("\n缺失值统计:")
    missing_values = data.isnull().sum()
    print(missing_values)
    missing_percentage = (missing_values / len(data)) * 100
    print(f"\n缺失值百分比:")
    print(missing_percentage)
    
    # 目标变量分布
    print("\n目标变量分布:")
    target_counts = data['target'].value_counts()
    print(target_counts)
    print(f"批准率: {target_counts[1] / len(data) * 100:.2f}%")
    
    # 可视化目标变量分布
    plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))
    sns.countplot(x='target', data=data)
    plt.title('信用卡申请批准/拒绝分布')
    plt.xlabel('申请结果 (1=批准, 0=拒绝)')
    plt.ylabel('数量')
    plt.save***REMOVED***g('data/target_distribution.png')
    print("目标变量分布图已保存到 data/target_distribution.png")
    
    # 检查数值特征的分布
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('target')  # 排除目标变量
    
    if numeric_columns:
        print("\n数值特征的基本统计量:")
        print(data[numeric_columns].describe())
        
        # 可视化数值特征
        plt.***REMOVED***gure(***REMOVED***gsize=(15, 10))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(3, 3, i)
            sns.histplot(data[col].dropna(), kde=True)
            plt.title(f'特征 {col} 分布')
        plt.tight_layout()
        plt.save***REMOVED***g('data/numeric_features_distribution.png')
        print("数值特征分布图已保存到 data/numeric_features_distribution.png")
    
    # 检查分类特征的分布
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        print("\n分类特征的值计数:")
        for col in categorical_columns:
            print(f"\n特征 {col} 的值计数:")
            print(data[col].value_counts())
            print(f"唯一值数量: {data[col].nunique()}")
        
        # 可视化前5个分类特征
        plt.***REMOVED***gure(***REMOVED***gsize=(15, 10))
        for i, col in enumerate(categorical_columns[:5], 1):
            plt.subplot(2, 3, i)
            value_counts = data[col].value_counts()
            value_counts.plot(kind='bar')
            plt.title(f'特征 {col} 分布')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.save***REMOVED***g('data/categorical_features_distribution.png')
        print("分类特征分布图已保存到 data/categorical_features_distribution.png")

def main():
    """主函数"""
    print("开始数据收集过程...")
    
    # 下载数据
    data = download_credit_card_data()
    
    if data is not None:
        # 分析数据
        analyze_data(data)
        print("数据收集和初步分析完成！")
    else:
        print("由于无法下载数据，数据收集过程失败。")

if __name__ == "__main__":
    main() 