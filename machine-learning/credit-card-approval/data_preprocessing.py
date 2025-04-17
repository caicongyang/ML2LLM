#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理模块：对信用卡审批数据集进行清洗和预处理
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

def load_data(***REMOVED***le_path='data/credit_card_applications.csv'):
    """
    加载数据集
    
    参数:
    ***REMOVED***le_path (str): 数据文件路径
    
    返回:
    pandas.DataFrame: 加载的数据集
    """
    try:
        data = pd.read_csv(***REMOVED***le_path)
        print(f"成功加载数据，形状为: {data.shape}")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def handle_missing_values(data):
    """
    处理数据集中的缺失值
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 处理后的数据集
    """
    if data is None:
        return None
    
    print("\n开始处理缺失值...")
    
    # 显示缺失值统计
    missing_values = data.isnull().sum()
    print("缺失值统计:")
    print(missing_values[missing_values > 0])
    
    # 复制数据，避免修改原始数据
    processed_data = data.copy()
    
    # 分离数值特征和分类特征
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = processed_data.select_dtypes(include=['object']).columns.tolist()
    
    # 对数值特征使用中位数填充
    if numeric_columns:
        print("使用中位数填充数值特征的缺失值...")
        imputer = SimpleImputer(strategy='median')
        processed_data[numeric_columns] = imputer.***REMOVED***t_transform(processed_data[numeric_columns])
    
    # 对分类特征使用众数填充
    if categorical_columns:
        print("使用众数填充分类特征的缺失值...")
        imputer = SimpleImputer(strategy='most_frequent')
        processed_data[categorical_columns] = imputer.***REMOVED***t_transform(processed_data[categorical_columns])
    
    # 检查是否还有缺失值
    remaining_missing = processed_data.isnull().sum().sum()
    print(f"填充后剩余的缺失值数量: {remaining_missing}")
    
    return processed_data

def detect_and_handle_outliers(data, contamination=0.05):
    """
    检测和处理异常值
    
    参数:
    data (pandas.DataFrame): 输入数据集
    contamination (float): 异常值比例的估计
    
    返回:
    pandas.DataFrame: 处理后的数据集
    """
    if data is None:
        return None
    
    print("\n开始检测和处理异常值...")
    
    # 复制数据，避免修改原始数据
    processed_data = data.copy()
    
    # 只对数值特征检测异常值
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_columns:
        numeric_columns.remove('target')  # 排除目标变量
    
    if not numeric_columns:
        print("没有数值特征可以检测异常值")
        return processed_data
    
    # 使用IsolationForest检测异常值
    print("使用IsolationForest检测异常值...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.***REMOVED***t_predict(processed_data[numeric_columns])
    
    # 标记异常值 (1=正常, -1=异常)
    processed_data['is_outlier'] = outliers
    
    # 统计异常值数量
    outlier_count = (outliers == -1).sum()
    print(f"检测到 {outlier_count} 个异常样本 (约 {outlier_count/len(processed_data)*100:.2f}%)")
    
    # 可视化异常值分布
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
    sns.countplot(x='is_outlier', data=processed_data)
    plt.title('异常值分布')
    plt.xlabel('是否为异常值 (1=正常, -1=异常)')
    plt.ylabel('数量')
    plt.save***REMOVED***g('data/outliers_distribution.png')
    print("异常值分布图已保存到 data/outliers_distribution.png")
    
    # 对异常值进行处理
    # 方法1: 移除异常值
    # processed_data = processed_data[processed_data['is_outlier'] == 1]
    
    # 方法2: 将异常值替换为边界值（截断）
    for col in numeric_columns:
        Q1 = processed_data[col].quantile(0.25)
        Q3 = processed_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 截断异常值
        processed_data.loc[(processed_data['is_outlier'] == -1) & (processed_data[col] < lower_bound), col] = lower_bound
        processed_data.loc[(processed_data['is_outlier'] == -1) & (processed_data[col] > upper_bound), col] = upper_bound
    
    # 移除临时的异常值标记列
    processed_data.drop('is_outlier', axis=1, inplace=True)
    
    return processed_data

def encode_categorical_features(data):
    """
    对分类特征进行编码
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 处理后的数据集
    tuple: (编码器字典, 分类特征列表)
    """
    if data is None:
        return None, None
    
    print("\n开始编码分类特征...")
    
    # 复制数据，避免修改原始数据
    processed_data = data.copy()
    
    # 获取分类特征列
    categorical_columns = processed_data.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_columns:
        print("没有分类特征需要编码")
        return processed_data, {}
    
    print(f"需要编码的分类特征: {categorical_columns}")
    
    # 创建编码器字典
    encoders = {}
    
    # 对每个分类特征进行标签编码
    for col in categorical_columns:
        encoder = LabelEncoder()
        processed_data[col] = encoder.***REMOVED***t_transform(processed_data[col])
        encoders[col] = encoder
        
        # 打印编码映射
        unique_values = data[col].unique()
        encoded_values = encoder.transform(unique_values)
        mapping = dict(zip(unique_values, encoded_values))
        print(f"特征 {col} 的编码映射: {mapping}")
    
    return processed_data, (encoders, categorical_columns)

def normalize_features(data):
    """
    对数值特征进行标准化
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 处理后的数据集
    object: 标准化器
    """
    if data is None:
        return None, None
    
    print("\n开始标准化数值特征...")
    
    # 复制数据，避免修改原始数据
    processed_data = data.copy()
    
    # 获取数值特征列
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_columns:
        numeric_columns.remove('target')  # 排除目标变量
    
    if not numeric_columns:
        print("没有数值特征需要标准化")
        return processed_data, None
    
    print(f"需要标准化的数值特征: {numeric_columns}")
    
    # 使用StandardScaler标准化数值特征
    scaler = StandardScaler()
    processed_data[numeric_columns] = scaler.***REMOVED***t_transform(processed_data[numeric_columns])
    
    # 显示标准化后的统计信息
    print("标准化后的数值特征统计信息:")
    print(processed_data[numeric_columns].describe())
    
    return processed_data, scaler

def check_data_quality(data, original_data):
    """
    检查数据质量，确保预处理后的数据质量良好
    
    参数:
    data (pandas.DataFrame): 处理后的数据集
    original_data (pandas.DataFrame): 原始数据集
    """
    if data is None or original_data is None:
        return
    
    print("\n检查数据质量...")
    
    # 检查数据形状变化
    print(f"原始数据形状: {original_data.shape}")
    print(f"处理后数据形状: {data.shape}")
    
    # 检查缺失值
    missing_values = data.isnull().sum().sum()
    print(f"处理后的缺失值数量: {missing_values}")
    
    # 检查目标变量分布变化
    original_target_dist = original_data['target'].value_counts(normalize=True)
    processed_target_dist = data['target'].value_counts(normalize=True)
    
    print("目标变量分布比较 (百分比):")
    comparison = pd.DataFrame({
        '原始数据': original_target_dist * 100,
        '处理后数据': processed_target_dist * 100
    })
    print(comparison)
    
    # 可视化比较
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='target', data=original_data)
    plt.title('原始数据目标分布')
    plt.xlabel('申请结果 (1=批准, 0=拒绝)')
    plt.ylabel('数量')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='target', data=data)
    plt.title('处理后数据目标分布')
    plt.xlabel('申请结果 (1=批准, 0=拒绝)')
    plt.ylabel('数量')
    
    plt.tight_layout()
    plt.save***REMOVED***g('data/target_distribution_comparison.png')
    print("目标分布比较图已保存到 data/target_distribution_comparison.png")

def preprocess_data(input_***REMOVED***le='data/credit_card_applications.csv', output_***REMOVED***le='data/processed_data.csv'):
    """
    数据预处理主函数
    
    参数:
    input_***REMOVED***le (str): 输入数据文件路径
    output_***REMOVED***le (str): 输出数据文件路径
    
    返回:
    tuple: (处理后的数据集, 编码器字典, 标准化器)
    """
    print("开始数据预处理过程...")
    
    # 加载数据
    original_data = load_data(input_***REMOVED***le)
    if original_data is None:
        return None, None, None
    
    # 处理缺失值
    data = handle_missing_values(original_data)
    
    # 处理异常值
    data = detect_and_handle_outliers(data)
    
    # 编码分类特征
    data, encoders = encode_categorical_features(data)
    
    # 标准化数值特征
    data, scaler = normalize_features(data)
    
    # 检查数据质量
    check_data_quality(data, original_data)
    
    # 保存处理后的数据
    if data is not None:
        data.to_csv(output_***REMOVED***le, index=False)
        print(f"处理后的数据已保存到 {output_***REMOVED***le}")
    
    print("数据预处理完成！")
    return data, encoders, scaler

def main():
    """主函数"""
    preprocess_data()

if __name__ == "__main__":
    main() 