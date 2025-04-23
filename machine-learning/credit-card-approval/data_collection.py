#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据收集模块：从UCI机器学习存储库获取信用卡审批数据集

本模块实现了以下功能：
1. 从UCI机器学习存储库下载信用卡审批数据集
2. 对数据进行基本预处理（处理缺失值，转换目标变量等）
3. 进行初步数据分析（统计特征，可视化分布等）
4. 生成数据洞察报告和可视化图表

数据集来源：https://archive.ics.uci.edu/ml/datasets/Credit+Approval
"""

import os                   # 用于操作文件系统，创建目录等
import pandas as pd         # 用于数据处理和分析
import numpy as np          # 用于数值计算
import requests             # 用于发送HTTP请求获取数据
from io import StringIO     # 用于将字符串转换为类文件对象
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns       # 用于高级数据可视化
import matplotlib           # 用于配置可视化参数

# 设置matplotlib显示中文字体，确保中文标签正确显示
# 按优先顺序尝试使用不同的字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决matplotlib中负号显示为方块的问题

def download_credit_card_data(save_path='machine-learning/credit-card-approval/data/'):
    """
    从UCI机器学习存储库下载信用卡审批数据集并进行初步处理
    
    该函数完成以下任务：
    1. 从UCI存储库下载原始信用卡审批数据
    2. 为数据集添加列名（特征名称）
    3. 处理数据中的缺失值（'?'替换为NaN）
    4. 将目标变量转换为二进制格式
    5. 将处理后的数据保存到本地CSV文件
    
    参数:
    save_path (str): 保存数据的本地路径，默认为'machine-learning/credit-card-approval/data/'
    
    返回:
    pandas.DataFrame: 下载并处理后的数据集，如果下载失败则返回None
    """
    print("正在从UCI下载信用卡审批数据集...")
    
    # 创建保存目录，exist_ok=True表示如果目录已存在则不会报错
    os.makedirs(save_path, exist_ok=True)
    
    # UCI数据集URL - 指向信用卡审批数据集的直接链接
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    
    try:
        # 发送HTTP GET请求下载数据
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败（状态码不是200），则抛出异常
        
        # 为数据集定义列名 - 原始数据没有列名，这里使用A1-A16作为占位符
        # 实际上，这些特征代表申请人的各种属性和信用信息
        column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                         'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
        
        # 将数据加载到pandas DataFrame
        # StringIO将文本转换为类文件对象，以便pd.read_csv可以处理
        # na_values='?'将所有问号替换为NaN（缺失值）
        # header=None表示原始数据没有列标题行
        data = pd.read_csv(StringIO(response.text), names=column_names, 
                           na_values='?', header=None)
        
        # 将最后一列（A16）重命名为目标变量'target'
        # 在这个数据集中，'+'表示信用卡申请被批准，'-'表示申请被拒绝
        data.rename(columns={'A16': 'target'}, inplace=True)
        
        # 将目标变量转换为二进制数值: 1表示批准，0表示拒绝
        # 这有助于后续的机器学习模型处理
        data['target'] = data['target'].map({'+': 1, '-': 0})
        
        # 保存处理后的原始数据到CSV文件
        # os.path.join用于生成跨平台的路径
        # index=False表示不保存DataFrame的行索引
        data.to_csv(os.path.join(save_path, 'credit_card_applications.csv'), index=False)
        print(f"数据已保存到 {os.path.join(save_path, 'credit_card_applications.csv')}")
        
        return data
    
    except requests.exceptions.RequestException as e:
        # 捕获并处理下载过程中可能发生的任何HTTP请求异常
        print(f"下载数据时发生错误: {e}")
        return None

def analyze_data(data, save_path='machine-learning/credit-card-approval/data/'):
    """
    对信用卡审批数据集进行全面的探索性数据分析
    
    该函数执行以下分析任务：
    1. 计算并显示数据集的基本统计信息（形状、特征数量等）
    2. 分析数据类型和缺失值情况
    3. 分析并可视化目标变量的分布
    4. 分析并可视化数值型特征的分布
    5. 分析并可视化分类型特征的分布
    6. 将所有可视化结果保存为高分辨率图像
    
    参数:
    data (pandas.DataFrame): 要分析的数据集
    save_path (str): 保存分析图表的路径，默认为'machine-learning/credit-card-approval/data/'
    """
    if data is None:
        print("没有数据可供分析")
        return
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # ========== 1. 数据集基本信息分析 ==========
    print("\n数据基本分析:")
    print(f"数据集形状: {data.shape}")
    print(f"特征数量: {data.shape[1] - 1}")  # 减去目标变量列
    print(f"样本数量: {data.shape[0]}")
    
    # ========== 2. 数据类型和缺失值分析 ==========
    # 检查每列的数据类型，了解哪些是数值型、哪些是分类型
    print("\n数据类型:")
    print(data.dtypes)
    
    # 检查每列的缺失值数量
    print("\n缺失值统计:")
    missing_values = data.isnull().sum()
    print(missing_values)
    
    # 计算每列缺失值的百分比，帮助评估缺失值的严重程度
    missing_percentage = (missing_values / len(data)) * 100
    print(f"\n缺失值百分比:")
    print(missing_percentage)
    
    # ========== 3. 目标变量分布分析 ==========
    print("\n目标变量分布:")
    target_counts = data['target'].value_counts()
    print(target_counts)
    # 计算批准率，评估数据集的类别平衡性
    print(f"批准率: {target_counts[1] / len(data) * 100:.2f}%")
    
    # 可视化目标变量分布 - 使用柱状图显示批准和拒绝的数量
    plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))  # 设置图形大小
    sns.countplot(x='target', data=data)  # 使用seaborn绘制计数图
    plt.title('信用卡申请批准/拒绝分布')
    plt.xlabel('申请结果 (1=批准, 0=拒绝)')
    plt.ylabel('数量')
    # 保存图像，dpi=300表示高分辨率
    plt.save***REMOVED***g(os.path.join(save_path, 'target_distribution.png'), dpi=300)
    print(f"目标变量分布图已保存到 {os.path.join(save_path, 'target_distribution.png')}")
    
    # ========== 4. 数值特征分析 ==========
    # 筛选所有数值型列（排除目标变量）
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('target')  # 排除目标变量
    
    if numeric_columns:
        # 计算数值特征的描述性统计量（均值、标准差、分位数等）
        print("\n数值特征的基本统计量:")
        print(data[numeric_columns].describe())
        
        # 可视化每个数值特征的分布
        plt.***REMOVED***gure(***REMOVED***gsize=(15, 10))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(3, 3, i)  # 创建3x3的子图网格
            # 使用直方图+密度曲线显示分布，dropna()去除缺失值
            sns.histplot(data[col].dropna(), kde=True)
            plt.title(f'特征 {col} 分布')
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.save***REMOVED***g(os.path.join(save_path, 'numeric_features_distribution.png'), dpi=300)
        print(f"数值特征分布图已保存到 {os.path.join(save_path, 'numeric_features_distribution.png')}")
    
    # ========== 5. 分类特征分析 ==========
    # 筛选所有对象（字符串）类型的列，这些通常是分类特征
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        print("\n分类特征的值计数:")
        for col in categorical_columns:
            # 显示每个分类特征的类别及其计数
            print(f"\n特征 {col} 的值计数:")
            print(data[col].value_counts())
            # 显示唯一值数量，评估基数（cardinality）
            print(f"唯一值数量: {data[col].nunique()}")
        
        # 可视化前5个分类特征的分布（如果分类特征太多，只展示部分）
        plt.***REMOVED***gure(***REMOVED***gsize=(15, 10))
        for i, col in enumerate(categorical_columns[:5], 1):
            plt.subplot(2, 3, i)
            # 计算每个类别的频率
            value_counts = data[col].value_counts()
            # 使用条形图显示分布
            value_counts.plot(kind='bar')
            plt.title(f'特征 {col} 分布')
            plt.xticks(rotation=45)  # 旋转x轴标签以避免重叠
        plt.tight_layout()
        plt.save***REMOVED***g(os.path.join(save_path, 'categorical_features_distribution.png'), dpi=300)
        print(f"分类特征分布图已保存到 {os.path.join(save_path, 'categorical_features_distribution.png')}")

def main():
    """
    主函数：协调数据收集和分析的完整流程
    
    该函数执行以下步骤：
    1. 调用download_credit_card_data()下载数据
    2. 如果数据下载成功，调用analyze_data()进行数据分析
    3. 打印流程完成状态信息
    """
    print("开始数据收集过程...")
    
    # 下载数据集
    data = download_credit_card_data()
    
    if data is not None:
        # 如果数据下载成功，进行数据分析
        analyze_data(data)
        print("数据收集和初步分析完成！")
    else:
        # 如果数据下载失败，打印错误信息
        print("由于无法下载数据，数据收集过程失败。")

# 当直接运行该脚本时执行main()函数
if __name__ == "__main__":
    main() 