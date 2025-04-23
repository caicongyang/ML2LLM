#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理模块：对信用卡审批数据集进行清洗和预处理

本模块实现以下功能：
1. 加载由data_collection模块收集的信用卡审批原始数据
2. 处理数据中的缺失值（使用中位数和众数填充）
3. 检测和处理异常值（使用隔离森林算法）
4. 对分类特征进行编码（使用标签编码）
5. 对数值特征进行标准化（使用Z-score标准化）
6. 检查预处理前后的数据质量
7. 保存处理后的数据集以供建模使用

预处理是机器学习流程中的关键步骤，可以显著提高模型性能并减少训练时间。
"""

import os                      # 用于文件和目录操作
import pandas as pd            # 用于数据处理和分析
import numpy as np             # 用于科学计算和数组操作
import matplotlib.pyplot as plt # 用于数据可视化
import seaborn as sns          # 用于高级数据可视化
from sklearn.impute import SimpleImputer       # 用于填充缺失值
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 用于特征编码和标准化
from sklearn.ensemble import IsolationForest   # 用于异常值检测

# 获取脚本所在目录的绝对路径，用于确保在任何目录运行脚本时都能找到正确的文件
# 这解决了相对路径在不同环境中可能导致的文件不存在问题
SCRIPT_DIR = os.path.dirname(os.path.abspath(__***REMOVED***le__))

# 设置matplotlib的中文字体支持，确保图表中的中文正确显示
# 按照优先级尝试不同的字体，适应不同操作系统环境
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 首选Arial Unicode MS，备选黑体和微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题，避免显示为方块

def get_absolute_path(relative_path):
    """
    将相对于脚本的路径转换为绝对路径
    
    在不同环境下运行脚本时，相对路径可能会导致文件找不到的问题。
    该函数确保无论从哪里运行脚本，都能正确找到所需的文件。
    
    参数:
    relative_path (str): 相对路径，如'data/***REMOVED***le.csv'
    
    返回:
    str: 完整的绝对路径，如'/home/user/project/data/***REMOVED***le.csv'
    """
    return os.path.join(SCRIPT_DIR, relative_path)

def load_data(***REMOVED***le_path='data/credit_card_applications.csv'):
    """
    加载信用卡审批数据集
    
    该函数负责从指定路径读取数据集CSV文件，并将其加载到pandas DataFrame中。
    如果文件不存在或读取过程中出现错误，将返回None并打印错误信息。
    
    参数:
    ***REMOVED***le_path (str): 数据文件路径，默认为'data/credit_card_applications.csv'
    
    返回:
    pandas.DataFrame: 加载的数据集，如果加载失败则返回None
    """
    # 转换为绝对路径，确保在任何环境下都能找到文件
    abs_***REMOVED***le_path = get_absolute_path(***REMOVED***le_path)
    try:
        # 尝试读取CSV文件到DataFrame
        data = pd.read_csv(abs_***REMOVED***le_path)
        print(f"成功加载数据，形状为: {data.shape}")
        return data
    except Exception as e:
        # 捕获并处理可能的异常，如文件不存在、格式错误等
        print(f"加载数据时出错: {e}")
        return None

def handle_missing_values(data):
    """
    处理数据集中的缺失值
    
    该函数对数据集中的缺失值进行智能填充：
    - 对数值型特征使用中位数填充，这比均值更稳健，不受极端值影响
    - 对分类型特征使用众数填充，保持类别的合理性
    
    处理缺失值是数据预处理的重要步骤，可以避免模型训练时出现错误，
    并提高模型的性能和泛化能力。
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 处理后的数据集，缺失值已被填充
    """
    if data is None:
        return None
    
    print("\n开始处理缺失值...")
    
    # 统计并显示各列的缺失值数量，帮助了解数据质量
    missing_values = data.isnull().sum()
    print("缺失值统计:")
    print(missing_values[missing_values > 0])  # 只显示有缺失值的列
    
    # 复制数据，避免修改原始数据
    # 这是一种良好的实践，保持原始数据的完整性
    processed_data = data.copy()
    
    # 根据数据类型将特征分为数值型和分类型
    # select_dtypes方法可以基于数据类型筛选列
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = processed_data.select_dtypes(include=['object']).columns.tolist()
    
    # 对数值特征使用中位数填充
    # 中位数比均值更稳健，不易受异常值影响
    if numeric_columns:
        print("使用中位数填充数值特征的缺失值...")
        imputer = SimpleImputer(strategy='median')  # 创建中位数填充器
        processed_data[numeric_columns] = imputer.***REMOVED***t_transform(processed_data[numeric_columns])
    
    # 对分类特征使用众数填充
    # 众数是分类数据中出现最频繁的值，是分类特征的合理估计
    if categorical_columns:
        print("使用众数填充分类特征的缺失值...")
        imputer = SimpleImputer(strategy='most_frequent')  # 创建众数填充器
        processed_data[categorical_columns] = imputer.***REMOVED***t_transform(processed_data[categorical_columns])
    
    # 验证填充效果，确保没有遗漏的缺失值
    remaining_missing = processed_data.isnull().sum().sum()
    print(f"填充后剩余的缺失值数量: {remaining_missing}")
    
    return processed_data

def detect_and_handle_outliers(data, contamination=0.05):
    """
    检测和处理数据集中的异常值
    
    该函数使用隔离森林(Isolation Forest)算法检测异常值，这是一种基于随机森林的无监督学习方法，
    特别适合于高维数据的异常检测。然后使用截断法（基于IQR）处理检测到的异常值。
    
    异常值处理对于提高模型稳定性和准确性非常重要，尤其是对于对异常值敏感的模型（如线性回归）。
    
    参数:
    data (pandas.DataFrame): 输入数据集
    contamination (float): 预期的异常值比例，默认为0.05（即5%）
                          这个参数影响模型对异常的判定阈值
    
    返回:
    pandas.DataFrame: 处理异常值后的数据集
    """
    if data is None:
        return None
    
    print("\n开始检测和处理异常值...")
    
    # 复制数据，保持原始数据的完整性
    processed_data = data.copy()
    
    # 只对数值特征检测异常值，因为隔离森林算法需要数值输入
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_columns:
        numeric_columns.remove('target')  # 排除目标变量，不对其进行异常值处理
    
    if not numeric_columns:
        print("没有数值特征可以检测异常值")
        return processed_data
    
    # 使用隔离森林检测异常值
    # 隔离森林特别适合高维数据的异常检测，原理是异常值通常更容易被"隔离"
    print("使用IsolationForest检测异常值...")
    iso_forest = IsolationForest(
        contamination=contamination,  # 预期的异常值比例
        random_state=42              # 固定随机种子，确保结果可重现
    )
    outliers = iso_forest.***REMOVED***t_predict(processed_data[numeric_columns])
    
    # 标记异常值: 1=正常, -1=异常（这是隔离森林的输出格式）
    processed_data['is_outlier'] = outliers
    
    # 统计异常值数量及占比
    outlier_count = (outliers == -1).sum()
    print(f"检测到 {outlier_count} 个异常样本 (约 {outlier_count/len(processed_data)*100:.2f}%)")
    
    # 可视化异常值分布，帮助理解异常检测的结果
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
    sns.countplot(x='is_outlier', data=processed_data)
    plt.title('异常值分布')
    plt.xlabel('是否为异常值 (1=正常, -1=异常)')
    plt.ylabel('数量')
    plt.save***REMOVED***g(get_absolute_path('data/outliers_distribution.png'))
    print(f"异常值分布图已保存到 {get_absolute_path('data/outliers_distribution.png')}")
    
    # 处理异常值 - 方法选择
    # 方法1: 移除异常值（适用于异常值较少且确定是噪声的情况）
    # processed_data = processed_data[processed_data['is_outlier'] == 1]
    
    # 方法2: 将异常值替换为边界值（截断法，基于IQR）
    # 这种方法保留了数据点但减少了极端值的影响
    for col in numeric_columns:
        # 计算Q1（第一四分位数）和Q3（第三四分位数）
        Q1 = processed_data[col].quantile(0.25)
        Q3 = processed_data[col].quantile(0.75)
        # 计算四分位距（IQR）
        IQR = Q3 - Q1
        # 定义上下边界（通常使用1.5倍IQR）
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 截断异常值：将低于下界的异常值设为下界，将高于上界的异常值设为上界
        processed_data.loc[(processed_data['is_outlier'] == -1) & (processed_data[col] < lower_bound), col] = lower_bound
        processed_data.loc[(processed_data['is_outlier'] == -1) & (processed_data[col] > upper_bound), col] = upper_bound
    
    # 移除临时的异常值标记列，保持数据集的整洁
    processed_data.drop('is_outlier', axis=1, inplace=True)
    
    return processed_data

def encode_categorical_features(data):
    """
    对分类特征进行编码
    
    该函数使用标签编码(Label Encoding)将分类特征转换为数值形式，
    这是许多机器学习算法处理分类数据的必要步骤。
    
    注意：标签编码适用于有序分类变量或二元分类变量。
    对于无序多类别变量，在某些情况下可能需要使用独热编码(One-Hot Encoding)。
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 编码后的数据集
    tuple: (编码器字典, 分类特征列表)，用于未来预测时进行一致的转换
    """
    if data is None:
        return None, None
    
    print("\n开始编码分类特征...")
    
    # 复制数据，避免修改原始数据
    processed_data = data.copy()
    
    # 识别所有分类特征（数据类型为object的列）
    categorical_columns = processed_data.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_columns:
        print("没有分类特征需要编码")
        return processed_data, {}
    
    print(f"需要编码的分类特征: {categorical_columns}")
    
    # 创建编码器字典，用于存储每个特征的编码器
    # 这对于将来对新数据进行一致的转换非常重要
    encoders = {}
    
    # 对每个分类特征进行标签编码
    # 标签编码将类别值转换为0到n-1之间的整数
    for col in categorical_columns:
        encoder = LabelEncoder()
        processed_data[col] = encoder.***REMOVED***t_transform(processed_data[col])
        encoders[col] = encoder
        
        # 打印编码映射，这有助于理解和解释转换后的数据
        unique_values = data[col].unique()
        encoded_values = encoder.transform(unique_values)
        mapping = dict(zip(unique_values, encoded_values))
        print(f"特征 {col} 的编码映射: {mapping}")
    
    return processed_data, (encoders, categorical_columns)

def normalize_features(data):
    """
    对数值特征进行标准化
    
    该函数使用Z-score标准化（也称为标准化缩放）将数值特征转换为均值为0、标准差为1的分布。
    标准化是许多机器学习算法的重要预处理步骤，特别是对于使用梯度下降的算法和基于距离的模型。
    
    标准化的好处:
    1. 使不同尺度的特征具有可比性
    2. 加速基于梯度的优化算法收敛
    3. 提高使用距离度量的算法（如KNN、SVM）的性能
    
    参数:
    data (pandas.DataFrame): 输入数据集
    
    返回:
    pandas.DataFrame: 标准化后的数据集
    object: 标准化器对象，用于对未来的预测数据进行一致的转换
    """
    if data is None:
        return None, None
    
    print("\n开始标准化数值特征...")
    
    # 复制数据，避免修改原始数据
    processed_data = data.copy()
    
    # 获取所有数值特征列，排除目标变量
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_columns:
        numeric_columns.remove('target')  # 不对目标变量进行标准化
    
    if not numeric_columns:
        print("没有数值特征需要标准化")
        return processed_data, None
    
    print(f"需要标准化的数值特征: {numeric_columns}")
    
    # 使用StandardScaler进行Z-score标准化
    # Z-score = (X - mean) / std，使特征均值为0，标准差为1
    scaler = StandardScaler()
    processed_data[numeric_columns] = scaler.***REMOVED***t_transform(processed_data[numeric_columns])
    
    # 显示标准化后的统计信息，验证是否成功（均值应接近0，标准差应接近1）
    print("标准化后的数值特征统计信息:")
    print(processed_data[numeric_columns].describe())
    
    return processed_data, scaler

def check_data_quality(data, original_data):
    """
    检查数据质量，确保预处理前后的数据质量良好
    
    该函数比较原始数据和处理后数据的关键统计特性，以验证预处理步骤是否保留了数据的本质特征，
    同时又改善了数据质量。主要检查：
    1. 数据形状变化
    2. 缺失值情况
    3. 目标变量分布变化
    
    参数:
    data (pandas.DataFrame): 处理后的数据集
    original_data (pandas.DataFrame): 原始数据集
    """
    if data is None or original_data is None:
        return
    
    print("\n检查数据质量...")
    
    # 检查数据形状变化，确保没有意外丢失或增加行/列
    print(f"原始数据形状: {original_data.shape}")
    print(f"处理后数据形状: {data.shape}")
    
    # 检查缺失值，确认所有缺失值已被处理
    missing_values = data.isnull().sum().sum()
    print(f"处理后的缺失值数量: {missing_values}")
    
    # 检查目标变量分布变化
    # 预处理不应显著改变目标变量的分布，否则可能引入偏差
    original_target_dist = original_data['target'].value_counts(normalize=True)
    processed_target_dist = data['target'].value_counts(normalize=True)
    
    print("目标变量分布比较 (百分比):")
    comparison = pd.DataFrame({
        '原始数据': original_target_dist * 100,
        '处理后数据': processed_target_dist * 100
    })
    print(comparison)
    
    # 可视化比较原始数据和处理后数据的目标变量分布
    # 这有助于直观地判断预处理是否影响了类别平衡
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 6))
    
    # 绘制原始数据的目标分布
    plt.subplot(1, 2, 1)
    sns.countplot(x='target', data=original_data)
    plt.title('原始数据目标分布')
    plt.xlabel('申请结果 (1=批准, 0=拒绝)')
    plt.ylabel('数量')
    
    # 绘制处理后数据的目标分布
    plt.subplot(1, 2, 2)
    sns.countplot(x='target', data=data)
    plt.title('处理后数据目标分布')
    plt.xlabel('申请结果 (1=批准, 0=拒绝)')
    plt.ylabel('数量')
    
    plt.tight_layout()
    # 确保输出目录存在
    output_dir = os.path.dirname(get_absolute_path('data/target_distribution_comparison.png'))
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存比较图表
    plt.save***REMOVED***g(get_absolute_path('data/target_distribution_comparison.png'))
    print(f"目标分布比较图已保存到 {get_absolute_path('data/target_distribution_comparison.png')}")

def preprocess_data(input_***REMOVED***le='data/credit_card_applications.csv', output_***REMOVED***le='data/processed_data.csv'):
    """
    数据预处理主函数：协调整个预处理流程
    
    该函数按照以下步骤处理数据：
    1. 加载原始数据
    2. 处理缺失值
    3. 检测和处理异常值
    4. 编码分类特征
    5. 标准化数值特征
    6. 检查数据质量
    7. 保存处理后的数据
    
    这个管道式设计使数据按顺序通过各个预处理步骤，每个步骤都改善数据的某个方面，
    从而生成高质量、适合机器学习算法使用的数据集。
    
    参数:
    input_***REMOVED***le (str): 输入数据文件路径，默认为'data/credit_card_applications.csv'
    output_***REMOVED***le (str): 输出数据文件路径，默认为'data/processed_data.csv'
    
    返回:
    tuple: (处理后的数据集, 编码器字典, 标准化器)，这些对象可用于处理将来的预测数据
    """
    print("开始数据预处理过程...")
    
    # 加载原始数据
    original_data = load_data(input_***REMOVED***le)
    if original_data is None:
        return None, None, None
    
    # 处理缺失值 - 第一步预处理
    # 缺失值处理通常是首要任务，因为许多算法不能处理缺失值
    data = handle_missing_values(original_data)
    
    # 处理异常值 - 第二步预处理
    # 在填充缺失值后处理异常值，避免缺失值被误判为异常
    data = detect_and_handle_outliers(data)
    
    # 编码分类特征 - 第三步预处理
    # 将分类变量转换为数值形式，使机器学习算法能够处理
    data, encoders = encode_categorical_features(data)
    
    # 标准化数值特征 - 第四步预处理
    # 在特征编码后进行标准化，确保所有特征在相同的尺度上
    data, scaler = normalize_features(data)
    
    # 检查数据质量 - 验证预处理效果
    # 确保预处理步骤提高了数据质量而未引入偏差
    check_data_quality(data, original_data)
    
    # 保存处理后的数据
    if data is not None:
        # 确保输出目录存在
        output_dir = os.path.dirname(get_absolute_path(output_***REMOVED***le))
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存处理后的数据到CSV文件
        abs_output_***REMOVED***le = get_absolute_path(output_***REMOVED***le)
        data.to_csv(abs_output_***REMOVED***le, index=False)
        print(f"处理后的数据已保存到 {abs_output_***REMOVED***le}")
    
    print("数据预处理完成！")
    # 返回处理后的数据以及转换器，供后续步骤使用
    return data, encoders, scaler

def main():
    """
    主函数：程序入口点
    
    当脚本直接运行时调用此函数，启动完整的数据预处理流程。
    简单调用preprocess_data函数，使用默认参数。
    """
    preprocess_data()

if __name__ == "__main__":
    main() 