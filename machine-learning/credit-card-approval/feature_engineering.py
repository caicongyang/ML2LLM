#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征工程模块：对预处理后的信用卡审批数据集进行特征选择、转换和创建

本模块实现以下功能：
1. 加载由data_preprocessing模块处理过的信用卡审批数据
2. 分析特征重要性（使用随机森林、ANOVA F值和互信息）
3. 基于重要性分数选择最有价值的特征
4. 创建多项式特征以捕获非线性关系
5. 应用主成分分析(PCA)进行降维
6. 保存特征工程处理后的数据和特征转换器

特征工程是机器学习流程中至关重要的环节，可以显著提高模型性能。通过特征选择减少噪声，
通过特征创建捕获更复杂的模式，通过降维减少过拟合风险。
"""

import os                    # 用于文件和目录操作
import pandas as pd          # 用于数据处理和分析
import numpy as np           # 用于科学计算和数组操作
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns        # 用于高级数据可视化
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE  # 特征选择工具
from sklearn.ensemble import RandomForestClassifier  # 用于特征重要性评估和RFE
from sklearn.decomposition import PCA               # 用于降维
from sklearn.preprocessing import PolynomialFeatures # 用于创建多项式特征
from sklearn.pipeline import Pipeline               # 用于构建特征处理流水线
import joblib                # 用于保存和加载模型和转换器

# 设置matplotlib的中文字体支持，确保图表中的中文正确显示
# 按照优先级尝试不同的字体，适应不同操作系统环境
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 首选Arial Unicode MS，备选黑体和微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题，避免显示为方块

# 获取脚本所在目录的绝对路径，用于确保在任何目录运行脚本时都能找到正确的文件
# 这解决了相对路径在不同环境中可能导致的文件不存在问题
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path):
    """
    将相对于脚本的路径转换为绝对路径
    
    在不同环境下运行脚本时，相对路径可能会导致文件找不到的问题。
    该函数确保无论从哪里运行脚本，都能正确找到所需的文件。
    
    参数:
    relative_path (str): 相对路径，如'data/file.csv'
    
    返回:
    str: 完整的绝对路径，如'/home/user/project/data/file.csv'
    """
    return os.path.join(SCRIPT_DIR, relative_path)

def load_processed_data(file_path='data/processed_data.csv'):
    """
    加载预处理后的信用卡审批数据集
    
    本函数负责从指定路径读取已经过预处理的数据集CSV文件，
    并将其加载到pandas DataFrame中。这是特征工程的第一步，
    确保我们使用的是已清洗和标准化的数据。
    
    参数:
    file_path (str): 数据文件路径，默认为'data/processed_data.csv'
    
    返回:
    pandas.DataFrame: 加载的预处理数据集，如果加载失败则返回None
    """
    # 转换为绝对路径，确保在任何环境下都能找到文件
    abs_file_path = get_absolute_path(file_path)
    try:
        # 尝试读取CSV文件到DataFrame
        data = pd.read_csv(abs_file_path)
        print(f"成功加载预处理后的数据，形状为: {data.shape}")
        return data
    except Exception as e:
        # 捕获并处理可能的异常，如文件不存在、格式错误等
        print(f"加载数据时出错: {e}")
        return None

def analyze_feature_importance(data):
    """
    分析特征重要性，使用多种方法评估特征对目标变量的贡献度
    
    本函数使用三种不同的方法评估特征重要性：
    1. 随机森林特征重要性 - 基于平均不纯度减少或预测精度贡献
    2. ANOVA F值 - 基于特征与目标变量之间的统计相关性
    3. 互信息 - 测量特征与目标变量之间的非线性关系
    
    使用多种方法可以提供更全面的视角，因为不同方法对不同类型的关系和数据分布敏感。
    
    参数:
    data (pandas.DataFrame): 输入数据集，包含特征和目标变量'target'
    
    返回:
    dict: 包含三种方法的特征重要性分数的字典
          {'random_forest': df1, 'anova': df2, 'mutual_info': df3}
    """
    if data is None:
        return None
    
    print("\n开始分析特征重要性...")
    
    # 分离特征和目标变量
    # 特征是所有非目标列，目标变量是'target'列
    X = data.drop('target', axis=1)  # 所有特征
    y = data['target']               # 目标变量（信用卡审批结果）
    
    # ========== 方法1：随机森林特征重要性 ==========
    # 随机森林是一种集成学习方法，可评估特征在决策中的重要性
    print("使用随机森林估计特征重要性...")
    rf = RandomForestClassifier(
        n_estimators=100,   # 使用100棵决策树构成森林
        random_state=42     # 设置随机种子，确保结果可重现
    )
    rf.fit(X, y)  # 训练随机森林模型
    
    # 获取特征重要性分数
    # 对于分类问题，这通常基于基尼不纯度或信息增益的平均减少
    importances = rf.feature_importances_
    
    # 创建特征重要性数据框，将特征名与其重要性分数配对
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,           # 特征名称
        'Importance': importances       # 重要性分数
    })
    
    # 按重要性降序排序，最重要的特征排在最前面
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    print("特征重要性 (随机森林):")
    print(feature_importance_df)
    
    # 可视化随机森林特征重要性
    # 使用条形图直观展示各特征的重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('特征重要性 (随机森林)')
    plt.tight_layout()
    plt.savefig(get_absolute_path('data/feature_importance_rf.png'))
    print("特征重要性图已保存到 data/feature_importance_rf.png")
    
    # ========== 方法2：ANOVA F值 ==========
    # ANOVA F值用于评估分类问题中特征与目标变量之间的线性关系
    print("\n使用ANOVA F-value估计特征重要性...")
    selector = SelectKBest(
        f_classif,      # ANOVA F值作为评分函数
        k='all'         # 评估所有特征而不进行筛选
    )
    selector.fit(X, y)  # 计算每个特征的F统计量
    
    # 获取F值和p值
    # F值越高，特征与目标的相关性越强
    # p值越低，相关性越显著（通常p<0.05被视为显著）
    anova_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': selector.scores_,      # F统计量
        'P_Value': selector.pvalues_      # 对应的p值
    })
    
    # 按F值降序排序
    anova_scores = anova_scores.sort_values('F_Score', ascending=False)
    
    print("特征重要性 (ANOVA F-value):")
    print(anova_scores)
    
    # 可视化ANOVA F值
    plt.figure(figsize=(12, 8))
    sns.barplot(x='F_Score', y='Feature', data=anova_scores)
    plt.title('特征重要性 (ANOVA F-value)')
    plt.tight_layout()
    plt.savefig(get_absolute_path('data/feature_importance_anova.png'))
    print("ANOVA F-value特征重要性图已保存到 data/feature_importance_anova.png")
    
    # ========== 方法3：互信息 ==========
    # 互信息可以捕获特征与目标变量之间的非线性关系
    # 对于机器学习中的复杂关系非常有用
    print("\n使用互信息估计特征重要性...")
    selector = SelectKBest(
        mutual_info_classif,    # 互信息作为评分函数
        k='all'                 # 评估所有特征
    )
    selector.fit(X, y)  # 计算每个特征的互信息分数
    
    # 获取互信息分数
    # 分数越高，特征包含的关于目标的信息越多
    mi_scores = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': selector.scores_      # 互信息分数
    })
    
    # 按互信息分数降序排序
    mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
    
    print("特征重要性 (互信息):")
    print(mi_scores)
    
    # 可视化互信息分数
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=mi_scores)
    plt.title('特征重要性 (互信息)')
    plt.tight_layout()
    plt.savefig(get_absolute_path('data/feature_importance_mi.png'))
    print("互信息特征重要性图已保存到 data/feature_importance_mi.png")
    
    # 返回包含所有重要性分数的字典，供后续特征选择使用
    return {
        'random_forest': feature_importance_df,  # 随机森林分数
        'anova': anova_scores,                   # ANOVA F值分数
        'mutual_info': mi_scores                 # 互信息分数
    }

def select_features(data, importance_scores, method='random_forest', threshold=0.01, k=None):
    """
    基于特征重要性选择最相关的特征子集
    
    特征选择是降低维度和减少过拟合的重要技术，它通过去除不相关或冗余特征来改善模型性能。
    本函数支持四种选择方法：
    1. 随机森林重要性 - 基于决策树的特征重要性
    2. ANOVA F值 - 基于统计显著性
    3. 互信息 - 基于信息理论的度量
    4. 递归特征消除(RFE) - 通过反复构建模型并移除最不重要的特征
    
    选择可以基于阈值（保留重要性高于某值的特征）或数量（保留前k个最重要特征）。
    
    参数:
    data (pandas.DataFrame): 输入数据集，包含特征和目标变量'target'
    importance_scores (dict): 由analyze_feature_importance返回的特征重要性分数
    method (str): 特征选择方法，可选'random_forest', 'anova', 'mutual_info', 'rfe'
    threshold (float): 特征重要性阈值（当基于阈值选择时使用）
    k (int): 要选择的特征数量（当基于数量选择时使用），如果为None则使用阈值
    
    返回:
    pandas.DataFrame: 只包含选定特征的数据集
    list: 选定的特征名称列表
    """
    if data is None or importance_scores is None:
        return None, None
    
    print(f"\n使用 {method} 方法选择特征...")
    
    # 分离特征和目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    
    selected_features = []  # 用于存储选定的特征
    
    # ========== 方法1：基于随机森林重要性选择特征 ==========
    if method == 'random_forest':
        if k is not None:
            # 基于数量选择 - 保留重要性排名前k的特征
            selected_features = importance_scores['random_forest']['Feature'].tolist()[:k]
        else:
            # 基于阈值选择 - 保留重要性高于阈值的特征
            selected_features = importance_scores['random_forest'].loc[
                importance_scores['random_forest']['Importance'] >= threshold, 'Feature'
            ].tolist()
    
    # ========== 方法2：基于ANOVA F值选择特征 ==========
    elif method == 'anova':
        if k is not None:
            # 基于数量选择 - 保留F值排名前k的特征
            selected_features = importance_scores['anova']['Feature'].tolist()[:k]
        else:
            # 基于p值选择 - 保留p值显著（小于0.05）的特征
            # 这是一种基于统计显著性的选择方法
            selected_features = importance_scores['anova'].loc[
                importance_scores['anova']['P_Value'] <= 0.05, 'Feature'
            ].tolist()
    
    # ========== 方法3：基于互信息选择特征 ==========
    elif method == 'mutual_info':
        if k is not None:
            # 基于数量选择 - 保留互信息排名前k的特征
            selected_features = importance_scores['mutual_info']['Feature'].tolist()[:k]
        else:
            # 基于阈值选择 - 保留互信息高于阈值的特征
            selected_features = importance_scores['mutual_info'].loc[
                importance_scores['mutual_info']['MI_Score'] >= threshold, 'Feature'
            ].tolist()
    
    # ========== 方法4：使用递归特征消除(RFE)选择特征 ==========
    elif method == 'rfe':
        # RFE是一种包装器方法，它通过反复训练模型、评估特征重要性并移除最不重要的特征
        print("使用递归特征消除 (RFE)...")
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        if k is None:
            # 如果未指定k，默认选择一半的特征
            k = X.shape[1] // 2
        
        # 创建RFE对象，指定要选择的特征数量
        rfe = RFE(estimator=estimator, n_features_to_select=k)
        # 应用RFE，拟合数据
        rfe.fit(X, y)
        
        # 获取选定的特征（RFE为每个特征创建一个boolean掩码）
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask].tolist()
    
    # 打印选定的特征列表
    print(f"选定的特征 ({len(selected_features)}):")
    print(selected_features)
    
    # 创建只包含选定特征和目标变量的数据集
    # 这个数据集将用于后续步骤，如多项式特征创建和PCA
    selected_data = data[selected_features + ['target']]
    
    return selected_data, selected_features

def create_polynomial_features(data, selected_features, degree=2):
    """
    创建多项式特征以捕获特征之间的非线性关系和交互效应
    
    多项式特征是处理非线性关系的强大技术，它通过创建原始特征的高阶项和交互项，
    使线性模型能够学习更复杂的非线性模式。对于信用卡审批等领域，特征之间的
    交互往往比单个特征更具预测价值（如收入与债务比率的交互）。
    
    例如，对于特征x1和x2，2阶多项式特征将创建: x1^2, x1*x2, x2^2
    
    参数:
    data (pandas.DataFrame): 输入数据集，包含特征和目标变量
    selected_features (list): 用于创建多项式特征的特征列表（通常是已选择的最重要特征）
    degree (int): 多项式阶数，默认为2（二阶多项式）
    
    返回:
    pandas.DataFrame: 增加了多项式特征的扩展数据集
    sklearn.preprocessing.PolynomialFeatures: 多项式转换器对象，用于将来转换新数据
    """
    if data is None or not selected_features:
        return None, None
    
    print(f"\n创建 {degree} 阶多项式特征...")
    
    # 复制数据，避免修改原始数据集
    enhanced_data = data.copy()
    
    # 选择要用于创建多项式特征的特征子集
    # 通常我们只对最重要的特征创建多项式，以避免特征爆炸
    X_subset = enhanced_data[selected_features]
    
    # 创建多项式特征
    # degree=2表示创建至多2阶的特征（如x^2, x*y）
    # include_bias=False表示不添加截距项（常数项1）
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(X_subset)
    
    # 获取生成的多项式特征数量（排除原始特征）
    # 多项式转换后的数据包含原始特征和新生成的多项式特征
    n_original_features = len(selected_features)
    n_poly_features = poly_features.shape[1] - n_original_features
    
    # 创建多项式特征的有意义的列名
    # 例如: "poly_Age^2" 或 "poly_Income^1_Debt^1"（Income*Debt）
    poly_feature_names = []
    powers = poly.powers_[1:]  # 跳过原始特征的幂次矩阵
    
    for i, power in enumerate(powers):
        name = "poly_"
        for j, feat in enumerate(selected_features):
            if power[j] > 0:
                name += f"{feat}^{power[j]}_"
        poly_feature_names.append(name[:-1])  # 移除最后一个下划线
    
    # 确保多项式特征名和实际生成的特征数量一致
    poly_feature_names = poly_feature_names[:n_poly_features]
    
    # 创建包含多项式特征的数据框
    # 只取多项式特征部分（排除已有的原始特征）
    poly_df = pd.DataFrame(
        poly_features[:, n_original_features:],  # 只取新生成的多项式特征部分
        columns=poly_feature_names
    )
    
    # 将多项式特征添加到原始数据集
    enhanced_data = pd.concat([enhanced_data, poly_df], axis=1)
    
    # 打印多项式特征创建的统计信息
    print(f"创建了 {poly_df.shape[1]} 个多项式特征")
    print(f"增强后的数据形状: {enhanced_data.shape}")
    
    # 返回增强的数据集和多项式转换器（用于将来转换新数据）
    return enhanced_data, poly

def apply_pca(data, n_components=None, variance_threshold=0.95):
    """
    应用主成分分析(PCA)进行降维，减少特征数量并捕获主要信息
    
    PCA是一种无监督的降维技术，通过线性变换将原始特征转换为一组相互正交的新特征（主成分）。
    这些主成分按照它们解释的方差量排序，前几个主成分通常包含数据中的大部分信息。
    
    PCA的主要优势：
    1. 降低维度，减少计算复杂度
    2. 减轻多重共线性问题
    3. 降低过拟合风险
    4. 可视化高维数据
    
    在创建多项式特征后应用PCA特别有用，因为多项式特征会显著增加特征空间维度，
    而这些特征往往存在高度相关性，PCA可以有效提取它们的主要信息同时降低维度。
    
    参数:
    data (pandas.DataFrame): 输入数据集，包含特征和目标变量'target'
    n_components (int): 要保留的主成分数量，如果为None则自动基于方差阈值确定
    variance_threshold (float): 要保留的方差比例阈值（0-1之间），默认为0.95
                               意味着保留能解释95%原始数据方差的主成分
    
    返回:
    pandas.DataFrame: PCA转换后的数据集，特征被替换为主成分
    sklearn.decomposition.PCA: PCA转换器对象，用于将来转换新数据
    """
    if data is None:
        return None, None
    
    print("\n应用主成分分析 (PCA)...")
    
    # 分离特征和目标变量
    # PCA只应用于特征，目标变量不参与转换
    X = data.drop('target', axis=1)
    y = data['target']
    
    # ========== 自动确定主成分数量 ==========
    if n_components is None:
        # 当没有明确指定主成分数量时，使用方差阈值自动确定
        # 首先创建一个不限制组件数量的PCA对象
        pca = PCA()
        pca.fit(X)
        
        # 计算累积解释方差比
        # explained_variance_ratio_包含每个主成分解释的方差比例
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 找到满足方差阈值的最小主成分数量
        # 即找到第一个累积方差比大于等于阈值的索引，再加1得到主成分数量
        n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        
        print(f"基于 {variance_threshold*100:.1f}% 的方差阈值，选择 {n_components} 个主成分")
    
    # ========== 应用PCA转换 ==========
    # 使用确定的主成分数量创建PCA对象并执行转换
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    # 创建主成分列名（PC1, PC2, ...）
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    
    # 创建PCA转换后的数据框
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    # 将目标变量添加回数据集
    pca_df['target'] = y.values
    
    # ========== 可视化解释的方差比例 ==========
    # 创建图表显示每个主成分解释的方差比例和累积方差比例
    plt.figure(figsize=(10, 6))
    
    # 条形图显示每个主成分解释的方差比例
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_
    )
    
    # 线图显示累积方差比例
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1), 
        np.cumsum(pca.explained_variance_ratio_), 
        'r-'
    )
    
    plt.xlabel('主成分')
    plt.ylabel('解释的方差比例')
    plt.title('PCA: 解释的方差比例')
    
    # 添加方差阈值参考线
    plt.axhline(y=variance_threshold, color='g', linestyle='--')
    plt.text(1, variance_threshold+0.02, f'{variance_threshold*100:.1f}% 方差阈值', color='g')
    
    plt.grid(True)
    plt.savefig(get_absolute_path('data/pca_explained_variance.png'))
    print("PCA解释的方差比例图已保存到 data/pca_explained_variance.png")
    
    # 输出降维结果的统计信息
    print(f"PCA转换后的数据形状: {pca_df.shape}")
    print(f"数据降维: {X.shape[1]} -> {X_pca.shape[1]} 特征")
    
    return pca_df, pca

def engineer_features(data, output_file='data/engineered_data.csv'):
    """
    特征工程主函数：协调整个特征工程流程
    
    该函数实现了一个完整的特征工程管道，按以下步骤处理数据：
    1. 分析特征重要性（使用多种方法）
    2. 选择最重要的特征子集
    3. 创建多项式特征以捕获非线性关系
    4. 应用PCA降维减少特征数量
    5. 保存工程处理后的数据和特征转换器
    
    特征工程是提高模型性能的关键步骤，通过创建更有信息量的特征表示，
    可以帮助模型发现数据中的复杂模式，同时通过降维减少过拟合风险。
    
    参数:
    data (pandas.DataFrame): 输入数据集，包含特征和目标变量
    output_file (str): 输出数据文件路径，默认为'data/engineered_data.csv'
    
    返回:
    tuple: (工程处理后的数据集, 选择的特征列表, 特征转换器元组)
           转换器元组包含(多项式转换器, PCA转换器)，用于将来处理新数据
    """
    if data is None:
        # 如果没有提供数据，尝试加载预处理数据
        data = load_processed_data()
        if data is None:
            # 如果加载失败，无法继续处理
            return None, None, None
    
    print("开始特征工程过程...")
    
    # ========== 阶段1: 特征重要性分析 ==========
    # 使用多种方法（随机森林、ANOVA F值、互信息）分析特征重要性
    importance_scores = analyze_feature_importance(data)
    
    # ========== 阶段2: 特征选择 ==========
    # 基于重要性分数选择最相关的特征
    # 这里我们使用随机森林方法选择前10个最重要的特征
    # 选择顶级特征可以减少噪声，提高模型泛化能力
    _, selected_features = select_features(
        data, 
        importance_scores, 
        method='random_forest',  # 使用随机森林特征重要性
        k=10                     # 选择前10个特征
    )
    
    # ========== 阶段3: 特征创建 - 多项式特征 ==========
    # 基于选定的特征创建多项式特征，捕获非线性关系
    # 通常在特征选择之后进行，以避免特征爆炸
    enhanced_data, poly_transformer = create_polynomial_features(
        data, 
        selected_features, 
        degree=2  # 创建2阶多项式特征
    )
    
    # ========== 阶段4: 降维 - PCA ==========
    # 应用PCA降维，减少特征数量但保留大部分信息
    # 这对于处理多项式特征创建后的高维数据特别有用
    final_data, pca_transformer = apply_pca(
        enhanced_data, 
        variance_threshold=0.95  # 保留95%的方差
    )
    
    # ========== 阶段5: 保存结果 ==========
    # 保存工程处理后的数据和转换器，供后续建模使用
    if final_data is not None:
        # 转换为绝对路径并保存数据
        abs_output_file = get_absolute_path(output_file)
        final_data.to_csv(abs_output_file, index=False)
        print(f"特征工程处理后的数据已保存到 {output_file}")
        
        # 保存特征转换器，用于将来处理新数据
        # 包括：选定的特征列表、多项式转换器和PCA转换器
        transformers = {
            'selected_features': selected_features,    # 选定的特征名称
            'poly_transformer': poly_transformer,      # 多项式转换器
            'pca_transformer': pca_transformer         # PCA转换器
        }
        joblib.dump(transformers, get_absolute_path('data/feature_transformers.pkl'))
        print("特征转换器已保存到 data/feature_transformers.pkl")
    
    print("特征工程完成！")
    
    # 返回处理后的数据和转换器，供后续建模步骤使用
    return final_data, selected_features, (poly_transformer, pca_transformer)

def main():
    """
    主函数：程序入口点
    
    当脚本被直接执行时调用此函数，它协调整个特征工程流程：
    1. 加载预处理数据
    2. 执行特征工程
    
    这是一个简单的入口点，适合命令行运行脚本时使用。
    """
    # 加载预处理后的数据集
    data = load_processed_data()
    
    # 如果数据加载成功，执行特征工程
    if data is not None:
        engineer_features(data)

if __name__ == "__main__":
    main() 