#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练模块：用于信用卡审批系统的模型选择、训练和保存
此脚本实现了完整的机器学习工作流程，包括:
- 数据加载和划分
- 多种分类模型的训练和评估
- 模型性能对比和可视化
- 超参数调优
- 最佳模型选择和保存
- 构建完整的机器学习流水线
"""

# 导入基础库
import os                  # 操作系统接口，用于文件和路径操作
import pandas as pd        # 数据分析和操作库
import numpy as np         # 科学计算库，提供多维数组对象支持
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns      # 基于matplotlib的数据可视化库，提供更美观的图表

# 导入模型评估和数据处理库
from sklearn.model_selection import train_test_split  # 数据集分割工具
from sklearn.model_selection import GridSearchCV      # 网格搜索交叉验证，用于超参数调优
from sklearn.model_selection import cross_val_score   # 交叉验证评分
from sklearn.model_selection import RandomizedSearchCV # 随机搜索交叉验证，比GridSearchCV更高效

# 导入评估指标相关库
from sklearn.metrics import accuracy_score   # 准确率：正确预测的比例
from sklearn.metrics import precision_score  # 精确率：预测为正的样本中实际为正的比例
from sklearn.metrics import recall_score     # 召回率：实际为正的样本中被正确预测的比例
from sklearn.metrics import f1_score         # F1分数：精确率和召回率的调和平均数
from sklearn.metrics import roc_auc_score    # ROC曲线下面积，衡量二分类模型性能
from sklearn.metrics import confusion_matrix  # 混淆矩阵，展示TP/TN/FP/FN的数量
from sklearn.metrics import classi***REMOVED***cation_report  # 分类报告，包含精确率、召回率等多项指标
from sklearn.metrics import roc_curve        # ROC曲线数据
from sklearn.metrics import precision_recall_curve  # 精确率-召回率曲线数据

# 导入各种分类模型
from sklearn.linear_model import LogisticRegression  # 逻辑回归，线性分类器
from sklearn.tree import DecisionTreeClassi***REMOVED***er      # 决策树
from sklearn.ensemble import RandomForestClassi***REMOVED***er  # 随机森林，多个决策树的集成
from sklearn.ensemble import GradientBoostingClassi***REMOVED***er  # 梯度提升，通过迭代优化弱学习器
from sklearn.svm import SVC                          # 支持向量机
from sklearn.neighbors import KNeighborsClassi***REMOVED***er   # K近邻分类器
from sklearn.naive_bayes import GaussianNB           # 高斯朴素贝叶斯
from sklearn.neural_network import MLPClassi***REMOVED***er     # 多层感知机 (神经网络)

# 导入管道和特征工程组件
from sklearn.pipeline import Pipeline               # 构建机器学习工作流的工具
from sklearn.feature_selection import SelectFromModel  # 基于模型的特征选择
from sklearn.preprocessing import PolynomialFeatures   # 多项式特征转换
from sklearn.preprocessing import StandardScaler       # 特征标准化
from sklearn.decomposition import PCA                  # 主成分分析，降维

# 导入模型持久化和性能工具
import joblib                # 用于序列化Python对象，保存模型
import time                  # 时间处理工具，用于计时
import pickle                # Python对象序列化工具，可保存模型
from scipy import stats      # 科学计算库，提供统计函数

# 尝试导入高级梯度提升模型，由于这些是可选的外部依赖，使用try-except处理
try:
    # XGBoost: 高效的梯度提升库
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    # LightGBM: 微软开发的高效梯度提升框架
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    # CatBoost: Yandex开发的梯度提升库，特别适合处理类别特征
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 设置中文字体支持，确保图表中文字显示正确
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 首选Arial Unicode MS，备选黑体和微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取脚本所在目录的绝对路径，用于确保在任何目录运行脚本时都能找到正确的文件
# __***REMOVED***le__是当前脚本的路径，abspath获取绝对路径，dirname获取目录名
SCRIPT_DIR = os.path.dirname(os.path.abspath(__***REMOVED***le__))

def get_absolute_path(relative_path):
    """
    将相对于脚本的路径转换为绝对路径
    
    这个函数确保无论从哪个目录运行脚本，都能正确找到相对于脚本位置的文件
    
    参数:
    relative_path (str): 相对路径，如'data/***REMOVED***le.csv'
    
    返回:
    str: 绝对路径，如'/home/user/project/data/***REMOVED***le.csv'
    """
    return os.path.join(SCRIPT_DIR, relative_path)

def load_engineered_data(***REMOVED***le_path='data/engineered_data.csv'):
    """
    加载特征工程后的数据集
    
    此函数负责从CSV文件中读取处理过的数据，并处理可能出现的错误
    
    参数:
    ***REMOVED***le_path (str): 数据文件相对路径，默认为'data/engineered_data.csv'
    
    返回:
    pandas.DataFrame: 加载的数据集，若加载失败则返回None
    """
    # 转换为绝对路径，确保从任何位置都能找到文件
    abs_***REMOVED***le_path = get_absolute_path(***REMOVED***le_path)
    try:
        # 尝试读取CSV文件
        data = pd.read_csv(abs_***REMOVED***le_path)
        print(f"成功加载特征工程后的数据，形状为: {data.shape}")
        return data
    except Exception as e:
        # 捕获并打印所有可能的异常
        print(f"加载数据时出错: {e}")
        return None

def split_data(data, test_size=0.2, random_state=42, y=None):
    """
    将数据集分割为训练集和测试集
    
    此函数支持两种输入形式:
    1. 完整DataFrame，包含特征和目标变量
    2. 单独的特征矩阵X和目标变量y
    
    参数:
    data (pandas.DataFrame): 输入数据集或特征矩阵X
    test_size (float): 测试集的比例，默认为0.2（20%）
    random_state (int): 随机种子，确保结果可重现，默认为42
    y (pandas.Series, optional): 如果data是特征矩阵X，则需要传入目标变量y
    
    返回:
    tuple: (X_train, X_test, y_train, y_test) - 训练和测试数据集
    """
    # 如果数据为空，则直接返回空结果
    if data is None:
        return None, None, None, None
    
    print(f"\n将数据分割为训练集 ({1-test_size:.0%}) 和测试集 ({test_size:.0%})...")
    
    # 检查是否传入了单独的目标变量y
    if y is None:
        # 如果没有传入y，则假设data是完整的DataFrame
        # 从数据中分离特征和目标变量
        X = data.drop('target', axis=1)  # 移除target列作为特征
        y = data['target']               # 使用target列作为目标变量
    else:
        # 如果传入了y，则data就是特征矩阵X
        X = data
    
    # 使用sklearn的train_test_split函数分割数据
    # stratify=y确保训练集和测试集中的类别分布与原数据集一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # 打印分割后的数据集大小
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    训练模型并评估性能
    
    此函数完成以下任务:
    1. 训练给定的分类模型
    2. 测量训练时间
    3. 在训练集和测试集上评估模型性能
    4. 计算多种评估指标
    5. 生成性能可视化（混淆矩阵、ROC曲线、精确率-召回率曲线）
    6. 保存可视化结果
    
    参数:
    model: 分类器模型实例，如LogisticRegression()
    X_train: 训练特征矩阵
    y_train: 训练目标变量
    X_test: 测试特征矩阵
    y_test: 测试目标变量
    model_name (str): 模型名称，用于输出和保存文件
    
    返回:
    tuple: (训练好的模型, 性能指标字典)
    """
    print(f"\n训练 {model_name} 模型...")
    
    # 记录开始时间，用于计算训练耗时
    start_time = time.time()
    
    # 训练模型 - 使用训练集拟合模型
    model.***REMOVED***t(X_train, y_train)
    
    # 计算训练时间
    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.2f} 秒")
    
    # 在训练集上评估模型性能
    y_train_pred = model.predict(X_train)  # 获取训练集预测标签
    # 如果模型支持概率预测，则获取预测概率；否则为None
    y_train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 在测试集上评估模型性能
    y_test_pred = model.predict(X_test)    # 获取测试集预测标签
    # 如果模型支持概率预测，则获取预测概率；否则为None
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 创建性能指标字典，用于存储各项评估指标
    metrics = {}
    
    # 计算训练集性能指标
    metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)    # 准确率
    metrics['train_precision'] = precision_score(y_train, y_train_pred)  # 精确率
    metrics['train_recall'] = recall_score(y_train, y_train_pred)        # 召回率
    metrics['train_f1'] = f1_score(y_train, y_train_pred)                # F1分数
    # 如果有概率预测，则计算ROC AUC
    if y_train_prob is not None:
        metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_prob)  # ROC曲线下面积
    
    # 计算测试集性能指标
    metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)     # 准确率
    metrics['test_precision'] = precision_score(y_test, y_test_pred)   # 精确率
    metrics['test_recall'] = recall_score(y_test, y_test_pred)         # 召回率
    metrics['test_f1'] = f1_score(y_test, y_test_pred)                 # F1分数
    # 如果有概率预测，则计算ROC AUC
    if y_test_prob is not None:
        metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_prob)   # ROC曲线下面积
    
    # 记录训练时间
    metrics['train_time'] = train_time
    
    # 输出性能指标
    print(f"\n{model_name} 性能指标:")
    print(f"训练集准确率: {metrics['train_accuracy']:.4f}")  # 训练集上的准确率
    print(f"测试集准确率: {metrics['test_accuracy']:.4f}")   # 测试集上的准确率
    print(f"测试集精确率: {metrics['test_precision']:.4f}")  # 测试集上的精确率
    print(f"测试集召回率: {metrics['test_recall']:.4f}")     # 测试集上的召回率
    print(f"测试集F1分数: {metrics['test_f1']:.4f}")         # 测试集上的F1分数
    if 'test_roc_auc' in metrics:
        print(f"测试集ROC AUC: {metrics['test_roc_auc']:.4f}")  # 测试集上的ROC AUC
    
    # 计算并可视化混淆矩阵
    # 混淆矩阵展示了预测类别与实际类别的对应关系，包含TP/TN/FP/FN
    cm = confusion_matrix(y_test, y_test_pred)
    plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))  # 设置图形大小
    # 使用seaborn的热图展示混淆矩阵，带数值标注
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['拒绝', '通过'], yticklabels=['拒绝', '通过'])
    plt.xlabel('预测标签')  # x轴标签
    plt.ylabel('真实标签')  # y轴标签
    plt.title(f'{model_name} 混淆矩阵')  # 图表标题
    # 保存混淆矩阵图像到文件
    plt.save***REMOVED***g(get_absolute_path(f'data/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    print(f"{model_name} 混淆矩阵已保存到 data/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    
    # 如果模型支持概率预测，绘制ROC曲线和精确率-召回率曲线
    if y_test_prob is not None:
        # 绘制ROC曲线
        # ROC曲线展示了不同阈值下的真正例率（TPR）和假正例率（FPR）
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)  # 计算ROC曲线坐标
        plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))  # 设置图形大小
        # 绘制ROC曲线，标注AUC值
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {metrics["test_roc_auc"] if "test_roc_auc" in metrics else "N/A":.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')  # 绘制对角线，代表随机猜测
        plt.xlabel('假正例率')  # x轴标签
        plt.ylabel('真正例率')  # y轴标签
        plt.title(f'{model_name} ROC曲线')  # 图表标题
        plt.legend(loc='lower right')  # 图例位置
        plt.grid(True)  # 显示网格
        # 保存ROC曲线图像到文件
        plt.save***REMOVED***g(get_absolute_path(f'data/{model_name.lower().replace(" ", "_")}_roc_curve.png'))
        print(f"{model_name} ROC曲线已保存到 data/{model_name.lower().replace(' ', '_')}_roc_curve.png")
        
        # 绘制精确率-召回率曲线
        # PR曲线展示了不同阈值下的精确率和召回率之间的权衡
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)  # 计算PR曲线坐标
        plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))  # 设置图形大小
        plt.plot(recall, precision, label=f'精确率-召回率曲线')  # 绘制PR曲线
        plt.xlabel('召回率')  # x轴标签
        plt.ylabel('精确率')  # y轴标签
        plt.title(f'{model_name} 精确率-召回率曲线')  # 图表标题
        plt.legend(loc='lower left')  # 图例位置
        plt.grid(True)  # 显示网格
        # 保存PR曲线图像到文件
        plt.save***REMOVED***g(get_absolute_path(f'data/{model_name.lower().replace(" ", "_")}_pr_curve.png'))
        print(f"{model_name} 精确率-召回率曲线已保存到 data/{model_name.lower().replace(' ', '_')}_pr_curve.png")
    
    return model, metrics

def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    训练多个模型并比较它们的性能
    
    此函数创建8种不同类型的分类模型，训练每个模型，比较它们的性能，
    并生成各种可视化和比较报告。这是模型选择阶段的核心函数。
    
    参数:
    X_train: 训练特征矩阵
    y_train: 训练目标变量
    X_test: 测试特征矩阵
    y_test: 测试目标变量
    
    返回:
    tuple: (所有模型字典, 所有性能指标字典)
    """
    print("\n训练多个模型并比较性能...")
    
    # 创建模型字典，包含8种常用的分类模型
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),  # 逻辑回归，max_iter增大以确保收敛
        'Decision Tree': DecisionTreeClassi***REMOVED***er(random_state=42),                   # 决策树
        'Random Forest': RandomForestClassi***REMOVED***er(random_state=42),                   # 随机森林
        'Gradient Boosting': GradientBoostingClassi***REMOVED***er(random_state=42),           # 梯度提升
        'SVM': SVC(random_state=42, probability=True),                             # 支持向量机，启用概率输出
        'K-Nearest Neighbors': KNeighborsClassi***REMOVED***er(),                              # K近邻
        'Naive Bayes': GaussianNB(),                                                # 高斯朴素贝叶斯
        'Neural Network': MLPClassi***REMOVED***er(random_state=42, max_iter=1000)             # 多层感知机（神经网络）
    }
    
    # 用于存储训练好的模型和性能指标
    trained_models = {}  # 存储每个训练好的模型
    all_metrics = {}     # 存储每个模型的性能指标
    
    # 训练并评估每个模型
    for name, model in models.items():
        # 调用train_model函数训练模型并获取性能指标
        trained_model, metrics = train_model(model, X_train, y_train, X_test, y_test, name)
        trained_models[name] = trained_model  # 保存训练好的模型
        all_metrics[name] = metrics           # 保存性能指标
    
    # 创建性能比较DataFrame，用于可视化和保存
    model_comparison = pd.DataFrame({
        'Model': list(all_metrics.keys()),                                       # 模型名称
        'Accuracy': [metrics['test_accuracy'] for metrics in all_metrics.values()],  # 准确率
        'Precision': [metrics['test_precision'] for metrics in all_metrics.values()], # 精确率
        'Recall': [metrics['test_recall'] for metrics in all_metrics.values()],       # 召回率
        'F1 Score': [metrics['test_f1'] for metrics in all_metrics.values()],         # F1分数
        'ROC AUC': [metrics.get('test_roc_auc', np.nan) for metrics in all_metrics.values()],  # ROC AUC，不支持的模型为NaN
        'Training Time (s)': [metrics['train_time'] for metrics in all_metrics.values()]        # 训练时间
    })
    
    # 按F1分数降序排序，突出显示性能最好的模型
    model_comparison = model_comparison.sort_values('F1 Score', ascending=False)
    
    # 打印模型性能比较表
    print("\n模型性能比较:")
    print(model_comparison)
    
    # 生成各种比较可视化图表
    
    # 准确率比较柱状图
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))  # 设置图形大小
    sns.barplot(x='Accuracy', y='Model', data=model_comparison)  # 创建水平柱状图
    plt.title('模型准确率比较')  # 图表标题
    plt.grid(True)  # 显示网格
    plt.save***REMOVED***g(get_absolute_path('data/model_accuracy_comparison.png'))  # 保存图像
    print("模型准确率比较图已保存到 data/model_accuracy_comparison.png")
    
    # F1分数比较柱状图
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='F1 Score', y='Model', data=model_comparison)
    plt.title('模型F1分数比较')
    plt.grid(True)
    plt.save***REMOVED***g(get_absolute_path('data/model_f1_comparison.png'))
    print("模型F1分数比较图已保存到 data/model_f1_comparison.png")
    
    # ROC AUC比较柱状图
    # 只包含支持概率预测的模型（dropna移除NaN值）
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='ROC AUC', y='Model', data=model_comparison.dropna(subset=['ROC AUC']))
    plt.title('模型ROC AUC比较')
    plt.grid(True)
    plt.save***REMOVED***g(get_absolute_path('data/model_roc_auc_comparison.png'))
    print("模型ROC AUC比较图已保存到 data/model_roc_auc_comparison.png")
    
    # 训练时间比较柱状图
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='Training Time (s)', y='Model', data=model_comparison)
    plt.title('模型训练时间比较')
    plt.grid(True)
    plt.save***REMOVED***g(get_absolute_path('data/model_training_time_comparison.png'))
    print("模型训练时间比较图已保存到 data/model_training_time_comparison.png")
    
    # 保存模型比较结果到CSV文件
    model_comparison.to_csv(get_absolute_path('data/model_comparison.csv'), index=False)
    print("模型比较结果已保存到 data/model_comparison.csv")
    
    return trained_models, all_metrics

def tune_hyperparameters(model_name, X_train, y_train, X_test, y_test):
    """
    为选定的模型进行超参数调优
    
    此函数使用随机搜索交叉验证(RandomizedSearchCV)来寻找模型的最佳超参数组合。
    RandomizedSearchCV比网格搜索更高效，因为它只评估部分参数组合。
    
    参数:
    model_name (str): 模型名称，如'Random Forest'
    X_train: 训练特征矩阵
    y_train: 训练目标变量
    X_test: 测试特征矩阵
    y_test: 测试目标变量
    
    返回:
    tuple: (调优后的模型, 最佳参数字典, 性能指标字典)
    """
    print(f"\n开始对 {model_name} 进行超参数调优...")
    
    # 为不同模型定义参数网格
    # 每个模型类型都有其特定的超参数集合
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],                        # 正则化强度，越小正则化越强
            'penalty': ['l1', 'l2', 'elasticnet', None],         # 正则化类型
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # 优化算法
            'max_iter': [1000]                                   # 最大迭代次数
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],                      # 树的数量
            'max_depth': [None, 10, 20, 30],                     # 树的最大深度
            'min_samples_split': [2, 5, 10],                     # 内部节点再划分所需最小样本数
            'min_samples_leaf': [1, 2, 4]                        # 叶节点最少样本数
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],                      # 树的数量
            'learning_rate': [0.01, 0.1, 0.2],                   # 学习率，控制每棵树的贡献
            'max_depth': [3, 5, 7],                              # 树的最大深度
            'subsample': [0.8, 1.0]                              # 样本随机采样比例
        },
        'SVM': {
            'C': [0.1, 1, 10],                                   # 正则化参数
            'kernel': ['linear', 'rbf', 'poly'],                 # 核函数类型
            'gamma': ['scale', 'auto', 0.1, 1]                   # 核系数，用于'rbf'和'poly'
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],                      # 树的数量
            'learning_rate': [0.01, 0.1, 0.2],                   # 学习率
            'max_depth': [3, 5, 7],                              # 树的最大深度
            'subsample': [0.8, 1.0],                             # 样本随机采样比例
            'colsample_bytree': [0.8, 1.0]                       # 特征随机采样比例
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],                      # 树的数量
            'learning_rate': [0.01, 0.1, 0.2],                   # 学习率
            'max_depth': [3, 5, 7],                              # 树的最大深度
            'subsample': [0.8, 1.0],                             # 样本随机采样比例
            'colsample_bytree': [0.8, 1.0]                       # 特征随机采样比例
        },
        'CatBoost': {
            'iterations': [50, 100, 200],                        # 树的数量
            'learning_rate': [0.01, 0.1, 0.2],                   # 学习率
            'depth': [4, 6, 8],                                  # 树的最大深度
            'l2_leaf_reg': [1, 3, 5, 7, 9]                       # L2正则化系数
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # 隐藏层大小
            'activation': ['relu', 'tanh'],                      # 激活函数
            'solver': ['adam', 'sgd'],                           # 权重优化器
            'alpha': [0.0001, 0.001, 0.01],                      # L2正则化系数
            'learning_rate': ['constant', 'adaptive']            # 学习率类型
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],                     # 树的最大深度
            'min_samples_split': [2, 5, 10],                     # 内部节点再划分所需最小样本数
            'min_samples_leaf': [1, 2, 4],                       # 叶节点最少样本数
            'criterion': ['gini', 'entropy']                     # 分裂标准，信息增益的计算方式
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9, 11],                     # 近邻数
            'weights': ['uniform', 'distance'],                  # 权重类型
            'p': [1, 2]                                          # 距离度量(1-曼哈顿距离，2-欧式距离)
        }
    }
    
    # 获取当前模型的参数网格
    if model_name not in param_grids:
        print(f"无法为 {model_name} 进行超参数调优，使用默认参数")
        return None, {}, {}
    
    param_grid = param_grids[model_name]
    
    # 根据模型名称获取模型实例
    model = get_model_instance(model_name)
    
    # 使用RandomizedSearchCV进行调优
    cv = 5           # 5折交叉验证
    n_iter = 20      # 随机搜索的迭代次数，评估20种参数组合
    
    # 创建随机搜索对象
    random_search = RandomizedSearchCV(
        model,                          # 基模型
        param_distributions=param_grid, # 参数分布/网格
        n_iter=n_iter,                  # 搜索迭代次数
        cv=cv,                          # 交叉验证折数
        scoring='f1',                   # 优化指标是F1分数
        n_jobs=-1,                      # 使用所有可用CPU核心
        random_state=42,                # 随机种子
        verbose=1                       # 输出详细信息
    )
    
    # 执行超参数搜索
    start_time = time.time()
    random_search.***REMOVED***t(X_train, y_train)  # 在训练数据上拟合
    search_time = time.time() - start_time
    
    # 输出搜索结果
    print(f"超参数搜索完成，耗时 {search_time:.2f} 秒")
    print(f"最佳参数: {random_search.best_params_}")  # 打印最佳参数组合
    print(f"最佳交叉验证得分: {random_search.best_score_:.4f}")  # 打印最佳交叉验证得分
    
    # 获取调优后的最佳模型
    best_model = random_search.best_estimator_
    
    # 在测试集上评估最佳模型的性能
    predictions = best_model.predict(X_test)  # 获取预测标签
    
    # 如果模型支持概率预测，则获取预测概率
    prob_predictions = None
    if hasattr(best_model, "predict_proba"):
        prob_predictions = best_model.predict_proba(X_test)[:, 1]
    
    # 计算各项性能指标
    metrics = calculate_metrics(y_test, predictions, prob_predictions)
    
    # 打印调优后模型的性能指标
    print("\n调优后的模型性能:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return best_model, random_search.best_params_, metrics

def get_model_instance(model_name):
    """
    根据模型名称返回相应的模型实例
    
    此函数为模型名称与实际模型类之间提供映射关系，
    确保超参数调优和其他函数能够正确获取模型实例。
    
    参数:
    model_name (str): 模型名称，如'Random Forest'
    
    返回:
    object: 模型实例，如RandomForestClassi***REMOVED***er对象
    """
    # 创建基础模型字典，包含常用分类模型
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),  # 逻辑回归
        'Random Forest': RandomForestClassi***REMOVED***er(random_state=42),                   # 随机森林
        'Gradient Boosting': GradientBoostingClassi***REMOVED***er(random_state=42),           # 梯度提升
        'SVM': SVC(probability=True, random_state=42),                             # 支持向量机
        'Decision Tree': DecisionTreeClassi***REMOVED***er(random_state=42),                   # 决策树
        'K-Nearest Neighbors': KNeighborsClassi***REMOVED***er(),                              # K近邻
        'MLP': MLPClassi***REMOVED***er(max_iter=1000, random_state=42)                       # 神经网络
    }
    
    # 添加可选模型（如果已导入）
    # 检查全局命名空间中是否存在这些类
    if 'XGBClassi***REMOVED***er' in globals():
        models['XGBoost'] = xgb.XGBClassi***REMOVED***er(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    if 'LGBMClassi***REMOVED***er' in globals():
        models['LightGBM'] = lgb.LGBMClassi***REMOVED***er(random_state=42)
    
    if 'CatBoostClassi***REMOVED***er' in globals():
        models['CatBoost'] = cb.CatBoostClassi***REMOVED***er(verbose=False, random_state=42)
        
    return models.get(model_name)  # 从字典中获取对应模型实例

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算各项评估指标
    
    此函数计算分类模型的多种性能指标，包括准确率、精确率、召回率、F1分数，
    如果提供了预测概率，还会计算ROC AUC。
    
    参数:
    y_true (array-like): 真实标签
    y_pred (array-like): 预测标签
    y_prob (array-like, optional): 预测概率，用于计算ROC AUC
    
    返回:
    dict: 包含各种指标的字典
    """
    # 创建指标字典
    metrics = {
        'test_accuracy': accuracy_score(y_true, y_pred),     # 准确率：预测正确的样本比例
        'test_precision': precision_score(y_true, y_pred),   # 精确率：预测为正的样本中实际为正的比例
        'test_recall': recall_score(y_true, y_pred),         # 召回率：实际为正的样本中被正确预测的比例
        'test_f1': f1_score(y_true, y_pred)                 # F1分数：精确率和召回率的调和平均数
    }
    
    # 如果提供了预测概率，则计算ROC AUC
    if y_prob is not None:
        metrics['test_roc_auc'] = roc_auc_score(y_true, y_prob)  # ROC曲线下面积
    
    return metrics

def save_model(model, ***REMOVED***le_path):
    """
    保存模型到指定路径
    
    此函数使用joblib库将训练好的模型序列化并保存到文件系统中，
    以便在将来可以加载模型进行预测而不需要重新训练。
    
    参数:
    model: 要保存的训练好的模型
    ***REMOVED***le_path (str): 模型保存路径，相对于脚本目录
    """
    # 获取绝对路径
    model_***REMOVED***le = get_absolute_path(***REMOVED***le_path)
    # 使用joblib.dump保存模型
    # joblib比pickle更高效，特别是对于包含大量numpy数组的模型
    joblib.dump(model, model_***REMOVED***le)
    print(f"模型已保存到 {model_***REMOVED***le}")

def train_and_save_best_model(data=None, target_metric='f1'):
    """
    训练多个模型，选择最佳模型，进行超参数调优，并保存完整处理流程
    
    此函数实现了端到端的模型训练流程：
    1. 准备数据：加载或使用传入的数据
    2. 划分训练集和测试集
    3. 训练多个候选模型
    4. 基于目标评估指标选择最佳模型
    5. 对最佳模型进行超参数调优
    6. 创建包含特征选择、特征工程和模型的完整流水线
    7. 评估并保存流水线模型
    
    参数:
    data (tuple or DataFrame): 包含特征和标签的数据，可以是DataFrame或(X, y)元组
    target_metric (str): 用于选择最佳模型的指标，默认为'f1'
    
    返回:
    tuple: (最佳模型流水线, 性能指标字典)
    """
    # 步骤1：准备数据
    if data is None:
        # 如果没有提供数据，则加载预处理过的数据
        data = load_engineered_data()
    
    # 检查data是否是DataFrame（而不是(X, y)元组）
    if isinstance(data, pd.DataFrame):
        # 如果是DataFrame，则分离X和y
        X = data.drop('target', axis=1)  # 特征矩阵
        y = data['target']               # 目标变量
    else:
        # 如果已经是(X, y)元组，则直接解包
        X, y = data
    
    # 步骤2：划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X, test_size=0.2, random_state=42, y=y)
    
    # 步骤3：训练多个基础模型
    print("训练多个基础模型...")
    # 训练所有候选模型并收集性能指标
    trained_models, all_metrics = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # 步骤4：选择最佳模型
    # 根据指定的评估指标选择性能最好的模型
    best_model_name = None
    best_metric_value = -float('inf')  # 初始化为负无穷，确保任何模型都能更新它
    
    # 遍历所有模型及其性能指标
    for model_name, metrics in all_metrics.items():
        # 构建评估指标的键名，如'test_f1'
        metric_key = f"test_{target_metric}"
        # 如果模型有该指标并且值更大，则更新最佳模型
        if metric_key in metrics and metrics[metric_key] > best_metric_value:
            best_metric_value = metrics[metric_key]
            best_model_name = model_name
    
    # 确保找到了最佳模型
    if best_model_name is None:
        raise ValueError(f"无法找到基于 {target_metric} 的最佳模型")
    
    # 输出最佳模型信息
    print(f"\n基于 {target_metric} 指标，最佳模型是: {best_model_name}")
    print(f"初始性能: {best_metric_value:.4f}")
    
    # 步骤5：对最佳模型进行超参数调优
    print(f"\n对最佳模型 {best_model_name} 进行超参数调优...")
    # 使用随机搜索调优超参数
    tuned_model, best_params, tuned_metrics = tune_hyperparameters(
        best_model_name, X_train, y_train, X_test, y_test
    )
    
    # 步骤6：保存调优后的最佳模型
    # 构建模型文件名，替换空格为下划线
    model_***REMOVED***lename = get_absolute_path(f"data/best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
    save_model(tuned_model, model_***REMOVED***lename)
    
    # 步骤7：创建完整的处理流程（机器学习流水线）
    print("\n创建完整的模型处理流程...")
    
    # 创建特征选择器 - 使用调优后的模型选择最重要的特征
    # 'median'阈值表示只保留重要性大于所有特征重要性中位数的特征
    feature_selector = SelectFromModel(tuned_model, threshold='median')
    feature_selector.***REMOVED***t(X_train, y_train)
    
    # 创建特征工程步骤
    
    # 多项式特征转换 - 创建特征的组合（交互项）
    # degree=2表示只考虑2阶交互项，interaction_only=True表示只包含交互项而不包含单个特征的平方
    poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    # 主成分分析(PCA) - 降维并捕获主要变异
    # 选择不超过10个主成分或原始特征数量(取较小值)
    pca = PCA(n_components=min(X_train.shape[1], 10))
    
    # 创建完整的处理流水线
    # 按顺序应用特征选择、特征组合、降维和最终模型
    ***REMOVED***nal_model = tuned_model
    complete_pipeline = Pipeline([
        ('feature_selector', feature_selector),  # 第一步：特征选择
        ('poly_features', poly_features),        # 第二步：创建特征交互项
        ('pca', pca),                           # 第三步：降维
        ('model', ***REMOVED***nal_model)                  # 第四步：调优后的模型
    ])
    
    # 在训练数据上拟合完整流程
    complete_pipeline.***REMOVED***t(X_train, y_train)
    
    # 评估完整流程在测试集上的性能
    pipeline_predictions = complete_pipeline.predict(X_test)
    pipeline_proba = None
    # 如果模型支持概率预测，获取预测概率
    if hasattr(complete_pipeline, "predict_proba"):
        pipeline_proba = complete_pipeline.predict_proba(X_test)[:, 1]
    
    # 计算并输出流水线模型的性能指标
    pipeline_metrics = calculate_metrics(y_test, pipeline_predictions, pipeline_proba)
    
    print("\n完整流程的性能:")
    for k, v in pipeline_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 保存完整的流水线模型
    pipeline_***REMOVED***lename = get_absolute_path('data/complete_model_pipeline.pkl')
    save_model(complete_pipeline, pipeline_***REMOVED***lename)
    
    return complete_pipeline, pipeline_metrics

def test_load_data():
    """
    测试数据加载和路径解析功能
    
    此函数用于验证数据加载过程是否正确，并输出数据的基本信息，
    包括形状、列名和目标变量分布。这对于快速检查数据是否可用很有用。
    
    返回:
    pandas.DataFrame: 加载的数据集
    """
    # 输出脚本目录和数据文件的绝对路径
    print(f"脚本目录: {SCRIPT_DIR}")
    print(f"engineered_data.csv 的绝对路径: {get_absolute_path('data/engineered_data.csv')}")
    
    # 尝试加载数据
    data = load_engineered_data()
    if data is not None:
        print("数据加载成功！")
        
        # 查看数据的基本统计信息
        print("\n数据基本信息:")
        print(f"形状: {data.shape}")  # 数据行数和列数
        print(f"列名: {data.columns.tolist()}")  # 所有特征名称
        
        # 计算并输出目标变量分布
        # normalize=True表示返回比例而不是计数，乘100转换为百分比
        target_counts = data['target'].value_counts(normalize=True) * 100
        print(f"目标变量分布:\n0 (拒绝): {target_counts[0]:.2f}%\n1 (批准): {target_counts[1]:.2f}%")
    else:
        print("数据加载失败！")
    
    return data

def main():
    """
    主函数，程序的入口点
    
    此函数实现了完整的模型训练流程，从数据加载到模型训练和保存。
    当脚本被直接运行时（而不是被导入），将执行此函数。
    """
    # 加载数据
    data = load_engineered_data()
    
    # 如果数据成功加载，则训练和保存最佳模型
    if data is not None:
        train_and_save_best_model(data)

if __name__ == "__main__":
    # 当脚本直接运行时执行以下代码
    
    # 先进行测试，确认数据可以正确加载
    test_load_data()
    
    # 正式运行模型训练
    print("\n开始正式模型训练过程...")
    main() 