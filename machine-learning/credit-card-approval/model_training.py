#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练模块：用于信用卡审批系统的模型选择、训练和保存
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classi***REMOVED***cation_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassi***REMOVED***er
from sklearn.ensemble import RandomForestClassi***REMOVED***er, GradientBoostingClassi***REMOVED***er
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassi***REMOVED***er
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassi***REMOVED***er
from sklearn.pipeline import Pipeline
import joblib
import time

def load_engineered_data(***REMOVED***le_path='data/engineered_data.csv'):
    """
    加载特征工程后的数据集
    
    参数:
    ***REMOVED***le_path (str): 数据文件路径
    
    返回:
    pandas.DataFrame: 加载的数据集
    """
    try:
        data = pd.read_csv(***REMOVED***le_path)
        print(f"成功加载特征工程后的数据，形状为: {data.shape}")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def split_data(data, test_size=0.2, random_state=42):
    """
    将数据集分割为训练集和测试集
    
    参数:
    data (pandas.DataFrame): 输入数据集
    test_size (float): 测试集的比例
    random_state (int): 随机种子
    
    返回:
    tuple: (X_train, X_test, y_train, y_test)
    """
    if data is None:
        return None, None, None, None
    
    print(f"\n将数据分割为训练集 ({1-test_size:.0%}) 和测试集 ({test_size:.0%})...")
    
    # 分离特征和目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    训练模型并评估性能
    
    参数:
    model: 分类器模型
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    model_name (str): 模型名称
    
    返回:
    tuple: (训练好的模型, 性能指标字典)
    """
    print(f"\n训练 {model_name} 模型...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.***REMOVED***t(X_train, y_train)
    
    # 计算训练时间
    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.2f} 秒")
    
    # 在训练集上评估
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 在测试集上评估
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 计算性能指标
    metrics = {}
    
    # 训练集性能
    metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    metrics['train_precision'] = precision_score(y_train, y_train_pred)
    metrics['train_recall'] = recall_score(y_train, y_train_pred)
    metrics['train_f1'] = f1_score(y_train, y_train_pred)
    if y_train_prob is not None:
        metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_prob)
    
    # 测试集性能
    metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
    metrics['test_precision'] = precision_score(y_test, y_test_pred)
    metrics['test_recall'] = recall_score(y_test, y_test_pred)
    metrics['test_f1'] = f1_score(y_test, y_test_pred)
    if y_test_prob is not None:
        metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_prob)
    
    # 训练时间
    metrics['train_time'] = train_time
    
    # 输出性能指标
    print(f"\n{model_name} 性能指标:")
    print(f"训练集准确率: {metrics['train_accuracy']:.4f}")
    print(f"测试集准确率: {metrics['test_accuracy']:.4f}")
    print(f"测试集精确率: {metrics['test_precision']:.4f}")
    print(f"测试集召回率: {metrics['test_recall']:.4f}")
    print(f"测试集F1分数: {metrics['test_f1']:.4f}")
    if 'test_roc_auc' in metrics:
        print(f"测试集ROC AUC: {metrics['test_roc_auc']:.4f}")
    
    # 计算并显示混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['拒绝', '通过'], yticklabels=['拒绝', '通过'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.save***REMOVED***g(f'data/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    print(f"{model_name} 混淆矩阵已保存到 data/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    
    # 如果模型支持概率预测，绘制ROC曲线
    if y_test_prob is not None:
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {metrics["test_roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title(f'{model_name} ROC曲线')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.save***REMOVED***g(f'data/{model_name.lower().replace(" ", "_")}_roc_curve.png')
        print(f"{model_name} ROC曲线已保存到 data/{model_name.lower().replace(' ', '_')}_roc_curve.png")
        
        # 绘制精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        plt.***REMOVED***gure(***REMOVED***gsize=(8, 6))
        plt.plot(recall, precision, label=f'精确率-召回率曲线')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title(f'{model_name} 精确率-召回率曲线')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.save***REMOVED***g(f'data/{model_name.lower().replace(" ", "_")}_pr_curve.png')
        print(f"{model_name} 精确率-召回率曲线已保存到 data/{model_name.lower().replace(' ', '_')}_pr_curve.png")
    
    return model, metrics

def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    训练多个模型并比较它们的性能
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    
    返回:
    tuple: (所有模型字典, 所有性能指标字典)
    """
    print("\n训练多个模型并比较性能...")
    
    # 创建模型列表
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassi***REMOVED***er(random_state=42),
        'Random Forest': RandomForestClassi***REMOVED***er(random_state=42),
        'Gradient Boosting': GradientBoostingClassi***REMOVED***er(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassi***REMOVED***er(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassi***REMOVED***er(random_state=42, max_iter=1000)
    }
    
    # 训练所有模型
    trained_models = {}
    all_metrics = {}
    
    for name, model in models.items():
        trained_model, metrics = train_model(model, X_train, y_train, X_test, y_test, name)
        trained_models[name] = trained_model
        all_metrics[name] = metrics
    
    # 比较模型性能
    model_comparison = pd.DataFrame({
        'Model': list(all_metrics.keys()),
        'Accuracy': [metrics['test_accuracy'] for metrics in all_metrics.values()],
        'Precision': [metrics['test_precision'] for metrics in all_metrics.values()],
        'Recall': [metrics['test_recall'] for metrics in all_metrics.values()],
        'F1 Score': [metrics['test_f1'] for metrics in all_metrics.values()],
        'ROC AUC': [metrics.get('test_roc_auc', np.nan) for metrics in all_metrics.values()],
        'Training Time (s)': [metrics['train_time'] for metrics in all_metrics.values()]
    })
    
    # 按F1分数排序
    model_comparison = model_comparison.sort_values('F1 Score', ascending=False)
    
    print("\n模型性能比较:")
    print(model_comparison)
    
    # 可视化模型比较
    # 准确率比较
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='Accuracy', y='Model', data=model_comparison)
    plt.title('模型准确率比较')
    plt.grid(True)
    plt.save***REMOVED***g('data/model_accuracy_comparison.png')
    print("模型准确率比较图已保存到 data/model_accuracy_comparison.png")
    
    # F1分数比较
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='F1 Score', y='Model', data=model_comparison)
    plt.title('模型F1分数比较')
    plt.grid(True)
    plt.save***REMOVED***g('data/model_f1_comparison.png')
    print("模型F1分数比较图已保存到 data/model_f1_comparison.png")
    
    # ROC AUC比较
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='ROC AUC', y='Model', data=model_comparison.dropna(subset=['ROC AUC']))
    plt.title('模型ROC AUC比较')
    plt.grid(True)
    plt.save***REMOVED***g('data/model_roc_auc_comparison.png')
    print("模型ROC AUC比较图已保存到 data/model_roc_auc_comparison.png")
    
    # 训练时间比较
    plt.***REMOVED***gure(***REMOVED***gsize=(12, 8))
    sns.barplot(x='Training Time (s)', y='Model', data=model_comparison)
    plt.title('模型训练时间比较')
    plt.grid(True)
    plt.save***REMOVED***g('data/model_training_time_comparison.png')
    print("模型训练时间比较图已保存到 data/model_training_time_comparison.png")
    
    # 保存模型比较结果
    model_comparison.to_csv('data/model_comparison.csv', index=False)
    print("模型比较结果已保存到 data/model_comparison.csv")
    
    return trained_models, all_metrics

def tune_hyperparameters(best_model_name, X_train, y_train, X_test, y_test):
    """
    使用网格搜索进行超参数调优
    
    参数:
    best_model_name (str): 最佳模型的名称
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    
    返回:
    tuple: (调优后的模型, 最佳参数, 性能指标)
    """
    print(f"\n对 {best_model_name} 进行超参数调优...")
    
    # 为不同的模型定义参数网格
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    # 为不同的模型创建基础模型
    base_models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassi***REMOVED***er(random_state=42),
        'Random Forest': RandomForestClassi***REMOVED***er(random_state=42),
        'Gradient Boosting': GradientBoostingClassi***REMOVED***er(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassi***REMOVED***er(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassi***REMOVED***er(random_state=42, max_iter=2000)
    }
    
    # 获取模型和参数网格
    if best_model_name not in param_grids:
        print(f"没有为 {best_model_name} 定义参数网格")
        return None, None, None
    
    base_model = base_models[best_model_name]
    param_grid = param_grids[best_model_name]
    
    # 创建网格搜索
    print("执行网格搜索，这可能需要一段时间...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行网格搜索
    grid_search.***REMOVED***t(X_train, y_train)
    
    # 计算训练时间
    train_time = time.time() - start_time
    print(f"超参数调优时间: {train_time:.2f} 秒")
    
    # 获取最佳模型和参数
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\n最佳参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 评估最佳模型
    tuned_model, metrics = train_model(
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
        f"Tuned {best_model_name}"
    )
    
    return tuned_model, best_params, metrics

def save_model(model, model_***REMOVED***le='data/best_model.pkl'):
    """
    保存模型到文件
    
    参数:
    model: 要保存的模型
    model_***REMOVED***le (str): 模型文件路径
    """
    try:
        joblib.dump(model, model_***REMOVED***le)
        print(f"模型已保存到 {model_***REMOVED***le}")
    except Exception as e:
        print(f"保存模型时出错: {e}")

def train_and_save_best_model(data=None, target_metric='f1'):
    """
    训练并保存最佳模型
    
    参数:
    data (pandas.DataFrame): 输入数据集
    target_metric (str): 用于选择最佳模型的指标 ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
    
    返回:
    tuple: (最佳模型, 性能指标)
    """
    if data is None:
        data = load_engineered_data()
        if data is None:
            return None, None
    
    print("开始模型训练过程...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = split_data(data)
    
    # 训练多个模型
    trained_models, all_metrics = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # 选择最佳模型
    metric_key = f'test_{target_metric}'
    best_model_name = max(all_metrics.items(), key=lambda x: x[1].get(metric_key, 0))[0]
    
    print(f"\n基于 {target_metric} 指标，最佳模型是: {best_model_name}")
    
    # 对最佳模型进行超参数调优
    tuned_model, best_params, tuned_metrics = tune_hyperparameters(
        best_model_name,
        X_train,
        y_train,
        X_test,
        y_test
    )
    
    # 比较原始模型和调优后的模型
    original_metric = all_metrics[best_model_name][metric_key]
    tuned_metric = tuned_metrics[metric_key]
    
    print(f"\n{best_model_name}:")
    print(f"原始模型 {target_metric}: {original_metric:.4f}")
    print(f"调优后的模型 {target_metric}: {tuned_metric:.4f}")
    print(f"改进: {(tuned_metric - original_metric) / original_metric:.2%}")
    
    # 保存最终模型
    ***REMOVED***nal_model = tuned_model if tuned_metric > original_metric else trained_models[best_model_name]
    model_type = "tuned" if tuned_metric > original_metric else "original"
    
    save_model(***REMOVED***nal_model, f'data/best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    
    # 保存模型元数据
    model_metadata = {
        'model_name': best_model_name,
        'model_type': model_type,
        'parameters': best_params if model_type == "tuned" else "default",
        'metrics': tuned_metrics if model_type == "tuned" else all_metrics[best_model_name]
    }
    
    pd.DataFrame([model_metadata]).to_json('data/best_model_metadata.json', orient='records')
    print("模型元数据已保存到 data/best_model_metadata.json")
    
    print("\n模型训练和选择完成！")
    
    return ***REMOVED***nal_model, tuned_metrics if model_type == "tuned" else all_metrics[best_model_name]

def main():
    """主函数"""
    data = load_engineered_data()
    if data is not None:
        train_and_save_best_model(data)

if __name__ == "__main__":
    main() 