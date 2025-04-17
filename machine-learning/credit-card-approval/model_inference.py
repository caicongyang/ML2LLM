#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型推理模块：用于信用卡审批系统的模型加载和预测
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_***REMOVED***le='data/best_model_random_forest.pkl'):
    """
    加载训练好的模型
    
    参数:
    model_***REMOVED***le (str): 模型文件路径
    
    返回:
    object: 加载的模型
    """
    try:
        model = joblib.load(model_***REMOVED***le)
        print(f"成功加载模型: {model_***REMOVED***le}")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def load_feature_transformers(***REMOVED***le_path='data/feature_transformers.pkl'):
    """
    加载特征转换器
    
    参数:
    ***REMOVED***le_path (str): 特征转换器文件路径
    
    返回:
    dict: 特征转换器字典
    """
    try:
        transformers = joblib.load(***REMOVED***le_path)
        print(f"成功加载特征转换器: {***REMOVED***le_path}")
        return transformers
    except Exception as e:
        print(f"加载特征转换器时出错: {e}")
        return None

def load_model_metadata(***REMOVED***le_path='data/best_model_metadata.json'):
    """
    加载模型元数据
    
    参数:
    ***REMOVED***le_path (str): 模型元数据文件路径
    
    返回:
    dict: 模型元数据
    """
    try:
        with open(***REMOVED***le_path, 'r') as f:
            metadata = json.load(f)
        print(f"成功加载模型元数据: {***REMOVED***le_path}")
        return metadata[0] if isinstance(metadata, list) else metadata
    except Exception as e:
        print(f"加载模型元数据时出错: {e}")
        return None

def preprocess_input(data, transformers):
    """
    使用保存的转换器预处理输入数据
    
    参数:
    data (pandas.DataFrame): 输入数据
    transformers (dict): 特征转换器字典
    
    返回:
    pandas.DataFrame: 预处理后的数据
    """
    if transformers is None:
        print("特征转换器不可用，无法预处理数据")
        return data
    
    processed_data = data.copy()
    
    try:
        # 应用特征选择
        if 'selected_features' in transformers:
            selected_features = transformers['selected_features']
            print(f"应用特征选择，从 {data.shape[1]} 个特征中选择 {len(selected_features)} 个")
            if all(feature in processed_data.columns for feature in selected_features):
                processed_data = processed_data[selected_features]
            else:
                print("警告: 某些选定的特征在输入数据中不存在")
        
        # 应用多项式特征
        if 'poly' in transformers and transformers['poly'] is not None:
            print("应用多项式特征转换")
            poly_features = transformers['poly'].transform(processed_data)
            poly_df = pd.DataFrame(
                poly_features, 
                columns=transformers.get('poly_feature_names', [f'poly_{i}' for i in range(poly_features.shape[1])])
            )
            processed_data = pd.concat([processed_data, poly_df], axis=1)
        
        # 应用PCA
        if 'pca' in transformers and transformers['pca'] is not None:
            print("应用PCA降维")
            pca_features = transformers['pca'].transform(processed_data)
            pca_columns = [f'pca_{i}' for i in range(pca_features.shape[1])]
            processed_data = pd.DataFrame(pca_features, columns=pca_columns)
        
        print(f"预处理完成，数据形状: {processed_data.shape}")
        return processed_data
        
    except Exception as e:
        print(f"预处理输入数据时出错: {e}")
        return data

def predict(model, data, threshold=0.5):
    """
    使用训练好的模型进行预测
    
    参数:
    model: 训练好的模型
    data (pandas.DataFrame): 预处理后的输入数据
    threshold (float): 分类阈值
    
    返回:
    tuple: (预测类别, 预测概率)
    """
    if model is None:
        print("模型不可用，无法进行预测")
        return None, None
    
    try:
        # 预测概率
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)[:, 1]
            print(f"生成概率预测，形状: {proba.shape}")
            
            # 根据阈值确定类别
            predictions = (proba >= threshold).astype(int)
            print(f"根据阈值 {threshold} 生成类别预测")
        else:
            # 直接预测类别
            predictions = model.predict(data)
            proba = None
            print("生成类别预测（模型不支持概率预测）")
        
        return predictions, proba
    
    except Exception as e:
        print(f"预测时出错: {e}")
        return None, None

def explain_prediction(model, data, prediction, probability=None, metadata=None):
    """
    解释预测结果
    
    参数:
    model: 训练好的模型
    data (pandas.DataFrame): 输入数据
    prediction (int): 预测类别
    probability (float): 预测概率
    metadata (dict): 模型元数据
    
    返回:
    dict: 预测解释
    """
    explanation = {
        'prediction': 'Approved' if prediction == 1 else 'Rejected',
        'con***REMOVED***dence': float(probability) if probability is not None else None,
    }
    
    # 添加模型信息
    if metadata is not None:
        explanation['model'] = {
            'name': metadata.get('model_name', 'Unknown'),
            'type': metadata.get('model_type', 'Unknown'),
            'performance': {
                'accuracy': metadata.get('metrics', {}).get('test_accuracy', None),
                'precision': metadata.get('metrics', {}).get('test_precision', None),
                'recall': metadata.get('metrics', {}).get('test_recall', None),
                'f1': metadata.get('metrics', {}).get('test_f1', None),
            }
        }
    
    # 对于某些模型，我们可以获取特征重要性
    if hasattr(model, 'feature_importances_'):
        # 获取特征重要性
        importances = model.feature_importances_
        feature_names = data.columns
        
        # 创建特征重要性字典
        feature_importance = dict(zip(feature_names, importances))
        
        # 按重要性排序
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 获取前5个最重要的特征
        top_features = sorted_importance[:5]
        
        explanation['important_features'] = {feature: float(importance) for feature, importance in top_features}
    
    return explanation

def visualize_prediction(prediction, probability=None, threshold=0.5):
    """
    可视化预测结果
    
    参数:
    prediction (int): 预测类别
    probability (float): 预测概率
    threshold (float): 分类阈值
    
    返回:
    None
    """
    if probability is None:
        print("无概率信息可供可视化")
        return
    
    # 创建条形图显示预测概率
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
    
    # 设置颜色
    colors = ['#FF9999', '#66B2FF'] if prediction == 1 else ['#66B2FF', '#FF9999']
    
    # 创建条形图
    bars = plt.bar([0, 1], [1-probability, probability], color=colors)
    
    # 添加阈值线
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'阈值 ({threshold})')
    
    # 添加标签和标题
    plt.xticks([0, 1], ['拒绝', '通过'])
    plt.ylabel('预测概率')
    plt.title('信用卡申请预测结果')
    
    # 在条形上方添加文本
    plt.text(0, 1-probability, f'{1-probability:.4f}', ha='center', va='bottom')
    plt.text(1, probability, f'{probability:.4f}', ha='center', va='bottom')
    
    # 添加预测结果说明
    result_text = '通过' if prediction == 1 else '拒绝'
    con***REMOVED***dence = probability if prediction == 1 else 1-probability
    plt.***REMOVED***gtext(0.5, 0.01, f'预测结果: {result_text} (置信度: {con***REMOVED***dence:.2f})', ha='center', fontsize=12)
    
    # 添加图例
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.save***REMOVED***g('data/prediction_visualization.png')
    print("预测结果可视化已保存到 data/prediction_visualization.png")
    
    plt.close()

def process_single_application(application_data, model=None, transformers=None, threshold=0.5):
    """
    处理单个信用卡申请
    
    参数:
    application_data (pandas.DataFrame): 申请数据
    model: 训练好的模型，如果为None则加载默认模型
    transformers: 特征转换器，如果为None则加载默认转换器
    threshold (float): 分类阈值
    
    返回:
    dict: 处理结果
    """
    # 加载模型和转换器（如果未提供）
    if model is None:
        model = load_model()
    
    if transformers is None:
        transformers = load_feature_transformers()
    
    metadata = load_model_metadata()
    
    # 预处理申请数据
    processed_data = preprocess_input(application_data, transformers)
    
    # 进行预测
    prediction, probability = predict(model, processed_data, threshold)
    
    if prediction is not None:
        # 获取第一个预测结果（对于单个申请）
        single_prediction = prediction[0]
        single_probability = probability[0] if probability is not None else None
        
        # 解释预测
        explanation = explain_prediction(
            model, 
            processed_data, 
            single_prediction, 
            single_probability, 
            metadata
        )
        
        # 可视化预测
        visualize_prediction(single_prediction, single_probability, threshold)
        
        return {
            'success': True,
            'prediction': int(single_prediction),
            'probability': float(single_probability) if single_probability is not None else None,
            'explanation': explanation
        }
    else:
        return {
            'success': False,
            'error': '预测过程中出错'
        }

def batch_process_applications(applications_data, model=None, transformers=None, threshold=0.5):
    """
    批量处理信用卡申请
    
    参数:
    applications_data (pandas.DataFrame): 批量申请数据
    model: 训练好的模型，如果为None则加载默认模型
    transformers: 特征转换器，如果为None则加载默认转换器
    threshold (float): 分类阈值
    
    返回:
    pandas.DataFrame: 处理结果
    """
    # 加载模型和转换器（如果未提供）
    if model is None:
        model = load_model()
    
    if transformers is None:
        transformers = load_feature_transformers()
    
    # 预处理申请数据
    processed_data = preprocess_input(applications_data, transformers)
    
    # 进行预测
    predictions, probabilities = predict(model, processed_data, threshold)
    
    if predictions is not None:
        # 创建结果DataFrame
        results = applications_data.copy()
        results['prediction'] = predictions
        if probabilities is not None:
            results['probability'] = probabilities
        
        # 添加决策结果
        results['decision'] = results['prediction'].map({0: '拒绝', 1: '通过'})
        
        # 保存批量处理结果
        results.to_csv('data/batch_processing_results.csv', index=False)
        print("批量处理结果已保存到 data/batch_processing_results.csv")
        
        # 生成批量结果摘要
        summary = {
            'total_applications': len(results),
            'approved': int(results['prediction'].sum()),
            'rejected': int((results['prediction'] == 0).sum()),
            'approval_rate': float(results['prediction'].mean())
        }
        
        # 可视化批量结果
        plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
        ax = sns.countplot(x='decision', data=results, palette=['#FF9999', '#66B2FF'])
        
        # 添加计数标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 5), 
                        textcoords = 'offset points')
        
        plt.title('信用卡申请批量处理结果')
        plt.ylabel('申请数量')
        plt.tight_layout()
        plt.save***REMOVED***g('data/batch_processing_summary.png')
        print("批量处理摘要图已保存到 data/batch_processing_summary.png")
        
        plt.close()
        
        return results, summary
    else:
        print("批量处理过程中出错")
        return None, None

def main():
    """主函数：示例用法"""
    # 示例：加载模型和转换器
    model = load_model()
    transformers = load_feature_transformers()
    
    if model is not None and transformers is not None:
        # 创建一个示例申请
        example_application = pd.DataFrame({
            # 根据实际特征调整
            'feature1': [0.5],
            'feature2': [0.7],
            'feature3': [-0.2],
            # ...添加更多特征
        })
        
        # 处理单个申请
        result = process_single_application(example_application, model, transformers)
        print("\n单个申请处理结果:")
        print(f"预测: {'通过' if result['prediction'] == 1 else '拒绝'}")
        print(f"概率: {result['probability']:.4f}")

if __name__ == "__main__":
    main() 