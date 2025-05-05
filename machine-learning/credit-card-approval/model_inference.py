#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型推理模块：用于信用卡审批系统的模型加载和预测

本模块提供了信用卡审批系统的模型推理功能，包括：
1. 加载预训练的机器学习模型
2. 加载特征转换器
3. 处理输入数据并进行特征转换
4. 执行模型推理获取预测结果
5. 提供结果解释和建议
"""

# 标准库导入
import os          # 用于文件和目录操作
import sys         # 用于系统相关功能和路径操作
import numpy as np  # 用于数值计算和数组操作
import pandas as pd  # 用于数据处理和分析
import joblib      # 用于模型和数据的序列化/反序列化
import pickle      # 用于对象的序列化/反序列化

# 可视化库导入
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib后端为Agg，防止在没有GUI的环境中显示错误
import matplotlib.pyplot as plt  # 用于数据可视化和绘图
import seaborn as sns  # 用于高级数据可视化

# 设置中文字体支持，确保图表中的中文能正确显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 设置优先使用的字体，按顺序尝试
plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号显示为方块的问题

# 获取脚本所在目录的绝对路径，确保在任何环境中都能正确加载相关文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 将父目录添加到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(SCRIPT_DIR))

def get_absolute_path(relative_path):
    """
    获取相对于脚本目录的绝对路径
    
    此函数将相对路径转换为基于脚本所在目录的绝对路径，
    确保在任何工作目录下运行脚本时，都能正确找到文件。
    
    参数:
    relative_path (str): 相对于脚本目录的相对路径
    
    返回:
    str: 计算得到的绝对路径
    
    示例:
    >>> get_absolute_path('data/model.pkl')
    '/absolute/path/to/script/data/model.pkl'
    """
    return os.path.join(SCRIPT_DIR, relative_path)

def load_model():
    """
    加载预训练的机器学习模型
    
    按优先级尝试加载不同类型的模型文件：
    1. 首先尝试加载逻辑回归模型
    2. 如果逻辑回归模型不存在，尝试加载随机森林模型
    3. 如果都不存在，则尝试加载任何以'best_model_'开头的模型文件
    
    返回:
    object or None: 加载成功则返回模型对象，否则返回None
    
    可能抛出的异常:
    - FileNotFoundError: 模型文件不存在
    - Exception: 加载模型时出现其他错误
    """
    try:
        # 尝试加载逻辑回归模型
        model_path = get_absolute_path('data/best_model_logistic_regression.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"成功加载模型: {model_path}")
            return model
        
        # 如果逻辑回归模型不存在，尝试加载随机森林模型
        model_path = get_absolute_path('data/best_model_random_forest.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"成功加载模型: {model_path}")
            return model
            
        # 如果找不到任何预期的模型文件，尝试查找其他可能的模型文件
        model_files = [f for f in os.listdir(get_absolute_path('data')) if f.startswith('best_model_') and f.endswith('.pkl')]
        if model_files:
            model_path = get_absolute_path(f'data/{model_files[0]}')
            model = joblib.load(model_path)
            print(f"成功加载模型: {model_path}")
            return model
            
        print(f"未找到任何模型文件")
        return None
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None

def load_feature_transformers():
    """
    加载特征转换器
    
    特征转换器通常包含数据预处理所需的所有组件，例如：
    - 数值特征和类别特征的列表
    - 缺失值填充器
    - 类别特征编码器
    - 特征缩放器
    - PCA转换器
    
    返回:
    dict or None: 包含所有特征转换组件的字典，加载失败则返回None
    
    可能抛出的异常:
    - FileNotFoundError: 特征转换器文件不存在
    - Exception: 加载特征转换器时出现其他错误
    """
    try:
        transformer_path = get_absolute_path('data/feature_transformers.pkl')
        if not os.path.exists(transformer_path):
            print(f"特征转换器文件不存在: {transformer_path}")
            return None
        transformers = joblib.load(transformer_path)
        print(f"成功加载特征转换器: {transformer_path}")
        return transformers
    except FileNotFoundError:
        print("特征转换器文件未找到。您需要先运行特征工程脚本生成特征转换器。")
        return None
    except Exception as e:
        print(f"加载特征转换器时出错: {str(e)}")
        return None

def test_load_model():
    """
    测试加载模型和转换器的功能
    
    此函数用于验证模型和特征转换器能否正确加载，
    并打印相关信息，便于调试和确认文件路径是否正确。
    
    返回:
    tuple: (model, transformers) 包含加载的模型和特征转换器对象
    
    示例输出:
    脚本目录: /path/to/script
    尝试从以下路径加载模型: /path/to/script/data/best_model_random_forest.pkl
    模型加载成功!
    尝试从以下路径加载特征转换器: /path/to/script/data/feature_transformers.pkl
    特征转换器加载成功!
    转换器包含以下组件: ['numerical_features', 'categorical_features', 'numerical_imputer', ...]
    """
    print(f"脚本目录: {SCRIPT_DIR}")
    
    # 尝试加载模型
    model = None
    try:
        model_path = get_absolute_path('data/best_model_random_forest.pkl')
        print(f"尝试从以下路径加载模型: {model_path}")
        model = joblib.load(model_path)
        print("模型加载成功!")
    except FileNotFoundError:
        print("模型文件未找到。您需要先运行模型训练脚本生成模型文件。")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
    
    # 尝试加载特征转换器
    transformers = None
    try:
        transformer_path = get_absolute_path('data/feature_transformers.pkl')
        print(f"尝试从以下路径加载特征转换器: {transformer_path}")
        transformers = joblib.load(transformer_path)
        print("特征转换器加载成功!")
        print(f"转换器包含以下组件: {list(transformers.keys())}")
    except FileNotFoundError:
        print("特征转换器文件未找到。您需要先运行特征工程脚本生成特征转换器。")
    except Exception as e:
        print(f"加载特征转换器时出错: {str(e)}")
    
    return model, transformers

def process_credit_application(application_data, model, transformers):
    """
    处理单个信用卡申请并预测结果
    
    此函数是整个信用卡审批流程的核心，它执行以下步骤：
    1. 接收申请数据并转换为DataFrame格式
    2. 使用特征转换器对数据进行预处理
    3. 调用模型进行预测
    4. 根据预测结果和申请数据生成解释和建议
    
    参数:
    application_data (dict): 单个申请的数据，包含所有特征字段
    model (object): 训练好的机器学习模型，如逻辑回归或随机森林
    transformers (dict): 特征转换器字典，包含所有预处理所需的组件
    
    返回:
    dict: 包含以下字段的结果字典：
        - approved (bool): 是否批准申请
        - approval_probability (float): 批准的概率值(0-1)
        - confidence (float): 预测的置信度(0-1)
        - decision_factors (list): 影响决策的关键因素列表
        - suggestions (list): 给申请人的建议列表
    
    异常处理:
    捕获所有异常并返回带有错误信息的拒绝结果
    """
    try:
        # 将单个应用转换为DataFrame，便于后续处理
        df = pd.DataFrame([application_data])
        
        # 应用特征转换
        # 以下代码假设transformers包含所有必要的转换器组件
        # 如：缺失值填充、类别编码、特征缩放等，最后可能还有PCA转换
        
        # 1. 分离数值和类别特征，不同特征类型需要不同的预处理方法
        numerical_features = transformers['numerical_features']
        categorical_features = transformers['categorical_features']
        
        # 2. 应用预处理转换
        # 2.1 处理数值特征：通常包括缺失值填充和特征缩放
        df_numerical = df[numerical_features].copy()
        if 'numerical_imputer' in transformers:
            # 使用训练集上拟合的填充器填充缺失值
            df_numerical = pd.DataFrame(
                transformers['numerical_imputer'].transform(df_numerical),
                columns=numerical_features
            )
        
        # 2.2 处理类别特征：通常包括缺失值填充和类别编码
        df_categorical = df[categorical_features].copy()
        if 'categorical_imputer' in transformers:
            # 使用训练集上拟合的填充器填充缺失值
            df_categorical = pd.DataFrame(
                transformers['categorical_imputer'].transform(df_categorical),
                columns=categorical_features
            )
        
        # 2.3 应用独热编码：将类别特征转换为数值特征
        if 'onehot_encoder' in transformers:
            # 使用训练集上拟合的编码器进行独热编码
            onehot_encoded = transformers['onehot_encoder'].transform(df_categorical)
            onehot_feature_names = transformers['onehot_feature_names']
            df_categorical_encoded = pd.DataFrame(
                onehot_encoded.toarray(),
                columns=onehot_feature_names
            )
        else:
            # 如果没有预先拟合的编码器，使用pandas内置的get_dummies方法
            df_categorical_encoded = pd.get_dummies(df_categorical)
            
        # 2.4 合并处理后的数值特征和类别特征
        df_preprocessed = pd.concat([df_numerical, df_categorical_encoded], axis=1)
        
        # 2.5 应用PCA变换（如果模型训练时使用了PCA）
        if 'pca' in transformers:
            pca_result = transformers['pca'].transform(df_preprocessed)
            transformed_data = pd.DataFrame(
                pca_result,
                columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
            )
        else:
            transformed_data = df_preprocessed
        
        # 3. 使用模型进行预测
        prediction = model.predict(transformed_data)[0]  # 获取预测的类别(0或1)
        
        # 4. 获取预测概率（如果模型支持）
        try:
            # 获取属于正类(批准)的概率
            probability = model.predict_proba(transformed_data)[0][1]  # 假设1代表"通过"
        except:
            # 如果模型不支持概率预测，则使用预测值作为概率
            probability = float(prediction)
        
        # 5. 计算置信度（预测的确定性程度）
        confidence = max(probability, 1 - probability)
        
        # 6. 准备决策因素和建议
        decision_factors = []
        suggestions = []
        
        # 7. 基于申请数据分析决策因素，并提供个性化建议
        # 7.1 检查年龄因素
        if application_data['A2'] < 25.0:
            decision_factors.append("申请人年龄较低")
            suggestions.append("建立更长的信用记录可能有助于提高批准率")
        
        # 7.2 检查信用评分相关指标
        if application_data['A11'] < 5:  # 较低的信用评分相关指标
            decision_factors.append("信用评分相关指标低于平均水平")
            suggestions.append("提高信用评分：按时支付账单并减少现有债务")
        elif application_data['A11'] > 10:
            decision_factors.append("优秀的信用评分相关指标")
        
        # 7.3 检查违约记录
        if application_data['A9'] == 't':  # t 表示有违约记录
            decision_factors.append("有过违约记录")
            suggestions.append("改善支付历史记录：确保所有账单按时支付")
        
        # 7.4 检查债务收入比
        if application_data['A3'] > 5.0:  # 较高的债务收入比
            decision_factors.append(f"债务收入比较高: {application_data['A3']:.2f}")
            suggestions.append("减少现有债务以改善债务收入比")
        
        # 7.5 检查就业历史
        if application_data['A8'] < 2.0:  # 就业年限短
            decision_factors.append("就业历史较短")
            suggestions.append("就业稳定性是重要因素，保持当前工作可能有助于未来申请")
        
        # 7.6 如果没有足够数据生成决策因素，提供一般性解释
        if not decision_factors:
            if prediction == 1:
                decision_factors.append("综合评估符合信用卡批准条件")
            else:
                decision_factors.append("综合评估未满足信用卡批准条件")
                suggestions.append("请考虑改善信用评分、减少债务或增加收入")
        
        # 7.7 如果没有足够数据生成建议，提供一般性建议
        if not suggestions and prediction == 0:
            suggestions.append("请联系客服获取更多关于您的申请被拒绝的信息")
        elif not suggestions:
            suggestions.append("继续保持良好的信用记录")
        
        # 8. 返回处理结果
        return {
            'approved': prediction == 1,           # 是否批准
            'approval_probability': float(probability),  # 批准的概率
            'confidence': float(confidence),       # 预测的置信度
            'decision_factors': decision_factors,  # 决策因素列表
            'suggestions': suggestions             # 建议列表
        }
        
    except Exception as e:
        # 处理过程中出现任何错误，记录错误并返回拒绝结果
        print(f"处理申请时出错: {str(e)}")
        return {
            'approved': False,
            'approval_probability': 0.0,
            'confidence': 0.0,
            'decision_factors': ["处理申请时出错"],
            'suggestions': ["请检查输入数据是否正确，然后重试"]
        }

def process_feature(raw_data, feature_transformers):
    """
    处理输入特征，返回原始特征和PCA特征
    
    此函数负责将输入的原始特征数据进行预处理，以便进行模型预测。
    主要有两条处理路径：
    1. 处理原始特征：确保格式和特征名称与模型期望的一致
    2. 生成PCA特征：如果模型使用PCA特征，将原始特征转换为PCA特征
    
    Args:
        raw_data (dict): 输入的原始特征数据，包含所有需要的特征字段
        feature_transformers (dict): 包含所有特征转换器的字典，如:
            - numerical_features: 数值特征列表
            - categorical_features: 类别特征列表 
            - selected_features: 模型使用的特征列表
            - imputers: 填充缺失值的转换器
            - encoders: 编码类别特征的转换器
            - pca_transformer: PCA降维转换器
    
    Returns:
        tuple: (原始特征DataFrame, PCA特征DataFrame)
            - 如果某一类特征处理失败，对应的返回值为None
            - 成功处理的特征将作为DataFrame返回，列名与模型期望的一致
    
    异常处理:
        - 捕获并记录所有处理过程中的异常
        - 当出现严重错误时返回(None, None)
        - 对于部分错误，尝试进行调整和修复以继续处理
    """
    try:
        # 1. 准备输入数据：将字典转换为DataFrame
        input_df = pd.DataFrame([raw_data])
        print("预处理后的输入数据:")
        print(input_df)
        
        # 2. 特征选择与兼容性处理
        # 检查特征转换器中是否有模型期望的特征列表
        if 'selected_features' in feature_transformers:
            selected_features = feature_transformers['selected_features']
            
            # 2.1 特征兼容性检查：确认输入数据包含所有模型需要的特征
            missing_features = [feat for feat in selected_features if feat not in input_df.columns]
            extra_features = [feat for feat in input_df.columns if feat not in selected_features]
            
            # 输出特征匹配情况，帮助调试
            if missing_features:
                print(f"警告: 输入数据缺少以下特征: {missing_features}")
            if extra_features:
                print(f"信息: 输入数据包含额外特征: {extra_features}")
            
            # 2.2 特征修复：为缺失的特征创建默认值
            for feat in selected_features:
                if feat not in input_df.columns:
                    print(f"警告: 输入数据缺少特征 '{feat}'，使用0填充")
                    input_df[feat] = 0  # 为缺失特征填充默认值0
                    
            # 2.3 提取模型所需的特征子集
            raw_features = input_df[selected_features].copy()
        else:
            # 如果没有特定的特征选择配置，则使用所有可用特征
            raw_features = input_df.copy()
            
        print(f"原始特征数据:")
        print(raw_features)
        
        # 3. PCA特征转换（如果模型使用PCA）
        pca_features = None
        if 'pca_transformer' in feature_transformers and feature_transformers['pca_transformer'] is not None:
            try:
                print("开始PCA特征转换...")
                pca_transformer = feature_transformers['pca_transformer']
                
                # 3.1 检查PCA转换器的特征兼容性
                # 确保输入特征与PCA转换器期望的特征匹配
                if hasattr(pca_transformer, 'feature_names_in_'):
                    required_features = pca_transformer.feature_names_in_
                    
                    # 检查特征兼容性，确保所有需要的特征都存在
                    if not all(feat in raw_features.columns for feat in required_features):
                        print("输入特征与PCA转换器期望的特征不匹配")
                        print(f"PCA转换器期望特征: {required_features}")
                        print(f"当前输入特征: {raw_features.columns.tolist()}")
                        
                        # 3.2 特征适配：调整输入特征以匹配PCA期望
                        # 为缺失的特征添加默认值
                        for feat in required_features:
                            if feat not in raw_features.columns:
                                raw_features[feat] = 0  # 用0填充缺失特征
                        
                        # 删除PCA不需要的多余特征
                        extra_cols = [col for col in raw_features.columns if col not in required_features]
                        if extra_cols:
                            raw_features = raw_features.drop(columns=extra_cols)
                            
                        # 确保列顺序与训练时一致，这对某些模型和转换器很重要
                        raw_features = raw_features[required_features]
                        
                        print("调整后的特征:")
                        print(raw_features)
                
                # 3.3 执行PCA转换：将原始特征转换为PCA特征
                pca_result = pca_transformer.transform(raw_features)
                
                # 3.4 创建PCA特征数据帧：为PCA结果创建有意义的列名
                pca_features = pd.DataFrame(
                    pca_result,
                    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
                )
                
                print(f"PCA转换后的特征:")
                print(pca_features)
            except Exception as e:
                # 捕获PCA转换过程中的任何错误
                print(f"PCA转换失败: {str(e)}")
                pca_features = None
        
        # 4. 返回处理结果：原始特征和PCA特征
        return raw_features, pca_features
    
    except Exception as e:
        # 捕获整个处理过程中的任何错误
        print(f"处理输入数据时出错: {str(e)}")
        return None, None

def make_prediction(model, features_tuple):
    """
    使用加载的模型进行预测，支持多种特征格式尝试
    
    此函数实现了一种灵活的预测策略，会尝试不同类型的特征进行预测：
    1. 首先尝试使用PCA特征进行预测(如果可用)
    2. 如果PCA特征预测失败，则尝试使用原始特征
    3. 如果两种方法都失败，返回保守的默认预测
    
    这种策略确保了预测过程的鲁棒性，能够适应不同类型的模型和特征。
    
    Args:
        model: 加载的机器学习模型，通常是scikit-learn的分类器
        features_tuple: 特征元组 (原始特征DataFrame, PCA特征DataFrame)
                        由process_feature函数生成
    
    Returns:
        tuple: 包含两个元素:
            - prediction: 预测结果 (1: 批准, 0: 拒绝)，失败时为None
            - probability: 批准的概率 (0.0-1.0)，失败时为None
    
    异常处理:
        - 捕获每种预测方法中可能出现的异常
        - 记录详细的错误信息以便调试
        - 在所有尝试都失败时返回默认的保守预测
    """
    if model is None:
        print("模型为空，无法进行预测")
        return None, None
        
    raw_features, pca_features = features_tuple
    
    if raw_features is None and pca_features is None:
        print("特征为空，无法进行预测")
        return None, None
    
    # 1. 首先尝试使用PCA特征进行预测 (通常更高效)
    if pca_features is not None:
        try:
            print("尝试使用PCA特征进行预测...")
            # 获取概率预测结果
            prediction_proba = model.predict_proba(pca_features)
            # 提取正类(批准)的概率
            approval_proba = prediction_proba[0][1]
            # 根据概率确定最终预测 (>=0.5为批准)
            prediction = 1 if approval_proba >= 0.5 else 0
            print(f"使用PCA特征预测成功，预测结果: {prediction}，概率: {approval_proba:.4f}")
            return prediction, approval_proba
        except Exception as e:
            print(f"使用PCA特征预测失败: {str(e)}")
            # 记录错误后继续尝试其他方法
    
    # 2. 如果PCA特征预测失败，尝试使用原始特征
    if raw_features is not None:
        try:
            print("尝试使用原始特征进行预测...")
            # 获取概率预测结果
            prediction_proba = model.predict_proba(raw_features)
            # 提取正类(批准)的概率
            approval_proba = prediction_proba[0][1]
            # 根据概率确定最终预测 (>=0.5为批准)
            prediction = 1 if approval_proba >= 0.5 else 0
            print(f"使用原始特征预测成功，预测结果: {prediction}，概率: {approval_proba:.4f}")
            return prediction, approval_proba
        except Exception as e:
            print(f"使用原始特征预测失败: {str(e)}")
            # 两种方法都失败，准备返回默认预测
    
    # 3. 如果两种方法都失败，返回默认的保守预测 (应急方案)
    print("所有预测尝试都失败，返回应急预测结果")
    print("注意：这不是基于模型的实际预测，而是系统默认的保守结果")
    print("需要重新训练模型或修复特征处理来获得准确的预测")
    # 作为应急方案，拒绝(0)被认为是更安全的结果
    return 0, 0.0

def explain_prediction(model, features, prediction, probability):
    """
    为预测结果提供解释，帮助用户理解决策依据
    
    此函数生成对模型预测结果的解释，包括影响决策的因素和给申请人的建议。
    在生产系统中，这部分可以扩展为更复杂的逻辑，例如使用SHAP值或
    其他可解释AI技术来提供更精确的特征重要性。
    
    参数:
    model: 训练好的模型，可用于提取特征重要性
    features: 预处理后的特征，可用于计算特征贡献
    prediction: 预测结果（0表示拒绝，1表示批准）
    probability: 预测概率，表示模型对结果的确信度
    
    返回:
    dict: 包含解释信息的字典，有以下字段:
        - decision_factors (list): 影响决策的关键因素列表
        - suggestions (list): 给申请人的建议列表
    
    注意:
    当前实现是简化版本，只提供基于预测结果的一般性解释。
    实际应用中可以扩展为基于特征重要性的详细解释。
    """
    # 创建空的决策因素和建议列表
    decision_factors = []
    suggestions = []
    
    # 根据预测结果提供不同的解释和建议
    if prediction == 1:  # 批准
        decision_factors.append("综合评估符合信用卡批准条件")
        suggestions.append("继续保持良好的信用记录")
    else:  # 拒绝
        decision_factors.append("综合评估未满足信用卡批准条件")
        suggestions.append("请考虑改善信用评分、减少债务或增加收入")
    
    # 返回解释结果
    return {
        "decision_factors": decision_factors,
        "suggestions": suggestions
    }

def preprocess_input(input_data, transformers):
    """
    预处理输入数据的兼容函数，支持多种输入格式
    
    此函数是process_feature的包装器，提供了更统一的接口，
    支持字典和DataFrame两种输入格式，并进行适当的转换。
    
    参数:
    input_data (dict or DataFrame): 输入数据，可以是字典或DataFrame
        - 如果是字典：直接转换为单行DataFrame
        - 如果是DataFrame：取第一行作为处理对象
    transformers (dict): 特征转换器，包含预处理所需的所有组件
    
    返回:
    tuple: 处理后的特征元组(原始特征, PCA特征)
        - 由process_feature函数处理并返回
    
    用途:
    - 提供一个统一接口，简化API调用
    - 处理不同格式的输入数据
    - 向后兼容旧版API
    """
    # 检查输入类型并进行适当的转换
    if isinstance(input_data, dict):
        # 如果输入是字典，转换为单行DataFrame
        input_data = pd.DataFrame([input_data])
    
    # 转发到process_feature函数处理
    # 将第一行转换为字典格式，这是process_feature所需的格式
    return process_feature(input_data.iloc[0].to_dict(), transformers)

def predict(model, features_tuple):
    """
    使用模型进行预测的兼容函数
    
    此函数是make_prediction的简化包装器，
    提供了向后兼容性和更简洁的接口。
    
    参数:
    model: 训练好的机器学习模型
    features_tuple: 特征元组(原始特征, PCA特征)
        - 由preprocess_input或process_feature函数返回
    
    返回:
    tuple: (预测结果, 预测概率)
        - 预测结果: 1表示批准，0表示拒绝
        - 预测概率: 表示批准的概率(0.0-1.0)
    
    用途:
    - 向后兼容旧版API
    - 提供简化的预测接口
    """
    # 直接调用已有的make_prediction函数
    return make_prediction(model, features_tuple)

def batch_process_applications(applications_df, model, transformers, threshold=0.5):
    """
    批量处理信用卡申请，用于处理多个申请的场景
    
    此函数用于处理包含多个申请记录的数据集，适用于批量预测场景。
    对每个申请进行单独处理，并生成汇总统计信息。
    
    参数:
    applications_df (DataFrame): 包含多个申请的DataFrame，每行一个申请
    model: 训练好的模型，用于预测每个申请的结果
    transformers: 特征转换器，用于预处理每个申请的特征
    threshold (float): 决策阈值，默认为0.5，用于概率转换为决策
    
    返回:
    tuple: (结果DataFrame, 汇总信息)
        - 结果DataFrame包含每个申请的预测结果、概率和解释
        - 汇总信息包含总申请数、批准数、拒绝数和批准率等统计数据
    
    处理流程:
    1. 遍历每个申请记录
    2. 对每个申请分别进行特征处理和预测
    3. 为每个预测生成解释
    4. 汇总所有结果并计算统计信息
    """
    # 初始化结果列表，用于存储每个申请的处理结果
    results = []
    
    # 逐行处理每个申请
    for _, row in applications_df.iterrows():
        # 将行数据转换为字典格式
        application_data = row.to_dict()
        
        # 处理特征并预测结果
        features_tuple = process_feature(application_data, transformers)
        prediction, probability = make_prediction(model, features_tuple)
        
        # 创建当前申请的结果字典
        result = {
            "prediction": int(prediction) if prediction is not None else 0,
            "probability": float(probability) if probability is not None else 0.0,
            "approved": bool(prediction == 1) if prediction is not None else False
        }
        
        # 添加预测解释
        explanation = explain_prediction(model, features_tuple, prediction, probability)
        result.update(explanation)
        
        # 将结果添加到结果列表
        results.append(result)
    
    # 创建结果DataFrame，便于后续分析和展示
    results_df = pd.DataFrame(results)
    
    # 创建汇总统计信息
    summary = {
        "total_applications": len(applications_df),  # 总申请数
        "approved": int(results_df["approved"].sum()),  # 批准数
        "rejected": int((~results_df["approved"]).sum()),  # 拒绝数
        "approval_rate": float(results_df["approved"].mean())  # 批准率
    }
    
    return results_df, summary

def process_single_application(application_df, model, transformers, threshold=0.5):
    """
    处理单个信用卡申请，提供完整的处理流程
    
    此函数整合了整个处理流程，包括特征处理、预测和结果解释，
    专门用于处理单个申请的场景，返回结构化的结果。
    
    参数:
    application_df (DataFrame): 包含单个申请的DataFrame，必须只有一行
    model: 训练好的机器学习模型
    transformers: 特征转换器，包含预处理所需的所有组件
    threshold (float): 决策阈值，默认为0.5
        - 如果预测概率 >= threshold，则批准申请
        - 如果预测概率 < threshold，则拒绝申请
    
    返回:
    dict: 包含以下字段的结果字典:
        - success (bool): 处理是否成功
        - prediction (int): 预测结果(1=批准, 0=拒绝)，仅当success=True
        - probability (float): 批准概率，仅当success=True
        - explanation (dict): 包含decision_factors和suggestions的解释，仅当success=True
        - error (str): 错误信息，仅当success=False
    
    处理流程:
    1. 验证输入数据格式(必须是单行DataFrame)
    2. 预处理申请数据提取特征
    3. 使用模型进行预测
    4. 根据阈值确定最终决策
    5. 生成解释和建议
    6. 返回结构化结果
    
    异常处理:
    捕获所有可能的异常，返回带有错误信息的失败结果
    """
    try:
        # 1. 验证输入是否只有一行
        if application_df.shape[0] != 1:
            return {
                'success': False,
                'error': 'Expected a single application'
            }
        
        # 2. 预处理数据，提取特征
        application_data = application_df.iloc[0].to_dict()
        features_tuple = process_feature(application_data, transformers)
        
        # 3. 使用模型进行预测
        prediction, probability = make_prediction(model, features_tuple)
        
        # 4. 检查预测是否成功
        if prediction is None:
            return {
                'success': False,
                'error': 'Failed to make prediction'
            }
        
        # 5. 使用阈值确定最终决策
        # 如果概率高于阈值则批准，否则拒绝
        final_prediction = 1 if probability >= threshold else 0
        
        # 6. 获取预测解释和建议
        explanation = explain_prediction(model, features_tuple, final_prediction, probability)
        
        # 7. 返回成功结果
        return {
            'success': True,               # 处理成功标志
            'prediction': final_prediction, # 最终预测结果
            'probability': float(probability), # 批准概率
            'explanation': explanation      # 解释和建议
        }
        
    except Exception as e:
        # 8. 处理过程中的任何异常，返回失败结果
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """
    主函数：演示如何使用此脚本进行信用卡申请预测
    
    此函数展示了完整的模型推理流程，包括：
    1. 准备示例输入数据
    2. 加载模型和特征转换器
    3. 处理特征并进行预测
    4. 解析预测结果并提供解释
    5. 展示决策因素和建议
    
    用途:
    - 作为整个系统的功能测试
    - 展示API的正确使用方法
    - 为开发人员提供使用示例
    
    运行方式:
    $ python model_inference.py
    """
    # 示例数据 - 使用正确的特征名称 (A1-A15)，基于原始数据集的特征格式
    sample_data = {
        'A1': 1,  # 性别 (0=女, 1=男)
        'A2': 35.0,  # 年龄
        'A3': 2.5,  # 债务/收入比例
        'A4': 0,  # 婚姻状况 (0=未婚, 1=已婚)
        'A5': 6,  # 银行客户状态
        'A6': 22,  # 教育水平/职业
        'A7': 21,  # 种族
        'A8': 4.5,  # 就业年限
        'A9': 1,  # 是否有违约记录 (1=是, 0=否)
        'A10': 0,  # 是否有正式工作 (1=是, 0=否)
        'A11': 5,  # 信用评分相关
        'A12': 0,  # 是否有驾照 (1=是, 0=否)
        'A13': 6,  # 公民身份
        'A14': 200.0,  # 邮政编码/特定地区特征
        'A15': 45000  # 收入
    }
    
    # 1. 加载模型和特征转换器
    print("正在加载模型和特征转换器...")
    model = load_model()
    feature_transformers = load_feature_transformers()
    
    # 检查模型和转换器是否成功加载
    if model is None or feature_transformers is None:
        print("无法加载模型或特征转换器，无法继续。")
        return
    
    try:
        # 2. 调试信息：显示模型期望的特征列表
        if 'selected_features' in feature_transformers:
            print("模型预期的特征:")
            print(feature_transformers['selected_features'])
        
        # 3. 特征处理：将原始数据转换为模型所需的特征格式
        print("正在处理特征数据...")
        features_tuple = process_feature(sample_data, feature_transformers)
        
        # 4. 模型预测：使用处理后的特征进行预测
        print("正在进行预测...")
        prediction, probability = make_prediction(model, features_tuple)
        
        # 5. 输出预测结果
        if prediction is not None:
            result = "批准" if prediction == 1 else "拒绝"
            print(f"\n信用卡申请预测结果: {result}")
            print(f"批准概率: {probability:.2%}")
            
            # 6. 生成决策因素和建议
            decision_factors = []
            suggestions = []
            
            # 6.1 基于输入数据和预测结果生成详细解释
            # 年龄因素
            if sample_data['A2'] < 25.0:
                decision_factors.append("申请人年龄较低")
                suggestions.append("建立更长的信用记录可能有助于提高批准率")
            
            # 信用评分因素
            if sample_data['A11'] < 5:  # 较低的信用评分相关指标
                decision_factors.append("信用评分相关指标低于平均水平")
                suggestions.append("提高信用评分：按时支付账单并减少现有债务")
            elif sample_data['A11'] > 10:
                decision_factors.append("优秀的信用评分相关指标")
            
            # 违约记录因素
            if sample_data['A9'] == 1:  # 1 表示有违约记录
                decision_factors.append("有过违约记录")
                suggestions.append("改善支付历史记录：确保所有账单按时支付")
            
            # 债务收入比因素
            if sample_data['A3'] > 5.0:  # 较高的债务收入比
                decision_factors.append(f"债务收入比较高: {sample_data['A3']:.2f}")
                suggestions.append("减少现有债务以改善债务收入比")
            
            # 就业稳定性因素
            if sample_data['A8'] < 2.0:  # 就业年限短
                decision_factors.append("就业历史较短")
                suggestions.append("就业稳定性是重要因素，保持当前工作可能有助于未来申请")
                
            # 6.2 如果没有足够数据生成决策因素，提供一般性解释
            if not decision_factors:
                if prediction == 1:
                    decision_factors.append("综合评估符合信用卡批准条件")
                else:
                    decision_factors.append("综合评估未满足信用卡批准条件")
                    suggestions.append("请考虑改善信用评分、减少债务或增加收入")
            
            # 6.3 如果没有足够数据生成建议，提供一般性建议
            if not suggestions and prediction == 0:
                suggestions.append("请联系客服获取更多关于您的申请被拒绝的信息")
            elif not suggestions:
                suggestions.append("继续保持良好的信用记录")
                
            # 7. 输出决策因素和建议
            print("\n决策因素:")
            for i, factor in enumerate(decision_factors, 1):
                print(f"{i}. {factor}")
                
            print("\n建议:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        else:
            print("无法完成预测")
            
    except Exception as e:
        print(f"运行出错: {str(e)}")

if __name__ == "__main__":
    main() 