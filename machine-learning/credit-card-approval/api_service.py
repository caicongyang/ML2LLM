#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信用卡审批系统API服务
提供RESTful API接口，用于模型推理和批量处理

本模块实现了一个基于Flask的Web服务，提供以下功能：
1. 加载和缓存预训练的机器学习模型
2. 提供健康检查接口，用于监控服务状态
3. 提供单个信用卡申请的预测接口
4. 提供批量信用卡申请的预测接口
5. 提供模型信息查询接口
6. 提供模型重新加载接口
"""

# 导入所需库
from flask import Flask, request, jsonify  # Flask Web框架核心组件
import pandas as pd                        # 数据处理库
import os                                  # 操作系统接口
import sys                                 # 系统特定参数和函数
import json                                # JSON数据处理
import joblib                              # 模型序列化/反序列化
import traceback                           # 异常追踪
from datetime import datetime              # 日期时间处理
import logging                             # 日志记录

# 导入自定义模型推理模块
# 首先添加父目录到系统路径，确保能找到模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_inference import (
    load_model,                  # 加载模型函数
    load_feature_transformers,   # 加载特征转换器函数
    preprocess_input,            # 输入预处理函数
    predict,                     # 预测函数
    explain_prediction,          # 预测解释函数
    process_single_application,  # 单个申请处理函数
    batch_process_applications   # 批量申请处理函数
)

# 配置日志记录
# 设置日志级别为INFO，同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,                        # 设置日志级别为INFO
    format='%(asctime)s [%(levelname)s] %(message)s',  # 日志格式：时间 [级别] 消息
    handlers=[
        logging.FileHandler('api_service.log'),  # 日志文件处理器
        logging.StreamHandler()                # 控制台处理器
    ]
)
logger = logging.getLogger(__name__)  # 获取当前模块的logger

# 初始化Flask应用
app = Flask(__name__)

# 全局变量，用于缓存模型和转换器
# 在服务启动时加载一次，避免每次请求都重新加载，提高性能
MODEL = None                # 机器学习模型
TRANSFORMERS = None         # 特征转换器
MODEL_METADATA = None       # 模型元数据
MODEL_LOAD_TIME = None      # 模型加载时间

def initialize_models():
    """
    初始化并加载模型和转换器
    
    此函数负责:
    1. 加载预训练的机器学习模型
    2. 加载特征转换器
    3. 加载模型元数据
    4. 记录模型加载时间
    
    返回:
    bool: 初始化成功返回True，失败返回False
    """
    global MODEL, TRANSFORMERS, MODEL_METADATA, MODEL_LOAD_TIME
    
    try:
        # 加载模型和转换器
        MODEL = load_model()  # 从磁盘加载序列化的模型
        TRANSFORMERS = load_feature_transformers()  # 加载特征转换器
        
        # 加载模型元数据（包含模型类型、训练时间、性能指标等信息）
        metadata_path = 'data/best_model_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # 如果元数据是列表形式，取第一个元素
                MODEL_METADATA = metadata[0] if isinstance(metadata, list) else metadata
        
        # 记录模型加载时间
        MODEL_LOAD_TIME = datetime.now()
        
        logger.info(f"模型和转换器加载成功，时间: {MODEL_LOAD_TIME}")
        return True
    except Exception as e:
        # 捕获并记录所有可能的异常
        logger.error(f"初始化模型时出错: {str(e)}")
        logger.error(traceback.format_exc())  # 记录完整的异常堆栈
        return False

# 在应用启动时初始化模型
# 使用应用上下文确保在Flask应用完全初始化后进行
with app.app_context():
    initialize_models()

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    
    用于监控系统是否正常运行，检查模型是否已正确加载。
    常用于Kubernetes等容器编排系统的健康检查。
    
    HTTP方法: GET
    
    返回:
    JSON: 包含服务状态信息的JSON响应
    """
    if MODEL is not None and TRANSFORMERS is not None:
        # 服务健康，返回正常状态
        return jsonify({
            'status': 'healthy',  # 服务状态
            'model_loaded': True,  # 模型已加载
            'model_type': MODEL_METADATA.get('model_type', 'Unknown') if MODEL_METADATA else 'Unknown',  # 模型类型
            'model_loaded_at': MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None  # 模型加载时间
        }), 200  # HTTP 200 OK
    else:
        # 服务不健康，可能是模型未加载
        return jsonify({
            'status': 'unhealthy',  # 服务状态
            'model_loaded': False,  # 模型未加载
            'error': 'Model or transformers not loaded'  # 错误信息
        }), 503  # HTTP 503 Service Unavailable

@app.route('/reload', methods=['POST'])
def reload_models():
    """
    重新加载模型和转换器
    
    当模型文件更新后，可通过此接口重新加载模型，无需重启服务。
    
    HTTP方法: POST
    
    返回:
    JSON: 包含重新加载结果的JSON响应
    """
    success = initialize_models()  # 调用初始化函数
    if success:
        # 重新加载成功
        return jsonify({
            'status': 'success',
            'message': 'Models reloaded successfully',
            'model_type': MODEL_METADATA.get('model_type', 'Unknown') if MODEL_METADATA else 'Unknown',
            'model_loaded_at': MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None
        }), 200  # HTTP 200 OK
    else:
        # 重新加载失败
        return jsonify({
            'status': 'error',
            'message': 'Failed to reload models'
        }), 500  # HTTP 500 Internal Server Error

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    单个预测接口
    
    处理单个信用卡申请的预测请求，返回预测结果、概率和解释。
    
    HTTP方法: POST
    
    请求体示例:
    {
        "application": {
            "feature1": 0.5,
            "feature2": 0.7,
            "feature3": -0.2,
            ...
        },
        "threshold": 0.5  // 可选，决策阈值
    }
    
    返回:
    JSON: 包含预测结果的JSON响应
    """
    # 检查模型是否已初始化
    if MODEL is None or TRANSFORMERS is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized'
        }), 503  # HTTP 503 Service Unavailable
    
    try:
        # 解析请求JSON数据
        data = request.get_json()
        
        # 验证请求数据格式
        if not data or 'application' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing application data'
            }), 400  # HTTP 400 Bad Request
        
        # 获取申请数据和阈值
        application_data = data['application']  # 申请数据
        threshold = data.get('threshold', 0.5)  # 决策阈值，默认0.5
        
        # 将申请数据转换为DataFrame（机器学习模型需要DataFrame格式）
        application_df = pd.DataFrame([application_data])
        
        # 处理申请：预处理、预测和解释
        result = process_single_application(
            application_df,
            model=MODEL,
            transformers=TRANSFORMERS,
            threshold=threshold
        )
        
        # 检查处理结果
        if result['success']:
            # 成功处理，返回预测结果
            return jsonify({
                'status': 'success',
                'result': {
                    'prediction': 'approved' if result['prediction'] == 1 else 'rejected',  # 预测结果
                    'probability': result['probability'],  # 批准的概率
                    'explanation': result['explanation']  # 预测结果的解释
                }
            }), 200  # HTTP 200 OK
        else:
            # 处理失败，返回错误信息
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Unknown error during prediction')
            }), 500  # HTTP 500 Internal Server Error
            
    except Exception as e:
        # 捕获并记录所有异常
        logger.error(f"预测接口出错: {str(e)}")
        logger.error(traceback.format_exc())  # 记录完整的异常堆栈
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500  # HTTP 500 Internal Server Error

@app.route('/batch-predict', methods=['POST'])
def batch_predict_endpoint():
    """
    批量预测接口
    
    同时处理多个信用卡申请，提高处理效率。
    适用于需要批量处理申请的场景。
    
    HTTP方法: POST
    
    请求体示例:
    {
        "applications": [
            {
                "application_id": "123",  // 可选，申请ID
                "feature1": 0.5,
                "feature2": 0.7,
                ...
            },
            {
                "application_id": "124",
                "feature1": -0.2,
                "feature2": 1.3,
                ...
            }
        ],
        "threshold": 0.5  // 可选，决策阈值
    }
    
    返回:
    JSON: 包含批量预测结果和汇总信息的JSON响应
    """
    # 检查模型是否已初始化
    if MODEL is None or TRANSFORMERS is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized'
        }), 503  # HTTP 503 Service Unavailable
    
    try:
        # 解析请求JSON数据
        data = request.get_json()
        
        # 验证请求数据格式
        if not data or 'applications' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing applications data'
            }), 400  # HTTP 400 Bad Request
        
        # 获取申请数据列表和阈值
        applications = data['applications']  # 申请数据列表
        threshold = data.get('threshold', 0.5)  # 决策阈值，默认0.5
        
        # 检查申请列表是否为空
        if not applications:
            return jsonify({
                'status': 'error',
                'message': 'Empty applications list'
            }), 400  # HTTP 400 Bad Request
        
        # 将申请数据列表转换为DataFrame
        applications_df = pd.DataFrame(applications)
        
        # 处理application_id（如果存在）
        # 预测时不需要application_id，但需要在结果中保留它
        has_app_id = 'application_id' in applications_df.columns
        app_ids = applications_df['application_id'].tolist() if has_app_id else None
        
        # 如果存在application_id列，在预测前暂时删除
        if has_app_id:
            applications_df_for_prediction = applications_df.drop('application_id', axis=1)
        else:
            applications_df_for_prediction = applications_df
        
        # 批量处理申请
        results_df, summary = batch_process_applications(
            applications_df_for_prediction,
            model=MODEL,
            transformers=TRANSFORMERS,
            threshold=threshold
        )
        
        # 检查处理结果
        if results_df is not None:
            # 如果存在application_id，添加回结果
            if has_app_id:
                results_df['application_id'] = app_ids
            
            # 将DataFrame转换为记录列表（更适合JSON序列化）
            results_list = results_df.to_dict('records')
            
            # 返回批量处理结果
            return jsonify({
                'status': 'success',
                'results': results_list,  # 每个申请的详细结果
                'summary': summary  # 批量处理的汇总信息（如批准率）
            }), 200  # HTTP 200 OK
        else:
            # 处理失败
            return jsonify({
                'status': 'error',
                'message': 'Error during batch processing'
            }), 500  # HTTP 500 Internal Server Error
            
    except Exception as e:
        # 捕获并记录所有异常
        logger.error(f"批量预测接口出错: {str(e)}")
        logger.error(traceback.format_exc())  # 记录完整的异常堆栈
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500  # HTTP 500 Internal Server Error

@app.route('/model-info', methods=['GET'])
def model_info_endpoint():
    """
    获取模型信息接口
    
    提供关于当前加载模型的详细信息，包括模型类型、特征重要性等。
    用于模型透明度和调试目的。
    
    HTTP方法: GET
    
    返回:
    JSON: 包含模型详细信息的JSON响应
    """
    # 检查模型和元数据是否已加载
    if MODEL is None or MODEL_METADATA is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized or metadata not available'
        }), 503  # HTTP 503 Service Unavailable
    
    try:
        # 获取模型类型信息（从模型类名）
        model_type = type(MODEL).__name__
        
        # 获取特征重要性（如果模型支持）
        feature_importance = None
        if hasattr(MODEL, 'feature_importances_'):  # 检查模型是否有特征重要性属性
            if TRANSFORMERS and 'selected_features' in TRANSFORMERS:
                features = TRANSFORMERS['selected_features']  # 获取特征名称
                importances = MODEL.feature_importances_  # 获取特征重要性值
                # 确保特征名称和重要性值长度一致
                if len(features) == len(importances):
                    # 创建特征名称到重要性的映射字典
                    feature_importance = dict(zip(features, importances.tolist()))
                    # 按重要性降序排序
                    feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                                 key=lambda item: item[1], 
                                                                 reverse=True)}
        
        # 构建完整的模型信息响应
        info = {
            'model_type': model_type,  # 模型类型
            'loaded_at': MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None,  # 模型加载时间
            'metadata': MODEL_METADATA,  # 模型元数据
            'feature_importance': feature_importance  # 特征重要性
        }
        
        # 返回模型信息
        return jsonify({
            'status': 'success',
            'model_info': info
        }), 200  # HTTP 200 OK
            
    except Exception as e:
        # 捕获并记录所有异常
        logger.error(f"获取模型信息接口出错: {str(e)}")
        logger.error(traceback.format_exc())  # 记录完整的异常堆栈
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500  # HTTP 500 Internal Server Error

def start_server(host='0.0.0.0', port=5001, debug=False):
    """
    启动Flask服务器
    
    配置并运行Flask Web服务器，处理API请求。
    
    参数:
    host (str): 服务器监听的主机地址，默认'0.0.0.0'表示监听所有网络接口
    port (int): 服务器监听的端口，默认5001
    debug (bool): 是否启用调试模式，默认False
    """
    logger.info(f"启动API服务，host={host}, port={port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # 当脚本直接运行时，启动服务器
    # debug=True表示开启调试模式，适用于开发环境，生产环境应设为False
    start_server(debug=True) 