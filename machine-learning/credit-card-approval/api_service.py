#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信用卡审批系统API服务
提供RESTful API接口，用于模型推理和批量处理
"""

from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
import json
import joblib
import traceback
from datetime import datetime
import logging

# 导入模型推理模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__***REMOVED***le__))))
from model_inference import (
    load_model, 
    load_feature_transformers, 
    preprocess_input, 
    predict, 
    explain_prediction,
    process_single_application,
    batch_process_applications
)

# 配置日志
logging.basicCon***REMOVED***g(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('api_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)

# 全局变量，用于缓存模型和转换器
MODEL = None
TRANSFORMERS = None
MODEL_METADATA = None
MODEL_LOAD_TIME = None

def initialize_models():
    """初始化并加载模型和转换器"""
    global MODEL, TRANSFORMERS, MODEL_METADATA, MODEL_LOAD_TIME
    
    try:
        # 加载模型和转换器
        MODEL = load_model()
        TRANSFORMERS = load_feature_transformers()
        
        # 加载模型元数据
        metadata_path = 'data/best_model_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                MODEL_METADATA = metadata[0] if isinstance(metadata, list) else metadata
        
        MODEL_LOAD_TIME = datetime.now()
        
        logger.info(f"模型和转换器加载成功，时间: {MODEL_LOAD_TIME}")
        return True
    except Exception as e:
        logger.error(f"初始化模型时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.before_***REMOVED***rst_request
def before_***REMOVED***rst_request():
    """首次请求前初始化"""
    initialize_models()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    if MODEL is not None and TRANSFORMERS is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_type': MODEL_METADATA.get('model_type', 'Unknown') if MODEL_METADATA else 'Unknown',
            'model_loaded_at': MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model or transformers not loaded'
        }), 503

@app.route('/reload', methods=['POST'])
def reload_models():
    """重新加载模型和转换器"""
    success = initialize_models()
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Models reloaded successfully',
            'model_type': MODEL_METADATA.get('model_type', 'Unknown') if MODEL_METADATA else 'Unknown',
            'model_loaded_at': MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to reload models'
        }), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    单个预测接口
    
    请求体示例:
    {
        "application": {
            "feature1": 0.5,
            "feature2": 0.7,
            "feature3": -0.2,
            ...
        },
        "threshold": 0.5  // 可选
    }
    """
    if MODEL is None or TRANSFORMERS is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'application' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing application data'
            }), 400
        
        # 获取申请数据和阈值
        application_data = data['application']
        threshold = data.get('threshold', 0.5)
        
        # 转换为DataFrame
        application_df = pd.DataFrame([application_data])
        
        # 处理申请
        result = process_single_application(
            application_df,
            model=MODEL,
            transformers=TRANSFORMERS,
            threshold=threshold
        )
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'result': {
                    'prediction': 'approved' if result['prediction'] == 1 else 'rejected',
                    'probability': result['probability'],
                    'explanation': result['explanation']
                }
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Unknown error during prediction')
            }), 500
            
    except Exception as e:
        logger.error(f"预测接口出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict_endpoint():
    """
    批量预测接口
    
    请求体示例:
    {
        "applications": [
            {
                "application_id": "123",
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
        "threshold": 0.5  // 可选
    }
    """
    if MODEL is None or TRANSFORMERS is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'applications' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing applications data'
            }), 400
        
        # 获取申请数据和阈值
        applications = data['applications']
        threshold = data.get('threshold', 0.5)
        
        if not applications:
            return jsonify({
                'status': 'error',
                'message': 'Empty applications list'
            }), 400
        
        # 转换为DataFrame
        applications_df = pd.DataFrame(applications)
        
        # 记录应用ID（如果存在）
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
        
        if results_df is not None:
            # 如果存在application_id，添加回结果
            if has_app_id:
                results_df['application_id'] = app_ids
            
            # 将DataFrame转换为记录列表
            results_list = results_df.to_dict('records')
            
            return jsonify({
                'status': 'success',
                'results': results_list,
                'summary': summary
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error during batch processing'
            }), 500
            
    except Exception as e:
        logger.error(f"批量预测接口出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info_endpoint():
    """获取模型信息接口"""
    if MODEL is None or MODEL_METADATA is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized or metadata not available'
        }), 503
    
    try:
        # 获取模型类型信息
        model_type = type(MODEL).__name__
        
        # 获取特征重要性（如果可用）
        feature_importance = None
        if hasattr(MODEL, 'feature_importances_'):
            if TRANSFORMERS and 'selected_features' in TRANSFORMERS:
                features = TRANSFORMERS['selected_features']
                importances = MODEL.feature_importances_
                if len(features) == len(importances):
                    feature_importance = dict(zip(features, importances.tolist()))
                    # 按重要性排序
                    feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                                 key=lambda item: item[1], 
                                                                 reverse=True)}
        
        # 构建模型信息响应
        info = {
            'model_type': model_type,
            'loaded_at': MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None,
            'metadata': MODEL_METADATA,
            'feature_importance': feature_importance
        }
        
        return jsonify({
            'status': 'success',
            'model_info': info
        }), 200
            
    except Exception as e:
        logger.error(f"获取模型信息接口出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def start_server(host='0.0.0.0', port=5000, debug=False):
    """启动Flask服务器"""
    logger.info(f"启动API服务，host={host}, port={port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # 初始化模型和转换器
    initialize_models()
    
    # 启动服务
    start_server(debug=True) 