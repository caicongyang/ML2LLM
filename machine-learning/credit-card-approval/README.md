# 信用卡审批系统

## 简介
本系统是一个基于机器学习的信用卡审批预测系统，通过分析申请人的个人信息和财务状况，预测信用卡申请的批准可能性。系统提供了RESTful API接口，方便集成到现有的业务系统中。

更详细的项目文档请参考 [credit-card-approval.md](./credit-card-approval.md)。

## 功能特点
- 提供单条和批量预测接口
- 支持模型热重载
- 提供模型信息和特征重要性查询
- 完善的健康检查和错误处理机制
- 详细的日志记录

## 技术栈
- Python 3.9+
- Flask (Web框架)
- Scikit-learn (机器学习库)
- Pandas & Numpy (数据处理)
- Joblib (模型序列化)
- SHAP (模型解释)

## 快速开始

### 安装依赖
```
pip install -r requirements.txt
```

### 模型训练流程
如需重新训练模型，请按顺序执行以下脚本：
```
python data_collection.py
python data_preprocessing.py
python feature_engineering.py
python model_training.py
```

### 启动API服务
开发环境：
```
python api_service.py
```

生产环境（使用Gunicorn）：
```
gunicorn -w 4 -b 0.0.0.0:5000 api_service:app
```

## API接口参考

### 健康检查
- **端点**：`/health`
- **方法**：GET
- **返回示例**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForest",
  "model_loaded_at": "2023-06-01T12:00:00"
}
```

### 模型重载
- **端点**：`/reload`
- **方法**：POST
- **返回示例**：
```json
{
  "status": "success",
  "message": "Models reloaded successfully",
  "model_type": "RandomForest",
  "model_loaded_at": "2023-06-01T12:30:00"
}
```

### 单条预测
- **端点**：`/predict`
- **方法**：POST
- **请求体示例**：
```json
{
  "application": {
    "feature1": 0.5,
    "feature2": 0.7,
    "feature3": -0.2
  },
  "threshold": 0.5
}
```
- **返回示例**：
```json
{
  "status": "success",
  "result": {
    "prediction": "approved",
    "probability": 0.78,
    "explanation": {
      "important_features": {
        "feature2": 0.45,
        "feature1": 0.30,
        "feature3": 0.15
      }
    }
  }
}
```

### 批量预测
- **端点**：`/batch-predict`
- **方法**：POST
- **请求体示例**：
```json
{
  "applications": [
    {
      "application_id": "123",
      "feature1": 0.5,
      "feature2": 0.7
    },
    {
      "application_id": "124",
      "feature1": -0.2,
      "feature2": 1.3
    }
  ],
  "threshold": 0.5
}
```
- **返回示例**：
```json
{
  "status": "success",
  "results": [
    {
      "application_id": "123",
      "prediction": "approved",
      "probability": 0.78
    },
    {
      "application_id": "124",
      "prediction": "rejected",
      "probability": 0.35
    }
  ],
  "summary": {
    "total": 2,
    "approved": 1,
    "rejected": 1
  }
}
```

### 模型信息
- **端点**：`/model-info`
- **方法**：GET
- **返回示例**：
```json
{
  "model_type": "RandomForest",
  "training_date": "2023-05-15",
  "performance": {
    "accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.83,
    "f1": 0.84
  },
  "feature_importance": {
    "feature2": 0.45,
    "feature1": 0.30,
    "feature3": 0.15
  }
}
```

## 测试
运行API服务测试：
```
python -m pytest tests/test_api_service.py
```

## 许可证
MIT 