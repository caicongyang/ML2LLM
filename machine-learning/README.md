# 信用卡审批系统

## 简介
本系统是一个基于机器学习的信用卡审批预测系统，通过分析申请人的个人信息和财务状况，预测信用卡申请的批准可能性。系统提供了RESTful API接口，方便集成到现有的业务系统中。

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

## 安装与启动
1. 安装依赖包：
```
pip install -r requirements.txt
```

2. 启动API服务：
```
python api_service.py
```

或使用Gunicorn（生产环境推荐）：
```
gunicorn -w 4 -b 0.0.0.0:5000 api_service:app
```

## API接口说明

### 健康检查
- 端点：`/health`
- 方法：GET
- 描述：检查API服务和模型是否正常运行

### 模型重载
- 端点：`/reload`
- 方法：POST
- 描述：重新加载机器学习模型和转换器

### 单条预测
- 端点：`/predict`
- 方法：POST
- 描述：对单个申请进行预测
- 请求体示例：
```json
{
  "age": 35,
  "income": 80000,
  "employment_years": 5,
  "credit_score": 720,
  "loan_amount": 15000,
  "debt_to_income_ratio": 0.3
}
```

### 批量预测
- 端点：`/batch-predict`
- 方法：POST
- 描述：对多个申请进行批量预测
- 请求体示例：包含多个申请对象的数组

### 模型信息
- 端点：`/model-info`
- 方法：GET
- 描述：获取当前加载的模型信息和特征重要性

## 测试
运行单元测试：
```
pytest test_api_service.py
```

## 许可证
MIT

## 联系方式
如有问题请联系：example@company.com 