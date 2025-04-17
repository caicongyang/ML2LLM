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

## 项目结构
```
credit-card-approval/
├── data/                          # 数据目录
│   ├── credit_card_applications.csv  # 原始数据
│   ├── processed_data.csv         # 预处理后的数据
│   ├── engineered_data.csv        # 特征工程后的数据
│   ├── feature_transformers.pkl   # 特征转换器
│   ├── best_model_random_forest.pkl  # 训练好的模型
│   └── best_model_metadata.json   # 模型元数据
├── data_collection.py              # 数据收集模块
├── data_preprocessing.py           # 数据预处理模块
├── feature_engineering.py          # 特征工程模块
├── model_training.py               # 模型训练模块
├── model_inference.py              # 模型推理模块
├── api_service.py                  # API服务模块
├── tests/                          # 测试目录
│   └── test_api_service.py         # API服务测试
└── requirements.txt                # 依赖包列表
```

## 安装与启动
1. 安装依赖包：
```
pip install -r requirements.txt
```

2. 训练模型（如果需要）：
```
python data_collection.py
python data_preprocessing.py
python feature_engineering.py
python model_training.py
```

3. 启动API服务：
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
  "application": {
    "feature1": 0.5,
    "feature2": 0.7,
    "feature3": -0.2
  },
  "threshold": 0.5
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

## 开发流程
1. 数据收集：从UCI机器学习存储库获取信用卡审批数据集
2. 数据预处理：处理缺失值、异常值、编码分类特征等
3. 特征工程：特征选择、创建新特征、应用PCA降维等
4. 模型训练：训练和评估多种机器学习模型
5. 模型推理：使用训练好的模型进行预测
6. API服务：提供RESTful API接口供外部调用

## 许可证
MIT 