# ML2LLM - 从机器学习到大型语言模型

ML2LLM（Machine Learning to Large Language Models）是一个面向Java工程师的学习项目，旨在帮助Java开发者快速理解并转型到大模型开发领域。本项目包含从传统机器学习到大型语言模型的完整发展历程，提供了丰富的教程和示例代码。

## 项目结构

### 机器学习部分 (machine-learning/)

此目录包含传统机器学习相关的教程和示例：

- **introduction.md**: 机器学习基础概念介绍
- **classi***REMOVED***cation.md**: 分类算法详解
- **regression.md**: 回归算法详解
- **clustering-algorithms.md**: 聚类算法详解
- **feature-extraction.md**: 特征提取技术
- **customer-segmentation.md**: 客户细分分析
- **reinforcement-learning.md**: 强化学习教程

#### 实例项目: 信用卡审批系统 (machine-learning/credit-card-approval/)

这是一个完整的机器学习应用示例，展示了如何构建信用卡审批预测系统：

- **data_collection.py**: 数据收集模块
- **data_preprocessing.py**: 数据预处理模块
- **feature_engineering.py**: 特征工程实现
- **model_training.py**: 模型训练逻辑
- **model_inference.py**: 模型推理实现
- **api_service.py**: API服务封装
- **README.md**: 项目说明文档
- **requirements.txt**: 项目依赖包

### 深度学习部分 (deep-learning/)

此目录包含深度学习相关的教程和示例：

- **introduction.md**: 深度学习基础介绍
- **neural-networks.md** 和 **neural-networks-simpli***REMOVED***ed.md**: 神经网络详解和简化版
- **cnn.md** 和 **cnn-simpli***REMOVED***ed.md**: 卷积神经网络详解和简化版
- **rnn.md** 和 **rnn-simpli***REMOVED***ed.md**: 循环神经网络详解和简化版
- **autoencoder.md** 和 **autoencoder_cn.md**: 自编码器详解（英文和中文版）
- **neural-network-training.md**: 神经网络训练指南
- **neural-network-evaluation.md**: 神经网络评估方法
- **gan.md**: 生成对抗网络教程

#### 实例项目: 产品推荐系统 (deep-learning/product-recommendation-system/)

基于深度学习的产品推荐系统实现：

- **recommend.py**: 推荐算法实现
- **evaluate_model.py**: 模型评估脚本
- **con***REMOVED***g.py**: 配置文件
- **training/**: 训练相关代码
- **models/**: 模型定义
- **utils/**: 工具函数
- **data/**: 示例数据
- **README.md**: 项目说明
- **TUTORIAL.md**: 使用教程
- **requirements.txt**: 项目依赖

#### 实例项目: 自动视觉检测系统 (deep-learning/auto-vision/)

基于深度学习的图像异常检测系统：

- **anomaly_detection.py**: 异常检测核心代码
- **extract_features.py**: 特征提取实现
- **run_autoencoder.py**: 自编码器运行脚本
- **src/**: 源代码目录
- **README.md**: 项目说明
- **requirements.txt**: 项目依赖

### 大型语言模型部分 (LLM/)

该部分将包含LLM相关的教程和示例（尚待开发）。

## 开发旅程

项目的 `journey.md` 是整个项目的出发领航，方便java开发者快速理解并转型到大模型开发领域

## 使用指南

每个示例项目都包含独立的README文件和requirements.txt，可以按照以下步骤运行：

1. 进入对应的项目目录
2. 安装依赖：`pip install -r requirements.txt`
3. 按照README中的说明运行示例代码

## 贡献指南

欢迎贡献更多的教程和示例！请通过Pull Request提交您的贡献。

## 许可证

本项目采用开源许可证，详情请参见LICENSE文件。 