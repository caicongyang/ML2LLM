# ML2LLM - 从机器学习到大型语言模型

ML2LLM（Machine Learning to Large Language Models）是一个面向Java工程师的学习项目，旨在帮助Java开发者快速理解并转型到大模型开发领域。本项目包含从传统机器学习到大型语言模型的完整发展历程，提供了丰富的教程和示例代码。

## 项目结构

### 机器学习部分 (machine-learning/)

此目录包含传统机器学习相关的教程和示例：

- **introduction.md**: 机器学习基础概念介绍
- **classification.md**: 分类算法详解
- **regression.md**: 回归算法详解
- **clustering-algorithms.md**: 聚类算法详解
- **feature-extraction.md**: 特征提取技术
- **customer-segmentation.md**: 客户细分分析
- **reinforcement-learning.md**: 强化学习教程

#### 实例项目: 信用卡审批系统 (machine-learning/credit-card-approval/)

这是一个完整的机器学习应用示例，展示了如何构建信用卡审批预测系统。

### 深度学习部分 (deep-learning/)

此目录包含深度学习相关的教程和示例：

- **introduction.md**: 深度学习基础介绍
- **neural-networks.md** 和 **neural-networks-simplified.md**: 神经网络详解和简化版
- **neural_network_fundamentals.md** 和 **neural_network_fundamentals_simplified.md**: 神经网络基础知识详解和简化版
- **cnn.md** 和 **cnn-simplified.md**: 卷积神经网络详解和简化版
- **rnn.md** 和 **rnn-simplified.md**: 循环神经网络详解和简化版
- **autoencoder.md** 和 **autoencoder_cn.md**: 自编码器详解（英文和中文版）
- **neural-network-training.md**: 神经网络训练指南
- **neural-network-evaluation.md**: 神经网络评估方法
- **gan.md**: 生成对抗网络教程

#### 实例项目: 产品推荐系统 (deep-learning/product-recommendation-system/)

基于深度学习的产品推荐系统实现。

#### 实例项目: 自动视觉检测系统 (deep-learning/auto-vision/)

基于深度学习的图像异常检测系统。

### 大型语言模型部分 (LLM/)

该部分包含LLM相关的教程和实例项目：

- **transformer.md** 和 **transformer-simplified.md**: Transformer架构详解和简化版
- **from_BERT_to_GPT2.md**: 从BERT到GPT2的发展历程
- **fine-tuning-simplified.md**: 模型微调简介

#### 实例项目: BERT (LLM/bert/)

BERT模型的实现和训练示例：
- 包含完整的训练、测试和推理代码
- 提供数据处理和模型构建示例

#### 实例项目: GPT2 (LLM/gpt2/)

GPT2模型训练与使用示例。

#### 实例项目: LoRA微调 (LLM/lora/)

低秩适应(LoRA)微调技术实现：
- 提供完整的微调流程和脚本
- 包含模型合并和推理示例
- 支持问答系统实现

## 开发旅程

项目的 `journey.md` 是整个项目的出发领航，方便Java开发者快速理解并转型到大模型开发领域。它涵盖了从传统机器学习到大语言模型的完整发展历程，包括：

1. 传统机器学习阶段
2. 神经网络复兴与深度学习阶段
3. 预训练与Transformer革命
4. 大型语言模型（LLM）时代
5. 多模态与AGI探索

## 使用指南

每个示例项目都包含独立的README文件和requirements.txt，可以按照以下步骤运行：

1. 进入对应的项目目录
2. 安装依赖：`pip install -r requirements.txt`
3. 按照README中的说明运行示例代码

## 贡献指南

欢迎贡献更多的教程和示例！请通过Pull Request提交您的贡献。

## 许可证

本项目采用开源许可证，详情请参见LICENSE文件。 