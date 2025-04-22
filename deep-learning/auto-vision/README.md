# 图像分类演示项目

## 项目简介

这是一个基于深度学习的图像分类演示项目，旨在展示如何使用不同类型的神经网络模型（包括CNN、自编码器等）进行图像分类任务。项目实现了从数据准备、模型定义、训练、评估到结果可视化的完整流程。

## 项目结构

```
demo-project/
├── README.md           # 项目说明文档
├── requirements.txt    # 项目依赖
├── data/               # 数据集目录
├── models/             # 预训练模型保存目录
├── src/                # 源代码
│   ├── __init__.py
│   ├── data_utils.py   # 数据加载和预处理
│   ├── models.py       # 模型定义
│   ├── train.py        # 模型训练
│   ├── evaluate.py     # 模型评估
│   └── visualize.py    # 结果可视化
└── main.py             # 主程序入口
```

## 功能特点

1. 支持多种经典深度学习模型：
   - 卷积神经网络(CNN)
   - 带自注意力机制的CNN
   - 自编码器(用于异常检测)

2. 数据处理功能：
   - 数据加载和批处理
   - 数据增强
   - 训练/验证/测试集划分

3. 模型训练与评估：
   - 不同优化器选择
   - 学习率调度
   - 混淆矩阵和分类报告

4. 结果可视化：
   - 训练过程曲线
   - 模型预测结果
   - 特征可视化

## 安装与使用

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- 其他依赖(详见requirements.txt)

### 安装步骤

1. 克隆项目
```bash
git clone [your-repo-url]
cd demo-project
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
```bash
python src/data_utils.py --download
```

### 使用方法

#### 训练模型

```bash
python main.py --mode train --model cnn --epochs 10 --batch-size 64
```

#### 评估模型

```bash
python main.py --mode evaluate --model cnn --model-path models/cnn_best.pth
```

#### 可视化结果

```bash
python main.py --mode visualize --model cnn --model-path models/cnn_best.pth
```

## 示例结果

训练一个简单的CNN模型在CIFAR-10数据集上，可以达到约85%的测试准确率。自编码器模型可用于异常检测任务，检测出与训练数据分布不同的异常图像。

## 贡献与许可

欢迎提交问题和改进建议。本项目使用MIT许可证。 