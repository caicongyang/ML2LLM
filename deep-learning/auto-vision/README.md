# 图像分类演示项目

## 项目简介

这是一个基于深度学习的图像分类演示项目，旨在展示如何使用不同类型的神经网络模型（包括CNN、自编码器等）进行图像分类任务。项目实现了从数据准备、模型定义、训练、评估到结果可视化的完整流程。

## 项目结构

```
auto-vision/
├── README.md           # 项目说明文档
├── requirements.txt    # 项目依赖
├── data/               # 数据集目录
├── models/             # 预训练模型保存目录
├── results/            # 评估结果和可视化输出目录
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
   - 基础卷积神经网络(basic_cnn)
   - 带自注意力机制的CNN(attention_cnn)
   - 基础自编码器(vanilla_ae)
   - 卷积自编码器(conv_ae)
   - 变分自编码器(vae)

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
   - 特征可视化(t-SNE)
   - 注意力热力图(attention_cnn)
   - 自编码器重建结果

## 安装与使用

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- matplotlib
- seaborn
- tqdm
- 其他依赖(详见requirements.txt)

### 安装步骤

1. 克隆项目
```bash
git clone [your-repo-url]
cd auto-vision
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据目录
```bash
mkdir -p data models results
```

### 使用方法

#### 训练模型

使用主程序`main.py`进行训练，可选的模型类型包括`basic_cnn`、`attention_cnn`、`vanilla_ae`、`conv_ae`和`vae`：

```bash
python main.py --model-type basic_cnn --epochs 10 --batch-size 64
```

#### 评估模型

评估已训练的模型有两种方式：

1. 使用`main.py`（训练后立即评估）：
```bash
python main.py --model-type basic_cnn --skip-training --model-dir ../models
```

2. 使用单独的评估脚本`evaluate.py`（推荐）：
```bash
python src/evaluate.py --model-type basic_cnn --model-path ../models/basic_cnn.pth --data-dir ../data/
```

评估脚本的其他选项：
```bash
python src/evaluate.py --help
```

#### 命令行参数

主程序`main.py`的主要参数：
- `--model-type`: 模型类型，可选 `basic_cnn`, `attention_cnn`, `vanilla_ae`, `conv_ae`, `vae`
- `--batch-size`: 批量大小，默认为64
- `--epochs`: 训练轮数，默认为10
- `--learning-rate`: 学习率，默认为0.001
- `--data-dir`: 数据目录，默认为项目根目录下的data目录
- `--model-dir`: 模型保存目录，默认为项目根目录下的models目录
- `--results-dir`: 结果保存目录，默认为项目根目录下的results目录
- `--skip-training`: 跳过训练，直接加载模型进行评估
- `--no-cuda`: 禁用CUDA

评估脚本`evaluate.py`的主要参数：
- `--model-type`: 模型类型，可选 `basic_cnn`, `attention_cnn`, `vanilla_ae`, `conv_ae`, `vae`
- `--model-path`: 模型文件路径(必需)
- `--data-dir`: 数据目录，默认为项目根目录下的data目录
- `--results-dir`: 结果保存目录，默认为项目根目录下的results目录
- `--batch-size`: 批量大小，默认为64
- `--no-cuda`: 禁用CUDA

## 示例结果

训练一个基础CNN模型在CIFAR-10数据集上，可以达到约85%的测试准确率。自编码器模型可用于图像重建和异常检测任务。

评估结果将保存在`results/[model_type]/`目录下，包括混淆矩阵、预测结果可视化、特征降维可视化等。

## 自定义扩展

您可以通过以下方式扩展项目：

1. 在`models.py`中添加新的模型架构
2. 在`train.py`中自定义训练流程
3. 在`data_utils.py`中支持更多数据集
4. 在`visualize.py`中增加新的可视化方法

## 贡献与许可

欢迎提交问题和改进建议。本项目使用MIT许可证。 