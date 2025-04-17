# ML2LLM
ML-to-LLM: A Beginner's Journey for Java Engineers

这个系列旨在帮助Java工程师快速理解和转型到大模型开发领域，从传统机器学习到大型语言模型的完整发展历程。

## 机器学习与大语言模型的发展历程

### 1. 传统机器学习阶段
**特点**：依赖人工特征工程，模型以统计方法和浅层模型为主。

**主要技术**：
- **监督学习**：
  - 线性回归、逻辑回归
  - 决策树（如ID3、C4.5）
  - 支持向量机（SVM）
  - 随机森林（Random Forest）
  - 梯度提升机（GBM、XGBoost、LightGBM）

- **无监督学习**：
  - 聚类（K-Means、层次聚类、DBSCAN）
  - 降维（PCA、t-SNE）
  - 关联规则（Apriori）

- **其他**：
  - 贝叶斯网络、隐马尔可夫模型（HMM）

**局限性**：特征工程繁琐，难以处理高维非结构化数据（如图像、文本）。

### 2. 神经网络复兴与深度学习阶段
**特点**：通过多层神经网络自动学习特征，解决复杂模式识别问题。

**关键里程碑**：
- **基础神经网络**：
  - 多层感知机（MLP）、反向传播算法（BP）

- **深度网络结构**：
  - 卷积神经网络（CNN）：LeNet、AlexNet（2012）、ResNet（图像领域）
  - 循环神经网络（RNN）：LSTM、GRU（时序数据，如文本、语音）
  - 自编码器（Autoencoder）、生成对抗网络（GAN）

- **技术突破**：
  - GPU加速训练
  - Dropout、Batch Normalization等优化技术
  - 端到端训练（End-to-End Learning）

**应用领域**：计算机视觉（CV）、语音识别（ASR）、自然语言处理（NLP）。

### 3. 预训练与Transformer革命
**特点**：模型规模大幅提升，依赖自监督预训练和注意力机制。

**关键技术与模型**：
- **Word Embedding**：
  - Word2Vec、GloVe（词向量表示）

- **上下文嵌入**：
  - ELMo（动态词向量）

- **Transformer架构（2017）**：
  - 自注意力机制（Self-Attention）
  - 并行化训练能力

- **预训练语言模型**：
  - GPT系列（单向语言模型，生成任务）
  - BERT（双向语言模型，理解任务）
  - RoBERTa、T5、BART等变体

- **技术突破**：
  - 大规模无监督预训练 + 下游任务微调（Pretrain-Finetune）
  - 迁移学习在NLP中的普及

### 4. 大型语言模型（LLM）时代
**特点**：模型参数规模爆炸（十亿至万亿级），涌现能力和通用性显著提升。

**代表性技术**：
- **模型架构**：
  - GPT-3（1750亿参数，Few-shot Learning）
  - PaLM、GPT-4、Claude、LLaMA（开源）
  - 混合专家模型（MoE，如Switch Transformer）

- **训练方法**：
  - 指令微调（Instruction Tuning）
  - 基于人类反馈的强化学习（RLHF，如ChatGPT）
  - 提示工程（Prompt Engineering）

- **关键技术**：
  - 缩放定律（Scaling Laws）
  - 分布式训练框架（如Megatron-LM、DeepSpeed）
  - 量化与推理优化（如LoRA、QLoRA）

**应用场景**：对话系统、代码生成、多模态交互等。

### 5. 多模态与AGI探索
**当前方向**：超越纯文本，融合视觉、音频等多模态数据，向通用人工智能（AGI）迈进。

**代表模型**：
- DALL·E（文本生成图像）
- GPT-4V（多模态理解）
- Sora（视频生成）
- 具身智能（Embodied AI）

## 项目目标

本项目将带领Java工程师从熟悉的编程环境逐步过渡到大模型开发，包括：
- 从Java视角理解机器学习核心概念
- 深度学习框架在Java中的应用
- 大模型接口调用与应用开发
- 模型部署与工程化实践
- LLM应用开发最佳实践

后续内容将陆续更新...
