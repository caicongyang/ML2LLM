# 商品推荐系统

基于深度学习的商品推荐系统，使用协同过滤（Collaborative Filtering）和神经网络实现个性化推荐功能。

## 项目概述

该系统通过分析用户的历史购买/浏览记录以及商品特征，为用户推荐最可能感兴趣的商品。系统结合了：

1. **协同过滤**：基于用户-商品交互数据
2. **神经网络**：捕捉用户和商品之间的复杂非线性关系
3. **嵌入层**：将用户和商品映射到共享的隐空间

## 功能特点

- 个性化商品推荐
- 冷启动问题处理
- 可解释的推荐结果
- 实时更新用户兴趣模型

## 项目结构

```
product-recommendation-system/
├── data/                   # 数据目录
├── models/                 # 模型定义
├── training/               # 训练脚本
├── utils/                  # 工具函数
├── config.py               # 配置文件
├── recommend.py            # 推荐API
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明
```

## 安装与使用

1. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

2. 数据预处理与训练：
   ```
   python training/train.py
   ```

3. 进行推荐：
   ```
   python recommend.py --user_id <用户ID>
   ```

## 技术栈

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Scikit-learn 