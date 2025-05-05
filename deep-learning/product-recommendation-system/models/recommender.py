"""
商品推荐系统的神经网络模型定义

这个模块定义了用于商品推荐系统的两种不同的神经网络模型：
1. NCFModel: 一个结合了矩阵分解和多层感知机的完整神经协同过滤模型
2. SimplifiedNCF: 一个简化版的神经协同过滤模型，适用于较小的数据集或资源有限的环境
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCFModel(nn.Module):
    """
    神经协同过滤模型 (Neural Collaborative Filtering)
    
    该模型结合了两种推荐系统的关键思想：
    1. 矩阵分解 (GMF: Generalized Matrix Factorization)：捕获用户和商品之间的隐性交互关系
    2. 多层感知机 (MLP: Multi-Layer Perceptron)：学习用户-商品交互的非线性特征

    模型架构：
    - 用户和商品的嵌入层：将ID映射到低维向量空间
    - GMF路径：用户和商品的嵌入向量元素乘法，捕获线性交互
    - MLP路径：用户和商品的嵌入向量拼接后通过多层神经网络，捕获非线性交互
    - 最终：GMF和MLP的输出拼接后通过一个全连接层进行评分预测

    参考论文：
    He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). 
    Neural collaborative filtering. In Proceedings of the 26th international 
    conference on world wide web (pp. 173-182).
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32], dropout=0.2):
        """
        初始化NCF模型
        
        参数:
            num_users (int): 用户数量，决定了用户嵌入矩阵的大小
            num_items (int): 商品数量，决定了商品嵌入矩阵的大小
            embedding_dim (int): 嵌入向量维度，影响模型表达能力和训练参数数量
            hidden_layers (list): 多层感知机部分的隐藏层神经元数量列表
                                每个元素表示一层的神经元数量
            dropout (float): Dropout比例，用于防止过拟合，取值范围0~1
        """
        super(NCFModel, self).__init__()
        
        # 多层感知机部分的嵌入层
        # 这些嵌入将用户和商品ID映射到隐空间，然后通过MLP处理
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 矩阵分解部分的嵌入层
        # 这些嵌入将进行元素乘法以模拟传统的矩阵分解方法
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 多层感知机部分 (MLP: Multi-Layer Perceptron)
        # 用于学习用户-商品交互的非线性模式
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * embedding_dim  # 用户和商品嵌入拼接
        
        # 构建MLP隐藏层
        # 每一层包含一个线性变换、ReLU激活函数和Dropout
        for i, next_size in enumerate(hidden_layers):
            self.mlp_layers.append(nn.Linear(input_size, next_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = next_size
        
        # 最终预测层 - 结合GMF和MLP的输出
        # 输入是GMF输出(embedding_dim)和MLP最后一层输出(hidden_layers[-1])的拼接
        self.final_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        
        对于线性层使用Xavier均匀初始化，这有助于保持每一层的方差稳定
        对于嵌入层使用正态分布初始化，标准差为0.01
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化对于tanh和sigmoid激活函数很有效，但对ReLU也有不错的效果
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0
            elif isinstance(m, nn.Embedding):
                # 嵌入层通常使用小的正态分布初始化
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices):
        """
        模型前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引，形状为 (batch_size,)
            item_indices (torch.LongTensor): 商品索引，形状为 (batch_size,)
            
        返回:
            torch.FloatTensor: 预测的用户-商品交互评分，值域在[0,1]之间，形状为 (batch_size,)
                             可以理解为用户对商品的喜好程度或评分的归一化值
        
        流程:
            1. GMF路径：用户和商品嵌入元素乘法
            2. MLP路径：用户和商品嵌入拼接，通过多层感知机处理
            3. 最终层：拼接GMF和MLP的输出，通过最终层预测评分
        """
        # GMF部分 - 矩阵分解风格的线性交互
        gmf_user_emb = self.gmf_user_embedding(user_indices)  # [batch_size, embedding_dim]
        gmf_item_emb = self.gmf_item_embedding(item_indices)  # [batch_size, embedding_dim]
        gmf_output = gmf_user_emb * gmf_item_emb  # 元素乘法 [batch_size, embedding_dim]
        
        # MLP部分 - 多层感知机捕获非线性特征
        mlp_user_emb = self.user_embedding(user_indices)  # [batch_size, embedding_dim]
        mlp_item_emb = self.item_embedding(item_indices)  # [batch_size, embedding_dim]
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)  # [batch_size, 2*embedding_dim]
        
        # 通过MLP层
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)  # 最终形状: [batch_size, hidden_layers[-1]]
        
        # 结合GMF和MLP的输出
        combined = torch.cat([gmf_output, mlp_input], dim=-1)  # [batch_size, embedding_dim + hidden_layers[-1]]
        
        # 最终预测
        prediction = self.final_layer(combined)  # [batch_size, 1]
        # sigmoid确保输出在0到1之间，适合评分预测
        return torch.sigmoid(prediction).squeeze()  # [batch_size]


class SimplifiedNCF(nn.Module):
    """
    简化版神经协同过滤模型
    
    该模型是NCF模型的简化版本，主要优化了以下方面：
    1. 移除了GMF部分，只保留了MLP组件
    2. 减少了隐藏层的数量，简化了网络结构
    3. 使用较小的嵌入维度

    适用场景：
    - 数据集较小，避免过拟合
    - 计算资源受限，需要更快的训练和推理
    - 对模型复杂度要求不高的简单推荐任务
    
    模型架构：
    - 用户和商品的嵌入层
    - 嵌入向量拼接
    - 一个带Dropout的隐藏层
    - 最终的预测层
    """
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64, dropout=0.2):
        """
        初始化简化版NCF模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 商品数量
            embedding_dim (int): 嵌入向量维度，默认比标准NCF小
            hidden_dim (int): 隐藏层神经元数量
            dropout (float): Dropout比例，防止过拟合
        """
        super(SimplifiedNCF, self).__init__()
        
        # 嵌入层 - 将用户和商品ID映射到低维向量空间
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 简化的多层感知机 - 只有一个隐藏层
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)  # 第一个全连接层
        self.dropout = nn.Dropout(dropout)                   # Dropout层防止过拟合
        self.fc2 = nn.Linear(hidden_dim, 1)                  # 输出层
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        
        与完整NCF模型使用相同的初始化策略:
        - 线性层使用Xavier均匀初始化
        - 嵌入层使用正态分布初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices):
        """
        模型前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引，形状为 (batch_size,)
            item_indices (torch.LongTensor): 商品索引，形状为 (batch_size,)
            
        返回:
            torch.FloatTensor: 预测的用户-商品交互评分，值域在[0,1]之间，形状为 (batch_size,)
        
        流程:
            1. 获取用户和商品的嵌入向量
            2. 拼接嵌入向量
            3. 通过简化的神经网络处理
            4. 使用sigmoid函数将输出映射到[0,1]区间
        """
        # 获取嵌入
        user_emb = self.user_embedding(user_indices)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_indices)  # [batch_size, embedding_dim]
        
        # 拼接嵌入
        x = torch.cat([user_emb, item_emb], dim=-1)  # [batch_size, 2*embedding_dim]
        
        # 前向传播 - 简化的MLP结构
        x = F.relu(self.fc1(x))  # ReLU激活函数
        x = self.dropout(x)      # 应用dropout防止过拟合
        x = self.fc2(x)          # 最终线性层
        
        # sigmoid确保输出在0到1之间
        return torch.sigmoid(x).squeeze()  # [batch_size] 