"""
商品推荐系统的神经网络模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCFModel(nn.Module):
    """
    神经协同过滤模型 (Neural Collaborative Filtering)
    结合矩阵分解和多层感知机实现深度推荐系统
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32], dropout=0.2):
        """
        初始化NCF模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 商品数量
            embedding_dim (int): 嵌入向量维度
            hidden_layers (list): 隐藏层神经元数量列表
            dropout (float): Dropout比例
        """
        super(NCFModel, self).__init__()
        
        # 嵌入层 - 将用户和商品ID映射到隐空间
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 矩阵分解部分 (GMF: Generalized Matrix Factorization)
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 多层感知机部分 (MLP: Multi-Layer Perceptron)
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        
        # 构建MLP隐藏层
        for i, next_size in enumerate(hidden_layers):
            self.mlp_layers.append(nn.Linear(input_size, next_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = next_size
        
        # 最终预测层 - 结合GMF和MLP的输出
        self.***REMOVED***nal_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices):
        """
        前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引
            item_indices (torch.LongTensor): 商品索引
            
        返回:
            torch.FloatTensor: 预测的用户-商品交互得分
        """
        # GMF部分
        gmf_user_emb = self.gmf_user_embedding(user_indices)
        gmf_item_emb = self.gmf_item_embedding(item_indices)
        gmf_output = gmf_user_emb * gmf_item_emb  # 元素乘法
        
        # MLP部分
        mlp_user_emb = self.user_embedding(user_indices)
        mlp_item_emb = self.item_embedding(item_indices)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        
        # 通过MLP层
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
        
        # 结合GMF和MLP的输出
        combined = torch.cat([gmf_output, mlp_input], dim=-1)
        
        # 最终预测
        prediction = self.***REMOVED***nal_layer(combined)
        return torch.sigmoid(prediction).squeeze()


class Simpli***REMOVED***edNCF(nn.Module):
    """
    简化版神经协同过滤模型
    适用于数据集较小或计算资源受限的场景
    """
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64, dropout=0.2):
        super(Simpli***REMOVED***edNCF, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 简化的多层感知机
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices):
        # 获取嵌入
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # 拼接嵌入
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # 前向传播
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x).squeeze() 