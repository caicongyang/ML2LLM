"""
商品推荐系统API - 基于神经协同过滤(NCF)模型的推荐服务
该模块提供了推荐系统的接口，用于加载训练好的模型并为用户生成个性化商品推荐
支持命令行调用和作为Python模块导入使用
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from models.recommender import NCFModel  # 导入自定义的NCF模型
from utils.data_utils import load_data  # 导入数据加载工具
from config import MODEL_CONFIG, RECOMMEND_CONFIG, MISC_CONFIG  # 导入配置参数


class RecommenderSystem:
    """
    推荐系统类 - 封装了模型加载、数据处理和推荐生成的核心功能
    实现了基于NCF模型的商品推荐逻辑，包括用户-商品评分预测和推荐结果解释
    """
    def __init__(self, model_path=None):
        """
        初始化推荐系统实例
        
        参数:
            model_path (str, optional): 预训练模型文件的路径. 如果未提供，则使用配置文件中的默认路径.
        """
        # 如果未提供模型路径，则使用配置中的默认路径
        if model_path is None:
            model_path = MODEL_CONFIG['model_save_path']
            
        # 加载数据集和ID映射字典（将原始ID映射到模型内部使用的连续整数ID）
        self.ratings_df, self.users_df, self.items_df, self.id_maps = load_data()
        
        # 获取用户和商品数量，用于初始化模型
        self.num_users = len(self.users_df)  # 用户总数
        self.num_items = len(self.items_df)  # 商品总数
        
        # 设置计算设备 - 优先使用GPU（如果可用且配置了使用GPU）
        self.device = torch.device(MISC_CONFIG['device'] if torch.cuda.is_available() and MISC_CONFIG['device'] == 'cuda' else 'cpu')
        
        # 加载预训练的推荐模型
        self.model = self._load_model(model_path)
        
        # 构建用户交互历史字典，用于过滤已交互商品
        self.user_interactions = self._get_user_interactions()

    def _load_model(self, model_path):
        """
        加载预训练的NCF模型
        
        参数:
            model_path (str): 模型权重文件的路径
            
        返回:
            NCFModel: 加载了预训练权重的模型实例
            
        异常:
            FileNotFoundError: 如果模型文件不存在
        """
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建与训练时相同结构的NCF模型
        model = NCFModel(
            num_users=self.num_users,             # 用户数量
            num_items=self.num_items,             # 商品数量
            embedding_dim=MODEL_CONFIG['embedding_dim'],  # 嵌入向量维度
            hidden_layers=MODEL_CONFIG['hidden_layers'],  # MLP层配置
            dropout=MODEL_CONFIG['dropout_rate']          # Dropout率
        ).to(self.device)  # 将模型移动到指定设备(CPU/GPU)
        
        # 加载预训练权重到模型中
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()  # 设置为评估模式，禁用BN和Dropout等训练特定操作
        
        return model
    
    def _get_user_interactions(self):
        """
        构建用户交互历史字典
        
        返回:
            dict: 用户交互历史字典, 格式为 {user_id: [item_id1, item_id2, ...]}
                 键为用户ID，值为该用户交互过的商品ID列表
        """
        user_interactions = {}
        # 遍历所有用户ID
        for user_id in self.ratings_df['user_id'].unique():
            # 获取该用户评分过的所有商品ID
            items = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
            user_interactions[user_id] = items
        
        return user_interactions
    
    def get_recommendations(self, user_id, top_k=None, exclude_interacted=True):
        """
        为指定用户生成个性化商品推荐
        
        参数:
            user_id (int): 目标用户ID
            top_k (int, optional): 要推荐的商品数量. 如果未提供，则使用配置文件中的默认值.
            exclude_interacted (bool): 是否从推荐结果中排除用户已交互过的商品
            
        返回:
            list: 推荐的商品列表，每个元素为(商品ID, 预测分数)的元组
                 按预测分数降序排列
                 
        异常:
            ValueError: 如果提供了无效的用户ID
        """
        # 如果未指定top_k，则使用配置中的默认值
        if top_k is None:
            top_k = RECOMMEND_CONFIG['top_k']
        
        # 验证用户ID是否存在于数据集中
        if user_id not in self.id_maps['user'].values():
            raise ValueError(f"无效的用户ID: {user_id}")
        
        # 获取用户已交互的商品列表（如果exclude_interacted=True）
        interacted_items = self.user_interactions.get(user_id, []) if exclude_interacted else []
        
        # 创建候选商品列表，排除用户已交互的商品
        candidate_items = [item for item in range(self.num_items) if item not in interacted_items]
        
        # 如果没有候选商品，返回空列表
        if not candidate_items:
            return []
        
        # 为每个候选商品创建用户-商品对，准备批量预测
        # 重复用户ID以匹配每个候选商品
        user_ids = torch.tensor([user_id] * len(candidate_items), dtype=torch.long).to(self.device)
        item_ids = torch.tensor(candidate_items, dtype=torch.long).to(self.device)
        
        # 使用模型批量预测所有候选商品的评分
        with torch.no_grad():  # 禁用梯度计算以提高推理速度
            predictions = self.model(user_ids, item_ids).cpu().numpy()
        
        # 创建包含商品ID和预测分数的DataFrame，便于排序
        pred_df = pd.DataFrame({
            'item_id': candidate_items,  # 内部商品ID
            'score': predictions         # 预测评分
        })
        
        # 按预测分数降序排序，并选取前top_k个商品
        top_items = pred_df.sort_values('score', ascending=False).head(top_k)
        
        # 将内部商品ID转换回原始商品ID
        # 创建从内部ID到原始ID的反向映射
        reverse_item_map = {v: k for k, v in self.id_maps['item'].items()}
        
        # 构建最终推荐结果列表
        recommendations = []
        for _, row in top_items.iterrows():
            item_id = row['item_id']  # 内部商品ID
            original_item_id = reverse_item_map.get(item_id)  # 原始商品ID
            score = row['score']  # 预测评分
            recommendations.append((original_item_id, score))
        
        return recommendations
    
    def explain_recommendation(self, user_id, item_id):
        """
        为推荐结果生成解释信息
        
        参数:
            user_id (int): 用户ID
            item_id (int): 被推荐的商品ID
            
        返回:
            dict: 包含推荐解释信息的字典
                可能包含的字段:
                - reason: 推荐理由
                - top_rated_items: 用户评分最高的商品列表
                - similarity_score: 相似度分数
                - error: 错误信息(当发生错误时)
        """
        # 这部分是一个简化的推荐解释实现
        # 注释说明了可能的扩展方向:
        # 1. 基于商品相似性的解释
        # 2. 基于用户嵌入向量分析的解释
        # 3. 基于协同过滤规则的解释
        
        # 验证用户ID是否有效
        if user_id not in self.id_maps['user'].values():
            return {"error": "无效的用户ID"}
        
        # 获取用户历史评分数据
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        # 如果用户没有历史评分，返回默认解释
        if len(user_ratings) == 0:
            return {"reason": "基于用户的兴趣模型"}
        
        # 获取用户评分最高的3个商品ID
        top_rated_items = user_ratings.sort_values('rating', ascending=False).head(3)['item_id'].tolist()
        
        # 构建解释信息字典
        # 注意：相似度分数是随机生成的示例值，实际应用中应该基于模型计算
        explanation = {
            "reason": "基于您对以下商品的喜好",
            "top_rated_items": top_rated_items,
            "similarity_score": round(float(np.random.uniform(0.7, 0.95)), 2)  # 示例相似度分数
        }
        
        return explanation
    
    
def main():
    """
    主函数 - 处理命令行参数并执行推荐过程
    支持通过命令行直接调用推荐功能
    """
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='商品推荐系统')
    parser.add_argument('--user_id', type=int, required=True, help='目标用户ID')
    parser.add_argument('--top_k', type=int, default=RECOMMEND_CONFIG['top_k'], help='推荐商品数量')
    parser.add_argument('--model_path', type=str, default=MODEL_CONFIG['model_save_path'], help='模型文件路径')
    parser.add_argument('--explain', action='store_true', help='是否为推荐结果提供解释')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 初始化推荐系统实例
        recommender = RecommenderSystem(args.model_path)
        
        # 为指定用户生成推荐
        recommendations = recommender.get_recommendations(args.user_id, args.top_k)
        
        # 打印推荐结果标题
        print(f"为用户 {args.user_id} 的推荐商品:")
        
        # 遍历并打印每个推荐商品
        for i, (item_id, score) in enumerate(recommendations, 1):
            # 打印商品ID和预测分数
            print(f"{i}. 商品ID: {item_id}, 预测分数: {score:.4f}")
            
            # 如果需要解释，则获取并打印推荐解释
            if args.explain:
                # 获取当前推荐项的解释
                explanation = recommender.explain_recommendation(args.user_id, item_id)
                
                # 打印基本解释理由
                print(f"   推荐理由: {explanation['reason']}")
                
                # 打印用户喜欢的相关商品(如果有)
                if 'top_rated_items' in explanation:
                    print(f"   基于您喜欢的商品: {explanation['top_rated_items']}")
                
                # 打印相似度分数(如果有)
                if 'similarity_score' in explanation:
                    print(f"   相似度: {explanation['similarity_score']}")
                    
    except Exception as e:
        # 捕获并打印任何错误
        print(f"错误: {str(e)}")


# 脚本入口点 - 当直接运行此脚本时执行main()函数
if __name__ == "__main__":
    main() 