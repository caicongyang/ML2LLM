"""
商品推荐系统API
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from models.recommender import NCFModel
from utils.data_utils import load_data
from con***REMOVED***g import MODEL_CONFIG, RECOMMEND_CONFIG, MISC_CONFIG


class RecommenderSystem:
    """推荐系统类"""
    def __init__(self, model_path=None):
        """
        初始化推荐系统
        
        参数:
            model_path (str, optional): 模型路径. 默认为配置中的路径.
        """
        if model_path is None:
            model_path = MODEL_CONFIG['model_save_path']
            
        # 加载数据和ID映射
        self.ratings_df, self.users_df, self.items_df, self.id_maps = load_data()
        
        # 加载模型
        self.num_users = len(self.users_df)
        self.num_items = len(self.items_df)
        
        # 设置设备
        self.device = torch.device(MISC_CONFIG['device'] if torch.cuda.is_available() and MISC_CONFIG['device'] == 'cuda' else 'cpu')
        
        # 初始化模型
        self.model = self._load_model(model_path)
        
        # 用户交互历史
        self.user_interactions = self._get_user_interactions()

    def _load_model(self, model_path):
        """加载模型"""
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化模型
        model = NCFModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            hidden_layers=MODEL_CONFIG['hidden_layers'],
            dropout=MODEL_CONFIG['dropout_rate']
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()  # 设置为评估模式
        
        return model
    
    def _get_user_interactions(self):
        """获取用户交互历史"""
        user_interactions = {}
        for user_id in self.ratings_df['user_id'].unique():
            # 获取用户评分过的商品
            items = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
            user_interactions[user_id] = items
        
        return user_interactions
    
    def get_recommendations(self, user_id, top_k=None, exclude_interacted=True):
        """
        为用户推荐商品
        
        参数:
            user_id (int): 用户ID
            top_k (int, optional): 推荐商品数量. 默认使用配置值.
            exclude_interacted (bool): 是否排除用户已交互的商品
            
        返回:
            list: 推荐的商品列表，每个元素为(商品ID, 预测分数)
        """
        if top_k is None:
            top_k = RECOMMEND_CONFIG['top_k']
        
        # 检查用户ID是否有效
        if user_id not in self.id_maps['user'].values():
            raise ValueError(f"无效的用户ID: {user_id}")
        
        # 用户已交互的商品
        interacted_items = self.user_interactions.get(user_id, []) if exclude_interacted else []
        
        # 创建候选商品列表 (排除已交互的)
        candidate_items = [item for item in range(self.num_items) if item not in interacted_items]
        
        if not candidate_items:
            return []
        
        # 创建用户-商品对
        user_ids = torch.tensor([user_id] * len(candidate_items), dtype=torch.long).to(self.device)
        item_ids = torch.tensor(candidate_items, dtype=torch.long).to(self.device)
        
        # 批量预测
        with torch.no_grad():
            predictions = self.model(user_ids, item_ids).cpu().numpy()
        
        # 创建预测结果数据框
        pred_df = pd.DataFrame({
            'item_id': candidate_items,
            'score': predictions
        })
        
        # 获取预测分数前K的商品
        top_items = pred_df.sort_values('score', ascending=False).head(top_k)
        
        # 转换为原始商品ID
        # 创建反向映射
        reverse_item_map = {v: k for k, v in self.id_maps['item'].items()}
        
        # 获取推荐结果
        recommendations = []
        for _, row in top_items.iterrows():
            item_id = row['item_id']
            original_item_id = reverse_item_map.get(item_id)
            score = row['score']
            recommendations.append((original_item_id, score))
        
        return recommendations
    
    def explain_recommendation(self, user_id, item_id):
        """
        解释推荐结果
        
        参数:
            user_id (int): 用户ID
            item_id (int): 商品ID
            
        返回:
            dict: 解释信息
        """
        # 这里可以实现一些解释推荐的逻辑
        # 例如，找出与当前商品相似的、用户曾经交互过的商品
        # 对于深度学习模型，可以尝试分析用户和商品的嵌入向量
        
        # 示例实现：获取用户历史交互中评分最高的商品
        if user_id not in self.id_maps['user'].values():
            return {"error": "无效的用户ID"}
        
        # 获取用户评分最高的几个商品
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        if len(user_ratings) == 0:
            return {"reason": "基于用户的兴趣模型"}
        
        top_rated_items = user_ratings.sort_values('rating', ascending=False).head(3)['item_id'].tolist()
        
        # 示例解释
        explanation = {
            "reason": "基于您对以下商品的喜好",
            "top_rated_items": top_rated_items,
            "similarity_score": round(float(np.random.uniform(0.7, 0.95)), 2)  # 示例相似度
        }
        
        return explanation
    
    
def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='商品推荐系统')
    parser.add_argument('--user_id', type=int, required=True, help='用户ID')
    parser.add_argument('--top_k', type=int, default=RECOMMEND_CONFIG['top_k'], help='推荐数量')
    parser.add_argument('--model_path', type=str, default=MODEL_CONFIG['model_save_path'], help='模型路径')
    parser.add_argument('--explain', action='store_true', help='是否解释推荐结果')
    
    args = parser.parse_args()
    
    try:
        # 初始化推荐系统
        recommender = RecommenderSystem(args.model_path)
        
        # 获取推荐
        recommendations = recommender.get_recommendations(args.user_id, args.top_k)
        
        # 打印推荐结果
        print(f"为用户 {args.user_id} 的推荐商品:")
        for i, (item_id, score) in enumerate(recommendations, 1):
            print(f"{i}. 商品ID: {item_id}, 预测分数: {score:.4f}")
            
            # 如果需要解释，打印解释
            if args.explain:
                explanation = recommender.explain_recommendation(args.user_id, item_id)
                print(f"   推荐理由: {explanation['reason']}")
                if 'top_rated_items' in explanation:
                    print(f"   基于您喜欢的商品: {explanation['top_rated_items']}")
                if 'similarity_score' in explanation:
                    print(f"   相似度: {explanation['similarity_score']}")
                    
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main() 