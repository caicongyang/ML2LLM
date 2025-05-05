"""
评估推荐系统模型性能
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.recommender import NCFModel
from utils.data_utils import load_data, split_data, create_data_loaders
from utils.metrics import evaluate_recommender
from config import MODEL_CONFIG, MISC_CONFIG, DATA_CONFIG

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置数据目录路径
DATA_DIR = os.path.join(CURRENT_DIR, 'data')


def generate_predictions(model, device, user_item_pairs):
    """
    使用模型生成预测评分
    
    参数:
        model: 推荐模型
        device: 设备(CPU/GPU)
        user_item_pairs: 包含(user_id, item_id)的数据框
        
    返回:
        pd.DataFrame: 包含(user_id, item_id, score)的预测数据框
    """
    # 转换为张量
    user_ids = torch.tensor(user_item_pairs['user_id'].values, dtype=torch.long).to(device)
    item_ids = torch.tensor(user_item_pairs['item_id'].values, dtype=torch.long).to(device)
    
    # 批量预测
    model.eval()
    with torch.no_grad():
        predictions = model(user_ids, item_ids).cpu().numpy()
    
    # 创建预测数据框
    pred_df = user_item_pairs.copy()
    pred_df['score'] = predictions
    
    return pred_df


def evaluate(model_path=None, metrics_k=10, plot=True):
    """
    评估模型性能
    
    参数:
        model_path (str, optional): 模型文件路径. 默认使用配置中的路径.
        metrics_k (int): 计算Top-K指标的K值
        plot (bool): 是否绘制评估结果图表
        
    返回:
        dict: 评估指标结果
    """
    if model_path is None:
        model_path = MODEL_CONFIG['model_save_path']
    
    # 设置设备
    device = torch.device(MISC_CONFIG['device'] if torch.cuda.is_available() and MISC_CONFIG['device'] == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    ratings_df, users_df, items_df, id_maps = load_data()
    
    # 获取用户和商品数量
    num_users = len(users_df)
    num_items = len(items_df)
    
    # 拆分数据
    train_df, test_df = split_data(ratings_df)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_layers=MODEL_CONFIG['hidden_layers'],
        dropout=MODEL_CONFIG['dropout_rate']
    ).to(device)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 生成预测
    print("生成预测...")
    pred_df = generate_predictions(model, device, test_df[['user_id', 'item_id']])
    
    # 评估模型性能
    print(f"评估模型性能 (k={metrics_k})...")
    metrics = evaluate_recommender(test_df, pred_df, k=metrics_k)
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Precision@{metrics_k}: {metrics[f'precision@{metrics_k}']:.4f}")
    print(f"Recall@{metrics_k}: {metrics[f'recall@{metrics_k}']:.4f}")
    print(f"NDCG@{metrics_k}: {metrics[f'ndcg@{metrics_k}']:.4f}")
    
    # 绘制评估结果图表
    if plot:
        plot_results(metrics, pred_df, test_df)
    
    return metrics


def plot_results(metrics, pred_df, test_df):
    """
    绘制评估结果图表
    
    参数:
        metrics (dict): 评估指标
        pred_df (pd.DataFrame): 预测数据
        test_df (pd.DataFrame): 测试数据
    """
    # 创建图表目录，使用data目录
    plots_dir = DATA_DIR
    os.makedirs(plots_dir, exist_ok=True)
    
    # 设置中文字体支持
    try:
        # 对于MacOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        # 对于Linux和Windows
        if not any([font in plt.matplotlib.font_manager.findSystemFonts() for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']]):
            raise ValueError("No suitable Chinese font found")
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
        use_chinese = True
    except:
        # 如果无法找到中文字体，使用英文标签
        use_chinese = False
    
    # 创建一个3x2的图表布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 评分分布图
    ax1 = axes[0, 0]
    ax1.hist(test_df['rating'], bins=10, alpha=0.5, label='True Ratings' if not use_chinese else '真实评分')
    ax1.hist(pred_df['score'], bins=10, alpha=0.5, label='Predicted Ratings' if not use_chinese else '预测评分')
    ax1.set_title('Rating Distribution Comparison' if not use_chinese else '评分分布对比')
    ax1.set_xlabel('Rating Value' if not use_chinese else '评分值')
    ax1.set_ylabel('Frequency' if not use_chinese else '频次')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 真实评分 vs 预测评分散点图
    ax2 = axes[0, 1]
    merged_df = pd.merge(test_df[['user_id', 'item_id', 'rating']], 
                         pred_df[['user_id', 'item_id', 'score']], 
                         on=['user_id', 'item_id'])
    ax2.scatter(merged_df['rating'], merged_df['score'], alpha=0.3)
    min_val = min(merged_df['rating'].min(), merged_df['score'].min()) - 0.5
    max_val = max(merged_df['rating'].max(), merged_df['score'].max()) + 0.5
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.set_title('True vs Predicted Ratings' if not use_chinese else '真实评分 vs 预测评分')
    ax2.set_xlabel('True Rating' if not use_chinese else '真实评分')
    ax2.set_ylabel('Predicted Rating' if not use_chinese else '预测评分')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 每个用户的平均评分
    ax3 = axes[1, 0]
    user_avg_true = test_df.groupby('user_id')['rating'].mean().reset_index()
    user_avg_pred = pred_df.groupby('user_id')['score'].mean().reset_index()
    merged_user_avg = pd.merge(user_avg_true, user_avg_pred, on='user_id')
    ax3.scatter(merged_user_avg['rating'], merged_user_avg['score'], alpha=0.5)
    min_val = min(merged_user_avg['rating'].min(), merged_user_avg['score'].min()) - 0.5
    max_val = max(merged_user_avg['rating'].max(), merged_user_avg['score'].max()) + 0.5
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax3.set_title('User Average Rating Comparison' if not use_chinese else '用户平均评分对比')
    ax3.set_xlabel('User Average True Rating' if not use_chinese else '用户真实平均评分')
    ax3.set_ylabel('User Average Predicted Rating' if not use_chinese else '用户预测平均评分')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 性能指标条形图
    ax4 = axes[1, 1]
    metrics_to_plot = {k: v for k, v in metrics.items() if k != 'rmse' and k != 'mae'}
    ax4.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    ax4.set_title('Performance Metrics' if not use_chinese else '性能指标')
    ax4.set_ylabel('Metric Value' if not use_chinese else '指标值')
    ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 设置主标题
    plt.suptitle('Recommendation System Model Evaluation Results' if not use_chinese else '推荐系统模型评估结果', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图表
    plt.savefig(os.path.join(plots_dir, 'evaluation_results.png'))
    print(f"评估结果图表已保存到: {os.path.join(plots_dir, 'evaluation_results.png')}")
    
    # 显示图表
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估推荐系统模型')
    parser.add_argument('--model_path', type=str, default=MODEL_CONFIG['model_save_path'], help='模型路径')
    parser.add_argument('--k', type=int, default=10, help='计算Top-K指标的K值')
    parser.add_argument('--no_plot', action='store_true', help='不绘制评估结果图表')
    
    args = parser.parse_args()
    
    evaluate(args.model_path, args.k, not args.no_plot)


if __name__ == "__main__":
    main() 