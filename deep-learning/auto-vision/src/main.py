#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度学习演示项目主程序
整合了模型训练、评估和可视化功能
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入自定义模块
from data_utils import load_cifar10, get_data_loaders
from models import get_model
from train import train_classi***REMOVED***er, train_autoencoder
from evaluate import evaluate_classi***REMOVED***er, evaluate_autoencoder
from visualize import (
    plot_training_history,
    visualize_predictions,
    visualize_features,
    visualize_attention,
    visualize_autoencoder_results,
    get_visualization_function
)


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_dirs(args):
    """创建输出目录"""
    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 创建结果保存目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 针对特定模型创建子目录
    model_results_dir = os.path.join(args.results_dir, args.model_type)
    os.makedirs(model_results_dir, exist_ok=True)
    
    return model_results_dir


def train_and_evaluate(args):
    """训练并评估模型"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    model_results_dir = create_output_dirs(args)
    
    # 加载数据
    print("正在加载CIFAR-10数据集...")
    train_dataset, val_dataset, test_dataset, class_names = load_cifar10(
        data_dir=args.data_dir, 
        val_size=0.1,
        apply_augmentation=True
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, 
        val_dataset, 
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 获取样本维度
    sample_data, _ = next(iter(train_loader))
    input_shape = sample_data.shape[1:]
    print(f"样本形状: {input_shape}")
    
    # 创建模型
    print(f"创建模型: {args.model_type}")
    model = get_model(args.model_type, input_shape=input_shape, num_classes=10 if args.model_type != 'autoencoder' else None)
    model = model.to(device)
    print(model)
    
    # 训练模型
    if not args.skip_training:
        print(f"开始训练 {args.model_type} 模型...")
        
        # 选择适当的训练函数
        if args.model_type == 'autoencoder':
            history = train_autoencoder(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                model_save_path=os.path.join(args.model_dir, f"{args.model_type}.pth")
            )
        else:
            history = train_classi***REMOVED***er(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                model_save_path=os.path.join(args.model_dir, f"{args.model_type}.pth")
            )
        
        # 绘制训练历史
        plot_training_history(
            history, 
            save_path=os.path.join(model_results_dir, 'training_history.png')
        )
    else:
        # 如果跳过训练，尝试加载已有模型
        model_path = os.path.join(args.model_dir, f"{args.model_type}.pth")
        if os.path.exists(model_path):
            print(f"加载已有模型 {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"找不到已有模型 {model_path}，无法跳过训练")
            return
    
    # 评估模型
    print(f"评估 {args.model_type} 模型...")
    if args.model_type == 'autoencoder':
        metrics = evaluate_autoencoder(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir=model_results_dir
        )
        print(f"测试集重建误差: {metrics['mse']:.4f}")
    else:
        metrics = evaluate_classi***REMOVED***er(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            save_dir=model_results_dir
        )
        print(f"测试集准确率: {metrics['accuracy']:.2f}%")
        print(f"测试集F1分数: {metrics['f1']:.4f}")
    
    # 可视化结果
    print("生成可视化结果...")
    
    # 特征可视化（对所有模型）
    visualize_features(
        model=model,
        data_loader=test_loader,
        device=device,
        method='tsne',
        save_path=os.path.join(model_results_dir, 'tsne_features.png')
    )
    
    # 根据模型类型选择可视化方法
    viz_func = get_visualization_function(args.model_type)
    
    if args.model_type == 'autoencoder':
        visualize_autoencoder_results(
            model=model,
            data_loader=test_loader,
            device=device,
            save_dir=model_results_dir,
            num_samples=10
        )
    elif args.model_type == 'attention_cnn':
        visualize_attention(
            model=model,
            data_loader=test_loader,
            device=device,
            class_names=class_names,
            save_dir=model_results_dir
        )
        # 额外添加常规预测可视化
        visualize_predictions(
            model=model,
            data_loader=test_loader,
            class_names=class_names,
            device=device,
            num_samples=16,
            save_path=os.path.join(model_results_dir, 'predictions.png')
        )
    else:
        visualize_predictions(
            model=model,
            data_loader=test_loader,
            class_names=class_names,
            device=device,
            num_samples=16,
            save_path=os.path.join(model_results_dir, 'predictions.png')
        )
    
    print(f"所有结果已保存到 {model_results_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='深度学习模型训练、评估与可视化演示')
    
    # 模型设置
    parser.add_argument('--model-type', type=str, default='basic_cnn',
                        choices=['basic_cnn', 'attention_cnn', 'autoencoder'],
                        help='模型类型 (默认: basic_cnn)')
    
    # 训练设置
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批量大小 (默认: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (默认: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')
    
    # 目录设置
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录 (默认: ./data)')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='模型保存目录 (默认: ./models)')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='结果保存目录 (默认: ./results)')
    
    # 其他选项
    parser.add_argument('--skip-training', action='store_true', default=False,
                        help='跳过训练，直接加载模型进行评估')
    
    args = parser.parse_args()
    
    # 打印参数设置
    print("参数设置:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # 训练和评估模型
    train_and_evaluate(args)


if __name__ == '__main__':
    main() 