#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自编码器运行脚本
提供简单的命令行接口运行自编码器模型的训练、评估和可视化
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from src.models import get_model
from src.train import train_autoencoder
from src.evaluate import evaluate_autoencoder
from src.visualize import visualize_autoencoder_results
from src.data_utils import create_noisy_data, get_data_loaders, get_sample_data


def main():
    parser = argparse.ArgumentParser(description='自编码器模型运行脚本')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'visualize'], 
                        help='运行模式')
    parser.add_argument('--model-type', type=str, default='vanilla_ae', 
                        choices=['vanilla_ae', 'conv_ae', 'vae'], 
                        help='自编码器模型类型')
    parser.add_argument('--latent-dim', type=int, default=128, 
                        help='潜在空间维度（仅用于vanilla_ae和vae）')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--noise-factor', type=float, default=0.3, 
                        help='噪声因子（用于去噪自编码器训练）')
    parser.add_argument('--use-noise', action='store_true', 
                        help='是否使用噪声数据进行训练（去噪自编码器）')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./output', 
                        help='输出目录')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='模型路径（用于评估和可视化）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备 (cuda|cpu)')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_cifar=True
    )
    
    # 创建模型
    print(f"创建{args.model_type}模型...")
    if args.model_type in ['vanilla_ae', 'vae']:
        model = get_model(args.model_type, in_channels=3, latent_dim=args.latent_dim)
    else:
        model = get_model(args.model_type, in_channels=3)
    model = model.to(device)
    
    # 根据模式运行
    if args.mode == 'train':
        print(f"开始训练{args.model_type}模型...")
        
        # 训练参数
        train_args = {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": 1e-5,
            "device": device,
            "log_dir": os.path.join(args.output_dir, 'logs'),
            "save_dir": os.path.join(args.output_dir, 'models'),
            "model_name": args.model_type,
            "is_vae": args.model_type == 'vae',
        }
        
        # 如果使用噪声，创建噪声数据加载器
        if args.use_noise:
            print(f"使用噪声因子 {args.noise_factor} 进行去噪自编码器训练")
            # 创建一个带噪声的训练加载器
            noisy_train_loader = create_noisy_data(train_loader, noise_factor=args.noise_factor)
            history = train_autoencoder(model, noisy_train_loader, val_loader, train_args)
        else:
            history = train_autoencoder(model, train_loader, val_loader, train_args)
        
        print(f"训练完成! 最终验证损失: {history['val_loss'][-1]:.6f}")
    
    elif args.mode == 'eval':
        if args.model_path is None:
            args.model_path = os.path.join(args.output_dir, 'models', f"{args.model_type}_best.pth")
            
        print(f"加载模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("评估模型...")
        eval_results = evaluate_autoencoder(model, test_loader, device)
        
        # 打印评估结果
        print("\n评估结果:")
        print(f"平均重构误差: {eval_results['mean_error']:.6f}")
        print(f"最大重构误差: {eval_results['max_error']:.6f}")
        print(f"最小重构误差: {eval_results['min_error']:.6f}")
        print(f"异常样本比例 (阈值={eval_results['threshold']:.6f}): {eval_results['anomaly_ratio']:.4f}")
    
    elif args.mode == 'visualize':
        if args.model_path is None:
            args.model_path = os.path.join(args.output_dir, 'models', f"{args.model_type}_best.pth")
            
        print(f"加载模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("可视化结果...")
        save_dir = os.path.join(args.output_dir, 'results', args.model_type)
        os.makedirs(save_dir, exist_ok=True)
        
        # 如果使用噪声，创建一个带噪声的测试加载器用于可视化
        if args.use_noise:
            print(f"使用噪声因子 {args.noise_factor} 创建噪声测试数据")
            noisy_test_loader = create_noisy_data(test_loader, noise_factor=args.noise_factor)
            visualize_autoencoder_results(model, noisy_test_loader, device, save_dir=save_dir)
        else:
            visualize_autoencoder_results(model, test_loader, device, save_dir=save_dir)
            
        print(f"可视化结果已保存到 {save_dir}")


if __name__ == "__main__":
    main() 