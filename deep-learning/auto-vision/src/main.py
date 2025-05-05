#!/usr/bin/env python  # 指定解释器路径
# -*- coding: utf-8 -*-  # 设置文件编码为UTF-8

"""
深度学习演示项目主程序
整合了模型训练、评估和可视化功能
"""

import os  # 导入操作系统模块，用于文件和目录操作
import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy数值计算库
import matplotlib.pyplot as plt  # 导入Matplotlib绘图库
from torch.utils.data import DataLoader  # 从PyTorch导入数据加载器

# 导入自定义模块
from data_utils import load_cifar10, get_data_loaders  # 导入数据处理工具
from models import get_model  # 导入模型获取函数
from train import train_classifier, train_autoencoder  # 导入训练函数
from evaluate import evaluate_classifier, evaluate_autoencoder  # 导入评估函数
from visualize import (  # 导入可视化相关函数
    plot_training_history,  # 绘制训练历史曲线
    visualize_predictions,  # 可视化模型预测结果
    visualize_features,  # 可视化特征降维效果
    visualize_attention,  # 可视化注意力图
    visualize_autoencoder_results  # 可视化自编码器结果
)


def set_seed(seed=42):  # 定义设置随机种子的函数，默认种子为42
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机种子
    torch.backends.cudnn.deterministic = True  # 设置cuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试模式


def create_output_dirs(args):  # 定义创建输出目录的函数
    """创建输出目录"""
    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)  # 创建模型保存目录，若已存在则不报错
    
    # 创建结果保存目录
    os.makedirs(args.results_dir, exist_ok=True)  # 创建结果保存目录，若已存在则不报错
    
    # 针对特定模型创建子目录
    model_results_dir = os.path.join(args.results_dir, args.model_type)  # 创建模型特定的结果子目录路径
    os.makedirs(model_results_dir, exist_ok=True)  # 创建模型特定的结果子目录，若已存在则不报错
    
    return model_results_dir  # 返回模型特定的结果目录路径


def train_and_evaluate(args):  # 定义训练和评估函数
    """训练并评估模型"""
    # 设置随机种子
    set_seed(args.seed)  # 调用set_seed函数设置随机种子
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")  # 根据CUDA可用性和用户参数设置计算设备
    print(f"使用设备: {device}")  # 打印所使用的计算设备信息
    
    # 创建输出目录
    model_results_dir = create_output_dirs(args)  # 调用create_output_dirs创建输出目录并获取模型结果目录路径
    
    # 加载数据
    print("正在加载CIFAR-10数据集...")  # 打印数据集加载信息
    train_dataset, val_dataset, test_dataset, class_names = load_cifar10(  # 加载CIFAR-10数据集
        data_dir=args.data_dir,  # 指定数据集存放目录
        val_size=0.1,  # 设置验证集比例为10%
        apply_augmentation=True  # 对训练集应用数据增强
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(  # 创建训练、验证和测试数据加载器
        train_dataset,  # 传入训练数据集
        val_dataset,  # 传入验证数据集
        test_dataset,  # 传入测试数据集
        batch_size=args.batch_size,  # 设置批次大小
        num_workers=args.num_workers  # 设置数据加载的工作线程数
    )
    
    # 获取样本维度
    sample_data, _ = next(iter(train_loader))  # 从训练加载器获取一批数据
    input_shape = sample_data.shape[1:]  # 获取输入数据形状（通道数、高度、宽度）
    print(f"样本形状: {input_shape}")  # 打印样本形状信息
    
    # 创建模型
    print(f"创建模型: {args.model_type}")  # 打印模型创建信息
    if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:  # 判断是否为自编码器类型模型
        model = get_model(args.model_type, in_channels=input_shape[0])  # 创建自编码器模型，指定输入通道数
    else:
        model = get_model(args.model_type, in_channels=input_shape[0], num_classes=10)  # 创建分类器模型，指定输入通道数和类别数
    model = model.to(device)  # 将模型移至指定计算设备（CPU或GPU）
    print(model)  # 打印模型结构
    
    # 训练模型
    if not args.skip_training:  # 如果不跳过训练阶段
        print(f"开始训练 {args.model_type} 模型...")  # 打印训练开始信息
        
        # 创建训练参数字典
        train_args = {  # 创建包含训练参数的字典
            "device": device,  # 设置计算设备
            "epochs": args.epochs,  # 设置训练轮数
            "lr": args.learning_rate,  # 设置学习率
            "weight_decay": 1e-5,  # 设置权重衰减（L2正则化系数）
            "log_dir": os.path.join(args.results_dir, 'logs'),  # 设置日志目录
            "save_dir": args.model_dir,  # 设置模型保存目录
            "model_name": args.model_type  # 设置模型名称
        }
        
        # 为VAE模型添加标志
        if args.model_type == 'vae':  # 如果是变分自编码器
            train_args["is_vae"] = True  # 添加VAE特定的标志
        
        # 选择适当的训练函数
        if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:  # 如果是自编码器类型
            history = train_autoencoder(  # 调用自编码器训练函数
                model=model,  # 传入模型
                train_loader=train_loader,  # 传入训练数据加载器
                val_loader=val_loader,  # 传入验证数据加载器
                args=train_args  # 传入训练参数
            )
        else:
            history = train_classifier(  # 调用分类器训练函数
                model=model,  # 传入模型
                train_loader=train_loader,  # 传入训练数据加载器
                val_loader=val_loader,  # 传入验证数据加载器
                args=train_args  # 传入训练参数
            )
        
        # 绘制训练历史
        plot_training_history(  # 绘制训练过程中的损失和指标变化曲线
            history,  # 传入训练历史记录
            save_path=os.path.join(model_results_dir, 'training_history.png')  # 设置保存路径
        )
    else:
        # 如果跳过训练，尝试加载已有模型
        model_path = os.path.join(args.model_dir, f"{args.model_type}_best.pth")  # 构建模型文件路径
        if os.path.exists(model_path):  # 检查模型文件是否存在
            print(f"跳过训练，加载已有模型: {model_path}")  # 打印模型加载信息
            model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])  # 加载模型状态字典
        else:
            print(f"错误：找不到已有模型 {model_path}，无法跳过训练并进行评估。请先训练模型。")  # 打印错误信息
            return  # 提前返回，终止函数执行
    
    # 评估模型
    print(f"评估 {args.model_type} 模型...")  # 打印评估开始信息
    if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:  # 如果是自编码器类型
        # 移除不支持的参数
        metrics = evaluate_autoencoder(  # 调用自编码器评估函数
            model=model,  # 传入模型
            test_loader=test_loader,  # 传入测试数据加载器
            device=device  # 传入计算设备
        )
        
        # 使用metrics中的数据可视化自编码器结果
        from evaluate import plot_autoencoder_results  # 导入自编码器结果绘制函数
        
        plot_autoencoder_results(  # 绘制自编码器结果对比图
            metrics['original_images'],  # 传入原始图像
            metrics['reconstructed_images'],  # 传入重建图像
            metrics['reconstruction_errors'],  # 传入重建误差
            save_dir=model_results_dir  # 设置保存目录
        )
        
        print(f"测试集平均重建误差: {metrics['mean_error']:.4f}")  # 打印平均重建误差
    else:
        # 移除不支持的参数
        metrics = evaluate_classifier(  # 调用分类器评估函数
            model=model,  # 传入模型
            test_loader=test_loader,  # 传入测试数据加载器
            device=device  # 传入计算设备
        )
        
        # 使用metrics中的数据绘制混淆矩阵和打印分类报告
        from evaluate import plot_confusion_matrix, print_classification_report  # 导入混淆矩阵和分类报告函数
        
        plot_confusion_matrix(  # 绘制混淆矩阵
            metrics['all_targets'],  # 传入真实标签
            metrics['all_preds'],  # 传入预测标签
            class_names,  # 传入类别名称
            save_path=os.path.join(model_results_dir, 'confusion_matrix.png')  # 设置保存路径
        )
        
        print_classification_report(  # 打印分类报告
            metrics['all_targets'],  # 传入真实标签
            metrics['all_preds'],  # 传入预测标签
            class_names  # 传入类别名称
        )
        
        print(f"测试集准确率: {metrics['accuracy']:.2%}")  # 打印测试集准确率（百分比格式）
        print(f"测试集F1分数 (Macro): {metrics['f1']:.4f}")  # 打印测试集F1分数
    
    # 可视化结果
    print("生成可视化结果...")  # 打印可视化开始信息
    
    # 特征可视化（对所有模型）
    try:  # 尝试进行特征可视化
        visualize_features(  # 调用特征可视化函数
            model=model,  # 传入模型
            data_loader=test_loader,  # 传入测试数据加载器
            device=device,  # 传入计算设备
            method='tsne',  # 设置降维方法为t-SNE
            save_path=os.path.join(model_results_dir, 'tsne_features.png')  # 设置保存路径
        )
    except Exception as e:  # 捕获可能的异常
        print(f"生成特征可视化时出错: {e}. 可能需要检查模型结构或使用 evaluate.py 中的 visualize_model_features 函数。")  # 打印错误信息
    
    # 根据模型类型选择专门的可视化方法
    if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:  # 如果是自编码器类型
        visualize_autoencoder_results(  # 调用自编码器结果可视化函数
            model=model,  # 传入模型
            data_loader=test_loader,  # 传入测试数据加载器
            device=device,  # 传入计算设备
            save_dir=model_results_dir,  # 设置保存目录
            num_samples=10  # 设置样本数量
        )
    elif args.model_type == 'attention_cnn':  # 如果是注意力CNN模型
        visualize_attention(  # 调用注意力可视化函数
            model=model,  # 传入模型
            data_loader=test_loader,  # 传入测试数据加载器
            device=device,  # 传入计算设备
            class_names=class_names,  # 传入类别名称
            save_dir=model_results_dir  # 设置保存目录
        )
        # 额外添加常规预测可视化
        visualize_predictions(  # 调用预测可视化函数
            model=model,  # 传入模型
            data_loader=test_loader,  # 传入测试数据加载器
            class_names=class_names,  # 传入类别名称
            device=device,  # 传入计算设备
            num_samples=16,  # 设置样本数量
            save_path=os.path.join(model_results_dir, 'predictions.png')  # 设置保存路径
        )
    else:  # 对于其他模型类型
        visualize_predictions(  # 调用预测可视化函数
            model=model,  # 传入模型
            data_loader=test_loader,  # 传入测试数据加载器
            class_names=class_names,  # 传入类别名称
            device=device,  # 传入计算设备
            num_samples=16,  # 设置样本数量
            save_path=os.path.join(model_results_dir, 'predictions.png')  # 设置保存路径
        )
    
    print(f"所有结果已保存到 {model_results_dir}")  # 打印结果保存信息


def main():  # 定义主函数
    """主函数"""
    # 获取当前脚本所在目录的上一级目录(auto-vision)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录路径
    
    parser = argparse.ArgumentParser(description='深度学习模型训练、评估与可视化演示')  # 创建命令行参数解析器
    
    # 模型设置
    parser.add_argument('--model-type', type=str, default='basic_cnn',  # 添加模型类型参数
                        choices=['basic_cnn', 'attention_cnn', 'vanilla_ae', 'conv_ae', 'vae'],  # 设置可选的模型类型
                        help='模型类型 (默认: basic_cnn)')  # 添加帮助信息
    
    # 训练设置
    parser.add_argument('--batch-size', type=int, default=64,  # 添加批次大小参数
                        help='批量大小 (默认: 64)')  # 添加帮助信息
    parser.add_argument('--epochs', type=int, default=10,  # 添加训练轮数参数
                        help='训练轮数 (默认: 10)')  # 添加帮助信息
    parser.add_argument('--learning-rate', type=float, default=0.001,  # 添加学习率参数
                        help='学习率 (默认: 0.001)')  # 添加帮助信息
    parser.add_argument('--seed', type=int, default=42,  # 添加随机种子参数
                        help='随机种子 (默认: 42)')  # 添加帮助信息
    parser.add_argument('--no-cuda', action='store_true', default=False,  # 添加禁用CUDA参数
                        help='禁用CUDA')  # 添加帮助信息
    parser.add_argument('--num-workers', type=int, default=4,  # 添加工作线程数参数
                        help='数据加载线程数 (默认: 4)')  # 添加帮助信息
    
    # 目录设置
    parser.add_argument('--data-dir', type=str, default=os.path.join(base_dir, 'data'),  # 添加数据目录参数
                        help='数据目录 (默认: {base_dir}/data)')  # 添加帮助信息
    parser.add_argument('--model-dir', type=str, default=os.path.join(base_dir, 'models'),  # 添加模型目录参数
                        help='模型保存目录 (默认: {base_dir}/models)')  # 添加帮助信息
    parser.add_argument('--results-dir', type=str, default=os.path.join(base_dir, 'results'),  # 添加结果目录参数
                        help='结果保存目录 (默认: {base_dir}/results)')  # 添加帮助信息
    
    # 其他选项
    parser.add_argument('--skip-training', action='store_true', default=False,  # 添加跳过训练参数
                        help='跳过训练，直接加载模型进行评估')  # 添加帮助信息
    
    args = parser.parse_args()  # 解析命令行参数
    
    # 打印参数设置
    print("参数设置:")  # 打印参数设置标题
    for k, v in vars(args).items():  # 遍历所有参数
        print(f"  {k}: {v}")  # 打印每个参数的名称和值
    
    # 训练和评估模型
    train_and_evaluate(args)  # 调用训练和评估函数


if __name__ == '__main__':  # 如果作为主程序运行
    main()  # 调用主函数 