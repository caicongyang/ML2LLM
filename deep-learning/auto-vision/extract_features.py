#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自编码器特征提取示例脚本
展示如何使用自编码器作为特征提取器，并用提取的特征训练简单分类器
该脚本首先加载预训练的自编码器模型，利用其编码器部分从图像中提取特征，
然后使用这些特征训练不同类型的分类器（SVM、随机森林、MLP），比较分类效果。
"""

# 导入必要的库
import os  # 用于文件和目录操作
import argparse  # 用于解析命令行参数
import numpy as np  # 用于数值计算和数组操作
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # PyTorch优化器模块
from torch.utils.data import DataLoader, TensorDataset  # 用于数据加载和数据集创建
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.ensemble import RandomForestClassi***REMOVED***er  # 随机森林分类器
from sklearn.metrics import accuracy_score, classi***REMOVED***cation_report, confusion_matrix  # 用于模型评估
import seaborn as sns  # 用于高级可视化
from tqdm import tqdm  # 用于进度条显示

# 从本地模块导入必要的函数
from src.models import get_model  # 获取模型的函数
from src.data_utils import get_data_loaders  # 获取数据加载器的函数


def extract_features(model, data_loader, device, latent_dim):
    """
    使用自编码器提取特征
    
    Args:
        model: 训练好的自编码器模型
        data_loader: 数据加载器
        device: 计算设备
        latent_dim: 潜在空间维度
        
    Returns:
        tuple: (特征, 标签)
    """
    # 将模型设置为评估模式，禁用dropout和批归一化的训练行为
    model.eval()
    # 初始化空列表，用于存储特征和标签
    features = []
    labels = []
    
    # 使用torch.no_grad()上下文管理器，避免计算梯度，节省内存
    with torch.no_grad():
        # 遍历数据加载器中的每一批数据
        for inputs, targets in tqdm(data_loader, desc="提取特征"):
            # 将输入数据移动到指定设备（CPU或GPU）
            inputs = inputs.to(device)
            
            # 根据模型类型提取特征
            if hasattr(model, 'encode'):
                # 检查模型是否有encode属性（VAE或标准自编码器）
                if isinstance(model.encode, torch.nn.Module):
                    # 如果encode是一个模块，直接调用它
                    z = model.encode(inputs)
                    # 处理VAE的情况，它返回均值和对数方差
                    if isinstance(z, tuple):
                        # 如果结果是元组，使用均值（第一个元素）作为特征
                        z = z[0]  # 使用均值作为特征
                else:
                    # 如果encode是一个方法，调用该方法
                    z = model.encode(inputs)
            else:
                # 卷积自编码器，使用编码器部分提取特征
                z = model.encoder(inputs)
                # 将特征展平为一维向量，便于后续处理
                z = z.view(z.size(0), -1)  # 将特征展平
            
            # 将提取的特征和对应标签添加到列表中
            # 将tensor转移到CPU并转换为numpy数组
            features.append(z.cpu().numpy())
            labels.append(targets.numpy())
    
    # 合并所有批次的特征和标签
    # np.vstack垂直堆叠数组，将所有批次的特征连接成一个大数组
    features = np.vstack(features)
    # np.concatenate连接数组，将所有批次的标签连接成一个大数组
    labels = np.concatenate(labels)
    
    # 返回提取的特征和对应的标签
    return features, labels


def train_classi***REMOVED***er(X_train, y_train, X_val, y_val, classi***REMOVED***er_type='svm'):
    """
    训练分类器
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        classi***REMOVED***er_type: 分类器类型，'svm'或'rf'
        
    Returns:
        训练好的分类器和特征缩放器
    """
    # 打印当前训练的分类器类型
    print(f"训练{classi***REMOVED***er_type}分类器...")
    
    # 标准化特征，使其均值为0，方差为1
    scaler = StandardScaler()
    # 对训练数据进行拟合和转换
    X_train_scaled = scaler.***REMOVED***t_transform(X_train)
    # 对验证数据仅进行转换（使用训练数据的统计量）
    X_val_scaled = scaler.transform(X_val)
    
    # 根据指定的分类器类型选择并训练分类器
    if classi***REMOVED***er_type == 'svm':
        # 创建支持向量机分类器，使用径向基函数核
        classi***REMOVED***er = SVC(kernel='rbf', C=10, gamma='scale')
        # 使用训练数据拟合分类器
        classi***REMOVED***er.***REMOVED***t(X_train_scaled, y_train)
    else:  # Random Forest
        # 创建随机森林分类器，使用100个决策树
        classi***REMOVED***er = RandomForestClassi***REMOVED***er(n_estimators=100, random_state=42)
        # 使用训练数据拟合分类器
        classi***REMOVED***er.***REMOVED***t(X_train_scaled, y_train)
    
    # 评估分类器在训练集和验证集上的性能
    # 在训练集上进行预测
    train_pred = classi***REMOVED***er.predict(X_train_scaled)
    # 在验证集上进行预测
    val_pred = classi***REMOVED***er.predict(X_val_scaled)
    
    # 计算训练集准确率
    train_acc = accuracy_score(y_train, train_pred)
    # 计算验证集准确率
    val_acc = accuracy_score(y_val, val_pred)
    
    # 打印训练集和验证集的准确率
    print(f"训练准确率: {train_acc:.4f}")
    print(f"验证准确率: {val_acc:.4f}")
    
    # 返回训练好的分类器和特征缩放器
    return classi***REMOVED***er, scaler


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    # 使用sklearn计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建一个新的图形，设置大小为10x8英寸
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    # 使用seaborn绘制热力图可视化混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # 设置x轴标签
    plt.xlabel('预测标签')
    # 设置y轴标签
    plt.ylabel('真实标签')
    # 设置图表标题
    plt.title('混淆矩阵')
    
    # 判断是否需要保存图形
    if save_path:
        # 调整图形布局
        plt.tight_layout()
        # 保存图形到指定路径
        plt.save***REMOVED***g(save_path)
        # 关闭当前图形
        plt.close()
        # 打印保存成功的消息
        print(f"混淆矩阵已保存到 {save_path}")
    else:
        # 如果不保存，则显示图形
        plt.show()


def main():
    """
    主函数，处理命令行参数，加载模型和数据，提取特征，训练分类器并评估性能
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='自编码器特征提取示例')
    # 添加模型类型参数，默认为vanilla_ae
    parser.add_argument('--model-type', type=str, default='vanilla_ae', 
                        choices=['vanilla_ae', 'conv_ae', 'vae'], 
                        help='自编码器模型类型')
    # 添加潜在空间维度参数，默认为128
    parser.add_argument('--latent-dim', type=int, default=128, 
                        help='潜在空间维度')
    # 添加预训练模型路径参数
    parser.add_argument('--model-path', type=str, default=None, 
                        help='预训练模型路径')
    # 添加分类器类型参数，默认为svm
    parser.add_argument('--classi***REMOVED***er', type=str, default='svm', 
                        choices=['svm', 'rf'], 
                        help='分类器类型')
    # 添加数据目录参数
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='数据目录')
    # 添加输出目录参数
    parser.add_argument('--output-dir', type=str, default='./output', 
                        help='输出目录')
    # 添加批次大小参数
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='批次大小')
    # 添加设备参数，默认使用CUDA(如果可用)，否则使用CPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备 (cuda|cpu)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建必要的目录
    # 创建数据目录
    os.makedirs(args.data_dir, exist_ok=True)
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 创建结果目录
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    # 设置计算设备
    device = torch.device(args.device)
    # 打印使用的设备信息
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    # 使用get_data_loaders函数获取训练、验证和测试数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,  # 数据目录
        batch_size=args.batch_size,  # 批次大小
        use_cifar=True  # 使用CIFAR-10数据集
    )
    
    # 创建模型
    print(f"创建{args.model_type}模型...")
    # 根据模型类型创建相应的自编码器模型
    if args.model_type in ['vanilla_ae', 'vae']:
        # 对于vanilla_ae和vae，指定输入通道数和潜在空间维度
        model = get_model(args.model_type, in_channels=3, latent_dim=args.latent_dim)
    else:
        # 对于conv_ae，只需指定输入通道数
        model = get_model(args.model_type, in_channels=3)
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 确定模型路径
    if args.model_path:
        # 如果提供了模型路径，直接使用
        model_path = args.model_path
    else:
        # 否则，使用默认路径
        model_path = os.path.join(args.output_dir, 'models', f"{args.model_type}_best.pth")
    
    # 加载预训练模型
    if os.path.exists(model_path):
        # 如果模型文件存在，打印加载信息
        print(f"加载预训练模型: {model_path}")
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        # 将权重加载到模型中
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果模型文件不存在，打印错误信息并退出
        print(f"未找到预训练模型: {model_path}")
        print("请先使用 run_autoencoder.py 训练自编码器模型")
        return
    
    # 提取特征
    # 从训练集提取特征
    print("从训练集提取特征...")
    X_train, y_train = extract_features(model, train_loader, device, args.latent_dim)
    
    # 从验证集提取特征
    print("从验证集提取特征...")
    X_val, y_val = extract_features(model, val_loader, device, args.latent_dim)
    
    # 从测试集提取特征
    print("从测试集提取特征...")
    X_test, y_test = extract_features(model, test_loader, device, args.latent_dim)
    
    # 打印特征的形状，了解特征维度
    print(f"特征形状: {X_train.shape}")
    
    # 训练分类器（SVM或随机森林）
    classi***REMOVED***er, scaler = train_classi***REMOVED***er(X_train, y_train, X_val, y_val, classi***REMOVED***er_type=args.classi***REMOVED***er)
    
    # 评估分类器在测试集上的性能
    # 对测试集特征进行标准化
    X_test_scaled = scaler.transform(X_test)
    # 使用分类器进行预测
    test_pred = classi***REMOVED***er.predict(X_test_scaled)
    # 计算测试集准确率
    test_acc = accuracy_score(y_test, test_pred)
    
    # 打印测试结果
    print(f"\n测试准确率: {test_acc:.4f}")
    print("\n分类报告:")
    # CIFAR-10数据集的类别名称
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    # 打印详细的分类报告
    print(classi***REMOVED***cation_report(y_test, test_pred, target_names=class_names))
    
    # 创建保存结果的目录
    save_dir = os.path.join(args.output_dir, 'results', f"{args.model_type}_{args.classi***REMOVED***er}")
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制混淆矩阵并保存
    plot_confusion_matrix(
        y_test, test_pred, class_names, 
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # 比较与原始数据的分类效果
    print("\n比较与原始数据的分类效果...")
    
    # 定义一个简单的MLP分类器类
    class SimpleClassi***REMOVED***er(nn.Module):
        """
        简单多层感知机(MLP)分类器
        
        这是一个用于特征分类的简单前馈神经网络。该网络由三个全连接层组成，
        中间使用ReLU激活函数和Dropout正则化。适用于对预先提取的特征进行分类，
        如自编码器生成的潜在向量。
        
        网络结构:
            1. 全连接层 (input_dim -> 512) + ReLU
            2. Dropout (p=0.5)
            3. 全连接层 (512 -> 256) + ReLU
            4. Dropout (p=0.5)
            5. 全连接层 (256 -> 10) - 输出层
            
        参数:
            input_dim (int): 输入特征的维度，通常是自编码器的潜在维度
        """
        def __init__(self, input_dim):
            """
            初始化SimpleClassi***REMOVED***er类
            
            Args:
                input_dim (int): 输入特征的维度
            """
            # 调用父类的初始化方法
            super(SimpleClassi***REMOVED***er, self).__init__()
            # 第一个全连接层，将输入特征映射到512维
            self.fc1 = nn.Linear(input_dim, 512)
            # 第二个全连接层，将512维特征映射到256维
            self.fc2 = nn.Linear(512, 256)
            # 输出层，将256维特征映射到10个类别（CIFAR-10数据集）
            self.fc3 = nn.Linear(256, 10)
            # Dropout层，用于防止过拟合，丢弃率为0.5
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            """
            前向传播函数
            
            参数:
                x (torch.Tensor): 输入特征张量，形状为 [batch_size, input_dim]
                
            返回:
                torch.Tensor: 类别预测logits，形状为 [batch_size, 10]
            """
            # 第一层：线性变换+ReLU激活+Dropout
            # 首先通过第一个全连接层
            x = self.fc1(x)
            # 应用ReLU激活函数
            x = torch.relu(x)
            # 应用Dropout正则化
            x = self.dropout(x)
            
            # 第二层：线性变换+ReLU激活+Dropout
            # 通过第二个全连接层
            x = self.fc2(x)
            # 应用ReLU激活函数
            x = torch.relu(x)
            # 应用Dropout正则化
            x = self.dropout(x)
            
            # 输出层：线性变换（不使用激活函数，因为后续通常会应用交叉熵损失函数）
            x = self.fc3(x)
            # 返回输出
            return x
    
    # 将numpy数组转换为PyTorch张量，以便创建PyTorch数据集
    # 转换训练特征
    X_train_tensor = torch.FloatTensor(X_train)
    # 转换训练标签
    y_train_tensor = torch.LongTensor(y_train)
    # 转换验证特征
    X_val_tensor = torch.FloatTensor(X_val)
    # 转换验证标签
    y_val_tensor = torch.LongTensor(y_val)
    # 转换测试特征
    X_test_tensor = torch.FloatTensor(X_test)
    # 转换测试标签
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建PyTorch数据集
    # 训练数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # 验证数据集
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # 测试数据集
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    # 训练数据加载器
    train_loader_features = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 验证数据加载器
    val_loader_features = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # 测试数据加载器
    test_loader_features = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 训练MLP分类器
    print("使用自编码器特征训练MLP分类器...")
    # 创建MLP分类器实例，输入维度为特征的维度
    mlp = SimpleClassi***REMOVED***er(X_train.shape[1]).to(device)
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 定义优化器为Adam，学习率为0.001
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    
    # 训练5个轮次
    for epoch in range(5):
        # 将模型设置为训练模式，启用dropout等
        mlp.train()
        # 初始化训练损失
        train_loss = 0.0
        # 初始化正确预测的样本数
        correct = 0
        # 初始化总样本数
        total = 0
        
        # 遍历训练数据加载器中的每一批数据
        for inputs, targets in train_loader_features:
            # 将输入和目标移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清空优化器中的梯度
            optimizer.zero_grad()
            # 前向传播，获取输出
            outputs = mlp(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            
            # 累加批次损失
            train_loss += loss.item()
            # 获取预测结果
            _, predicted = outputs.max(1)
            # 累加批次样本数
            total += targets.size(0)
            # 累加正确预测数
            correct += predicted.eq(targets).sum().item()
        
        # 验证阶段
        # 将模型设置为评估模式，禁用dropout等
        mlp.eval()
        # 初始化验证损失
        val_loss = 0.0
        # 初始化验证正确预测的样本数
        val_correct = 0
        # 初始化验证总样本数
        val_total = 0
        
        # 在不计算梯度的情况下进行前向传播
        with torch.no_grad():
            # 遍历验证数据加载器中的每一批数据
            for inputs, targets in val_loader_features:
                # 将输入和目标移动到指定设备
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播，获取输出
                outputs = mlp(inputs)
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 累加批次损失
                val_loss += loss.item()
                # 获取预测结果
                _, predicted = outputs.max(1)
                # 累加批次样本数
                val_total += targets.size(0)
                # 累加正确预测数
                val_correct += predicted.eq(targets).sum().item()
        
        # 打印每个轮次的训练和验证结果
        print(f"Epoch {epoch+1}/5 - 训练损失: {train_loss/len(train_loader_features):.4f}, "
              f"训练准确率: {100.*correct/total:.2f}%, "
              f"验证损失: {val_loss/len(val_loader_features):.4f}, "
              f"验证准确率: {100.*val_correct/val_total:.2f}%")
    
    # 评估MLP分类器在测试集上的性能
    # 将模型设置为评估模式
    mlp.eval()
    # 初始化测试正确预测的样本数
    test_correct = 0
    # 初始化测试总样本数
    test_total = 0
    
    # 在不计算梯度的情况下进行前向传播
    with torch.no_grad():
        # 遍历测试数据加载器中的每一批数据
        for inputs, targets in test_loader_features:
            # 将输入和目标移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播，获取输出
            outputs = mlp(inputs)
            # 获取预测结果
            _, predicted = outputs.max(1)
            # 累加批次样本数
            test_total += targets.size(0)
            # 累加正确预测数
            test_correct += predicted.eq(targets).sum().item()
    
    # 打印MLP分类器在测试集上的准确率
    print(f"\nMLP分类器测试准确率: {100.*test_correct/test_total:.2f}%")
    # 打印比较信息，对比MLP和之前训练的分类器(SVM或随机森林)的性能
    print(f"对比: 使用{args.classi***REMOVED***er}分类器的测试准确率为 {test_acc*100:.2f}%")
    
    # 打印实验完成的消息
    print(f"特征提取和分类实验完成! 结果已保存到 {save_dir}")


# 判断是否直接运行该脚本
if __name__ == "__main__":
    # 调用主函数
    main() 