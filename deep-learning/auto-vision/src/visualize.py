"""
结果可视化模块
包含以下功能：
1. 训练曲线绘制
2. 预测结果可视化
3. 特征可视化（t-SNE和PCA）
4. 模型注意力可视化
5. 自编码器重建结果可视化
"""

# 导入操作系统相关的功能，用于处理文件路径和目录创建
import os
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 导入Matplotlib绘图库，用于生成可视化图表
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的函数式接口，提供激活函数和池化操作等
import torch.nn.functional as F
# 导入t-SNE降维算法，用于高维数据的非线性降维可视化
from sklearn.manifold import TSNE
# 导入PCA降维算法，用于高维数据的线性降维可视化
from sklearn.decomposition import PCA
# 导入Seaborn库，提供基于matplotlib的高级统计图形绘制
import seaborn as sns
# 导入进度条库，用于显示长时间运行任务的进度
from tqdm import tqdm


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 包含训练历史的字典
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 创建一个新的图形，设置宽度为15英寸，高度为5英寸，用于容纳多个子图
    plt.figure(figsize=(15, 5))
    
    # 创建第一个子图（1行2列布局的第1个）用于绘制损失曲线
    plt.subplot(1, 2, 1)
    # 绘制训练损失曲线，使用history字典中的train_loss键对应的值
    plt.plot(history['train_loss'], label='Training Loss')
    # 绘制验证损失曲线，使用history字典中的val_loss键对应的值
    plt.plot(history['val_loss'], label='Validation Loss')
    # 设置x轴标签为"Epoch"（训练轮次）
    plt.xlabel('Epoch')
    # 设置y轴标签为"Loss"（损失值）
    plt.ylabel('Loss')
    # 设置图表标题
    plt.title('Training and Validation Loss')
    # 添加图例，显示每条曲线代表的含义
    plt.legend()
    # 添加网格线，使图表更易读
    plt.grid(True)
    
    # 检查history字典中是否包含准确率数据
    if 'train_acc' in history and 'val_acc' in history:
        # 创建第二个子图（1行2列布局的第2个）用于绘制准确率曲线
        plt.subplot(1, 2, 2)
        # 绘制训练准确率曲线
        plt.plot(history['train_acc'], label='Training Accuracy')
        # 绘制验证准确率曲线
        plt.plot(history['val_acc'], label='Validation Accuracy')
        # 设置x轴标签
        plt.xlabel('Epoch')
        # 设置y轴标签，说明单位是百分比
        plt.ylabel('Accuracy (%)')
        # 设置图表标题
        plt.title('Training and Validation Accuracy')
        # 添加图例
        plt.legend()
        # 添加网格线
        plt.grid(True)
    
    # 判断是保存图像还是显示图像
    if save_path:
        # 自动调整子图之间的间距，使布局更紧凑美观
        plt.tight_layout()
        # 将图像保存到指定路径
        plt.savefig(save_path)
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"训练历史曲线已保存到 {save_path}")
    else:
        # 自动调整子图间距
        plt.tight_layout()
        # 显示图像（不保存）
        plt.show()


def visualize_predictions(model, data_loader, class_names, device, num_samples=16, save_path=None):
    """
    可视化模型预测结果
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        class_names: 类别名称
        device: 计算设备
        num_samples: 可视化样本数量
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 将模型设置为评估模式，禁用如dropout等训练特定的层
    model.eval()
    # 初始化空列表，用于存储待可视化的图像
    images = []
    # 初始化空列表，用于存储图像的真实标签
    true_labels = []
    # 初始化空列表，用于存储模型的预测标签
    pred_labels = []
    
    # 使用torch.no_grad()上下文管理器，在预测时不计算梯度，提高速度和减少内存使用
    with torch.no_grad():
        # 遍历数据加载器中的每一批数据
        for inputs, targets in data_loader:
            # 如果已收集足够的样本，则停止遍历
            if len(images) >= num_samples:
                break
                
            # 将输入数据和目标标签移动到指定设备（CPU或GPU）
            inputs, targets = inputs.to(device), targets.to(device)
            # 使用模型对输入数据进行前向传播，获取输出
            outputs = model(inputs)
            # 对输出进行argmax操作，获取每个样本预测概率最高的类别索引
            _, preds = torch.max(outputs, 1)
            
            # 计算当前批次可以取多少样本
            current_batch_size = inputs.size(0)  # 当前批次的样本数
            remaining = num_samples - len(images)  # 还需要多少样本
            samples_to_take = min(remaining, current_batch_size)  # 取两者中的较小值
            
            # 将选取的样本数据添加到收集列表中
            # 将输入数据移回CPU并转换为NumPy数组后添加到images列表
            images.extend(inputs.cpu().numpy()[:samples_to_take])
            # 将真实标签移回CPU并转换为NumPy数组后添加到true_labels列表
            true_labels.extend(targets.cpu().numpy()[:samples_to_take])
            # 将预测标签移回CPU并转换为NumPy数组后添加到pred_labels列表
            pred_labels.extend(preds.cpu().numpy()[:samples_to_take])
    
    # 创建大小为15x12英寸的图形，用于展示多个图像及其预测结果
    plt.figure(figsize=(15, 12))
    # 遍历收集的样本数据
    for i, (image, true_label, pred_label) in enumerate(zip(images, true_labels, pred_labels)):
        # 如果达到指定的样本数量，则停止循环
        if i >= num_samples:
            break
            
        # 转置图像通道，从PyTorch的(C,H,W)格式转为Matplotlib需要的(H,W,C)格式
        # C=通道数，H=高度，W=宽度
        image = np.transpose(image, (1, 2, 0))
        
        # 将图像像素值归一化到[0,1]范围，便于显示
        image = (image - image.min()) / (image.max() - image.min())
        
        # 在图形中创建第i+1个子图（4x4网格布局）
        plt.subplot(4, 4, i + 1)
        # 显示图像
        plt.imshow(image)
        
        # 设置子图标题，显示真实标签和预测标签
        # 如果预测正确，标题显示为绿色；如果预测错误，标题显示为红色
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                 color=title_color)
        # 关闭坐标轴刻度，使图像更清晰
        plt.axis('off')
    
    # 判断是保存图像还是显示图像
    if save_path:
        # 自动调整子图间距
        plt.tight_layout()
        # 将图像保存到指定路径
        plt.savefig(save_path)
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"预测结果可视化已保存到 {save_path}")
    else:
        # 自动调整子图间距
        plt.tight_layout()
        # 显示图像（不保存）
        plt.show()


def visualize_features(model, data_loader, device, method='tsne', save_path=None):
    """
    可视化特征
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        method: 降维方法，'tsne' 或 'pca'
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 将模型设置为评估模式，禁用如dropout等训练特定的层
    model.eval()
    # 初始化空列表，用于存储提取的特征
    features = []
    # 初始化空列表，用于存储对应的标签
    labels = []
    
    # 在不计算梯度的情况下提取特征，提高速度和减少内存使用
    with torch.no_grad():
        # 使用tqdm包装数据加载器，显示进度条
        for inputs, targets in tqdm(data_loader, desc=f"Extracting features for {method}"):
            # 将输入数据和目标标签移动到指定设备（CPU或GPU）
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 根据模型类型采用不同的特征提取方法
            if hasattr(model, 'fc2') and hasattr(model, 'fc1'):
                # 对于CNN模型，通过逐层前向传播提取最后一个全连接层之前的特征
                # 经过第一个卷积层
                x = model.conv1(inputs)
                # 经过第一个批归一化层
                x = model.bn1(x)
                # 应用ReLU激活函数
                x = F.relu(x)
                # 经过第一个池化层
                x = model.pool1(x)
                
                # 经过第二个卷积层
                x = model.conv2(x)
                # 经过第二个批归一化层
                x = model.bn2(x)
                # 应用ReLU激活函数
                x = F.relu(x)
                # 经过第二个池化层
                x = model.pool2(x)
                
                # 经过第三个卷积层
                x = model.conv3(x)
                # 经过第三个批归一化层
                x = model.bn3(x)
                # 应用ReLU激活函数
                x = F.relu(x)
                # 经过第三个池化层
                x = model.pool3(x)
                
                # 将特征图展平为一维向量
                x = x.view(x.size(0), -1)
                # 经过第一个全连接层
                x = model.fc1(x)
                # 应用ReLU激活函数
                x = F.relu(x)
                
                # 将特征移回CPU并转换为NumPy数组
                batch_features = x.cpu().numpy()
            elif hasattr(model, 'encode') and callable(getattr(model, 'encode')):
                # 对于自编码器，直接使用编码器提取潜在特征
                batch_features = model.encode(inputs).cpu().numpy()
            else:
                # 如果无法识别模型类型，打印错误消息并退出函数
                print("无法识别的模型类型，无法提取特征")
                return
            
            # 将当前批次的特征添加到特征列表
            features.append(batch_features)
            # 将当前批次的标签添加到标签列表
            labels.append(targets.cpu().numpy())
    
    # 使用np.vstack合并所有批次的特征，形成一个大矩阵
    features = np.vstack(features)
    # 使用np.concatenate合并所有批次的标签
    labels = np.concatenate(labels)
    
    # 根据指定的方法进行降维
    if method == 'tsne':
        # 如果使用t-SNE方法，打印相关信息
        print("使用t-SNE进行降维...")
        # 使用t-SNE算法将高维特征降至2维，设置随机种子确保结果可复现
        embedding = TSNE(n_components=2, random_state=42).fit_transform(features)
    else:  # pca
        # 如果使用PCA方法，打印相关信息
        print("使用PCA进行降维...")
        # 使用PCA算法将高维特征降至2维，设置随机种子确保结果可复现
        embedding = PCA(n_components=2, random_state=42).fit_transform(features)
    
    # 创建一个新的图形，大小为10x8英寸
    plt.figure(figsize=(10, 8))
    # 使用散点图可视化降维后的特征，点的颜色根据类别标签着色
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.5)
    # 添加颜色条，标明不同颜色对应的类别
    plt.colorbar(scatter, label='Class')
    # 设置图表标题，显示使用的降维方法
    plt.title(f'Feature Visualization using {method.upper()}')
    # 设置x轴标签
    plt.xlabel('Dimension 1')
    # 设置y轴标签
    plt.ylabel('Dimension 2')
    
    # 判断是保存图像还是显示图像
    if save_path:
        # 自动调整图表布局
        plt.tight_layout()
        # 将图像保存到指定路径
        plt.savefig(save_path)
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"特征可视化已保存到 {save_path}")
    else:
        # 自动调整图表布局
        plt.tight_layout()
        # 显示图像（不保存）
        plt.show()


def visualize_attention(model, data_loader, device, class_names, save_dir=None):
    """
    可视化注意力模型
    
    Args:
        model: 带注意力的CNN模型
        data_loader: 数据加载器
        device: 计算设备
        class_names: 类别名称
        save_dir: 保存目录，若为None则显示图像而不保存
    """
    # 检查模型是否有注意力模块，如果没有则打印错误消息并返回
    if not hasattr(model, 'attention'):
        print("该模型没有注意力模块，无法可视化注意力")
        return
    
    # 如果指定了保存目录，确保该目录存在，不存在则创建
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 将模型设置为评估模式，禁用如dropout等训练特定的层
    model.eval()
    
    # 从数据加载器中获取一批数据
    iterator = iter(data_loader)  # 创建数据加载器的迭代器
    inputs, targets = next(iterator)  # 获取第一批数据
    # 将输入数据和目标标签移动到指定设备（CPU或GPU）
    inputs, targets = inputs.to(device), targets.to(device)
    
    # 只处理前8个样本，避免图形过于复杂
    inputs = inputs[:8]
    targets = targets[:8]
    
    # 将原始输入图像保存为NumPy数组，用于后续可视化
    orig_images = inputs.cpu().numpy()
    
    # 创建一个大尺寸的图形，用于并排显示原始图像和注意力图
    plt.figure(figsize=(20, 12))
    
    # 在不计算梯度的情况下获取中间层特征和注意力
    with torch.no_grad():
        # 前向传播到注意力层前
        # 经过第一个卷积层
        x = model.conv1(inputs)
        # 经过第一个批归一化层
        x = model.bn1(x)
        # 应用ReLU激活函数
        x = F.relu(x)
        # 经过第一个池化层
        x = model.pool1(x)
        
        # 经过第二个卷积层
        x = model.conv2(x)
        # 经过第二个批归一化层
        x = model.bn2(x)
        # 应用ReLU激活函数
        x = F.relu(x)
        # 经过第二个池化层
        x = model.pool2(x)
        
        # 保存注意力层前的特征，用于后续计算注意力权重
        pre_attention_features = x.clone()
        
        # 获取经过注意力模块后的输出
        attention_output = model.attention(x)
        
        # 计算注意力权重：注意力模块的输出与输入之间的差异
        attention_weights = attention_output - pre_attention_features
        # 取绝对值并在通道维度上求平均，得到每个空间位置的注意力强度
        attention_weights = attention_weights.abs().mean(dim=1, keepdim=True)
        
        # 对注意力权重进行归一化处理，使其范围在[0,1]之间
        # 找出每个样本的最小注意力值
        attention_min = attention_weights.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        # 找出每个样本的最大注意力值
        attention_max = attention_weights.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        # 使用最大最小值归一化
        attention_weights = (attention_weights - attention_min) / (attention_max - attention_min + 1e-8)
        
        # 使用双线性插值将注意力图上采样到原始图像大小
        attention_weights = F.interpolate(
            attention_weights,  # 输入的注意力权重
            size=(inputs.size(2), inputs.size(3)),  # 目标大小为原始图像的尺寸
            mode='bilinear',  # 使用双线性插值
            align_corners=False  # 不对齐角点，避免边缘效应
        ).cpu().numpy()  # 将结果转换为NumPy数组
    
    # 绘制原始图像和对应的注意力图
    for i in range(len(inputs)):
        # 绘制原始图像
        ax = plt.subplot(2, 8, i + 1)  # 在第一行创建子图
        # 转置图像通道，从(C,H,W)转为(H,W,C)
        image = np.transpose(orig_images[i], (1, 2, 0))
        # 归一化图像像素值到[0,1]范围
        image = (image - image.min()) / (image.max() - image.min())
        # 显示归一化后的图像
        plt.imshow(image)
        # 设置标题显示类别名称
        plt.title(f"Class: {class_names[targets[i]]}")
        # 关闭坐标轴刻度
        plt.axis('off')
        
        # 绘制注意力图
        ax = plt.subplot(2, 8, i + 9)  # 在第二行创建子图
        # 获取当前样本的注意力权重
        attention = attention_weights[i, 0]
        # 使用热力图显示注意力权重
        plt.imshow(attention, cmap='jet')
        # 设置标题
        plt.title("Attention Map")
        # 关闭坐标轴刻度
        plt.axis('off')
    
    # 判断是保存图像还是显示图像
    if save_dir:
        # 自动调整子图布局
        plt.tight_layout()
        # 将图像保存到指定目录下的文件中
        plt.savefig(os.path.join(save_dir, 'attention_visualization.png'))
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"注意力可视化已保存到 {save_dir}")
    else:
        # 自动调整子图布局
        plt.tight_layout()
        # 显示图像（不保存）
        plt.show()


def visualize_autoencoder_results(model, data_loader, device, save_dir=None, num_samples=10):
    """
    可视化自编码器结果
    
    Args:
        model: 自编码器模型
        data_loader: 数据加载器
        device: 计算设备
        save_dir: 保存目录，若为None则显示图像而不保存
        num_samples: 可视化样本数量
    """
    # 检查模型是否有decode方法，如果没有则打印错误消息并返回
    if not hasattr(model, 'decode') or not callable(getattr(model, 'decode')):
        print("该模型不是自编码器，无法可视化重建结果")
        return
    
    # 如果指定了保存目录，确保该目录存在，不存在则创建
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 将模型设置为评估模式，禁用如dropout等训练特定的层
    model.eval()
    
    # 从数据加载器中获取一批数据
    iterator = iter(data_loader)  # 创建数据加载器的迭代器
    inputs, _ = next(iterator)  # 获取第一批数据，忽略标签
    # 将输入数据移动到指定设备（CPU或GPU）
    inputs = inputs.to(device)
    
    # 只处理指定数量的样本
    inputs = inputs[:num_samples]
    
    # 在不计算梯度的情况下进行前向传播，获取重建图像
    with torch.no_grad():
        reconstructed = model(inputs)
    
    # 计算每个样本的重建误差：输入与重建图像之间的均方误差
    # reduction='none'表示不对结果求平均，返回每个元素的误差
    # 然后在各维度上求平均，得到每个样本的总体重建误差
    reconstruction_errors = F.mse_loss(reconstructed, inputs, reduction='none').mean([1, 2, 3]).cpu().numpy()
    
    # 创建一个大尺寸的图形，用于并排显示原始图像和重建图像
    plt.figure(figsize=(20, 4))
    
    # 遍历每个样本，绘制原始图像和对应的重建图像
    for i in range(num_samples):
        # 绘制原始图像
        ax = plt.subplot(2, num_samples, i + 1)  # 在第一行创建子图
        if inputs.size(1) == 1:  # 如果是灰度图像（通道数为1）
            # 直接显示第一个通道的数据
            plt.imshow(inputs[i, 0].cpu().numpy(), cmap='gray')
        else:  # 如果是彩色图像
            # 转置图像通道，从(C,H,W)转为(H,W,C)
            image = np.transpose(inputs[i].cpu().numpy(), (1, 2, 0))
            # 归一化图像像素值到[0,1]范围
            plt.imshow((image - image.min()) / (image.max() - image.min()))
        # 设置标题
        plt.title("Original")
        # 关闭坐标轴刻度
        plt.axis('off')
        
        # 绘制重建图像
        ax = plt.subplot(2, num_samples, i + num_samples + 1)  # 在第二行创建子图
        if reconstructed.size(1) == 1:  # 如果是灰度图像
            # 直接显示第一个通道的数据
            plt.imshow(reconstructed[i, 0].cpu().numpy(), cmap='gray')
        else:  # 如果是彩色图像
            # 转置图像通道
            image = np.transpose(reconstructed[i].cpu().numpy(), (1, 2, 0))
            # 归一化图像像素值
            plt.imshow((image - image.min()) / (image.max() - image.min()))
        # 设置标题，显示重建误差
        plt.title(f"Reconstructed\nError: {reconstruction_errors[i]:.4f}")
        # 关闭坐标轴刻度
        plt.axis('off')
    
    # 判断是保存图像还是显示图像
    if save_dir:
        # 自动调整子图布局
        plt.tight_layout()
        # 将图像保存到指定目录下的文件中
        plt.savefig(os.path.join(save_dir, 'reconstruction_results.png'))
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"重建结果可视化已保存到 {save_dir}")
    else:
        # 自动调整子图布局
        plt.tight_layout()
        # 显示图像（不保存）
        plt.show()
    
    # 尝试可视化潜在空间的分布（仅适用于二维潜在空间或可降维至二维的情况）
    try:
        # 初始化空列表，用于收集潜在向量
        latent_vectors = []
        # 初始化空列表，用于收集对应的标签
        labels = []
        
        # 在不计算梯度的情况下进行潜在空间提取
        with torch.no_grad():
            # 使用tqdm包装数据加载器，显示进度条
            for batch_inputs, batch_labels in tqdm(data_loader, desc="生成潜在空间可视化"):
                # 将输入数据移动到指定设备
                batch_inputs = batch_inputs.to(device)
                # 使用模型的编码器获取潜在向量
                latent = model.encode(batch_inputs)
                # 将潜在向量添加到收集列表
                latent_vectors.append(latent.cpu().numpy())
                # 将标签添加到收集列表
                labels.append(batch_labels.numpy())
        
        # 使用np.vstack合并所有批次的潜在向量，形成一个大矩阵
        latent_vectors = np.vstack(latent_vectors)
        # 使用np.concatenate合并所有批次的标签
        labels = np.concatenate(labels)
        
        # 如果潜在空间维度不是2，则使用t-SNE降维
        if latent_vectors.shape[1] != 2:
            # 打印提示信息
            print(f"潜在空间维度为 {latent_vectors.shape[1]}，使用t-SNE降维...")
            # 使用t-SNE将潜在向量降至2维
            latent_vectors = TSNE(n_components=2, random_state=42).fit_transform(latent_vectors)
        
        # 创建一个新的图形，用于可视化潜在空间分布
        plt.figure(figsize=(10, 8))
        # 使用散点图可视化潜在空间，点的颜色根据类别标签着色
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='viridis', alpha=0.5)
        # 添加颜色条，标明不同颜色对应的类别
        plt.colorbar(scatter, label='Class')
        # 设置图表标题
        plt.title('Latent Space Distribution')
        # 设置x轴标签
        plt.xlabel('Dimension 1')
        # 设置y轴标签
        plt.ylabel('Dimension 2')
        
        # 判断是保存图像还是显示图像
        if save_dir:
            # 自动调整图表布局
            plt.tight_layout()
            # 将图像保存到指定目录下的文件中
            plt.savefig(os.path.join(save_dir, 'latent_space.png'))
            # 关闭当前图形，释放内存
            plt.close()
            # 打印保存成功的消息
            print(f"潜在空间可视化已保存到 {save_dir}")
        else:
            # 自动调整图表布局
            plt.tight_layout()
            # 显示图像（不保存）
            plt.show()
    except Exception as e:
        # 如果在生成潜在空间可视化过程中出错，捕获异常并打印错误消息
        print(f"生成潜在空间可视化时出错: {e}")


def get_visualization_function(model_name):
    """
    根据模型名称获取对应的可视化函数
    
    Args:
        model_name: 模型名称
        
    Returns:
        对应的可视化函数
    """
    # 根据模型名称返回相应的可视化函数
    if model_name == 'autoencoder':
        # 如果是自编码器模型，返回自编码器结果可视化函数
        return visualize_autoencoder_results
    elif model_name == 'attention_cnn':
        # 如果是带注意力的CNN模型，返回注意力可视化函数
        return visualize_attention
    else:
        # 对于其他模型（如普通CNN），返回预测结果可视化函数
        return visualize_predictions 