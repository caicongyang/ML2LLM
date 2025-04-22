"""
结果可视化模块
包含以下功能：
1. 训练曲线绘制
2. 预测结果可视化
3. 特征可视化（t-SNE和PCA）
4. 模型注意力可视化
5. 自编码器重建结果可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 包含训练历史的字典
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 创建图像
    plt.***REMOVED***gure(***REMOVED***gsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线（如果有）
    if 'train_acc' in history and 'val_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    # 保存或显示图像
    if save_path:
        plt.tight_layout()
        plt.save***REMOVED***g(save_path)
        plt.close()
        print(f"训练历史曲线已保存到 {save_path}")
    else:
        plt.tight_layout()
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
    model.eval()
    images = []
    true_labels = []
    pred_labels = []
    
    # 获取样本和预测结果
    with torch.no_grad():
        for inputs, targets in data_loader:
            if len(images) >= num_samples:
                break
                
            # 前向传播
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 收集样本
            current_batch_size = inputs.size(0)
            remaining = num_samples - len(images)
            samples_to_take = min(remaining, current_batch_size)
            
            images.extend(inputs.cpu().numpy()[:samples_to_take])
            true_labels.extend(targets.cpu().numpy()[:samples_to_take])
            pred_labels.extend(preds.cpu().numpy()[:samples_to_take])
    
    # 绘制预测结果
    plt.***REMOVED***gure(***REMOVED***gsize=(15, 12))
    for i, (image, true_label, pred_label) in enumerate(zip(images, true_labels, pred_labels)):
        if i >= num_samples:
            break
            
        # 转置图像通道 (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # 图像归一化到 [0, 1] 范围
        image = (image - image.min()) / (image.max() - image.min())
        
        # 绘制图像
        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        
        # 设置标题：显示真实标签和预测标签
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                 color=title_color)
        plt.axis('off')
    
    # 保存或显示图像
    if save_path:
        plt.tight_layout()
        plt.save***REMOVED***g(save_path)
        plt.close()
        print(f"预测结果可视化已保存到 {save_path}")
    else:
        plt.tight_layout()
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
    model.eval()
    features = []
    labels = []
    
    # 提取特征
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Extracting features for {method}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 对于CNN模型，提取最后一个全连接层之前的特征
            if hasattr(model, 'fc2') and hasattr(model, 'fc1'):
                # 前向传播到倒数第二层
                x = model.conv1(inputs)
                x = model.bn1(x)
                x = F.relu(x)
                x = model.pool1(x)
                
                x = model.conv2(x)
                x = model.bn2(x)
                x = F.relu(x)
                x = model.pool2(x)
                
                x = model.conv3(x)
                x = model.bn3(x)
                x = F.relu(x)
                x = model.pool3(x)
                
                x = x.view(x.size(0), -1)
                x = model.fc1(x)
                x = F.relu(x)
                
                batch_features = x.cpu().numpy()
            elif hasattr(model, 'encode') and callable(getattr(model, 'encode')):
                # 自编码器提取潜在特征
                batch_features = model.encode(inputs).cpu().numpy()
            else:
                # 不知道如何提取特征，跳过
                print("无法识别的模型类型，无法提取特征")
                return
            
            features.append(batch_features)
            labels.append(targets.cpu().numpy())
    
    # 合并所有批次的特征和标签
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # 降维
    if method == 'tsne':
        print("使用t-SNE进行降维...")
        embedding = TSNE(n_components=2, random_state=42).***REMOVED***t_transform(features)
    else:  # pca
        print("使用PCA进行降维...")
        embedding = PCA(n_components=2, random_state=42).***REMOVED***t_transform(features)
    
    # 绘制降维结果
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Class')
    plt.title(f'Feature Visualization using {method.upper()}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # 保存或显示图像
    if save_path:
        plt.tight_layout()
        plt.save***REMOVED***g(save_path)
        plt.close()
        print(f"特征可视化已保存到 {save_path}")
    else:
        plt.tight_layout()
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
    if not hasattr(model, 'attention'):
        print("该模型没有注意力模块，无法可视化注意力")
        return
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # 获取一批数据
    iterator = iter(data_loader)
    inputs, targets = next(iterator)
    inputs, targets = inputs.to(device), targets.to(device)
    
    # 只处理前8个样本
    inputs = inputs[:8]
    targets = targets[:8]
    
    # 保存原始输入图像
    orig_images = inputs.cpu().numpy()
    
    # 绘制注意力图
    plt.***REMOVED***gure(***REMOVED***gsize=(20, 12))
    
    # 获取中间层特征和注意力
    with torch.no_grad():
        # 前向传播到注意力层
        x = model.conv1(inputs)
        x = model.bn1(x)
        x = F.relu(x)
        x = model.pool1(x)
        
        x = model.conv2(x)
        x = model.bn2(x)
        x = F.relu(x)
        x = model.pool2(x)
        
        # 保存注意力前的特征
        pre_attention_features = x.clone()
        
        # 获取注意力输出
        attention_output = model.attention(x)
        
        # 计算注意力权重
        attention_weights = attention_output - pre_attention_features
        attention_weights = attention_weights.abs().mean(dim=1, keepdim=True)
        
        # 归一化注意力权重
        attention_min = attention_weights.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        attention_max = attention_weights.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        attention_weights = (attention_weights - attention_min) / (attention_max - attention_min + 1e-8)
        
        # 上采样注意力图到原始图像大小
        attention_weights = F.interpolate(
            attention_weights, 
            size=(inputs.size(2), inputs.size(3)),
            mode='bilinear', 
            align_corners=False
        ).cpu().numpy()
    
    # 绘制原始图像和注意力图
    for i in range(len(inputs)):
        # 原始图像
        ax = plt.subplot(2, 8, i + 1)
        image = np.transpose(orig_images[i], (1, 2, 0))
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image)
        plt.title(f"Class: {class_names[targets[i]]}")
        plt.axis('off')
        
        # 注意力图
        ax = plt.subplot(2, 8, i + 9)
        attention = attention_weights[i, 0]
        plt.imshow(attention, cmap='jet')
        plt.title("Attention Map")
        plt.axis('off')
    
    # 保存或显示图像
    if save_dir:
        plt.tight_layout()
        plt.save***REMOVED***g(os.path.join(save_dir, 'attention_visualization.png'))
        plt.close()
        print(f"注意力可视化已保存到 {save_dir}")
    else:
        plt.tight_layout()
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
    if not hasattr(model, 'decode') or not callable(getattr(model, 'decode')):
        print("该模型不是自编码器，无法可视化重建结果")
        return
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # 获取一批数据
    iterator = iter(data_loader)
    inputs, _ = next(iterator)
    inputs = inputs.to(device)
    
    # 只处理指定数量的样本
    inputs = inputs[:num_samples]
    
    # 前向传播获取重建图像
    with torch.no_grad():
        reconstructed = model(inputs)
    
    # 计算重建误差
    reconstruction_errors = F.mse_loss(reconstructed, inputs, reduction='none').mean([1, 2, 3]).cpu().numpy()
    
    # 绘制原始图像和重建图像对比
    plt.***REMOVED***gure(***REMOVED***gsize=(20, 4))
    
    for i in range(num_samples):
        # 原始图像
        ax = plt.subplot(2, num_samples, i + 1)
        if inputs.size(1) == 1:  # 灰度图像
            plt.imshow(inputs[i, 0].cpu().numpy(), cmap='gray')
        else:  # 彩色图像
            image = np.transpose(inputs[i].cpu().numpy(), (1, 2, 0))
            plt.imshow((image - image.min()) / (image.max() - image.min()))
        plt.title("Original")
        plt.axis('off')
        
        # 重建图像
        ax = plt.subplot(2, num_samples, i + num_samples + 1)
        if reconstructed.size(1) == 1:  # 灰度图像
            plt.imshow(reconstructed[i, 0].cpu().numpy(), cmap='gray')
        else:  # 彩色图像
            image = np.transpose(reconstructed[i].cpu().numpy(), (1, 2, 0))
            plt.imshow((image - image.min()) / (image.max() - image.min()))
        plt.title(f"Reconstructed\nError: {reconstruction_errors[i]:.4f}")
        plt.axis('off')
    
    # 保存或显示图像
    if save_dir:
        plt.tight_layout()
        plt.save***REMOVED***g(os.path.join(save_dir, 'reconstruction_results.png'))
        plt.close()
        print(f"重建结果可视化已保存到 {save_dir}")
    else:
        plt.tight_layout()
        plt.show()
    
    # 可视化潜在空间的分布（仅适用于二维潜在空间）
    try:
        # 收集潜在向量
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_labels in tqdm(data_loader, desc="生成潜在空间可视化"):
                batch_inputs = batch_inputs.to(device)
                # 编码获取潜在向量
                latent = model.encode(batch_inputs)
                latent_vectors.append(latent.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        # 合并所有批次
        latent_vectors = np.vstack(latent_vectors)
        labels = np.concatenate(labels)
        
        # 如果潜在空间维度不是2，则使用t-SNE降维
        if latent_vectors.shape[1] != 2:
            print(f"潜在空间维度为 {latent_vectors.shape[1]}，使用t-SNE降维...")
            latent_vectors = TSNE(n_components=2, random_state=42).***REMOVED***t_transform(latent_vectors)
        
        # 绘制潜在空间分布
        plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Class')
        plt.title('Latent Space Distribution')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # 保存或显示图像
        if save_dir:
            plt.tight_layout()
            plt.save***REMOVED***g(os.path.join(save_dir, 'latent_space.png'))
            plt.close()
            print(f"潜在空间可视化已保存到 {save_dir}")
        else:
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"生成潜在空间可视化时出错: {e}")


def get_visualization_function(model_name):
    """
    根据模型名称获取对应的可视化函数
    
    Args:
        model_name: 模型名称
        
    Returns:
        对应的可视化函数
    """
    if model_name == 'autoencoder':
        return visualize_autoencoder_results
    elif model_name == 'attention_cnn':
        return visualize_attention
    else:
        return visualize_predictions 