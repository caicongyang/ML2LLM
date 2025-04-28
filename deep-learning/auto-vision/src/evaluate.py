"""
模型评估模块
包含以下功能：
1. 分类模型评估（准确率、精确率、召回率、F1得分）
2. 自编码器模型评估（重建误差、异常检测）
3. 混淆矩阵计算和可视化
"""

# 导入操作系统相关功能，用于文件和目录操作
import os
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 导入Matplotlib绘图库，用于生成可视化图表
import matplotlib.pyplot as plt
# 导入PyTorch库，深度学习框架的核心
import torch
# 导入PyTorch的函数式接口，提供激活函数、损失函数等
import torch.nn.functional as F
# 从scikit-learn导入分类评估指标：准确率、精确率、召回率、F1分数
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 从scikit-learn导入混淆矩阵、分类报告、ROC曲线和AUC计算工具
from sklearn.metrics import confusion_matrix, classi***REMOVED***cation_report, roc_curve, auc
# 导入Seaborn库，提供基于matplotlib的高级统计图形绘制
import seaborn as sns
# 导入进度条库，用于显示长时间运行任务的进度
from tqdm import tqdm
# 从本地的visualize模块导入可视化函数
from visualize import (
    visualize_predictions,  # 可视化预测结果
    visualize_features,  # 可视化特征
    visualize_attention,  # 可视化注意力图
    visualize_autoencoder_results  # 可视化自编码器结果
)


def evaluate_classi***REMOVED***er(model, test_loader, device):
    """
    评估分类器模型
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        包含各种评估指标的字典
    """
    # 将模型设置为评估模式，禁用dropout和批归一化的更新
    model.eval()
    # 初始化列表，用于存储所有预测结果
    all_preds = []
    # 初始化列表，用于存储所有真实标签
    all_targets = []
    # 初始化测试损失
    test_loss = 0.0
    
    # 使用torch.no_grad()上下文管理器，不计算梯度，以节省内存并加速计算
    with torch.no_grad():
        # 使用tqdm包装测试数据加载器，以显示评估进度条
        for inputs, targets in tqdm(test_loader, desc="正在评估"):
            # 将输入数据和真实标签移动到指定的计算设备（如GPU或CPU）
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 对输入数据进行前向传播，获取模型输出
            outputs = model(inputs)
            # 使用交叉熵损失函数计算模型输出与真实标签之间的损失
            loss = F.cross_entropy(outputs, targets)
            
            # 累加损失，乘以批次大小以得到批次总损失
            test_loss += loss.item() * inputs.size(0)
            
            # 获取预测结果（概率最高的类别的索引）
            _, preds = torch.max(outputs, 1)
            
            # 将当前批次的预测结果和真实标签收集到列表中
            # 需要将tensor从计算设备移到CPU，并转换为NumPy数组
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算整个测试集的平均损失
    test_loss = test_loss / len(test_loader.dataset)
    
    # 计算各种评估指标
    # 创建一个字典来存储评估结果
    metrics = {
        'loss': test_loss,  # 平均损失
        'accuracy': accuracy_score(all_targets, all_preds),  # 准确率
        'precision': precision_score(all_targets, all_preds, average='macro'),  # 精确率（宏平均）
        'recall': recall_score(all_targets, all_preds, average='macro'),  # 召回率（宏平均）
        'f1': f1_score(all_targets, all_preds, average='macro'),  # F1分数（宏平均）
        'all_preds': all_preds,  # 所有预测标签的列表
        'all_targets': all_targets  # 所有真实标签的列表
    }
    
    # 返回包含评估指标的字典
    return metrics


def evaluate_autoencoder(model, test_loader, device, threshold=None):
    """
    评估自编码器模型
    
    Args:
        model: 待评估的自编码器模型
        test_loader: 测试数据加载器
        device: 计算设备
        threshold: 异常检测阈值，若为None则自动计算
        
    Returns:
        包含重建误差和异常检测结果的字典
    """
    # 将模型设置为评估模式
    model.eval()
    # 初始化列表，用于存储每个样本的重建误差
    reconstruction_errors = []
    # 初始化列表，用于存储部分原始图像（用于可视化）
    original_images = []
    # 初始化列表，用于存储部分重建图像（用于可视化）
    reconstructed_images = []
    
    # 在不计算梯度的情况下进行前向传播
    with torch.no_grad():
        # 使用tqdm包装测试数据加载器，显示评估进度条
        for inputs, _ in tqdm(test_loader, desc="正在评估自编码器"):
            # 自编码器评估通常不需要标签，因此用_忽略
            # 将输入数据移动到指定设备
            inputs = inputs.to(device)
            
            # 前向传播获取重建图像
            outputs = model(inputs)
            
            # 计算每个样本的重建误差（均方误差）
            # reduction='none'表示不进行聚合，返回每个元素的误差
            # .mean([1, 2, 3])在通道、高度、宽度维度上求平均，得到每个样本的误差
            batch_errors = F.mse_loss(outputs, inputs, reduction='none').mean([1, 2, 3])
            
            # 将当前批次的重建误差收集到列表中
            # 需要将tensor从计算设备移到CPU，并转换为NumPy数组
            reconstruction_errors.extend(batch_errors.cpu().numpy())
            
            # 收集部分原始图像和重建图像，用于后续可视化（最多20张）
            # 将图像从计算设备移回CPU
            original_images.extend(inputs.cpu())
            reconstructed_images.extend(outputs.cpu())
    
    # 将重建误差列表转换为NumPy数组，方便后续计算
    reconstruction_errors = np.array(reconstruction_errors)
    
    # 如果没有提供异常检测阈值
    if threshold is None:
        # 使用重建误差的95%分位数作为默认阈值
        threshold = np.percentile(reconstruction_errors, 95)
    
    # 根据阈值判断哪些样本是异常（重建误差大于阈值）
    anomalies = reconstruction_errors > threshold
    
    # 整理评估结果到字典中
    results = {
        'reconstruction_errors': reconstruction_errors,  # 所有样本的重建误差数组
        'threshold': threshold,  # 使用的异常检测阈值
        'anomalies': anomalies,  # 布尔数组，标记每个样本是否为异常
        'anomaly_ratio': np.mean(anomalies),  # 异常样本的比例
        'mean_error': np.mean(reconstruction_errors),  # 平均重建误差
        'max_error': np.max(reconstruction_errors),  # 最大重建误差
        'min_error': np.min(reconstruction_errors),  # 最小重建误差
        'original_images': original_images[:20],  # 保存前20张原始图像用于可视化
        'reconstructed_images': reconstructed_images[:20]  # 保存前20张重建图像用于可视化
    }
    
    # 返回包含评估结果的字典
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 使用scikit-learn计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建一个新的图形，设置大小为10x8英寸
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    # 使用Seaborn的heatmap函数绘制混淆矩阵的热力图
    # annot=True表示在单元格中显示数值
    # fmt='d'表示数值格式为整数
    # cmap='Blues'设置颜色映射为蓝色系
    # xticklabels和yticklabels设置轴标签为类别名称
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # 设置x轴标签
    plt.xlabel('预测标签')
    # 设置y轴标签
    plt.ylabel('真实标签')
    # 设置图表标题
    plt.title('混淆矩阵')
    
    # 判断是否需要保存图像
    if save_path:
        # 自动调整子图布局，防止标签重叠
        plt.tight_layout()
        # 将图像保存到指定的路径
        plt.save***REMOVED***g(save_path)
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"混淆矩阵已保存到 {save_path}")
    else:
        # 如果不保存，则直接显示图像
        plt.show()


def plot_autoencoder_results(original_images, reconstructed_images, reconstruction_errors, save_dir=None):
    """
    绘制自编码器结果
    
    Args:
        original_images: 原始图像列表
        reconstructed_images: 重建图像列表
        reconstruction_errors: 重建误差列表
        save_dir: 保存目录，若为None则显示图像而不保存
    """
    # 如果指定了保存目录
    if save_dir:
        # 创建保存目录，如果目录已存在则不报错
        os.makedirs(save_dir, exist_ok=True)
    
    # 绘制原始图像和重建图像的对比图
    # 获取要显示的样本数量，最多10个
    n = min(10, len(original_images))
    # 创建一个宽度为20英寸，高度为4英寸的图形，用于并排显示图像
    plt.***REMOVED***gure(***REMOVED***gsize=(20, 4))
    
    # 遍历要显示的样本
    for i in range(n):
        # 绘制原始图像
        # 创建子图，布局为2行n列，当前是第i+1个子图（第一行）
        ax = plt.subplot(2, n, i + 1)
        # 显示原始图像，squeeze()移除单维度条目，cmap='gray'表示使用灰度颜色映射
        # 注意：这里假设了输入是灰度图，对于彩色图需要调整
        plt.imshow(original_images[i].squeeze().numpy(), cmap='gray')
        # 设置子图标题为"Original"
        plt.title(f"Original")
        # 关闭坐标轴显示
        plt.axis('off')
        
        # 绘制重建图像
        # 创建子图，布局为2行n列，当前是第i+1+n个子图（第二行）
        ax = plt.subplot(2, n, i + 1 + n)
        # 显示重建图像
        plt.imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
        # 设置子图标题，显示"Reconstructed"和对应的重建误差
        plt.title(f"Reconstructed\nError: {reconstruction_errors[i]:.4f}")
        # 关闭坐标轴显示
        plt.axis('off')
    
    # 判断是保存图像还是显示图像
    if save_dir:
        # 自动调整子图布局
        plt.tight_layout()
        # 将图像保存到指定目录下的文件中
        plt.save***REMOVED***g(os.path.join(save_dir, 'reconstruction_comparison.png'))
        # 关闭当前图形
        plt.close()
        # 打印保存成功的消息
        print(f"重建结果对比图已保存到 {save_dir}")
    else:
        # 自动调整子图布局
        plt.tight_layout()
        # 显示图像
        plt.show()
    
    # 绘制重建误差的分布直方图
    # 创建一个宽度为10英寸，高度为6英寸的图形
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 6))
    # 使用Seaborn的histplot函数绘制直方图，kde=True表示同时绘制核密度估计曲线
    sns.histplot(reconstruction_errors, kde=True)
    # 设置x轴标签
    plt.xlabel('重建误差')
    # 设置y轴标签
    plt.ylabel('频率')
    # 设置图表标题
    plt.title('重建误差分布')
    
    # 判断是保存图像还是显示图像
    if save_dir:
        # 自动调整子图布局
        plt.tight_layout()
        # 将图像保存到指定目录下的文件中
        plt.save***REMOVED***g(os.path.join(save_dir, 'error_distribution.png'))
        # 关闭当前图形
        plt.close()
        # 打印保存成功的消息
        print(f"重建误差分布图已保存到 {save_dir}")
    else:
        # 自动调整子图布局
        plt.tight_layout()
        # 显示图像
        plt.show()


def print_classi***REMOVED***cation_report(y_true, y_pred, class_names):
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    """
    # 使用scikit-learn生成分类报告，包含精确率、召回率、F1分数等指标
    # target_names指定类别名称，digits指定小数点后保留的位数
    report = classi***REMOVED***cation_report(y_true, y_pred, target_names=class_names, digits=4)
    # 打印报告标题
    print("\n分类报告:")
    # 打印生成的分类报告
    print(report)


def get_evaluation_function(model_name):
    """
    根据模型名称获取对应的评估函数
    
    Args:
        model_name: 模型名称
        
    Returns:
        对应的评估函数
    """
    # 判断模型名称是否为'autoencoder'
    if model_name == 'autoencoder':
        # 如果是自编码器，返回自编码器的评估函数
        return evaluate_autoencoder
    else:
        # 对于其他所有模型（假定为分类器），返回分类器的评估函数
        return evaluate_classi***REMOVED***er 


# 添加自定义的特征可视化函数
def visualize_model_features(model, data_loader, device, method='tsne', save_path=None):
    """
    更通用的特征可视化函数，适用于不同结构的模型
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        method: 降维方法，'tsne' 或 'pca'
        save_path: 保存路径，若为None则显示图像而不保存
    """
    # 导入降维算法和绘图库
    from sklearn.manifold import TSNE  # 导入t-SNE降维算法
    from sklearn.decomposition import PCA  # 导入PCA降维算法
    import matplotlib.pyplot as plt  # 导入matplotlib绘图库
    import numpy as np  # 导入NumPy库
    import torch.nn.functional as F  # 导入PyTorch的函数式接口
    from tqdm import tqdm  # 导入进度条库
    
    # 将模型设置为评估模式
    model.eval()
    # 初始化列表，用于存储提取的特征
    features = []
    # 初始化列表，用于存储对应的标签
    labels = []
    
    # 定义一个嵌套的辅助函数，用于以通用方式提取特征
    def extract_features(model, inputs):
        """
        尝试从不同类型的模型中提取特征。
        优先尝试自编码器的encode方法，然后尝试获取分类层前的特征。
        
        Args:
            model: 模型实例
            inputs: 输入数据
        
        Returns:
            提取的特征张量，如果无法提取则返回None
        """
        # 检查模型是否有可调用的encode方法（适用于标准AE和VAE）
        if hasattr(model, 'encode') and callable(getattr(model, 'encode')):
            # 如果是自编码器模型，调用encode方法获取潜在表示
            return model.encode(inputs)
        else:
            # 对于分类器模型，尝试获取最后一个分类层之前的特征
            # 假设最后一层是全连接层，通过注册钩子(hook)来获取其输入
            # 初始化变量用于存储提取的特征
            extracted_features = None
            
            # 定义钩子函数，该函数将在指定层的前向传播完成后被调用
            def hook_fn(module, input, output):
                # 使用nonlocal关键字声明extracted_features不是局部变量
                nonlocal extracted_features
                # 获取指定层的输入（通常是一个元组，取第一个元素）
                extracted_features = input[0]  # 获取最后一层的输入
            
            # 尝试为常见的分类层注册钩子
            # 如果模型有fc2层（通常是最后的分类层）
            if hasattr(model, 'fc2'):
                # 在fc2层注册前向钩子
                hook = model.fc2.register_forward_hook(hook_fn)
                # 运行模型的前向传播以触发钩子
                _ = model(inputs)  # 运行前向传播
                # 移除钩子，避免影响后续操作
                hook.remove()  # 移除钩子
                # 返回通过钩子获取的特征
                return extracted_features
            # 如果模型有classi***REMOVED***er属性且是Sequential类型（PyTorch常用结构）
            elif hasattr(model, 'classi***REMOVED***er') and isinstance(model.classi***REMOVED***er, torch.nn.Sequential):
                # 获取Sequential中的最后一层
                last_layer = model.classi***REMOVED***er[-1]
                # 在最后一层注册前向钩子
                hook = last_layer.register_forward_hook(hook_fn)
                # 运行模型的前向传播以触发钩子
                _ = model(inputs)
                # 移除钩子
                hook.remove()
                # 返回通过钩子获取的特征
                return extracted_features
            # 如果以上方法都不适用，尝试直接使用模型前向传播的输出作为特征
            else:
                # 运行模型的前向传播
                outputs = model(inputs)
                # 如果输出是一个元组（例如，某些模型可能返回多个值）
                if isinstance(outputs, tuple):
                    # 假设第一个元素是主要的特征输出
                    return outputs[0]
                # 否则，直接返回输出
                return outputs
    
    # 提取特征的主循环
    # 在不计算梯度的情况下进行
    with torch.no_grad():
        # 使用tqdm包装数据加载器，显示特征提取进度
        for inputs, batch_labels in tqdm(data_loader, desc=f"正在为 {method} 提取特征"):
            # 将输入数据移动到指定设备
            inputs = inputs.to(device)
            # 尝试使用辅助函数提取特征
            try:
                # 调用特征提取辅助函数
                batch_features = extract_features(model, inputs)
                
                # 检查是否成功提取到特征
                if batch_features is not None:
                    # 如果特征张量是多维的（例如，卷积层的输出）
                    if len(batch_features.shape) > 2:
                        # 将特征展平为二维张量 [batch_size, num_features]
                        batch_features = torch.flatten(batch_features, start_dim=1)
                    
                    # 将提取的特征（转换为NumPy数组）添加到列表
                    features.append(batch_features.cpu().numpy())
                    # 将对应的标签（转换为NumPy数组）添加到列表
                    labels.append(batch_labels.numpy())
            except Exception as e:
                # 如果在特征提取过程中发生异常，打印错误消息并退出函数
                print(f"特征提取失败: {e}")
                return
    
    # 检查是否成功提取到任何特征
    if not features:
        # 如果没有提取到特征，打印消息并退出函数
        print("无法提取特征")
        return
    
    # 合并所有批次的特征和标签
    # 使用np.vstack将所有批次的特征垂直堆叠成一个大矩阵
    features = np.vstack(features)
    # 使用np.concatenate将所有批次的标签连接成一个数组
    labels = np.concatenate(labels)
    
    # 根据指定的方法进行降维
    if method == 'tsne':
        # 如果使用t-SNE方法，打印相关信息
        print("使用t-SNE进行降维...")
        # 使用t-SNE算法将高维特征降至2维
        embedding = TSNE(n_components=2, random_state=42).***REMOVED***t_transform(features)
    else:  # pca
        # 如果使用PCA方法，打印相关信息
        print("使用PCA进行降维...")
        # 使用PCA算法将高维特征降至2维
        embedding = PCA(n_components=2, random_state=42).***REMOVED***t_transform(features)
    
    # 绘制降维后的特征分布图
    # 创建一个新的图形，大小为10x8英寸
    plt.***REMOVED***gure(***REMOVED***gsize=(10, 8))
    # 使用散点图可视化降维后的特征，点的颜色根据类别标签着色
    # cmap='viridis'指定颜色映射方案，alpha=0.5设置点的透明度
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.5)
    # 添加颜色条，标明不同颜色对应的类别
    plt.colorbar(scatter, label='Class')
    # 设置图表标题，显示使用的降维方法
    plt.title(f'使用 {method.upper()} 的特征可视化')
    # 设置x轴标签
    plt.xlabel('维度 1')
    # 设置y轴标签
    plt.ylabel('维度 2')
    
    # 判断是保存图像还是显示图像
    if save_path:
        # 自动调整图表布局
        plt.tight_layout()
        # 将图像保存到指定路径
        plt.save***REMOVED***g(save_path)
        # 关闭当前图形，释放内存
        plt.close()
        # 打印保存成功的消息
        print(f"特征可视化已保存到 {save_path}")
    else:
        # 自动调整图表布局
        plt.tight_layout()
        # 显示图像（不保存）
        plt.show()


if __name__ == "__main__":
    """
    独立运行评估模块
    支持直接评估已训练好的模型，无需重新训练
    """
    # 导入必要的标准库和本地模块
    import os  # 用于文件和目录操作
    import argparse  # 用于解析命令行参数
    import torch  # 导入PyTorch库
    from models import get_model  # 从本地models模块导入获取模型函数
    from data_utils import load_cifar10, get_data_loaders  # 从本地data_utils模块导入数据加载函数
    # 从本地visualize模块导入所有可视化函数
    from visualize import (
        visualize_predictions,
        visualize_features,
        visualize_attention,
        visualize_autoencoder_results
    )
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='模型评估')
    
    # 添加模型类型参数
    parser.add_argument('--model-type', type=str, required=True, 
                        choices=['basic_cnn', 'attention_cnn', 'vanilla_ae', 'conv_ae', 'vae'],
                        help='模型类型 (必需)')
    # 添加模型文件路径参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径 (必需)')
    
    # 添加数据目录参数
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录，默认为 "./data"')
    # 添加批次大小参数
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批量大小，默认为 64')
    # 添加数据加载线程数参数
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数，默认为 4')
    
    # 添加结果保存目录参数
    parser.add_argument('--results-dir', type=str, default=None,
                        help='结果保存目录，默认为项目根目录下的 "results" 文件夹')
    # 添加禁用CUDA的标志
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA，即使CUDA可用也强制使用CPU')
    
    # 解析命令行传入的参数
    args = parser.parse_args()
    
    # 设置计算设备
    # 检查CUDA是否可用并且用户没有指定禁用CUDA
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    # 根据检查结果设置设备为cuda或cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    # 打印当前使用的设备信息
    print(f"使用设备: {device}")
    
    # 获取项目根目录的绝对路径
    # os.path.abspath(__***REMOVED***le__) 获取当前脚本的绝对路径
    # os.path.dirname() 获取路径的目录部分
    # 两次调用dirname()是为了从src目录返回到项目根目录auto-vision
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__***REMOVED***le__)))
    
    # 设置默认的结果保存目录
    # 如果用户没有通过命令行指定结果目录
    if args.results_dir is None:
        # 将结果目录设置为项目根目录下的 'results' 文件夹
        args.results_dir = os.path.join(base_dir, 'results')
    
    # 根据模型类型创建具体的结果子目录
    # 例如，结果将保存在 'results/basic_cnn/' 或 'results/vanilla_ae/' 中
    results_dir = os.path.join(args.results_dir, args.model_type)
    # 创建结果目录，如果目录已存在则忽略
    os.makedirs(results_dir, exist_ok=True)
    
    # 打印最终结果保存路径
    print(f"结果将保存到: {results_dir}")
    
    # 加载CIFAR-10数据集
    print("正在加载CIFAR-10数据集...")
    # 调用load_cifar10函数加载数据集
    # _ 表示忽略训练集和验证集，因为评估只需要测试集
    # apply_augmentation=False 表示在评估时不应用数据增强
    _, _, test_dataset, class_names = load_cifar10(
        data_dir=args.data_dir,  # 数据目录
        val_size=0.1,  # 验证集比例（虽然不用验证集，但函数需要此参数）
        apply_augmentation=False  # 不应用数据增强
    )
    
    # 直接为测试集创建数据加载器，不使用 get_data_loaders 函数
    # 这是因为 get_data_loaders 可能需要训练集和验证集，而评估模式下它们是None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,  # 使用加载的测试数据集
        batch_size=args.batch_size,  # 设置批次大小
        shuffle=False,  # 评估时不需要打乱数据顺序
        num_workers=args.num_workers,  # 设置数据加载的工作线程数
        pin_memory=True  # 如果使用GPU，将数据加载到CUDA固定内存中，可以加速数据传输
    )
    
    # 创建模型实例
    print(f"创建模型: {args.model_type}")
    # 从测试数据加载器中获取一个批次的数据，以确定输入形状
    sample_data, _ = next(iter(test_loader))
    # 获取输入数据的形状（忽略批次维度）
    input_shape = sample_data.shape[1:]
    
    # 根据模型类型调用 get_model 函数创建模型
    if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:
        # 对于自编码器类型，通常只需要输入通道数
        model = get_model(args.model_type, in_channels=input_shape[0])
    else:
        # 对于分类器类型，需要输入通道数和类别数量（CIFAR-10为10类）
        model = get_model(args.model_type, in_channels=input_shape[0], num_classes=10)
    
    # 加载预训练的模型权重
    print(f"加载模型权重: {args.model_path}")
    try:
        # 加载模型文件，map_location=device确保模型加载到正确的设备上
        checkpoint = torch.load(args.model_path, map_location=device)
        # 检查加载的文件是包含状态字典的检查点还是直接的状态字典
        if 'model_state_dict' in checkpoint:
            # 如果是检查点，加载'model_state_dict'键对应的值
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果直接是状态字典，直接加载
            model.load_state_dict(checkpoint)
        # 打印加载成功的消息
        print("模型加载成功!")
    except Exception as e:
        # 如果加载过程中发生任何异常
        # 打印加载失败的消息和错误详情
        print(f"加载模型失败: {e}")
        # 退出脚本
        exit(1)
    
    # 将模型移动到指定的计算设备
    model = model.to(device)
    # 将模型设置为评估模式
    model.eval()
    
    # 开始评估模型
    print(f"评估 {args.model_type} 模型...")
    # 根据模型类型选择不同的评估流程
    if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:
        # 如果是自编码器类型
        # 调用自编码器评估函数
        metrics = evaluate_autoencoder(
            model=model,  # 传入模型实例
            test_loader=test_loader,  # 传入测试数据加载器
            device=device  # 传入计算设备
        )
        
        # 可视化自编码器的结果
        # 调用自编码器结果绘图函数
        plot_autoencoder_results(
            metrics['original_images'],  # 传入原始图像列表
            metrics['reconstructed_images'],  # 传入重建图像列表
            metrics['reconstruction_errors'],  # 传入重建误差列表
            save_dir=results_dir  # 指定保存目录
        )
        
        # 打印自编码器的评估指标
        print(f"测试集平均重建误差: {metrics['mean_error']:.4f}")
        print(f"检测到的异常比例: {metrics['anomaly_ratio']:.2%}")
        print(f"最大重建误差: {metrics['max_error']:.4f}")
        print(f"最小重建误差: {metrics['min_error']:.4f}")
    else:
        # 如果是分类器类型
        # 调用分类器评估函数
        metrics = evaluate_classi***REMOVED***er(
            model=model,  # 传入模型实例
            test_loader=test_loader,  # 传入测试数据加载器
            device=device  # 传入计算设备
        )
        
        # 绘制混淆矩阵并保存
        plot_confusion_matrix(
            metrics['all_targets'],  # 传入真实标签列表
            metrics['all_preds'],   # 传入预测标签列表
            class_names,  # 传入类别名称列表
            save_path=os.path.join(results_dir, 'confusion_matrix.png')  # 指定保存路径
        )
        
        # 打印详细的分类报告
        print_classi***REMOVED***cation_report(
            metrics['all_targets'],  # 传入真实标签列表
            metrics['all_preds'],   # 传入预测标签列表
            class_names  # 传入类别名称列表
        )
        
        # 打印分类器的主要评估指标
        print(f"测试集损失: {metrics['loss']:.4f}")
        print(f"测试集准确率: {metrics['accuracy']:.2%}")
        print(f"测试集精确率 (Macro): {metrics['precision']:.4f}")
        print(f"测试集召回率 (Macro): {metrics['recall']:.4f}")
        print(f"测试集F1分数 (Macro): {metrics['f1']:.4f}")
    
    # 生成附加的可视化结果
    print("生成其他可视化结果...")
    
    # 尝试进行特征可视化（对所有模型类型都尝试）
    try:
        # 调用我们定义的更通用的特征可视化函数
        visualize_model_features(
            model=model,  # 传入模型实例
            data_loader=test_loader,  # 传入测试数据加载器
            device=device,  # 传入计算设备
            method='tsne',  # 使用t-SNE方法进行降维
            save_path=os.path.join(results_dir, 'tsne_features.png')  # 指定保存路径
        )
    except Exception as e:
        # 如果特征可视化失败，打印错误消息
        print(f"生成特征可视化失败: {e}")
        print("这可能是因为模型结构与特征提取函数的期望不匹配，请检查模型的属性和层名称。")
    
    # 根据模型类型选择特定的可视化方法
    try:
        # 如果是自编码器类型
        if args.model_type in ['vanilla_ae', 'conv_ae', 'vae']:
            # 调用自编码器结果可视化函数（可能与上面的plot_autoencoder_results有些重叠，但这里使用visualize模块的函数）
            visualize_autoencoder_results(
                model=model,  # 传入模型实例
                data_loader=test_loader,  # 传入测试数据加载器
                device=device,  # 传入计算设备
                save_dir=results_dir,  # 指定保存目录
                num_samples=10  # 指定可视化样本数量
            )
        # 如果是带注意力的CNN模型
        elif args.model_type == 'attention_cnn':
            # 调用注意力可视化函数
            visualize_attention(
                model=model,  # 传入模型实例
                data_loader=test_loader,  # 传入测试数据加载器
                device=device,  # 传入计算设备
                class_names=class_names,  # 传入类别名称列表
                save_dir=results_dir  # 指定保存目录
            )
            # 对于注意力模型，通常也希望看到普通的预测结果可视化
            visualize_predictions(
                model=model,  # 传入模型实例
                data_loader=test_loader,  # 传入测试数据加载器
                class_names=class_names,  # 传入类别名称列表
                device=device,  # 传入计算设备
                num_samples=16,  # 指定可视化样本数量
                save_path=os.path.join(results_dir, 'predictions.png')  # 指定保存路径
            )
        # 对于其他（分类器）模型
        else:
            # 调用预测结果可视化函数
            visualize_predictions(
                model=model,  # 传入模型实例
                data_loader=test_loader,  # 传入测试数据加载器
                class_names=class_names,  # 传入类别名称列表
                device=device,  # 传入计算设备
                num_samples=16,  # 指定可视化样本数量
                save_path=os.path.join(results_dir, 'predictions.png')  # 指定保存路径
            )
    except Exception as e:
        # 如果生成特定可视化失败，打印错误消息
        print(f"生成模型特定可视化失败: {e}")
    
    # 打印评估完成的消息和结果保存路径
    print(f"评估完成! 所有结果已保存到 {results_dir}") 