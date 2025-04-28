#!/usr/bin/env python  # 指定解释器路径
# -*- coding: utf-8 -*-  # 设置文件编码为UTF-8

"""
模型定义模块
包含分类器和自编码器模型的定义
"""

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口，包含激活函数等


class BasicCNN(nn.Module):  # 定义基础卷积神经网络类，继承自nn.Module
    """
    基础卷积神经网络分类器
    
    这是一个简单的CNN模型，用于图像分类任务。它由三个卷积层组成，每个卷积层后跟批归一化和
    最大池化操作。最后通过两个全连接层输出类别预测概率。
    
    网络结构:
        1. 卷积层 (3->32) + 批归一化 + ReLU + 最大池化
        2. 卷积层 (32->64) + 批归一化 + ReLU + 最大池化
        3. 卷积层 (64->128) + 批归一化 + ReLU + 最大池化
        4. 全连接层 (2048->512) + ReLU
        5. Dropout (p=0.5)
        6. 全连接层 (512->num_classes)
        
    参数:
        num_classes (int): 分类类别数量，默认为10 (CIFAR-10)
        in_channels (int): 输入图像的通道数，默认为3 (RGB图像)
    """
    
    def __init__(self, num_classes=10, in_channels=3):  # 初始化函数，设置默认参数
        super(BasicCNN, self).__init__()  # 调用父类初始化
        # 第一个卷积块：输入通道 -> 32个特征图
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 第一个卷积层，保持输入尺寸不变
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化提高训练稳定性
        
        # 第二个卷积块：32 -> 64个特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 第二个卷积层，保持输入尺寸不变
        self.bn2 = nn.BatchNorm2d(64)  # 对64个特征图进行批归一化
        
        # 第三个卷积块：64 -> 128个特征图
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 第三个卷积层，保持输入尺寸不变
        self.bn3 = nn.BatchNorm2d(128)  # 对128个特征图进行批归一化
        
        # 最大池化层用于减小特征图尺寸
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义池化操作，将特征图尺寸减半
        
        # 经过3次下采样，图像大小变为原来的1/8
        # 对于32x32的输入，最终得到4x4的特征图
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 全连接层，连接展平的特征图到512个神经元
        self.dropout = nn.Dropout(0.5)  # 使用Dropout防止过拟合
        self.fc2 = nn.Linear(512, num_classes)  # 输出层，映射到类别数
        
    def forward(self, x):  # 定义前向传播函数
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, in_channels, height, width]
            
        返回:
            torch.Tensor: 类别预测logits，形状为 [batch_size, num_classes]
        """
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 卷积->归一化->激活->池化
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 卷积->归一化->激活->池化
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 卷积->归一化->激活->池化
        
        # 展平特征图为一维向量
        x = x.view(-1, 128 * 4 * 4)  # 将三维特征图展平成一维向量，-1表示自动计算批次大小
        
        # 全连接层
        x = F.relu(self.fc1(x))  # 全连接层后接ReLU激活函数
        x = self.dropout(x)  # 应用dropout
        x = self.fc2(x)  # 输出层
        return x  # 返回模型输出


class VanillaAutoencoder(nn.Module):  # 定义基础自编码器类，继承自nn.Module
    """
    基础自编码器模型
    
    这是一个标准的自编码器，具有卷积编码器和反卷积解码器。该模型将输入压缩到低维潜在空间，
    然后尝试从潜在空间重构原始输入。适用于特征学习、降维和图像重构任务。
    
    网络结构:
        编码器:
            1. 卷积层 (in_channels->32) + ReLU [32x32 -> 16x16]
            2. 卷积层 (32->64) + ReLU [16x16 -> 8x8]
            3. 卷积层 (64->128) + ReLU [8x8 -> 4x4]
            4. 全连接层 (128*4*4 -> latent_dim)
            
        解码器:
            1. 全连接层 (latent_dim -> 128*4*4)
            2. 转置卷积层 (128->64) + ReLU [4x4 -> 8x8]
            3. 转置卷积层 (64->32) + ReLU [8x8 -> 16x16]
            4. 转置卷积层 (32->in_channels) + Tanh [16x16 -> 32x32]
    
    参数:
        in_channels (int): 输入图像的通道数，默认为3 (RGB图像)
        latent_dim (int): 潜在空间维度，默认为128
    """
    
    def __init__(self, in_channels=3, latent_dim=128):  # 初始化函数，设置默认参数
        super(VanillaAutoencoder, self).__init__()  # 调用父类初始化
        
        # 编码器 - 将输入压缩到潜在空间
        self.encoder = nn.Sequential(  # 定义编码器为一个序列模块
            # 第一层: 输入 -> 32特征图，尺寸减半 (32x32 -> 16x16)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 使用步长2减小特征图尺寸
            nn.ReLU(True),  # ReLU激活函数，inplace=True节省内存
            
            # 第二层: 32 -> 64特征图，尺寸再减半 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 第二个卷积层，继续减小尺寸
            nn.ReLU(True),  # ReLU激活函数
            
            # 第三层: 64 -> 128特征图，尺寸再减半 (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 第三个卷积层，继续减小尺寸
            nn.ReLU(True)  # ReLU激活函数
        )
        
        # 全连接层到潜在空间 - 将卷积特征映射到潜在向量
        self.fc_encoder = nn.Linear(128 * 4 * 4, latent_dim)  # 全连接层，将特征图压缩到潜在空间
        
        # 全连接层从潜在空间 - 将潜在向量映射回卷积特征
        self.fc_decoder = nn.Linear(latent_dim, 128 * 4 * 4)  # 全连接层，从潜在空间映射回特征图形状
        
        # 解码器 - 从潜在空间重构输入
        self.decoder = nn.Sequential(  # 定义解码器为一个序列模块
            # 第一层: 128 -> 64特征图，尺寸增倍 (4x4 -> 8x8)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积，上采样
            nn.ReLU(True),  # ReLU激活函数
            
            # 第二层: 64 -> 32特征图，尺寸增倍 (8x8 -> 16x16)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积，上采样
            nn.ReLU(True),  # ReLU激活函数
            
            # 第三层: 32 -> 输入通道，尺寸增倍 (16x16 -> 32x32)
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积，上采样
            nn.Tanh()  # 输出范围限制在[-1,1]
        )
        
    def encode(self, x):  # 定义编码函数
        """
        编码函数：将输入图像编码到潜在空间
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        返回:
            torch.Tensor: 潜在空间表示，形状为 [batch_size, latent_dim]
        """
        x = self.encoder(x)  # 应用编码器
        x = x.view(x.size(0), -1)  # 展平特征图，保留批次维度
        x = self.fc_encoder(x)  # 映射到潜在空间
        return x  # 返回潜在空间表示
        
    def decode(self, z):  # 定义解码函数
        """
        解码函数：从潜在空间重构图像
        
        参数:
            z (torch.Tensor): 潜在空间表示，形状为 [batch_size, latent_dim]
            
        返回:
            torch.Tensor: 重构的图像，形状为 [batch_size, in_channels, height, width]
        """
        z = self.fc_decoder(z)  # 从潜在空间映射回卷积特征
        z = z.view(z.size(0), 128, 4, 4)  # 重塑为卷积特征图形状
        z = self.decoder(z)  # 应用解码器
        return z  # 返回重构的图像
    
    def forward(self, x):  # 定义前向传播函数
        """
        前向传播函数：编码然后解码
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        返回:
            torch.Tensor: 重构的图像，形状同输入
        """
        z = self.encode(x)  # 编码
        return self.decode(z)  # 解码并返回重构


class ConvAutoencoder(nn.Module):  # 定义卷积自编码器类，继承自nn.Module
    """
    卷积自编码器
    
    这是一个纯卷积的自编码器实现，使用卷积和最大池化进行编码，转置卷积进行解码。
    它没有使用全连接层，因此保留了更多的空间信息。适用于图像去噪、特征提取和压缩。
    
    网络结构:
        编码器:
            1. 卷积层 (in_channels->16) + ReLU [32x32]
            2. 最大池化 [32x32 -> 16x16]
            3. 卷积层 (16->32) + ReLU [16x16]
            4. 最大池化 [16x16 -> 8x8]
            5. 卷积层 (32->64) + ReLU [8x8]
            6. 最大池化 [8x8 -> 4x4]
            
        解码器:
            1. 转置卷积层 (64->32) + ReLU [4x4 -> 8x8]
            2. 转置卷积层 (32->16) + ReLU [8x8 -> 16x16]
            3. 转置卷积层 (16->in_channels) + Tanh [16x16 -> 32x32]
    
    参数:
        in_channels (int): 输入图像的通道数，默认为3 (RGB图像)
    """
    
    def __init__(self, in_channels=3):  # 初始化函数，设置默认参数
        super(ConvAutoencoder, self).__init__()  # 调用父类初始化
        
        # 编码器 - 采用卷积和池化组合
        self.encoder = nn.Sequential(  # 定义编码器为一个序列模块
            # 第一个卷积块：输入通道 -> 16特征图，保持尺寸
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # 卷积层，padding=1保持尺寸
            nn.ReLU(True),  # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16，最大池化减小尺寸
            
            # 第二个卷积块：16 -> 32特征图，保持尺寸
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 卷积层，padding=1保持尺寸
            nn.ReLU(True),  # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8，最大池化减小尺寸
            
            # 第三个卷积块：32 -> 64特征图，保持尺寸
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 卷积层，padding=1保持尺寸
            nn.ReLU(True),  # ReLU激活函数
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4，最大池化减小尺寸
        )
        
        # 解码器 - 使用转置卷积进行上采样
        self.decoder = nn.Sequential(  # 定义解码器为一个序列模块
            # 第一层: 64 -> 32特征图，尺寸增倍 (4x4 -> 8x8)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 转置卷积，进行2倍上采样
            nn.ReLU(True),  # ReLU激活函数
            
            # 第二层: 32 -> 16特征图，尺寸增倍 (8x8 -> 16x16)
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 转置卷积，进行2倍上采样
            nn.ReLU(True),  # ReLU激活函数
            
            # 第三层: 16 -> 输入通道，尺寸增倍 (16x16 -> 32x32)
            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),  # 转置卷积，进行2倍上采样
            nn.Tanh()  # 输出范围限制在[-1,1]
        )
        
    def forward(self, x):  # 定义前向传播函数
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        返回:
            torch.Tensor: 重构的图像，形状同输入
        """
        x = self.encoder(x)  # 编码
        x = self.decoder(x)  # 解码
        return x  # 返回重构的图像


class VariationalAutoencoder(nn.Module):  # 定义变分自编码器类，继承自nn.Module
    """
    变分自编码器 (VAE)
    
    VAE是一种生成模型，它学习输入数据的概率分布，而不仅仅是直接映射。它通过学习均值和方差参数，
    然后使用这些参数采样潜在向量，最后解码生成新样本。VAE适用于图像生成、异常检测和数据插值。
    
    网络结构:
        编码器:
            1. 卷积层 (in_channels->32) + 批归一化 + LeakyReLU [32x32 -> 16x16]
            2. 卷积层 (32->64) + 批归一化 + LeakyReLU [16x16 -> 8x8]
            3. 卷积层 (64->128) + 批归一化 + LeakyReLU [8x8 -> 4x4]
            4. 平坦化层 (128*4*4)
            5. 两个并行的全连接层:
               - 均值层 (mu): 128*4*4 -> latent_dim
               - 对数方差层 (logvar): 128*4*4 -> latent_dim
            
        重参数化技巧:
            z = mu + eps * std，其中 eps ~ N(0,1), std = exp(0.5 * logvar)
            
        解码器:
            1. 全连接层 (latent_dim -> 128*4*4)
            2. 转置卷积层 (128->64) + 批归一化 + LeakyReLU [4x4 -> 8x8]
            3. 转置卷积层 (64->32) + 批归一化 + LeakyReLU [8x8 -> 16x16]
            4. 转置卷积层 (32->in_channels) + Tanh [16x16 -> 32x32]
    
    参数:
        in_channels (int): 输入图像的通道数，默认为3 (RGB图像)
        latent_dim (int): 潜在空间维度，默认为128
    """
    
    def __init__(self, in_channels=3, latent_dim=128):  # 初始化函数，设置默认参数
        super(VariationalAutoencoder, self).__init__()  # 调用父类初始化
        
        # 编码器 - 将输入映射到潜在分布参数
        self.encoder = nn.Sequential(  # 定义编码器为一个序列模块
            # 第一层: 输入 -> 32特征图，尺寸减半 (32x32 -> 16x16)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 使用步长2减小特征图尺寸
            nn.BatchNorm2d(32),  # 批归一化提高训练稳定性
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU避免死神经元
            
            # 第二层: 32 -> 64特征图，尺寸再减半 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 第二个卷积层，继续减小尺寸
            nn.BatchNorm2d(64),  # 对64个特征图进行批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            
            # 第三层: 64 -> 128特征图，尺寸再减半 (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 第三个卷积层，继续减小尺寸
            nn.BatchNorm2d(128),  # 对128个特征图进行批归一化
            nn.LeakyReLU(0.2, inplace=True)  # LeakyReLU激活函数
        )
        
        # 展平特征图
        self.flatten = nn.Flatten()  # 定义展平层，将特征图转为一维向量
        
        # 计算均值和对数方差 - 变分自编码器的关键部分
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # 均值层，映射到潜在空间
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # 对数方差层，映射到潜在空间
        
        # 从潜在空间到解码器
        self.fc_decoder = nn.Linear(latent_dim, 128 * 4 * 4)  # 全连接层，将潜在向量映射回特征图形状
        
        # 解码器 - 从潜在空间重构输入
        self.decoder = nn.Sequential(  # 定义解码器为一个序列模块
            # 第一层: 128 -> 64特征图，尺寸增倍 (4x4 -> 8x8)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积，上采样
            nn.BatchNorm2d(64),  # 对64个特征图进行批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            
            # 第二层: 64 -> 32特征图，尺寸增倍 (8x8 -> 16x16)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积，上采样
            nn.BatchNorm2d(32),  # 对32个特征图进行批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            
            # 第三层: 32 -> 输入通道，尺寸增倍 (16x16 -> 32x32)
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积，上采样
            nn.Tanh()  # 输出范围限制在[-1,1]
        )
        
    def encode(self, x):  # 定义编码函数
        """
        编码函数：将输入图像编码到两个参数向量（均值和对数方差）
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (均值向量, 对数方差向量)，各自形状为 [batch_size, latent_dim]
        """
        x = self.encoder(x)  # 应用编码器
        x = self.flatten(x)  # 展平特征图
        mu = self.fc_mu(x)  # 计算均值
        logvar = self.fc_logvar(x)  # 计算对数方差
        return mu, logvar  # 返回均值和对数方差
    
    def reparameterize(self, mu, logvar):  # 定义重参数化函数
        """
        重参数化技巧：允许在反向传播中对随机采样进行梯度传播
        
        参数:
            mu (torch.Tensor): 均值向量，形状为 [batch_size, latent_dim]
            logvar (torch.Tensor): 对数方差向量，形状为 [batch_size, latent_dim]
            
        返回:
            torch.Tensor: 采样的潜在向量，形状为 [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布采样
        return mu + eps * std  # 返回采样结果
    
    def decode(self, z):  # 定义解码函数
        """
        解码函数：从潜在空间采样重构图像
        
        参数:
            z (torch.Tensor): 潜在空间向量，形状为 [batch_size, latent_dim]
            
        返回:
            torch.Tensor: 重构的图像，形状为 [batch_size, in_channels, height, width]
        """
        z = self.fc_decoder(z)  # 从潜在空间映射回特征图
        z = z.view(-1, 128, 4, 4)  # 重塑为4x4的特征图
        z = self.decoder(z)  # 应用解码器
        return z  # 返回重构的图像
    
    def forward(self, x):  # 定义前向传播函数
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (重构图像, 均值向量, 对数方差向量)
        """
        mu, logvar = self.encode(x)  # 得到潜在分布参数
        z = self.reparameterize(mu, logvar)  # 从分布中采样
        x_recon = self.decode(z)  # 解码重构
        return x_recon, mu, logvar  # 返回重构图像和分布参数


def get_model(model_name, **kwargs):  # 定义获取模型函数
    """
    根据模型名称获取相应的模型实例
    
    模型工厂函数，根据名称返回对应的模型类实例。支持以下模型:
    - basic_cnn: 基础CNN分类器
    - vanilla_ae: 基础自编码器
    - conv_ae: 卷积自编码器
    - vae: 变分自编码器
    
    参数:
        model_name (str): 模型名称
        **kwargs: 传递给模型构造函数的参数
        
    返回:
        nn.Module: 模型实例
        
    异常:
        ValueError: 如果指定的模型名称不存在
    """
    models = {  # 创建模型名称到模型类的映射字典
        'basic_cnn': BasicCNN,  # 基础CNN分类器
        'vanilla_ae': VanillaAutoencoder,  # 基础自编码器
        'conv_ae': ConvAutoencoder,  # 卷积自编码器
        'vae': VariationalAutoencoder  # 变分自编码器
    }
    
    if model_name not in models:  # 检查模型名称是否存在
        raise ValueError(f"模型 '{model_name}' 不存在，可用模型: {list(models.keys())}")  # 抛出错误
    
    return models[model_name](**kwargs)  # 返回实例化的模型对象


def count_parameters(model):  # 定义计算模型参数数量的函数
    """
    计算模型的可训练参数数量
    
    参数:
        model (nn.Module): 要计算参数的模型
        
    返回:
        int: 模型可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 累加所有需要梯度的参数数量


if __name__ == "__main__":  # 主程序入口
    # 测试模型定义模块
    print("测试模型定义模块...")  # 打印测试开始信息
    
    # 创建测试张量，大小为 [batch_size, channels, height, width]
    x = torch.randn(4, 3, 32, 32)  # 创建随机输入张量，模拟4张32x32的RGB图像
    
    # 测试基础CNN分类器
    basic_cnn = get_model('basic_cnn', num_classes=10)  # 创建基础CNN模型
    print(f"\n基础CNN分类器参数数量: {count_parameters(basic_cnn):,}")  # 打印参数数量
    y = basic_cnn(x)  # 前向传播
    print(f"输出形状: {y.shape}")  # 打印输出形状
    
    # 测试基础自编码器
    vanilla_ae = get_model('vanilla_ae', latent_dim=128)  # 创建基础自编码器
    print(f"\n基础自编码器参数数量: {count_parameters(vanilla_ae):,}")  # 打印参数数量
    output = vanilla_ae(x)  # 前向传播
    print(f"输出形状: {output.shape}")  # 打印输出形状
    
    # 测试卷积自编码器
    conv_ae = get_model('conv_ae')  # 创建卷积自编码器
    print(f"\n卷积自编码器参数数量: {count_parameters(conv_ae):,}")  # 打印参数数量
    output = conv_ae(x)  # 前向传播
    print(f"输出形状: {output.shape}")  # 打印输出形状
    
    # 测试变分自编码器
    vae = get_model('vae', latent_dim=64)  # 创建变分自编码器
    print(f"\n变分自编码器参数数量: {count_parameters(vae):,}")  # 打印参数数量
    output, mu, logvar = vae(x)  # 前向传播
    print(f"重构输出形状: {output.shape}")  # 打印重构输出形状
    print(f"均值形状: {mu.shape}")  # 打印均值形状
    print(f"对数方差形状: {logvar.shape}")  # 打印对数方差形状
    
    print("\n模型定义模块测试完成!")  # 打印测试完成信息 