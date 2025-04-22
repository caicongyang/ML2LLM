#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型定义模块
包含分类器和自编码器模型的定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    """基础CNN分类器"""
    
    def __init__(self, num_classes=10, in_channels=3):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 经过3次下采样，图像大小变为原来的1/8
        # 对于32x32的输入，最终得到4x4的特征图
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VanillaAutoencoder(nn.Module):
    """基础自编码器"""
    
    def __init__(self, in_channels=3, latent_dim=128):
        super(VanillaAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 8x8
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 4x4
            nn.ReLU(True)
        )
        
        # 全连接层到潜在空间
        self.fc_encoder = nn.Linear(128 * 4 * 4, latent_dim)
        
        # 全连接层从潜在空间
        self.fc_decoder = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.Tanh()  # 输出范围限制在[-1,1]
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)
        return x
        
    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(z.size(0), 128, 4, 4)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class ConvAutoencoder(nn.Module):
    """卷积自编码器"""
    
    def __init__(self, in_channels=3):
        super(ConvAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # 32x32
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 8x8
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)   # 4x4
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 8x8
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 16x16
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),  # 32x32
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VariationalAutoencoder(nn.Module):
    """变分自编码器 (VAE)"""
    
    def __init__(self, in_channels=3, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.flatten = nn.Flatten()
        
        # 计算均值和对数方差
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # 从潜在空间到解码器
        self.fc_decoder = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 128, 4, 4)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def get_model(model_name, **kwargs):
    """
    根据模型名称获取相应的模型实例
    
    Args:
        model_name (str): 模型名称
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        nn.Module: 模型实例
    """
    models = {
        'basic_cnn': BasicCNN,
        'vanilla_ae': VanillaAutoencoder,
        'conv_ae': ConvAutoencoder,
        'vae': VariationalAutoencoder
    }
    
    if model_name not in models:
        raise ValueError(f"模型 '{model_name}' 不存在，可用模型: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def count_parameters(model):
    """
    计算模型的参数数量
    
    Args:
        model (nn.Module): 要计算参数的模型
        
    Returns:
        int: 模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型定义模块
    print("测试模型定义模块...")
    
    # 创建测试张量，大小为 [batch_size, channels, height, width]
    x = torch.randn(4, 3, 32, 32)
    
    # 测试基础CNN分类器
    basic_cnn = get_model('basic_cnn', num_classes=10)
    print(f"\n基础CNN分类器参数数量: {count_parameters(basic_cnn):,}")
    y = basic_cnn(x)
    print(f"输出形状: {y.shape}")
    
    # 测试基础自编码器
    vanilla_ae = get_model('vanilla_ae', latent_dim=128)
    print(f"\n基础自编码器参数数量: {count_parameters(vanilla_ae):,}")
    output = vanilla_ae(x)
    print(f"输出形状: {output.shape}")
    
    # 测试卷积自编码器
    conv_ae = get_model('conv_ae')
    print(f"\n卷积自编码器参数数量: {count_parameters(conv_ae):,}")
    output = conv_ae(x)
    print(f"输出形状: {output.shape}")
    
    # 测试变分自编码器
    vae = get_model('vae', latent_dim=64)
    print(f"\n变分自编码器参数数量: {count_parameters(vae):,}")
    output, mu, logvar = vae(x)
    print(f"重构输出形状: {output.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"对数方差形状: {logvar.shape}")
    
    print("\n模型定义模块测试完成!") 