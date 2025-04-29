#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT模型训练脚本
用于训练一个基于BERT的二分类模型，通过增量微调方式实现情感分析
"""

# 导入所需的库
import torch  # PyTorch深度学习框架
from MyData import MyDataset  # 导入自定义数据集类
from torch.utils.data import DataLoader  # 数据加载器
from net import Model  # 导入自定义模型结构
from transformers import BertTokenizer  # BERT分词器
from torch.optim import AdamW  # Adam优化器的权重衰减版本

# 定义计算设备，优先使用GPU，无GPU则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练的总轮次
EPOCH = 30000  # 可根据需要调整

# 初始化BERT分词器，加载预训练的中文BERT模型
token = BertTokenizer.from_pretrained("bert-base-chinese")

def collate_fn(data):
    """
    数据整理函数，用于将原始数据转换为模型输入格式
    
    Args:
        data: 数据集返回的原始数据批次
        
    Returns:
        input_ids: 文本的token ID序列
        attention_mask: 注意力掩码，用于标识真实token和填充token
        token_type_ids: token类型ID，用于区分不同句子
        labels: 标签张量
    """
    # 提取文本和标签
    sents = [i[0] for i in data]  # 提取每个样本的文本
    label = [i[1] for i in data]  # 提取每个样本的标签
    
    # 使用BERT分词器对文本进行编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 文本列表
        truncation=True,                 # 启用截断
        max_length=500,                  # 最大序列长度为500
        padding="max_length",            # 使用最大长度填充
        return_tensors="pt",             # 返回PyTorch张量
        return_length=True               # 返回序列长度
    )
    
    # 从编码结果中获取所需的张量
    input_ids = data["input_ids"]           # token的数字编码
    attention_mask = data["attention_mask"]  # 注意力掩码
    token_type_ids = data["token_type_ids"]  # token类型编码
    labels = torch.LongTensor(label)         # 将标签转换为长整型张量

    return input_ids, attention_mask, token_type_ids, labels


# 创建训练数据集和数据加载器
train_dataset = MyDataset("train")  # 实例化训练集
train_loader = DataLoader(
    dataset=train_dataset,   # 数据集对象
    batch_size=100,          # 批次大小
    shuffle=True,            # 随机打乱数据
    drop_last=True,          # 舍弃最后一个不完整批次的数据，防止形状出错
    collate_fn=collate_fn    # 使用自定义的数据处理函数
)

if __name__ == '__main__':
    # 开始训练流程
    print(DEVICE)  # 打印使用的设备信息
    
    # 实例化模型并移动到指定设备
    model = Model().to(DEVICE)
    
    # 定义优化器，使用AdamW（带权重衰减的Adam）
    optimizer = AdamW(model.parameters())
    
    # 定义损失函数，使用交叉熵损失（适用于分类问题）
    loss_func = torch.nn.CrossEntropyLoss()

    # 开始训练循环
    for epoch in range(EPOCH):  # 外循环：遍历所有训练轮次
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):  # 内循环：遍历数据批次
            # 将数据移动到指定设备上
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播：将数据输入模型，得到预测输出
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            # 计算损失：模型输出与真实标签之间的差异
            loss = loss_func(out, labels)
            
            # 反向传播与优化：更新模型参数
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            # 每隔5个批次输出一次训练信息
            if i % 5 == 0:
                # 计算准确率
                out = out.argmax(dim=1)  # 获取预测的类别
                acc = (out == labels).sum().item() / len(labels)  # 计算准确率
                print(f"epoch:{epoch}, batch:{i}, loss:{loss.item():.4f}, acc:{acc:.4f}")
                
        # 每训练完一个epoch，保存一次模型参数
        torch.save(model.state_dict(), f"params/{epoch}_bert.pth")
        print(f"Epoch {epoch} 完成，参数保存成功！")