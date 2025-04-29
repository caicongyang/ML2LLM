from transformers import AdamW
from transformers.optimization import get_scheduler
import torch
from data import MyDataset  # 导入自定义的数据集类
from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入transformers的模型和分词器类
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载器类

# 实例化自定义数据集
dataset = MyDataset()  # 创建数据集对象

# 加载预训练的分词器，用于文本编码
tokenizer = AutoTokenizer.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
# 加载预训练的模型，用于语言模型任务
model = AutoModelForCausalLM.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# 定义一个函数，用于将文本数据转换为模型所需的格式
def collate_fn(data):
    # 使用分词器对数据进行编码，并填充或截断到固定长度
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,  # 填充序列
                                       truncation=True,  # 截断序列
                                       max_length=512,  # 最大长度
                                       return_tensors='pt')  # 返回PyTorch张量
    # 复制输入ID作为标签，用于语言模型训练
    data['labels'] = data['input_ids'].clone()
    return data

# 使用DataLoader创建数据加载器，用于批量加载数据
loader = DataLoader(
    dataset=dataset,  # 指定数据集
    batch_size=2,  # 指定批量大小
    shuffle=True,  # 打乱数据
    drop_last=True,  # 如果最后一个批次的数据量小于batch_size，则丢弃
    collate_fn=collate_fn  # 指定如何从数据集中收集样本到批次中
)
print(f"数据的长度：{len(loader)}")  # 打印数据加载器中的批次数量

# 定义训练函数
def train():
    # 定义训练参数
    EPOCH = 3000  # 训练轮数
    global model  # 使用全局模型变量
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU
    model = model.to(DEVICE)  # 将模型移动到指定设备

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)  # 使用AdamW优化器，并设置学习率
    # 定义学习率调度器
    scheduler = get_scheduler(name="linear",  # 线性调度器
                              num_warmup_steps=0,  # 预热步数
                              num_training_steps=len(loader),  # 总训练步数
                              optimizer=optimizer)
    model.train()  # 将模型设置为训练模式
    for epoch in range(EPOCH):  # 循环每一轮训练
        for i, data in enumerate(loader):  # 遍历数据加载器中的批次
            for k in data.keys():  # 将数据移动到指定设备
                data[k] = data[k].to(DEVICE)
            out = model(**data)  # 前向传播
            loss = out['loss']  # 获取损失

            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率

            optimizer.zero_grad()  # 清空优化器的梯度
            model.zero_grad()  # 清空模型的梯度

            if i % 50 == 0:  # 每隔50个批次打印一次信息
                labels = data["labels"][:, 1:]  # 获取真实标签，忽略<bos>标记
                out = out["logits"].argmax(dim=2)[:,:-1]  # 获取预测结果，忽略<eos>标记

                select = labels != 0  # 选择非填充的标签
                labels = labels[select]  # 应用选择
                out = out[select]  # 应用选择
                del select  # 删除不再使用的select
                # 计算准确率
                acc = (labels == out).sum().item() / labels.numel()  # 计算准确率的公式
                lr = optimizer.state_dict()["param_groups"][0]['lr']  # 获取当前学习率

                # 打印训练信息
                print(f"epoch:{epoch},batch:{i},loss:{loss.item()},lr:{lr},acc:{acc}")

        # 保存最后一轮模型参数
        torch.save(model.state_dict(), "params/net.pt")  # 保存模型参数到指定路径
        print("权重保存成功！")  # 打印成功信息

# 当该脚本作为主程序运行时，调用训练函数
if __name__ == '__main__':
    train()  # 开始训练过程
