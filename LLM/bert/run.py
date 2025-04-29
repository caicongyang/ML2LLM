# 导入必要的库
import torch  # PyTorch深度学习框架
from net import Model  # 导入自定义的模型结构
from transformers import BertTokenizer  # 导入BERT分词器
import os  # 导入os模块，用于处理文件路径

# 设置运行设备，优先使用GPU，如果没有则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)  # 打印当前使用的设备信息

# 初始化BERT分词器，使用预训练的中文BERT模型
token = BertTokenizer.from_pretrained("bert-base-chinese")

# 定义输出标签的映射关系
names = ["负向评价", "正向评价"]  # 索引0表示负向，1表示正向

# 实例化模型并移至指定设备
model = Model().to(DEVICE)

def collate_fn(data):
    """
    数据预处理函数
    Args:
        data: 输入的文本数据
    Returns:
        input_ids: 文本的token ID序列
        attention_mask: 注意力掩码，用于标识真实token和填充token
        token_type_ids: token类型ID，用于区分不同句子
    """
    sents = []
    sents.append(data)  # 将输入文本转换为列表格式
    
    # 使用BERT分词器对文本进行编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 输入文本列表
        truncation=True,  # 启用截断
        max_length=500,   # 最大序列长度为500
        padding="max_length",  # 使用最大长度填充
        return_tensors="pt",  # 返回PyTorch张量
        return_length=True    # 返回序列长度
    )
    
    # 从编码结果中获取所需的张量
    input_ids = data["input_ids"]  # token的数字编码
    attention_mask = data["attention_mask"]  # 注意力掩码
    token_type_ids = data["token_type_ids"]  # token类型编码

    return input_ids, attention_mask, token_type_ids

def test():
    """
    模型测试函数：进行交互式的文本情感预测
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__***REMOVED***le__))
    # 加载训练好的模型参数，使用相对路径
    model_path = os.path.join(current_dir, "params/1_bert.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # 将模型设置为评估模式（关闭dropout等训练特定操作）
    model.eval()

    # 开始交互式测试循环
    while True:
        # 获取用户输入
        data = input("请输入测试数据（输入'q'退出）：")
        if data == 'q':
            print("测试结束")
            break
            
        # 对输入文本进行预处理
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        
        # 将数据转移到指定设备上
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)

        # 使用模型进行预测
        with torch.no_grad():  # 关闭梯度计算，节省内存
            out = model(input_ids, attention_mask, token_type_ids)  # 获取模型输出
            out = out.argmax(dim=1)  # 获取预测类别（0或1）
            print("模型判定：", names[out], "\n")  # 输出预测结果

# 程序入口点
if __name__ == '__main__':
    test()