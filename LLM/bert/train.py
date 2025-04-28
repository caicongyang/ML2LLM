#模型训练
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from torch.optim import AdamW

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次
EPOCH= 30000

token = BertTokenizer.from_pretrained(r"bert-base-chinese")

def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids,attention_mask,token_type_ids,labels


#创建数据集
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载进来的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    #定义优化器
    optimizer = AdamW(model.parameters())
    #定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            #将数据存放到DEVICE上
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),labels.to(DEVICE)
            #前向计算（将数据输入模型，得到输出）
            out = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            #根据输出，计算损失
            loss = loss_func(out,labels)
            #根据损失，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #每隔5个批次输出训练信息
            if i%5==0:
                out = out.argmax(dim=1)
                acc = (out==labels).sum().item()/len(labels)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        #每训练完一轮，保存一次参数
        torch.save(model.state_dict(),f"params/{epoch}_bert.pth")
        print(epoch,"参数保存成功！")