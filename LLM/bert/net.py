from transformers import BertModel
import torch

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#加载预训练模型 - 直接从Hugging Face下载
pretrained = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)
print(pretrained)
#定义下游任务（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(768,2)

    def forward(self,input_ids,attention_mask,token_type_ids):
        #冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0])
        return out
