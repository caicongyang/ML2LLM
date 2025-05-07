#中文文言文生成


from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-ancient")
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-ancient")
print(model)

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device="cuda")

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("于是者", max_length=100, do_sample=True))