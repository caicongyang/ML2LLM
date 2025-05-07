from transformers import AutoTokenizer, BertTokenizer

#加载字典和分词器
token = BertTokenizer.from_pretrained("bert-base-chinese")
# print(token)

sents = ["价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所"]

#批量编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0]],
    add_special_tokens=True,
    #当句子长度大于max_length时，截断
    truncation=True,
    max_length=50,
    #一律补0到max_length长度
    padding="max_length",
    #可取值为tf,pt,np,默认为list
    return_tensors=None,
    #返回attention_mask
    return_attention_mask=True,
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    #返回length长度
    return_length=True
)
#input_ids 就是编码后的词
#token_type_ids第一个句子和特殊符号的位置是0，第二个句子的位置1（）只针对于上下文编码
#special_tokens_mask 特殊符号的位置是1，其他位置是0
print(out)
for k,v in out.items():
    print(k,";",v)

#解码文本数据
# print(token.decode(out["input_ids"][0]),token.decode(out["input_ids"][1]))


