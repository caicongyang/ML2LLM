from datasets import load_dataset, load_from_disk
import os

# #在线加载数据
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1", split="train")
# print(dataset)
# #转存为CSV格式
# dataset.to_csv(path_or_buf="data/hermes-function-calling-v1.csv")
# #加载csv格式数据
# dataset = load_dataset(path="csv", data_***REMOVED***les="data/hermes-function-calling-v1.csv")
# # print(dataset)

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__***REMOVED***le__))
# 加载缓存数据
data_path = os.path.join(current_dir, "data/ChnSentiCorp")
dataset = load_from_disk(data_path)
print(dataset)

test_data = dataset["train"]
for data in test_data:
    print(data)