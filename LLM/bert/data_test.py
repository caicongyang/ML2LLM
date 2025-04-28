from datasets import load_dataset,load_from_disk

# #在线加载数据
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
# print(dataset)
# #转存为CSV格式
# dataset.to_csv(path_or_buf=r"/Users/caicongyang/IdeaProjects/tom/ML2LLM/LLM/bert/data/hermes-function-calling-v1.csv")
#加载csv格式数据
# dataset = load_dataset(path="csv",data_***REMOVED***les=r"/Users/caicongyang/IdeaProjects/tom/ML2LLM/LLM/bert/data/hermes-function-calling-v1.csv")
# # print(dataset)
#加载缓存数据
dataset = load_from_disk(r"/Users/caicongyang/IdeaProjects/tom/ML2LLM/LLM/bert/data/ChnSentiCorp")
print(dataset)

test_data = dataset["train"]
for data in test_data:
    print(data)