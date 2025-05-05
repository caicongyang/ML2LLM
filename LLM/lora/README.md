# LoRA 大型语言模型微调

本目录包含使用LoRA（低秩适应）技术微调大型语言模型（LLMs）的脚本和资源。

## 什么是LoRA？

LoRA是一种高效的微调方法，在论文[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)中提出。它通过以下方式显著减少了可训练参数的数量：

- 冻结预训练模型的权重
- 在Transformer架构的每一层注入可训练的秩分解矩阵
- 只训练这些较小的低秩矩阵，而不是所有模型参数

这种方法有几个优点：
- 大幅减少内存需求（减少10-10000倍的训练参数）
- 可在消费级硬件上进行微调
- 生成较小的适配器文件（通常 < 100MB）
- 允许在推理时切换不同的适配

## 环境要求

```
transformers>=4.31.0
peft>=0.4.0
accelerate>=0.21.0
datasets>=2.13.0
bitsandbytes>=0.40.0  # 用于量化训练
torch>=2.0.1
```

## 使用方法

### 基本用法

主脚本`lora_fine_tuning.py`处理整个微调过程：

```bash
python lora_fine_tuning.py \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --dataset_path "your_dataset.json" \
  --output_dir "./lora-llama-2-output" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --learning_rate 3e-4
```

### 使用4位量化训练

对于较大的模型，可以使用4位量化来减少显存使用：

```bash
python lora_fine_tuning.py \
  --model_name_or_path "meta-llama/Llama-2-13b-hf" \
  --dataset_path "your_dataset.json" \
  --output_dir "./lora-llama-2-13b-output" \
  --lora_rank 16 \
  --use_4bit \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4
```

## 数据集格式

脚本支持以下数据集格式：

1. **本地JSON文件**：带有"text"字段的JSON文件（或通过`--text_column`指定）
2. **本地CSV文件**：带有"text"列的CSV文件（或通过`--text_column`指定）
3. **Hugging Face数据集**：来自Hugging Face Hub的数据集ID

JSON格式示例：
```json
[
  {"text": "您的训练示例文本在这里"},
  {"text": "另一个训练示例"}
]
```

## 高级选项

脚本支持多种微调参数，包括：

- `--lora_rank`：LoRA分解矩阵的秩（默认：8）
- `--lora_alpha`：LoRA alpha参数/缩放因子（默认：16）
- `--lora_dropout`：LoRA层的丢弃概率（默认：0.05）
- `--target_modules`：应用LoRA的模块名称列表（如未指定则自动检测）
- `--learning_rate`：学习率（默认：3e-4）
- `--num_train_epochs`：训练轮数（默认：3）
- `--max_seq_length`：最大序列长度（默认：512）
- `--use_8bit` / `--use_4bit`：启用8位或4位量化
- `--gradient_accumulation_steps`：累积的更新步数（默认：8）

运行`python lora_fine_tuning.py --help`获取完整的参数列表。

## 使用微调后的模型

微调后，您的LoRA适配器将保存在指定的输出目录中。您可以使用以下方式进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./lora-llama-2-output/final")

# 生成文本
input_text = "您的提示文本"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 获得更好结果的技巧

1. **针对正确的模块**：不同的模型架构有不同的模块名称。脚本尝试检测常见架构，但您可能需要手动指定`--target_modules`。

2. **调整LoRA秩**：更高的秩(r)意味着更高的容量但适配器更大。大多数任务可以从8-16开始。

3. **平衡学习率**：LoRA通常使用比全量微调稍高的学习率效果更好。尝试2e-4到5e-4。

4. **优化序列长度**：使用适合您任务的序列长度来提高效率。 