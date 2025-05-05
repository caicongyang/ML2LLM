# 上传模型到 Hugging Face Hub 指南

这个简单的工具可以帮助您将已合并的模型快速上传到 Hugging Face Hub。

## 准备工作

1. 确保您已经完成了 LoRA 模型与基础模型的合并
2. 获取 Hugging Face API 令牌: https://huggingface.co/settings/tokens
3. 安装必要的依赖:
   ```bash
   pip install huggingface_hub
   ```

## 方法一: 使用 Python 脚本

`upload_to_huggingface.py` 脚本提供了完整的上传功能。

### 基本使用

```bash
***REMOVED***
  --model_path "./exported_model" \
  --hf_model_name "your-username/your-model-name" \
  --hf_token "your_huggingface_token"
```

### 参数说明

- `--model_path`: 已合并模型的本地路径
- `--hf_model_name`: Hugging Face Hub 上的模型名称 (例如: 'username/model-name')
- `--hf_token`: Hugging Face API 令牌
- `--private`: (可选) 设置仓库为私有
- `--commit_message`: (可选) 自定义提交信息，默认为"上传合并后的模型"

### 创建私有仓库

如果您想将模型上传到私有仓库:

```bash
***REMOVED***
  --model_path "./exported_model" \
  --hf_model_name "your-username/your-model-name" \
  --hf_token "your_huggingface_token" \
  --private
```

## 方法二: 使用 Shell 脚本

为了更加方便，您可以使用 `upload_to_huggingface.sh` 脚本。

### 使用步骤

1. 编辑脚本中的参数:
   ```bash
   MODEL_PATH="./exported_model"  # 修改为您的模型路径
   HF_MODEL_NAME="your-username/your-model-name"  # 修改为您的 Hugging Face 用户名和模型名称
   HF_TOKEN="your_huggingface_token"  # 修改为您的 API 令牌
   ```

2. 运行脚本:
   ```bash
   bash upload_to_huggingface.sh
   ```

## 上传后的使用

上传成功后，您可以在 Python 代码中直接使用这个模型:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("your-username/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name")

# 使用模型进行推理
inputs = tokenizer("你好，请问你是谁？", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 注意事项

1. 根据模型大小和网络速度，上传过程可能需要较长时间
2. 请确保您有足够的磁盘空间和稳定的网络连接
3. 如果上传中断，可以再次运行脚本，Hugging Face Hub 支持断点续传
4. 大型模型建议使用 Git LFS 进行上传，`huggingface_hub` 库会自动处理这一点 