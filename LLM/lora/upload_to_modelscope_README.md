# 上传模型到魔塔社区(ModelScope)指南

这个简单的工具可以帮助您将已合并的模型快速上传到魔塔社区(ModelScope)。

## 准备工作

1. 确保您已经完成了 LoRA 模型与基础模型的合并
2. 获取魔塔社区(ModelScope) API 令牌: https://modelscope.cn/my/myaccesstoken
3. 安装必要的依赖:
   ```bash
   pip install modelscope
   ```

## 方法一: 使用 Python 脚本

`upload_to_modelscope.py` 脚本提供了完整的上传功能。

### 基本使用

```bash
python upload_to_modelscope.py \
  --model_path "./exported_model" \
  --model_name "my-lora-model" \
  --access_token "your_modelscope_token"
```

### 参数说明

- `--model_path`: 已合并模型的本地路径
- `--model_name`: 模型名称
- `--access_token`: 魔塔社区 API 令牌
- `--model_id`: (可选) 模型ID，格式为 'username/model-name'，若不指定，将使用登录用户名与model_name组合
- `--private`: (可选) 设置仓库为私有
- `--model_type`: (可选) 模型类型，可选 "nlp", "cv", "audio", "multi-modal", "scientific"，默认为 "nlp"
- `--model_task`: (可选) 模型任务类型，如 'text-generation'，默认为 'text-generation'
- `--model_description`: (可选) 模型描述，默认为 "基于LoRA微调并合并后的模型"
- `--model_tags`: (可选) 模型标签，以逗号分隔，默认为 "lora,llm,fine-tuned"
- `--commit_message`: (可选) 提交信息，默认为 "上传合并后的LoRA模型"

### 创建私有仓库

如果您想将模型上传到私有仓库:

```bash
python upload_to_modelscope.py \
  --model_path "./exported_model" \
  --model_name "my-lora-model" \
  --access_token "your_modelscope_token" \
  --private
```

## 方法二: 使用 Shell 脚本

为了更加方便，您可以使用 `upload_to_modelscope.sh` 脚本。

### 使用步骤

1. 编辑脚本中的参数:
   ```bash
   MODEL_PATH="./exported_model"  # 修改为您的模型路径
   MODEL_NAME="my-lora-model"     # 修改为您的模型名称
   ACCESS_TOKEN="your_modelscope_token"  # 修改为您的 API 令牌
   MODEL_TYPE="nlp"  # 模型类型
   MODEL_TASK="text-generation"  # 模型任务
   MODEL_DESCRIPTION="这是一个基于LoRA微调并合并后的语言模型"  # 模型描述
   MODEL_TAGS="lora,llm,fine-tuned"  # 模型标签
   ```

2. 运行脚本:
   ```bash
   bash upload_to_modelscope.sh
   ```

## 上传后的使用

上传成功后，您可以在 Python 代码中直接使用这个模型:

```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 加载模型
model_id = "username/my-lora-model"  # 替换为您的用户名和模型名称
pipe = pipeline(Tasks.text_generation, model=model_id)

# 使用模型进行推理
result = pipe("你好，请问你是谁？")
print(result)
```

## 注意事项

1. 根据模型大小和网络速度，上传过程可能需要较长时间
2. 请确保您有足够的磁盘空间和稳定的网络连接
3. 上传前脚本会自动为您创建配置文件和README.md
4. 魔塔社区要求模型有明确的任务类型和说明，因此请确保提供适当的参数
5. 如果上传中断，可以重新运行脚本继续上传
6. 上传成功后，请访问魔塔社区检查模型信息，确保一切正常

## 魔塔社区(ModelScope)与Hugging Face的区别

1. 魔塔社区需要指定更详细的模型类型和任务信息
2. 魔塔社区使用不同的API访问方式
3. 魔塔社区需要configuration.json配置文件（脚本会自动创建）
4. 访问模型的URL格式为 `https://modelscope.cn/models/username/model-name` 