# LoRA 模型导出工具 (LoRA Model Export Tools)

此脚本提供了导出 LoRA 微调模型的两种方式:
1. 上传到 Hugging Face Hub
2. 转换为 GGUF 格式以供 Ollama 使用

## 依赖要求 (Requirements)

安装所需依赖:

```bash
pip install transformers peft huggingface_hub torch
```

对于 GGUF 转换，需要:
```bash
pip install llama-cpp-python
```

要与 Ollama 一起使用，请从以下地址安装 Ollama: https://ollama.ai/

## 使用方法 (Usage)

### 基本用法 (Basic Usage)

首先，将 LoRA 权重与基础模型合并:

```bash
python export_lora_model.py \
  --base_model_path "path/to/base/model" \
  --lora_model_path "path/to/lora/adapter" \
  --output_dir "./exported_model"
```

### 导出到 Hugging Face (Export to Hugging Face)

```bash
python export_lora_model.py \
  --base_model_path "path/to/base/model" \
  --lora_model_path "path/to/lora/adapter" \
  --export_hf \
  --hf_model_name "username/model-name" \
  --hf_token "your_huggingface_token"
```

您可以从以下地址获取 Hugging Face 令牌: https://huggingface.co/settings/tokens

### 转换为 GGUF 格式供 Ollama 使用 (Convert to GGUF for Ollama)

```bash
python export_lora_model.py \
  --base_model_path "path/to/base/model" \
  --lora_model_path "path/to/lora/adapter" \
  --export_gguf \
  --quantize "q4_0" \
  --ollama_model_name "my-lora-model"
```

### 同时导出两种格式 (Export to Both Formats)

您可以结合两种导出方法:

```bash
python export_lora_model.py \
  --base_model_path "path/to/base/model" \
  --lora_model_path "path/to/lora/adapter" \
  --export_hf \
  --hf_model_name "username/model-name" \
  --hf_token "your_huggingface_token" \
  --export_gguf \
  --quantize "q4_0" \
  --ollama_model_name "my-lora-model"
```

## 量化选项 (Quantization Options)

转换为 GGUF 时，您可以选择不同的量化方法:

- `q4_0`: 4 位量化 (最快，准确度最低)
- `q4_1`: 4 位量化，精度更高
- `q5_0`: 5 位量化
- `q5_1`: 5 位量化，精度更高
- `q8_0`: 8 位量化 (较慢，更准确)
- `f16`: 16 位浮点 (高精度，内存占用更大)
- `f32`: 32 位浮点 (最高精度，内存占用最大)

## 与 Ollama 一起使用 (Using with Ollama)

导出为 GGUF 并创建 Ollama 模型后，您可以使用以下命令运行:

```bash
ollama run my-lora-model
```

或者在您的应用程序中使用 Ollama API。 