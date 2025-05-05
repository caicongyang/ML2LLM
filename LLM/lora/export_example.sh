***REMOVED***

# 导出 LoRA 微调模型的示例脚本
# 此脚本展示了使用 export_lora_model.py 工具的不同方式
# Example script for exporting LoRA ***REMOVED***ne-tuned models
# This script shows different ways to use the export_lora_model.py utility

# 设置模型路径
# Set your model paths
BASE_MODEL="llama-2-7b-chat-hf"  # 或使用本地路径，如 "./models/llama-2-7b-chat-hf"
LORA_MODEL="./my-lora-adapter"   # 微调后的 LoRA 权重路径
OUTPUT_DIR="./exported-model"    # 保存合并模型的目录

# 示例 1: 基本合并，不导出
# EXAMPLE 1: Basic merge without export
echo "示例 1: 将 LoRA 模型与基础模型合并"
python export_lora_model.py \
  --base_model_path $BASE_MODEL \
  --lora_model_path $LORA_MODEL \
  --output_dir $OUTPUT_DIR

# 示例 2: 上传到 Hugging Face Hub
# 将 "your-username/your-model-name" 替换为您自己的 Hugging Face 模型 ID
# EXAMPLE 2: Upload to Hugging Face Hub
# Replace "your-username/your-model-name" with your own Hugging Face model ID
echo "示例 2: 导出到 Hugging Face Hub"
python export_lora_model.py \
  --base_model_path $BASE_MODEL \
  --lora_model_path $LORA_MODEL \
  --output_dir $OUTPUT_DIR \
  --export_hf \
  --hf_model_name "your-username/your-model-name" \
  --hf_token "your_huggingface_token"  # 替换为您的实际令牌或使用环境变量

# 示例 3: 转换为 GGUF 格式供 Ollama 使用，使用 4 位量化
# EXAMPLE 3: Convert to GGUF format for Ollama with 4-bit quantization
echo "示例 3: 转换为 GGUF 格式供 Ollama 使用"
python export_lora_model.py \
  --base_model_path $BASE_MODEL \
  --lora_model_path $LORA_MODEL \
  --output_dir $OUTPUT_DIR \
  --export_gguf \
  --quantize "q4_0" \
  --ollama_model_name "my-custom-lora"

# 示例 4: 同时导出到 Hugging Face 和 GGUF 格式
# EXAMPLE 4: Export to both Hugging Face and GGUF format
echo "示例 4: 同时导出到 Hugging Face 和 GGUF 格式"
python export_lora_model.py \
  --base_model_path $BASE_MODEL \
  --lora_model_path $LORA_MODEL \
  --output_dir $OUTPUT_DIR \
  --export_hf \
  --hf_model_name "your-username/your-model-name" \
  --hf_token "your_huggingface_token" \
  --export_gguf \
  --quantize "q5_1" \
  --ollama_model_name "my-custom-lora-q5"

# 示例 5: 更高质量的 GGUF 转换 (8 位)
# EXAMPLE 5: Higher quality GGUF conversion (8-bit)
echo "示例 5: 更高质量的 GGUF 转换 (8 位)"
python export_lora_model.py \
  --base_model_path $BASE_MODEL \
  --lora_model_path $LORA_MODEL \
  --output_dir "${OUTPUT_DIR}-q8" \
  --export_gguf \
  --quantize "q8_0" \
  --ollama_model_name "my-custom-lora-q8"

echo "导出示例完成!"
echo "要使用 Ollama 模型，请运行: ollama run my-custom-lora" 