#!/bin/bash

# 将合并后的模型上传到 Hugging Face Hub 的简单脚本

# 载入环境变量
if [ -f ".env" ]; then
    echo "正在加载.env文件中的环境变量..."
    export $(grep -v '^#' .env | xargs)
elif [ -f "$(dirname "$0")/.env" ]; then
    echo "正在加载$(dirname "$0")/.env文件中的环境变量..."
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

# 设置参数 - 请在运行前修改这些值
MODEL_PATH="/root/autodl-tmp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-risk"  # 已合并模型的本地路径
HF_MODEL_NAME="caicongyang/DeepSeek-R1-Distill-Qwen-1.5B-risk"  # 更改为您的 Hugging Face 用户名和模型名称

# 从环境变量获取token，如果环境变量不存在则提示用户输入
if [ -z "$HF_TOKEN" ]; then
    echo "警告: 未找到HF_TOKEN环境变量。请确保在.env文件中设置或直接输入。"
    read -p "请输入您的Hugging Face API令牌: " HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo "错误: 必须提供Hugging Face API令牌才能上传模型。"
        echo "您可以从 https://huggingface.co/settings/tokens 获取令牌"
        exit 1
    fi
fi

# 显示上传信息
echo "准备上传模型到 Hugging Face..."
echo "模型路径: $MODEL_PATH"
echo "目标仓库: $HF_MODEL_NAME"

# 执行上传
python upload_to_huggingface.py \
  --model_path "$MODEL_PATH" \
  --hf_model_name "$HF_MODEL_NAME" \
  --hf_token "$HF_TOKEN"

# 如果您想创建私有仓库，请添加 --private 参数
# 例如:
# python upload_to_huggingface.py \
#   --model_path "$MODEL_PATH" \
#   --hf_model_name "$HF_MODEL_NAME" \
#   --hf_token "$HF_TOKEN" \
#   --private

echo "上传完成，请检查脚本输出以确认是否成功。"
echo "如果上传成功，您的模型应该可以在以下地址访问："
echo "https://huggingface.co/$HF_MODEL_NAME" 