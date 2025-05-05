#!/bin/bash

# 这是展示如何使用LoRA微调脚本的示例脚本

# 安装
echo "正在安装所需的包..."
pip install -r requirements.txt

# 示例1：使用Llama-2-7b进行基本LoRA微调
echo "示例1：基本LoRA微调设置..."
echo "python lora_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-7b-hf\" \
  --dataset_path \"path/to/your_dataset.json\" \
  --output_dir \"./lora-llama-2-output\" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --max_seq_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --fp16"

# 示例2：使用4位量化微调更大的模型
echo "示例2：使用4位量化微调更大的模型..."
echo "python lora_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-13b-hf\" \
  --dataset_path \"path/to/your_dataset.json\" \
  --output_dir \"./lora-llama-2-13b-output\" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --use_4bit \
  --gradient_accumulation_steps 16 \
  --per_device_train_batch_size 1 \
  --learning_rate 2e-4 \
  --max_seq_length 1024 \
  --fp16"

# 示例3：使用Hugging Face数据集微调
echo "示例3：使用Hugging Face数据集微调..."
echo "python lora_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-7b-hf\" \
  --dataset_path \"tatsu-lab/alpaca\" \
  --text_column \"text\" \
  --output_dir \"./lora-llama-2-alpaca\" \
  --lora_rank 8 \
  --num_train_epochs 2 \
  --learning_rate 3e-4 \
  --fp16"

# 示例4：使用8位量化 
echo "示例4：使用8位量化..."
echo "python lora_fine_tuning.py \
  --model_name_or_path \"mistralai/Mistral-7B-v0.1\" \
  --dataset_path \"path/to/your_dataset.json\" \
  --output_dir \"./lora-mistral-output\" \
  --lora_rank 8 \
  --use_8bit \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 2 \
  --learning_rate 3e-4 \
  --max_seq_length 512 \
  --fp16"

echo "要运行这些示例中的任何一个，复制命令并直接执行它。"
echo "确保将'path/to/your_dataset.json'替换为您实际的数据集路径。" 