#!/bin/bash

# LLaMA-Factory LoRA微调示例脚本
# 此脚本展示了如何使用llama_factory_fine_tuning.py进行LoRA微调

# 首先安装LLaMA-Factory（如果尚未安装）
echo "安装LLaMA-Factory依赖..."
echo "pip install 'llm-tuner[deepspeed]'"
echo "或者从源码安装: git clone https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory && pip install -e ."
echo ""

# 示例1：基本的微调场景
echo "示例1：基本的LoRA微调"
echo "python llama_factory_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-7b-hf\" \
  --dataset_path \"path/to/your_data.json\" \
  --dataset_format \"alpaca\" \
  --output_dir \"./llama-factory-output\" \
  --lora_rank 8 \
  --lora_alpha 32 \
  --epochs 3 \
  --batch_size 8 \
  --micro_batch_size 1 \
  --learning_rate 5e-5"

# 示例2：使用4位量化训练大型模型（节省显存）
echo -e "\n示例2：使用4位量化训练大型模型"
echo "python llama_factory_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-13b-hf\" \
  --dataset_path \"path/to/your_data.json\" \
  --dataset_format \"alpaca\" \
  --output_dir \"./llama-factory-13b-output\" \
  --lora_rank 16 \
  --lora_alpha 64 \
  --quantization \"4bit\" \
  --batch_size 4 \
  --micro_batch_size 1 \
  --learning_rate 2e-5"

# 示例3：使用ShareGPT对话格式数据集
echo -e "\n示例3：使用ShareGPT对话格式数据集"
echo "python llama_factory_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-7b-chat-hf\" \
  --dataset_path \"path/to/sharegpt_data.json\" \
  --dataset_format \"sharegpt\" \
  --template \"llama2\" \
  --output_dir \"./llama-factory-chat-output\" \
  --lora_rank 8 \
  --lora_alpha 32 \
  --epochs 2 \
  --batch_size 8 \
  --cutoff_len 2048"

# 示例4：指定不同的目标模块，并导出合并模型
echo -e "\n示例4：指定不同的目标模块，并导出合并模型"
echo "python llama_factory_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-7b-hf\" \
  --dataset_path \"path/to/your_data.json\" \
  --dataset_format \"alpaca\" \
  --output_dir \"./llama-factory-output\" \
  --lora_target \"q_proj\" \"v_proj\" \"k_proj\" \"o_proj\" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --export_dir \"./llama-factory-merged\""

# 示例5：使用Wandb跟踪训练过程
echo -e "\n示例5：使用Wandb跟踪训练过程"
echo "python llama_factory_fine_tuning.py \
  --model_name_or_path \"meta-llama/Llama-2-7b-hf\" \
  --dataset_path \"path/to/your_data.json\" \
  --dataset_format \"alpaca\" \
  --output_dir \"./llama-factory-output\" \
  --use_wandb \
  --wandb_project \"my_llama_project\""

# 示例数据格式说明
echo -e "\n数据格式说明："
echo "1. Alpaca格式示例："
echo '{
  "instruction": "写一个关于春天的诗歌",
  "input": "",
  "output": "春风轻拂面，花香满园芳。碧水映蓝天，燕归旧时房。"
}'

echo -e "\n2. ShareGPT格式示例："
echo '{
  "conversations": [
    {"from": "human", "value": "你好，请介绍一下自己。"},
    {"from": "assistant", "value": "我是一个AI助手，由LLaMA模型微调而来..."}
  ]
}'

echo -e "\n使用提示："
echo "1. 对于不同的基础模型，您可能需要调整目标模块(lora_target)参数"
echo "2. 对于有对话性质的数据，推荐使用相应的模板(template)参数"
echo "3. 确保您拥有足够的GPU内存，或使用量化选项减少内存占用"
echo "4. 实际使用前请替换数据集路径为您自己的数据集"
echo "5. LLaMA-Factory还支持WebUI界面，可以考虑使用 'python -m llmtuner' 启动" 