#!/bin/bash

# LoRA权重合并示例脚本
# 此脚本展示了如何使用merge_lora.py合并LoRA适配器与基础模型

# 安装所需的依赖（如有需要）
# pip install -r requirements.txt

# 示例1：基本合并（使用默认参数）
echo "示例1：基本的LoRA合并"
echo "python merge_lora.py \
  --base_model_path \"meta-llama/Llama-2-7b-hf\" \
  --lora_adapter_path \"./lora-output/final\" \
  --output_path \"./merged-model\""

# 示例2：使用4位精度加载大型模型进行合并（节省显存）
echo -e "\n示例2：使用4位精度加载进行合并（适用于大型模型）"
echo "python merge_lora.py \
  --base_model_path \"meta-llama/Llama-2-13b-hf\" \
  --lora_adapter_path \"./lora-output/final\" \
  --output_path \"./merged-model-13b\" \
  --load_in_4bit"

# 示例3：合并之前测试模型（使用自定义提示词）
echo -e "\n示例3：合并前测试模型效果"
echo "python merge_lora.py \
  --base_model_path \"meta-llama/Llama-2-7b-hf\" \
  --lora_adapter_path \"./lora-output/final\" \
  --output_path \"./merged-model-test\" \
  --test_merge \
  --test_prompt \"用中文介绍一下量子计算的基本原理\""

# 示例4：指定特定设备（如使用特定GPU或CPU）
echo -e "\n示例4：指定特定设备进行合并"
echo "python merge_lora.py \
  --base_model_path \"meta-llama/Llama-2-7b-hf\" \
  --lora_adapter_path \"./lora-output/final\" \
  --output_path \"./merged-model-cpu\" \
  --device \"cpu\""

echo -e "\n使用说明："
echo "1. 选择上述示例命令之一，根据您的需求调整参数"
echo "2. 确保您已经有权限访问基础模型和LoRA适配器"
echo "3. 复制命令并在终端中执行"
echo "4. 合并完成后，您可以像使用普通的Hugging Face模型一样使用合并模型" 