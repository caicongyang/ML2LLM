***REMOVED***

# LoRA微调脚本 - 针对区块链安全专家数据集
# 此脚本演示如何使用转换后的blockchain_peft.jsonl文件进行LoRA微调

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 如果有多个GPU，指定要使用的GPU

# 设置参数
MODEL_NAME="internlm/internlm2-chat-7b"  # 替换为您想要使用的基础模型
DATASET_PATH="LLM/lora/blockchain_peft.jsonl"  # 转换后的数据集路径
OUTPUT_DIR="LLM/lora/output/blockchain-expert"  # 输出目录

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行微调脚本
python LLM/lora/lora_***REMOVED***ne_tuning.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_path $DATASET_PATH \
  --output_dir $OUTPUT_DIR \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_seq_length 2048 \
  --fp16 \
  --text_column "text" \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 3

echo "LoRA微调完成! 模型保存在 $OUTPUT_DIR/***REMOVED***nal" 