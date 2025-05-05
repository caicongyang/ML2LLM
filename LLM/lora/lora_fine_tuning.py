#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA (低秩适应) 微调脚本

本脚本实现了针对大型语言模型(LLMs)的LoRA微调。
LoRA是一种高效的微调方法，通过学习模型权重的低秩分解矩阵，
而不是微调整个模型，从而显著减少了可训练参数的数量。

用法:
    python lora_fine_tuning.py --model_name_or_path <基础模型> --dataset_path <数据集路径> 
    --output_dir <输出目录> --lora_rank <秩> --lora_alpha <alpha值> --lora_dropout <dropout值>

作者: ML2LLM 团队
"""

import os
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LLM的LoRA微调脚本")
    
    # 基础模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型的路径或huggingface.co/models的模型标识符")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="如果与模型路径不同，则指定分词器名称或路径")
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集文件路径或huggingface数据集标识符")
    parser.add_argument("--text_column", type=str, default="text",
                        help="包含文本数据的列名")
    
    # LoRA特定参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA分解矩阵的秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha参数 - 缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA层的丢弃概率")
    parser.add_argument("--target_modules", type=str, default=None, nargs="+",
                        help="应用LoRA适配的模块名称列表")
    
    # 训练超参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="训练时每个GPU/TPU的批量大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="评估时每个GPU/TPU的批量大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="应用于模型参数的权重衰减")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练的总轮数")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="模型能处理的最大序列长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="反向传播前累积的更新步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="学习率预热的训练步数比例")
    
    # 输出/保存参数
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存模型检查点的目录")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="每X个更新步骤记录一次指标")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="每X个更新步骤保存一次检查点")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="保留的检查点最大数量")
    
    # 量化参数
    parser.add_argument("--use_4bit", action="store_true",
                        help="使用4位精度加载模型(使用bitsandbytes)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="使用8位精度加载模型(使用bitsandbytes)")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="可重现性的随机种子")
    parser.add_argument("--fp16", action="store_true",
                        help="使用混合精度训练")
    parser.add_argument("--bf16", action="store_true",
                        help="使用bfloat16混合精度训练(如果可用)")
    
    return parser.parse_args()

def prepare_dataset(tokenizer, args):
    """
    加载并预处理用于训练的数据集。
    
    参数:
        tokenizer: 用于预处理的分词器
        args: 命令行参数
        
    返回:
        预处理后的数据集和数据整理器
    """
    # 从文件或Hugging Face hub加载数据集
    if os.path.exists(args.dataset_path):
        dataset_format = args.dataset_path.split(".")[-1]
        if dataset_format == "json":
            dataset = load_dataset("json", data_files=args.dataset_path)
        elif dataset_format == "csv":
            dataset = load_dataset("csv", data_files=args.dataset_path)
        else:
            raise ValueError(f"不支持的数据集格式: {dataset_format}")
    else:
        # 从Hugging Face数据集hub加载
        dataset = load_dataset(args.dataset_path)
    
    # 如果数据集没有'train'分割，则分割数据集
    if "train" not in dataset:
        # 将数据集分割为训练集和验证集
        dataset = dataset["train"].train_test_split(test_size=0.05, seed=args.seed)
    
    print(f"数据集已加载: {dataset}")
    
    # 定义输入标记化函数
    def tokenize_function(examples):
        # 使用适当的填充和截断对文本进行标记化
        result = tokenizer(
            examples[args.text_column],
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    # 标记化数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != args.text_column],
        desc="正在标记化数据集",
    )
    
    # 创建用于语言建模的数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 我们在做因果语言建模，而不是掩码语言建模
    )
    
    return tokenized_dataset, data_collator

def main():
    """执行LoRA微调的主函数。"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子以确保可重现性
    set_seed(args.seed)
    
    # 加载分词器
    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
    )
    
    # 确保分词器有填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 确定如何加载模型(量化或全精度)
    model_kwargs = {"trust_remote_code": True}
    if args.use_4bit:
        print("正在加载4位精度模型...")
        from bitsandbytes.nn import Linear4bit
        from transformers import BitsAndBytesConfig
        
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        is_quantized = True
    elif args.use_8bit:
        print("正在加载8位精度模型...")
        model_kwargs["load_in_8bit"] = True
        is_quantized = True
    else:
        print("正在加载全精度模型...")
        is_quantized = False
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    
    # 为训练准备模型(对于量化模型)
    if is_quantized:
        model = prepare_model_for_kbit_training(model)
    
    # 如果未指定，为一些常见的LLM架构设置默认目标模块
    if args.target_modules is None:
        # 这些是针对不同模型架构的示例
        if "llama" in args.model_name_or_path.lower():
            args.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in args.model_name_or_path.lower():
            args.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "qwen" in args.model_name_or_path.lower() or "qwen2" in args.model_name_or_path.lower():
            args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt-neox" in args.model_name_or_path.lower():
            args.target_modules = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
        elif "falcon" in args.model_name_or_path.lower():
            args.target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:
            # 其他架构的默认值
            args.target_modules = ["query", "key", "value", "attention.output.dense", "output.dense"]
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    
    # 将LoRA适配器添加到模型中
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    # 准备数据集
    tokenized_dataset, data_collator = prepare_dataset(tokenizer, args)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=4,
        group_by_length=True,
        report_to="tensorboard",
        run_name=f"lora-{args.model_name_or_path.split('/')[-1]}",
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", tokenized_dataset.get("test", None)),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 训练模型
    print("开始LoRA微调...")
    trainer.train()
    
    # 保存最终模型和分词器
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 保存LoRA模型和分词器
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"LoRA微调完成。模型已保存到 {final_output_dir}")

if __name__ == "__main__":
    main() 