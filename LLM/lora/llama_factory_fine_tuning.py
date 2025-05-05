#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LLaMA-Factory的LoRA微调脚本

本脚本使用LLaMA-Factory框架对大型语言模型进行LoRA微调。
LLaMA-Factory框架提供了更集成、更易用的微调体验，特别适合指令微调场景。

用法:
    python llama_factory_fine_tuning.py --model_name_or_path <基础模型> 
                                        --dataset_path <数据集路径> 
                                        --output_dir <输出目录>
                                        
作者: ML2LLM 团队
"""

import os
import argparse
import torch
import json
import logging
from typing import Optional, Dict, Any, List

# 导入LLaMA-Factory相关模块
from llmtuner import train
from llmtuner.webui.utils import get_model_path
from llmtuner.extras.constants import TRAINING_STAGES

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用LLaMA-Factory进行LoRA微调")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="基础模型路径或HuggingFace模型名称")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，用于保存模型和检查点")
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集文件路径或目录")
    parser.add_argument("--dataset_format", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt", "belle", "baize", "moss", "wizard", "raw"],
                        help="数据集格式 (alpaca, sharegpt, belle等)")
    parser.add_argument("--template", type=str, default="default",
                        help="提示模板类型")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA适配器的秩")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout概率")
    parser.add_argument("--lora_target", type=str, nargs="+", 
                        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="要应用LoRA的目标模块")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批量大小")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="梯度累积的微批量大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--cutoff_len", type=int, default=1024,
                        help="输入序列的截断长度")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步骤")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存检查点的步骤")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热步骤比例")
    
    # 量化参数
    parser.add_argument("--quantization", type=str, default=None, 
                        choices=[None, "4bit", "8bit"],
                        help="模型量化类型 (None, 4bit, 8bit)")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用Weights & Biases记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="llama_factory_lora",
                        help="W&B项目名称")
    parser.add_argument("--device", type=str, default=None,
                        help="设备('cpu', 'cuda', 'cuda:0'等)")
    parser.add_argument("--export_dir", type=str, default=None,
                        help="导出合并后模型的目录，如果不指定则不合并")
    
    return parser.parse_args()

def prepare_dataset(dataset_path: str, dataset_format: str) -> List[Dict[str, Any]]:
    """
    准备数据集，将数据加载为LLaMA-Factory可处理的格式
    
    参数:
        dataset_path: 数据集路径
        dataset_format: 数据集格式
        
    返回:
        处理后的数据集
    """
    logger.info(f"正在加载数据集: {dataset_path}")
    
    # 如果是目录，则加载该目录下所有json文件
    if os.path.isdir(dataset_path):
        data_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                     if f.endswith('.json')]
        all_data = []
        for file in data_files:
            with open(file, 'r', encoding='utf-8') as f:
                all_data.extend(json.load(f))
        return all_data
    
    # 如果是单个文件
    elif os.path.isfile(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.endswith('.json'):
                return json.load(f)
            elif dataset_path.endswith('.jsonl'):
                return [json.loads(line) for line in f.readlines() if line.strip()]
    
    raise ValueError(f"无法加载数据集: {dataset_path}。支持的格式: .json, .jsonl")

def setup_llama_factory_args(args) -> Dict[str, Any]:
    """
    将命令行参数转换为LLaMA-Factory所需的配置格式
    
    参数:
        args: 命令行参数
        
    返回:
        LLaMA-Factory配置字典
    """
    # 确定设备配置
    if args.device:
        device_map = args.device
    else:
        device_map = "auto"
    
    # 构建LLaMA-Factory的模型参数
    model_args = {
        "model_name_or_path": args.model_name_or_path,
        "cache_dir": None,
        "use_fast_tokenizer": True,
        "trust_remote_code": True,
        "device_map": device_map,
        "finetuning_type": "lora",  # 使用LoRA
        "resume_lora_training": True,
        "use_auth_token": False
    }
    
    # 使用量化
    if args.quantization == "4bit":
        model_args.update({
            "quantization_bit": 4,
            "double_quantization": True,
            "quant_type": "nf4"
        })
    elif args.quantization == "8bit":
        model_args.update({
            "load_in_8bit": True
        })

    # 数据参数
    data_args = {
        "dataset": args.dataset_format,
        "dataset_dir": os.path.dirname(args.dataset_path),
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "val_size": args.val_size
    }
    
    # 训练参数
    training_args = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.batch_size // args.micro_batch_size,
        "per_device_eval_batch_size": args.micro_batch_size,
        "evaluation_strategy": "steps",
        "eval_steps": args.save_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "logging_steps": args.logging_steps,
        "report_to": "wandb" if args.use_wandb else "none",
        "overwrite_output_dir": True,
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "remove_unused_columns": False
    }
    
    # LoRA配置
    lora_args = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.lora_target,
        "task_type": "CAUSAL_LM"
    }
    
    # 整合为LLaMA-Factory格式
    llama_factory_args = {
        "stage": "sft",  # 监督微调阶段
        "do_train": True,
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args,
        "lora_args": lora_args
    }
    
    return llama_factory_args

def main():
    """主函数：解析参数并运行LLaMA-Factory训练"""
    args = parse_args()
    
    # 打印配置信息
    logger.info("="*50)
    logger.info("使用LLaMA-Factory进行LoRA微调")
    logger.info("="*50)
    logger.info(f"基础模型: {args.model_name_or_path}")
    logger.info(f"数据集路径: {args.dataset_path}")
    logger.info(f"数据集格式: {args.dataset_format}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"LoRA参数: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备数据集
    try:
        dataset = prepare_dataset(args.dataset_path, args.dataset_format)
        logger.info(f"成功加载数据集，共 {len(dataset)} 条样本")
        
        # 保存处理后的数据集到LLaMA-Factory格式文件
        dataset_dir = os.path.join(args.output_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_file = os.path.join(dataset_dir, f"{args.dataset_format}.json")
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"数据集已保存到: {dataset_file}")
        
    except Exception as e:
        logger.error(f"处理数据集时出错: {e}")
        return
    
    # 设置LLaMA-Factory参数
    llama_factory_args = setup_llama_factory_args(args)
    
    # 将参数保存到文件，用于记录
    with open(os.path.join(args.output_dir, "training_config.json"), 'w', encoding='utf-8') as f:
        json.dump(llama_factory_args, f, ensure_ascii=False, indent=2)
    
    # 使用LLaMA-Factory进行训练
    try:
        logger.info("开始训练...")
        train.main(llama_factory_args)
        logger.info(f"训练完成！模型已保存到 {args.output_dir}")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        return
    
    # 如果指定了导出目录，则合并LoRA权重并导出完整模型
    if args.export_dir:
        try:
            logger.info(f"正在将LoRA权重合并到基础模型并导出到: {args.export_dir}")
            os.makedirs(args.export_dir, exist_ok=True)
            
            from llmtuner.model import get_model_and_tokenizer
            from llmtuner.extras.misc import prepare_model_for_export
            
            # 加载带有LoRA的模型
            model_args = llama_factory_args["model_args"]
            model_args["checkpoint_dir"] = args.output_dir
            model, tokenizer = get_model_and_tokenizer(model_args)
            
            # 合并权重并导出
            prepare_model_for_export(model, tokenizer, args.export_dir)
            logger.info(f"模型合并并导出完成！路径: {args.export_dir}")
        except Exception as e:
            logger.error(f"导出模型时出错: {e}")
    
    logger.info("="*50)
    logger.info("完成所有操作！")
    logger.info("="*50)

if __name__ == "__main__":
    main() 