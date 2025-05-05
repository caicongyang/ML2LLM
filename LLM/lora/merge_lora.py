#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA权重合并脚本

此脚本用于将LoRA适配器权重与基础模型合并，生成一个完整的微调模型。
合并后的模型可以像普通模型一样使用，无需额外的PEFT库支持。

用法:
    python merge_lora.py --base_model_path <基础模型路径> 
                        --lora_adapter_path <LoRA适配器路径> 
                        --output_path <输出路径>

作者: ML2LLM 团队
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="合并LoRA适配器与基础模型")
    
    # 必需参数
    parser.add_argument("--base_model_path", type=str, required=True,
                      help="基础模型的路径或Hugging Face模型ID (例如 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                      help="LoRA适配器的路径")
    parser.add_argument("--output_path", type=str, required=True,
                      help="合并后模型的保存路径")
    
    # 可选参数
    parser.add_argument("--load_in_8bit", action="store_true",
                      help="以8位精度加载模型（仅用于检查，合并后仍会以全精度保存）")
    parser.add_argument("--load_in_4bit", action="store_true",
                      help="以4位精度加载模型（仅用于检查，合并后仍会以全精度保存）")
    parser.add_argument("--device", type=str, default="auto",
                      help="使用的设备，'auto'表示自动检测")
    parser.add_argument("--test_merge", action="store_true",
                      help="保存前测试合并模型")
    parser.add_argument("--test_prompt", type=str, default="你好，请介绍一下自己。",
                      help="用于测试的提示词")
    parser.add_argument("--save_tokenizer", action="store_true", default=True,
                      help="是否保存分词器")
    
    return parser.parse_args()

def print_section(title):
    """打印带有分隔符的节标题"""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50 + "\n")

def main():
    """主函数"""
    args = parse_args()
    
    print_section("参数信息")
    print(f"基础模型路径: {args.base_model_path}")
    print(f"LoRA适配器路径: {args.lora_adapter_path}")
    print(f"输出路径: {args.output_path}")
    
    # 加载LoRA配置以显示信息
    print_section("加载LoRA配置")
    try:
        peft_config = PeftConfig.from_pretrained(args.lora_adapter_path)
        print(f"LoRA配置: ")
        print(f"  - 任务类型: {peft_config.task_type}")
        print(f"  - 目标模块: {peft_config.target_modules}")
        print(f"  - LoRA秩 (r): {peft_config.r}")
        print(f"  - LoRA Alpha: {peft_config.lora_alpha}")
        print(f"  - LoRA Dropout: {peft_config.lora_dropout}")
    except Exception as e:
        print(f"读取LoRA配置时出错: {e}")
        print("将继续尝试加载适配器...")
    
    # 设置模型加载参数
    print_section("加载基础模型")
    model_kwargs = {"trust_remote_code": True}
    
    # 配置设备映射
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    elif args.device == "cpu":
        model_kwargs["device_map"] = "cpu"
    else:
        # 指定特定GPU
        model_kwargs["device_map"] = args.device
    
    # 配置加载精度
    if args.load_in_8bit and args.load_in_4bit:
        print("警告: 同时指定了8位和4位精度，将使用4位精度")
        args.load_in_8bit = False
        
    if args.load_in_4bit:
        print("使用4位精度加载基础模型...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        print("使用8位精度加载基础模型...")
        model_kwargs["load_in_8bit"] = True
    else:
        print("使用全精度加载基础模型...")
    
    # 加载基础模型
    print(f"正在加载基础模型: {args.base_model_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            **model_kwargs
        )
        print("基础模型加载成功！")
    except Exception as e:
        print(f"加载基础模型时出错: {e}")
        return
    
    # 加载分词器
    print("正在加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("分词器加载成功！")
    except Exception as e:
        print(f"加载分词器时出错: {e}")
        tokenizer = None
    
    # 加载LoRA适配器
    print_section("加载并合并LoRA适配器")
    print(f"正在加载LoRA适配器: {args.lora_adapter_path}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_adapter_path,
            is_trainable=False  # 确保我们不进入训练模式
        )
        print("LoRA适配器加载成功！")
    except Exception as e:
        print(f"加载LoRA适配器时出错: {e}")
        return
    
    # 打印当前可训练参数信息
    model.print_trainable_parameters()
    
    # 合并模型权重
    print("正在合并LoRA权重与基础模型...")
    try:
        merged_model = model.merge_and_unload()
        print("LoRA权重已成功合并！")
    except Exception as e:
        print(f"合并模型时出错: {e}")
        return
    
    # 如果需要，测试合并后的模型
    if args.test_merge and tokenizer is not None:
        print_section("测试合并后的模型")
        try:
            print(f"测试提示词: '{args.test_prompt}'")
            merged_model.eval()
            
            inputs = tokenizer(args.test_prompt, return_tensors="pt").to(merged_model.device)
            with torch.no_grad():
                outputs = merged_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"模型响应: '{response}'")
        except Exception as e:
            print(f"测试模型时出错: {e}")
    
    # 保存合并后的模型
    print_section("保存合并后的模型")
    print(f"正在保存合并后的模型到: {args.output_path}")
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(args.output_path, exist_ok=True)
        
        # 确保节省硬盘空间，使用最佳的保存设置
        merged_model.save_pretrained(
            args.output_path,
            safe_serialization=True
        )
        print("合并后的模型已保存！")
        
        # 保存分词器
        if args.save_tokenizer and tokenizer is not None:
            tokenizer.save_pretrained(args.output_path)
            print("分词器已保存！")
            
    except Exception as e:
        print(f"保存模型时出错: {e}")
        return
    
    print_section("完成")
    print(f"成功将LoRA适配器 '{args.lora_adapter_path}' 与基础模型 '{args.base_model_path}' 合并。")
    print(f"合并后的模型保存在: '{args.output_path}'")
    print("您现在可以像使用普通模型一样使用此合并模型:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"model = AutoModelForCausalLM.from_pretrained('{args.output_path}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{args.output_path}')")
    print("```")

if __name__ == "__main__":
    main() 