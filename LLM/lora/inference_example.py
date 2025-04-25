#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA微调模型推理示例

本脚本演示了如何加载和使用LoRA微调过的模型进行推理。
它提供了不同场景的示例，包括将LoRA权重与基础模型合并
或保持它们分离的方式。

用法:
    python inference_example.py --base_model <基础模型> --lora_model <lora模型> --prompt "您的提示文本"

作者: ML2LLM 团队
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftCon***REMOVED***g

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用LoRA微调模型进行推理")
    
    parser.add_argument("--base_model", type=str, required=True,
                        help="基础模型名称或路径(例如, 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--lora_model", type=str, required=True,
                        help="LoRA适配器权重的路径")
    parser.add_argument("--prompt", type=str, default="告诉我关于你自己的信息。",
                        help="用于生成的输入提示")
    parser.add_argument("--max_length", type=int, default=512,
                        help="生成文本的最大长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p(核采样)参数")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k采样参数")
    parser.add_argument("--merge_lora", action="store_true",
                        help="将LoRA权重与基础模型合并")
    parser.add_argument("--use_4bit", action="store_true",
                        help="以4位精度加载模型")
    parser.add_argument("--use_8bit", action="store_true",
                        help="以8位精度加载模型")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    print(f"正在从{args.base_model}加载分词器...")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 确保分词器有填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 根据指定加载基础模型（可能使用量化）
    print(f"正在从{args.base_model}加载基础模型...")
    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    
    if args.use_4bit:
        print("使用4位量化...")
        from transformers import BitsAndBytesCon***REMOVED***g
        model_kwargs["quantization_con***REMOVED***g"] = BitsAndBytesCon***REMOVED***g(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.use_8bit:
        print("使用8位量化...")
        model_kwargs["load_in_8bit"] = True
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs
    )
    
    # 加载LoRA配置以检查参数
    peft_con***REMOVED***g = PeftCon***REMOVED***g.from_pretrained(args.lora_model)
    print(f"LoRA配置已加载: target_modules={peft_con***REMOVED***g.target_modules}, r={peft_con***REMOVED***g.r}, lora_alpha={peft_con***REMOVED***g.lora_alpha}")
    
    # 加载LoRA模型
    print(f"正在从{args.lora_model}加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    
    # 如果指定了合并权重（仅适用于非量化模型）
    if args.merge_lora and not (args.use_4bit or args.use_8bit):
        print("正在将LoRA权重与基础模型合并...")
        model = model.merge_and_unload()
        print("模型合并成功！")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 打印模型和可训练参数信息
    print(f"模型已加载: {model.__class__.__name__}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    
    # 准备输入提示
    print("\n" + "="*50)
    print(f"使用提示生成: {args.prompt}")
    print("="*50 + "\n")
    
    # 标记化输入
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 解码并打印生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 