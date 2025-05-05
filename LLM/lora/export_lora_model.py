#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导出 LoRA 微调后的模型，支持两种方式:
1. 上传到 Hugging Face Hub
2. 转换为 GGUF 格式以便 Ollama 使用
"""

import os
import argparse
import logging
from pathlib import Path
import subprocess
import shutil
from typing import Optional, Union, List
from huggingface_hub import HfApi, create_repo, login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoCon***REMOVED***g
from peft import PeftModel, PeftCon***REMOVED***g

# 设置日志
logging.basicCon***REMOVED***g(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="导出 LoRA 微调后的模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="LoRA 适配器模型路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exported_model",
        help="合并后模型的保存目录"
    )
    parser.add_argument(
        "--export_hf",
        action="store_true",
        help="是否导出到 Hugging Face Hub"
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        help="Hugging Face Hub 上的模型名称 (例如: 'username/model-name')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face API 令牌"
    )
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="是否导出为 GGUF 格式以供 Ollama 使用"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        default="q4_0",
        help="GGUF 转换的量化类型"
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        help="Ollama 模型的名称"
    )
    return parser.parse_args()

def merge_lora_to_base_model(base_model_path: str, lora_model_path: str, output_dir: str) -> str:
    """
    将 LoRA 权重合并到基础模型中
    
    参数:
        base_model_path: 基础模型路径
        lora_model_path: LoRA 适配器路径
        output_dir: 输出目录
        
    返回:
        合并后模型的路径
    """
    logger.info(f"正在从 {base_model_path} 加载基础模型")
    
    # 加载配置
    peft_con***REMOVED***g = PeftCon***REMOVED***g.from_pretrained(lora_model_path)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 加载 LoRA 模型
    logger.info(f"正在从 {lora_model_path} 加载 LoRA 适配器")
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    # 合并权重
    logger.info("正在将 LoRA 权重与基础模型合并")
    model = model.merge_and_unload()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并后的模型
    logger.info(f"正在将合并后的模型保存到 {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # 保存配置
    con***REMOVED***g = AutoCon***REMOVED***g.from_pretrained(base_model_path, trust_remote_code=True)
    con***REMOVED***g.save_pretrained(output_dir)
    
    return output_dir

def export_to_huggingface(model_path: str, hf_model_name: str, token: str) -> None:
    """
    将合并后的模型导出到 Hugging Face Hub
    
    参数:
        model_path: 模型路径
        hf_model_name: Hugging Face 模型名称
        token: Hugging Face API 令牌
    """
    logger.info(f"正在登录 Hugging Face Hub")
    login(token=token)
    
    logger.info(f"正在创建/获取仓库: {hf_model_name}")
    create_repo(hf_model_name, private=False, exist_ok=True)
    
    logger.info(f"正在上传模型到 Hugging Face Hub: {hf_model_name}")
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_model_name,
        commit_message="上传合并后的 LoRA 模型"
    )
    
    logger.info(f"模型已成功上传到: https://huggingface.co/{hf_model_name}")

def convert_to_gguf(model_path: str, output_dir: str, quantize: str) -> str:
    """
    使用 llama.cpp 将模型转换为 GGUF 格式
    
    参数:
        model_path: 模型路径
        output_dir: 输出目录
        quantize: 量化类型
        
    返回:
        GGUF 文件路径
    """
    logger.info("正在将模型转换为 GGUF 格式")
    
    # 检查是否安装了 llama.cpp 工具
    try:
        subprocess.run(["which", "python3", "-m", "llama_cpp.server"], check=True)
    except subprocess.CalledProcessError:
        logger.info("正在安装 llama-cpp-python 及 GGUF 支持")
        subprocess.run(["pip", "install", "llama-cpp-python"], check=True)
    
    # 创建转换临时目录
    gguf_output_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    # 使用 llama.cpp 或 transformers 的 convert.py 转换为 GGUF
    gguf_path = os.path.join(gguf_output_dir, f"model-{quantize}.gguf")
    
    logger.info(f"正在将模型转换为 GGUF 格式，量化类型: {quantize}")
    try:
        subprocess.run([
            "python3", "-m", "llama_cpp.convert_hf_to_gguf",
            "--model-dir", model_path,
            "--out***REMOVED***le", gguf_path,
            "--outtype", quantize
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"转换为 GGUF 时出错: {e}")
        raise
    
    logger.info(f"模型已转换为 GGUF 格式: {gguf_path}")
    return gguf_path

def create_ollama_model(gguf_path: str, model_name: str) -> None:
    """
    从 GGUF 文件创建 Ollama 模型
    
    参数:
        gguf_path: GGUF 文件路径
        model_name: Ollama 模型名称
    """
    logger.info(f"正在创建 Ollama 模型: {model_name}")
    
    # 创建 Model***REMOVED***le
    model***REMOVED***le_content = f"""
FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "User:"
PARAMETER stop "Assistant:"
SYSTEM You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
"""
    
    model***REMOVED***le_path = os.path.join(os.path.dirname(gguf_path), "Model***REMOVED***le")
    with open(model***REMOVED***le_path, "w") as f:
        f.write(model***REMOVED***le_content)
    
    # 创建 Ollama 模型
    try:
        subprocess.run([
            "ollama", "create", model_name,
            "-f", model***REMOVED***le_path
        ], check=True)
        logger.info(f"Ollama 模型已创建: {model_name}")
        logger.info(f"现在您可以使用以下命令运行模型: ollama run {model_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"创建 Ollama 模型时出错: {e}")
        logger.info("如果未安装 Ollama，请从 https://ollama.ai/ 安装")

def main():
    """主函数"""
    args = parse_args()
    
    # 将 LoRA 模型与基础模型合并
    merged_model_path = merge_lora_to_base_model(
        args.base_model_path,
        args.lora_model_path,
        args.output_dir
    )
    
    # 导出到 Hugging Face Hub
    if args.export_hf:
        if not args.hf_model_name or not args.hf_token:
            logger.error("要导出到 Hugging Face，请提供 --hf_model_name 和 --hf_token 参数")
        else:
            export_to_huggingface(merged_model_path, args.hf_model_name, args.hf_token)
    
    # 导出为 GGUF 格式以供 Ollama 使用
    if args.export_gguf:
        gguf_path = convert_to_gguf(merged_model_path, args.output_dir, args.quantize)
        
        # 如果提供了名称，则创建 Ollama 模型
        if args.ollama_model_name:
            create_ollama_model(gguf_path, args.ollama_model_name)
        else:
            logger.info(f"GGUF 模型已保存到: {gguf_path}")
            logger.info("要与 Ollama 一起使用，请提供 --ollama_model_name 参数")

if __name__ == "__main__":
    main() 