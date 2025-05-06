#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将模型转换为GGUF格式并创建Ollama模型的工具类
"""

import os
import logging
import subprocess
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelToOllamaConverter:
    """将模型转换为GGUF格式并创建Ollama模型的工具类"""
    
    def __init__(self, check_dependencies: bool = True):
        """
        初始化转换器
        
        参数:
            check_dependencies: 是否检查依赖项
        """
        if check_dependencies:
            self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """检查必要的依赖项"""
        logger.info("检查依赖项...")
        
        # 检查llama-cpp-python是否安装
        try:
            subprocess.run(["which", "python", "-m", "llama_cpp.server"], check=True, capture_output=True)
            logger.info("llama-cpp-python已安装")
        except subprocess.CalledProcessError:
            logger.info("正在安装llama-cpp-python...")
            subprocess.run(["pip", "install", "llama-cpp-python"], check=True)
        
        # 检查Ollama是否安装
        try:
            subprocess.run(["which", "ollama"], check=True, capture_output=True)
            logger.info("Ollama已安装")
        except subprocess.CalledProcessError:
            logger.warning("未找到Ollama，请从https://ollama.ai/安装")
    
    def convert_to_gguf(self, 
                      model_path: str, 
                      output_dir: str, 
                      quantize: str = "q4_0") -> str:
        """
        使用llama.cpp将模型转换为GGUF格式
        
        参数:
            model_path: 模型路径
            output_dir: 输出目录
            quantize: 量化类型，可选值: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32
            
        返回:
            GGUF文件路径
        """
        logger.info("正在将模型转换为GGUF格式")
        
        # 创建转换临时目录
        gguf_output_dir = os.path.join(output_dir, "gguf")
        os.makedirs(gguf_output_dir, exist_ok=True)
        
        # 使用llama.cpp转换为GGUF
        gguf_path = os.path.join(gguf_output_dir, f"model-{quantize}.gguf")
        
        logger.info(f"正在将模型转换为GGUF格式，量化类型: {quantize}")
        try:
            subprocess.run([
                "python", "-m", "llama_cpp.convert_hf_to_gguf",
                "--model-dir", model_path,
                "--outfile", gguf_path,
                "--outtype", quantize
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"转换为GGUF时出错: {e}")
            raise
        
        logger.info(f"模型已转换为GGUF格式: {gguf_path}")
        return gguf_path
    
    def create_ollama_model(self, 
                          gguf_path: str, 
                          model_name: str,
                          system_prompt: Optional[str] = None,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          top_k: int = 40,
                          repeat_penalty: float = 1.1,
                          stop_words: Optional[list] = None) -> None:
        """
        从GGUF文件创建Ollama模型
        
        参数:
            gguf_path: GGUF文件路径
            model_name: Ollama模型名称
            system_prompt: 系统提示词，默认为通用助手提示
            temperature: 温度参数
            top_p: top-p参数
            top_k: top-k参数
            repeat_penalty: 重复惩罚参数
            stop_words: 停止词列表，默认为["User:", "Assistant:"]
        """
        logger.info(f"正在创建Ollama模型: {model_name}")
        
        # 准备系统提示词
        if system_prompt is None:
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
        
        # 准备停止词
        if stop_words is None:
            stop_words = ["User:", "Assistant:"]
        
        # 创建Modelfile内容
        modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k {top_k}
PARAMETER repeat_penalty {repeat_penalty}
"""
        
        # 添加停止词
        for stop_word in stop_words:
            modelfile_content += f'PARAMETER stop "{stop_word}"\n'
        
        # 添加系统提示词
        modelfile_content += f"SYSTEM {system_prompt}\n"
        
        # 写入Modelfile
        modelfile_path = os.path.join(os.path.dirname(gguf_path), "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        # 创建Ollama模型
        try:
            subprocess.run([
                "ollama", "create", model_name,
                "-f", modelfile_path
            ], check=True)
            logger.info(f"Ollama模型已创建: {model_name}")
            logger.info(f"现在您可以使用以下命令运行模型: ollama run {model_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"创建Ollama模型时出错: {e}")
            logger.info("如果未安装Ollama，请从https://ollama.ai/安装")

def main():
    """主函数，包含硬编码的参数和转换流程"""
    # 配置参数（直接硬编码在代码中）
    # ===================================================================
    # 必需参数
    model_path = "/Users/caicongyang/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-risk"  # 替换为您的模型路径
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B-risk"             # 替换为您想要的Ollama模型名称
    
    # 可选参数
    output_dir = "./output"             # 输出目录
    quantize = "q4_0"                   # 量化类型: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32
    system_prompt = "你是一个有帮助、尊重和诚实的助手。请始终尽可能有帮助地回答，同时注意安全。"
    temperature = 0.7                   # 温度参数(0.0-1.0)
    top_p = 0.9                         # top-p参数(0.0-1.0)
    top_k = 40                          # top-k参数
    repeat_penalty = 1.1                # 重复惩罚参数
    # ===================================================================
    
    # 创建转换器实例
    converter = ModelToOllamaConverter()
    
    # 转换为GGUF
    gguf_path = converter.convert_to_gguf(
        model_path=model_path,
        output_dir=output_dir,
        quantize=quantize
    )
    
    # 创建Ollama模型
    converter.create_ollama_model(
        gguf_path=gguf_path,
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty
    )

if __name__ == "__main__":
    main() 