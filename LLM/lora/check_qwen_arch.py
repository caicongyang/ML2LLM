#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AutoConfig
import json
from huggingface_hub import hf_hub_download

def main():
    print("获取 Qwen2.5-0.5B 模型配置...")
    
    # 使用AutoConfig获取模型配置
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    print("\n模型配置:")
    print(config)
    
    # 尝试下载并查看模型架构文件
    try:
        model_info_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B", 
            filename="config.json"
        )
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            
        print("\n模型架构详细信息:")
        print(json.dumps(model_info, indent=2))
        
        # 根据模型架构信息确定可能的目标模块
        print("\n推荐的LoRA目标模块:")
        if "qwen2" in model_info.get("model_type", "").lower():
            # Qwen2 系列模型的目标模块
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            print(target_modules)
        else:
            print("未能识别的模型类型，无法确定推荐的目标模块")
    
    except Exception as e:
        print(f"获取模型架构详细信息失败: {e}")

if __name__ == "__main__":
    main() 