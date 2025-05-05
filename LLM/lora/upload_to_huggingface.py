#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将已合并的模型上传到 Hugging Face Hub 的简单脚本
"""

import os
import argparse
import logging
from huggingface_hub import HfApi, create_repo, login

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="上传已合并的模型到 Hugging Face Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="已合并模型的本地路径"
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        required=True,
        help="Hugging Face Hub 上的模型名称 (例如: 'username/model-name')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face API 令牌"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="设置仓库为私有"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="上传合并后的模型",
        help="提交信息"
    )
    return parser.parse_args()

def upload_to_huggingface(
    model_path: str, 
    hf_model_name: str, 
    token: str, 
    private: bool = False,
    commit_message: str = "上传合并后的模型"
) -> None:
    """
    将合并后的模型上传到 Hugging Face Hub
    
    参数:
        model_path: 模型路径
        hf_model_name: Hugging Face 模型名称
        token: Hugging Face API 令牌
        private: 是否设置为私有仓库
        commit_message: 提交信息
    """
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return
    
    logger.info(f"正在登录 Hugging Face Hub")
    login(token=token)
    
    logger.info(f"正在创建/获取仓库: {hf_model_name}")
    create_repo(hf_model_name, private=private, exist_ok=True)
    
    logger.info(f"正在上传模型到 Hugging Face Hub: {hf_model_name}")
    logger.info(f"这可能需要一些时间，取决于模型大小和您的网络速度...")
    
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_model_name,
        commit_message=commit_message
    )
    
    logger.info(f"模型已成功上传到: https://huggingface.co/{hf_model_name}")
    logger.info(f"您现在可以使用 from_pretrained('{hf_model_name}') 加载此模型")

def main():
    """主函数"""
    args = parse_args()
    
    # 上传到 Hugging Face Hub
    upload_to_huggingface(
        model_path=args.model_path,
        hf_model_name=args.hf_model_name,
        token=args.hf_token,
        private=args.private,
        commit_message=args.commit_message
    )

if __name__ == "__main__":
    main() 