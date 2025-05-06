#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从Hugging Face下载模型的类
支持下载Qwen/Qwen3-1.7B等模型
"""

import os
import logging
from typing import Optional, Union, Dict, Any
from huggingface_hub import snapshot_download, HfApi, login

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HuggingFaceModelDownloader:
    """Hugging Face模型下载器类"""
    
    def __init__(self, cache_dir: Optional[str] = None, use_auth_token: Optional[str] = None):
        """
        初始化下载器
        
        参数:
            cache_dir: 模型缓存目录，如果不指定则使用默认的~/.cache/huggingface/hub
            use_auth_token: Hugging Face API令牌，用于下载需要访问权限的模型
        """
        self.cache_dir = cache_dir
        self.use_auth_token = use_auth_token
        
        # 如果提供了令牌，先尝试登录
        if self.use_auth_token:
            try:
                login(token=self.use_auth_token)
                logger.info("成功登录Hugging Face")
            except Exception as e:
                logger.warning(f"登录Hugging Face失败: {e}")
    
    def download_model(self, model_id: str, revision: Optional[str] = None, 
                      local_dir: Optional[str] = None, 
                      local_dir_use_symlinks: bool = True,
                      **kwargs) -> str:
        """
        下载模型
        
        参数:
            model_id: Hugging Face模型ID，如'Qwen/Qwen3-1.7B'
            revision: 模型版本，如'main'、特定的commit hash或tag
            local_dir: 下载到的本地目录，如不指定则使用cache_dir
            local_dir_use_symlinks: 是否使用符号链接以节省空间
            **kwargs: 传递给snapshot_download的其他参数
            
        返回:
            下载的模型本地路径
        """
        try:
            target_dir = local_dir or self.cache_dir
            logger.info(f"正在下载模型 {model_id}{'@'+revision if revision else ''} 到 {target_dir or '默认缓存目录'}")
            
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir if not local_dir else None,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                token=self.use_auth_token,
                **kwargs
            )
            
            logger.info(f"模型下载完成: {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"下载模型 {model_id} 失败: {e}")
            raise
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型的元信息
        
        参数:
            model_id: Hugging Face模型ID
            
        返回:
            模型的元信息字典
        """
        try:
            api = HfApi(token=self.use_auth_token)
            model_info = api.model_info(model_id)
            return {
                'id': model_info.id,
                'sha': model_info.sha,
                'tags': model_info.tags,
                'pipeline_tag': model_info.pipeline_tag,
                'siblings': [s.rfilename for s in model_info.siblings],
                'author': model_info.author,
                'last_modified': model_info.last_modified,
                'private': model_info.private,
            }
        except Exception as e:
            logger.error(f"获取模型 {model_id} 信息失败: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    # 从环境变量获取token（可选）
    import os
    hf_token = os.environ.get("HF_TOKEN")
    
    # 创建下载器实例
    downloader = HuggingFaceModelDownloader(
        cache_dir="/root/autodl-tmp/models",  # 根据需要修改为您的缓存目录
        use_auth_token=hf_token  # 如果需要下载私有模型或有访问限制的模型，请提供token
    )
    
    # 下载Qwen3-1.7B模型
    model_path = downloader.download_model(
        model_id="Qwen/Qwen3-1.7B",
        local_dir="/Users/caicongyang/Downloads/models/Qwen3-1.7B",  # 根据需要修改为您的缓存目录
        # 可选参数:
        # revision="main",  # 指定特定版本
        # local_files_only=False,  # 是否只使用本地文件
        # resume_download=True,  # 是否断点续传
        # proxies=None,  # 代理设置
        # etag_timeout=100,  # etag超时设置
        # force_download=False,  # 是否强制重新下载
    )
    
    print(f"模型已下载到: {model_path}")
    
    # 获取模型信息
    try:
        model_info = downloader.get_model_info("Qwen/Qwen3-1.7B")
        print(f"模型信息: {model_info}")
    except Exception as e:
        print(f"获取模型信息失败: {e}") 