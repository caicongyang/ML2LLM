#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将已合并的模型上传到魔塔社区(ModelScope)的简单脚本
根据 https://modelscope.cn/docs/models/upload 官方文档编写
"""

import os
import argparse
import logging
import json
import sys
import time
from pathlib import Path

# 导入ModelScope相关模块
try:
    from modelscope.hub import snapshot_download
    from modelscope.hub.api import HubApi
    from modelscope.utils.constant import ModelFile
    from modelscope.hub.constants import Licenses, ModelVisibility
except ImportError:
    print("请先安装ModelScope: pip install modelscope")
    sys.exit(1)

# 设置日志
logging.basicCon***REMOVED***g(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="上传已合并的模型到魔塔社区(ModelScope)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="已合并模型的本地路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="魔塔社区上的模型名称"
    )
    parser.add_argument(
        "--chinese_name",
        type=str,
        default=None,
        help="模型的中文名称"
    )
    parser.add_argument(
        "--access_token",
        type=str,
        required=True,
        help="魔塔社区 API 令牌"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="模型ID，格式为: 'username/model-name'。若不指定，将使用默认用户名与model_name组合"
    )
    parser.add_argument(
        "--username",
        type=str,
        default="caicongyang",
        help="魔塔社区用户名，默认为caicongyang"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="设置仓库为私有"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="nlp/text-generation",
        help="模型类型和任务，格式为'类型/任务'，例如'nlp/text-generation'，'cv/image-generation'"
    )
    parser.add_argument(
        "--license",
        type=str,
        default="Apache-2.0",
        choices=["Apache-2.0", "MIT", "BSD", "GPL", "LGPL", "CC-BY-NC", "CC-BY-SA", "CC-BY", "CC0"],
        help="模型许可证，默认为Apache-2.0"
    )
    parser.add_argument(
        "--model_description",
        type=str,
        default="基于LoRA微调并合并后的模型",
        help="模型描述"
    )
    parser.add_argument(
        "--model_tags",
        type=str,
        default="lora,llm,***REMOVED***ne-tuned",
        help="模型标签，以逗号分隔"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="上传合并后的LoRA模型",
        help="提交信息"
    )
    parser.add_argument(
        "--con***REMOVED***guration_***REMOVED***le",
        type=str,
        default=None,
        help="自定义配置文件路径，不提供则自动生成"
    )
    return parser.parse_args()

def create_con***REMOVED***guration(args, model_path):
    """创建模型的配置文件"""
    # 解析model_type参数
    if '/' in args.model_type:
        type_parts = args.model_type.split('/')
        model_category = type_parts[0]
        task_name = type_parts[1] if len(type_parts) > 1 else None
    else:
        model_category = args.model_type
        task_name = "text-generation" if model_category == "nlp" else None
    
    # 创建基础配置
    con***REMOVED***g = {
        "task": task_name,
        "framework": "pytorch",
        "model": {
            "type": "language-model"
        },
        "pipeline": {
            "type": task_name or "text-generation"
        },
        "library_name": "transformers"
    }
    
    # 检查是否存在配置文件
    con***REMOVED***g_path = os.path.join(model_path, ModelFile.CONFIGURATION)
    if os.path.exists(con***REMOVED***g_path):
        logger.info(f"发现现有配置文件: {con***REMOVED***g_path}")
        try:
            with open(con***REMOVED***g_path, 'r', encoding='utf-8') as f:
                existing_con***REMOVED***g = json.load(f)
                # 合并配置
                con***REMOVED***g.update(existing_con***REMOVED***g)
                logger.info("已合并现有配置文件")
        except Exception as e:
            logger.warning(f"读取现有配置文件失败: {e}")
    
    # 如果提供了自定义配置文件，使用它覆盖
    if args.con***REMOVED***guration_***REMOVED***le and os.path.exists(args.con***REMOVED***guration_***REMOVED***le):
        try:
            with open(args.con***REMOVED***guration_***REMOVED***le, 'r', encoding='utf-8') as f:
                custom_con***REMOVED***g = json.load(f)
                con***REMOVED***g.update(custom_con***REMOVED***g)
                logger.info(f"已合并自定义配置文件: {args.con***REMOVED***guration_***REMOVED***le}")
        except Exception as e:
            logger.warning(f"读取自定义配置文件失败: {e}")
    
    # 创建README.md
    readme_content = f"""# {args.model_name}

## 模型描述
{args.model_description}

## 使用方法

### 使用ModelScope加载模型
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "{args.model_id or (args.username + '/' + args.model_name)}"
pipe = pipeline(Tasks.text_generation, model=model_id)
result = pipe("你好，请问你是谁？")
print(result)
```

### 使用Transformers加载模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{args.model_id or (args.username + '/' + args.model_name)}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("你好，请问你是谁？", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
"""
    
    return con***REMOVED***g, readme_content

def get_license_enum(license_str):
    """将许可证字符串转换为Licenses枚举值"""
    license_mapping = {
        "Apache-2.0": Licenses.APACHE_V2,
        "MIT": Licenses.MIT,
        "BSD": Licenses.BSD,
        "GPL": Licenses.GPL_V3,
        "LGPL": Licenses.LGPL_V3,
        "CC-BY-NC": Licenses.CC_BY_NC_SA_4_0,
        "CC-BY-SA": Licenses.CC_BY_SA_4_0,
        "CC-BY": Licenses.CC_BY_4_0,
        "CC0": Licenses.CC0_1_0
    }
    return license_mapping.get(license_str, Licenses.APACHE_V2)

def upload_to_modelscope(
    model_path: str, 
    model_name: str,
    access_token: str,
    model_id: str = None,
    username: str = "caicongyang",
    chinese_name: str = None,
    private: bool = False,
    model_type: str = "nlp/text-generation",
    license_str: str = "Apache-2.0",
    model_description: str = "基于LoRA微调并合并后的模型",
    model_tags: str = "lora,llm,***REMOVED***ne-tuned",
    commit_message: str = "上传合并后的LoRA模型",
    con***REMOVED***guration_***REMOVED***le: str = None
) -> None:
    """
    将合并后的模型上传到魔塔社区
    
    参数:
        model_path: 模型路径
        model_name: 模型名称
        access_token: 魔塔社区 API 令牌
        model_id: 模型ID，格式为: 'username/model-name'
        username: 魔塔社区用户名
        chinese_name: 模型的中文名称
        private: 是否设置为私有仓库
        model_type: 模型类型和任务，格式为'类型/任务'
        license_str: 模型许可证
        model_description: 模型描述
        model_tags: 模型标签，以逗号分隔
        commit_message: 提交信息
        con***REMOVED***guration_***REMOVED***le: 自定义配置文件路径
    """
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return
    
    # 初始化参数
    args = argparse.Namespace()
    args.model_name = model_name
    args.model_type = model_type
    args.model_description = model_description
    args.con***REMOVED***guration_***REMOVED***le = con***REMOVED***guration_***REMOVED***le
    
    # 获取或创建模型ID
    if not model_id:
        model_id = f"{username}/{model_name}"
        logger.info(f"使用用户名 '{username}' 创建模型ID: {model_id}")
    args.model_id = model_id
    args.username = username
    
    # 为了使用准确的model_id进行README.md生成
    logger.info(f"正在准备上传模型到: {model_id}")
    
    # 创建配置文件和README
    con***REMOVED***g, readme_content = create_con***REMOVED***guration(args, model_path)
    
    # 写入配置文件到模型目录
    con***REMOVED***g_path = os.path.join(model_path, ModelFile.CONFIGURATION)
    with open(con***REMOVED***g_path, 'w', encoding='utf-8') as f:
        json.dump(con***REMOVED***g, f, ensure_ascii=False, indent=2)
    logger.info(f"已创建配置文件: {con***REMOVED***g_path}")
    
    # 写入README文件
    readme_path = os.path.join(model_path, ModelFile.README)
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    logger.info(f"已创建README文件: {readme_path}")
    
    # 上传模型
    logger.info(f"正在上传模型到魔塔社区: {model_id}")
    logger.info(f"这可能需要一些时间，取决于模型大小和您的网络速度...")
    
    # 处理模型标签
    tags = [tag.strip() for tag in model_tags.split(',') if tag.strip()]
    
    # 初始化API
    hub_api = HubApi()
    hub_api.login(access_token)
    
    try:
        # 步骤1: 创建模型库
        logger.info(f"步骤1: 创建模型库 '{model_id}'")
        visibility = ModelVisibility.PRIVATE if private else ModelVisibility.PUBLIC
      
        
        try:
            hub_api.create_model(
                model_id=model_id,
                visibility=visibility,
                chinese_name=chinese_name
            )
            logger.info(f"✅ 成功创建模型库: {model_id}")
        except Exception as e:
            logger.warning(f"创建模型库时出现警告 (可能仓库已存在): {e}")
        
        # 步骤2: 上传模型文件
        logger.info(f"步骤2: 上传模型文件到 '{model_id}'")
        hub_api.upload_folder(
            repo_id=model_id,
            folder_path=model_path,
            commit_message=commit_message
        )
        logger.info(f"✅ 成功上传所有模型文件")
        
        # 步骤3: 更新模型信息
        try:
            logger.info("步骤3: 更新模型元数据")
            
            # 更新模型描述
            api_url = f"https://api.modelscope.cn/api/v1/models/{model_id}/description"
            response = hub_api.do_put(api_url, json={"description": model_description})
            if response.status_code == 200:
                logger.info("✅ 成功更新模型描述")
            else:
                logger.warning(f"更新模型描述失败: {response.status_code} {response.text}")
            
            # 更新模型标签
            if tags:
                api_url = f"https://api.modelscope.cn/api/v1/models/{model_id}/tags"
                response = hub_api.do_put(api_url, json={"tags": tags})
                if response.status_code == 200:
                    logger.info(f"✅ 成功更新模型标签: {tags}")
                else:
                    logger.warning(f"更新模型标签失败: {response.status_code} {response.text}")
        except Exception as e:
            logger.warning(f"更新模型元数据失败: {e}")
            logger.info("模型已上传，但元数据可能需要手动更新。")
        
        # 显示成功信息
        logger.info("=" * 50)
        logger.info(f"模型已成功上传到: https://modelscope.cn/models/{model_id}")
        logger.info(f"您现在可以使用 Model.from_pretrained('{model_id}') 加载此模型")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"上传模型时出错: {e}")
        raise

def main():
    """主函数"""
    args = parse_args()
    
    # 上传到魔塔社区
    upload_to_modelscope(
        model_path=args.model_path,
        model_name=args.model_name,
        access_token=args.access_token,
        model_id=args.model_id,
        username=args.username,
        chinese_name=args.chinese_name,
        private=args.private,
        model_type=args.model_type,
        license_str=args.license,
        model_description=args.model_description,
        model_tags=args.model_tags,
        commit_message=args.commit_message,
        con***REMOVED***guration_***REMOVED***le=args.con***REMOVED***guration_***REMOVED***le
    )

if __name__ == "__main__":
    main() 