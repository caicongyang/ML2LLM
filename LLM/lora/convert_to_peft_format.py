#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据格式转换脚本：将对话格式的JSONL文件转换为PEFT支持的格式

本脚本将包含多轮对话的JSONL文件转换为PEFT微调所需的格式。
转换后的数据可直接用于lora_***REMOVED***ne_tuning.py脚本进行LoRA微调。

用法:
    python convert_to_peft_format.py --input_***REMOVED***le <输入JSONL文件> --output_***REMOVED***le <输出文件> 
    [--system_template <系统消息模板>] [--conversation_template <对话模板>]

作者: ML2LLM 团队
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将对话JSONL转换为PEFT支持的格式")
    
    parser.add_argument("--input_***REMOVED***le", type=str, required=True,
                        help="输入的JSONL文件路径，包含对话数据")
    parser.add_argument("--output_***REMOVED***le", type=str, required=True,
                        help="输出文件的路径，格式为JSONL")
    parser.add_argument("--system_template", type=str, default="<|system|>\n{system_content}\n",
                        help="系统消息的模板格式")
    parser.add_argument("--user_template", type=str, default="<|user|>\n{user_content}\n",
                        help="用户消息的模板格式")
    parser.add_argument("--assistant_template", type=str, default="<|assistant|>\n{assistant_content}\n",
                        help="助手消息的模板格式")
    parser.add_argument("--conversation_template", type=str, default=None,
                        help="整体对话的模板格式，如Alpaca或ChatML等")
    parser.add_argument("--add_eos", action="store_true", default=False,
                        help="是否在每个对话结束添加EOS标记")
    
    return parser.parse_args()

def read_jsonl(***REMOVED***le_path: str) -> List[Dict[str, Any]]:
    """
    读取JSONL文件
    
    参数:
        ***REMOVED***le_path: JSONL文件路径
        
    返回:
        包含JSON对象的列表
    """
    data = []
    with open(***REMOVED***le_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: 解析行时出错: {e}, 行内容: {line}")
    return data

def format_message(message: Dict[str, str], system_template: str, user_template: str, assistant_template: str) -> str:
    """
    根据角色格式化单条消息
    
    参数:
        message: 包含role和content的消息字典
        system_template: 系统消息模板
        user_template: 用户消息模板
        assistant_template: 助手消息模板
        
    返回:
        格式化后的消息文本
    """
    role = message.get("role", "").lower()
    content = message.get("content", "")
    
    if role == "system":
        return system_template.format(system_content=content)
    elif role == "user":
        return user_template.format(user_content=content)
    elif role == "assistant":
        return assistant_template.format(assistant_content=content)
    else:
        print(f"警告: 未知角色 '{role}'，将被忽略")
        return ""

def format_conversation(messages: List[Dict[str, str]], system_template: str, user_template: str, 
                        assistant_template: str, add_eos: bool = False, conversation_template: Optional[str] = None) -> str:
    """
    格式化一组消息为完整对话
    
    参数:
        messages: 消息列表
        system_template: 系统消息模板
        user_template: 用户消息模板
        assistant_template: 助手消息模板
        add_eos: 是否在对话结束添加EOS标记
        conversation_template: 对话整体模板
        
    返回:
        格式化后的对话文本
    """
    formatted_messages = []
    
    for message in messages:
        formatted_message = format_message(message, system_template, user_template, assistant_template)
        if formatted_message:
            formatted_messages.append(formatted_message)
    
    conversation = "".join(formatted_messages)
    
    if add_eos:
        conversation += "<|endoftext|>"
    
    if conversation_template:
        conversation = conversation_template.format(conversation=conversation)
    
    return conversation

def convert_to_peft_format(input_data: List[Dict[str, Any]], system_template: str, user_template: str, 
                          assistant_template: str, add_eos: bool = False, conversation_template: Optional[str] = None) -> List[Dict[str, str]]:
    """
    将对话数据转换为PEFT支持的格式
    
    参数:
        input_data: 输入的对话数据列表
        system_template: 系统消息模板
        user_template: 用户消息模板
        assistant_template: 助手消息模板
        add_eos: 是否在对话结束添加EOS标记
        conversation_template: 对话整体模板
        
    返回:
        转换后的数据列表，每项包含'text'字段
    """
    peft_data = []
    
    for item in input_data:
        messages = item.get("messages", [])
        if not messages:
            continue
        
        conversation = format_conversation(
            messages, 
            system_template, 
            user_template, 
            assistant_template, 
            add_eos, 
            conversation_template
        )
        
        peft_data.append({"text": conversation})
    
    return peft_data

def write_jsonl(data: List[Dict[str, str]], ***REMOVED***le_path: str):
    """
    将数据写入JSONL文件
    
    参数:
        data: 要写入的数据列表
        ***REMOVED***le_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(***REMOVED***le_path)), exist_ok=True)
    
    with open(***REMOVED***le_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    """主函数"""
    args = parse_args()
    
    print(f"正在读取输入文件: {args.input_***REMOVED***le}")
    input_data = read_jsonl(args.input_***REMOVED***le)
    print(f"已加载 {len(input_data)} 条对话")
    
    print("正在转换为PEFT格式...")
    peft_data = convert_to_peft_format(
        input_data,
        args.system_template,
        args.user_template,
        args.assistant_template,
        args.add_eos,
        args.conversation_template
    )
    
    print(f"正在写入输出文件: {args.output_***REMOVED***le}")
    write_jsonl(peft_data, args.output_***REMOVED***le)
    
    print(f"转换完成！转换了 {len(peft_data)} 条对话到PEFT格式")
    print(f"输出文件路径: {args.output_***REMOVED***le}")
    
    # 显示示例
    if peft_data:
        print("\n数据示例:")
        print("-" * 50)
        sample = peft_data[0]["text"]
        preview_length = min(500, len(sample))
        print(f"{sample[:preview_length]}{'...' if len(sample) > preview_length else ''}")
        print("-" * 50)

if __name__ == "__main__":
    main() 