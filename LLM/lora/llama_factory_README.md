# 基于LLaMA-Factory的LoRA微调

本目录包含使用LLaMA-Factory框架进行LoRA微调的脚本。LLaMA-Factory是一个专为大型语言模型微调设计的框架，提供了更简单、更集成的微调体验。

## LLaMA-Factory的优势

与基于原生PEFT的实现相比，LLaMA-Factory提供以下优势：

1. **更完整的训练流程**：集成了数据处理、训练、评估和推理等完整流程
2. **更友好的用户界面**：提供WebUI界面，使没有编程经验的用户也能轻松操作
3. **对多种数据格式的支持**：内置支持多种对话和指令数据格式（Alpaca、ShareGPT、BELLE等）
4. **丰富的预设模板**：针对不同模型提供了优化的提示模板
5. **更全面的功能**：支持监督微调(SFT)、奖励模型训练(RM)和强化学习人类反馈(RLHF)

## 安装依赖

使用前需要安装LLaMA-Factory依赖：

```bash
# 方法1：通过PyPI安装
pip install 'llm-tuner[deepspeed]'

# 方法2：从源码安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 主要脚本说明

- **llama_factory_***REMOVED***ne_tuning.py**：基于LLaMA-Factory进行LoRA微调的主要脚本
- **llama_factory_example.sh**：提供多种使用场景的示例命令

## 数据格式

LLaMA-Factory支持多种数据格式，最常用的有：

### Alpaca格式（指令微调）
```json
{
  "instruction": "任务指令",
  "input": "任务输入（可选）",
  "output": "期望输出"
}
```

### ShareGPT格式（对话微调）
```json
{
  "conversations": [
    {"from": "human", "value": "用户消息"},
    {"from": "assistant", "value": "助手回复"},
    {"from": "human", "value": "用户跟进问题"},
    {"from": "assistant", "value": "助手回复"}
  ]
}
```

## 使用示例

### 基本微调

```bash
python llama_factory_***REMOVED***ne_tuning.py \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --dataset_path "path/to/your_data.json" \
  --dataset_format "alpaca" \
  --output_dir "./llama-factory-output" \
  --lora_rank 8 \
  --lora_alpha 32 \
  --epochs 3
```

### 使用量化训练大模型

```bash
python llama_factory_***REMOVED***ne_tuning.py \
  --model_name_or_path "meta-llama/Llama-2-13b-hf" \
  --dataset_path "path/to/your_data.json" \
  --dataset_format "alpaca" \
  --output_dir "./llama-factory-13b-output" \
  --lora_rank 16 \
  --quantization "4bit" \
  --batch_size 4
```

更多示例请参考 `llama_factory_example.sh` 文件。

## 导出合并模型

微调完成后，可以将LoRA权重合并到原始模型中：

```bash
python llama_factory_***REMOVED***ne_tuning.py \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --dataset_path "path/to/your_data.json" \
  --output_dir "./llama-factory-output" \
  --export_dir "./merged-model"
```

## 使用WebUI界面

LLaMA-Factory还提供了WebUI界面，更适合不熟悉命令行的用户：

```bash
# 启动WebUI界面
python -m llmtuner
```

通过WebUI界面，您可以：
- 上传和处理数据集
- 配置训练参数
- 监控训练进度
- 使用微调后的模型对话

## 更多资源

- [LLaMA-Factory官方文档](https://github.com/hiyouga/LLaMA-Factory#readme)
- [支持的模型列表](https://github.com/hiyouga/LLaMA-Factory#supported-models)
- [支持的数据集格式](https://github.com/hiyouga/LLaMA-Factory#data-format) 