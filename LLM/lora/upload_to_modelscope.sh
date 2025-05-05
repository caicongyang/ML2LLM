#!/bin/bash

# 将合并后的模型上传到魔塔社区(ModelScope)的简单脚本
# 根据 https://modelscope.cn/docs/models/upload 官方文档编写

# 载入环境变量
if [ -f ".env" ]; then
    echo "正在加载.env文件中的环境变量..."
    export $(grep -v '^#' .env | xargs)
elif [ -f "$(dirname "$0")/.env" ]; then
    echo "正在加载$(dirname "$0")/.env文件中的环境变量..."
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

# 设置参数 - 请在运行前修改这些值
MODEL_PATH="/root/autodl-tmp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-risk"  # 已合并模型的本地路径
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B-risk"  # 模型名称
USERNAME="caicongyang"  # 您的魔塔社区用户名
CHINESE_NAME="DeepSeek-R1精炼风险模型"  # 模型的中文名称，可选

# 从环境变量获取token，如果环境变量不存在则提示用户输入
if [ -z "$MS_TOKEN" ]; then
    echo "警告: 未找到MS_TOKEN环境变量。请确保在.env文件中设置或直接输入。"
    read -p "请输入您的ModelScope API令牌: " MS_TOKEN
    if [ -z "$MS_TOKEN" ]; then
        echo "错误: 必须提供ModelScope API令牌才能上传模型。"
        echo "您可以从 https://modelscope.cn/my/myaccesstoken 获取令牌"
        exit 1
    fi
fi

# 其他可选参数
MODEL_TYPE="nlp/text-generation"  # 模型类型和任务，格式为'类型/任务'
LICENSE="Apache-2.0"  # 模型许可证，可选: Apache-2.0, MIT, BSD, GPL, LGPL, CC-BY-NC, CC-BY-SA, CC-BY, CC0
MODEL_DESCRIPTION="这是一个基于LoRA微调并合并后的语言模型"  # 模型描述
MODEL_TAGS="lora,llm,fine-tuned"  # 模型标签，以逗号分隔
COMMIT_MESSAGE="上传合并后的LoRA模型"  # 提交信息
IS_PRIVATE=false  # 是否为私有模型，设置为true或false

# 显示上传信息
echo "准备上传模型到魔塔社区(ModelScope)..."
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "中文名称: $CHINESE_NAME"
echo "用户名: $USERNAME"
echo "模型类型: $MODEL_TYPE"
echo "许可证: $LICENSE"

# 检查脚本路径
SCRIPT_PATH="upload_to_modelscope.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$(dirname "$0")/upload_to_modelscope.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "错误: 找不到上传脚本 upload_to_modelscope.py"
        exit 1
    fi
fi

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
if ! pip list 2>/dev/null | grep -q modelscope; then
    echo "正在安装 modelscope 依赖..."
    pip install modelscope
fi

# 确保安装必要的包
pip install --quiet pyyaml requests tqdm

# 设置私有参数
PRIVATE_FLAG=""
if [ "$IS_PRIVATE" = true ]; then
    PRIVATE_FLAG="--private"
    echo "模型将设置为私有"
fi

# 设置中文名称参数
CHINESE_NAME_FLAG=""
if [ -n "$CHINESE_NAME" ]; then
    CHINESE_NAME_FLAG="--chinese_name \"$CHINESE_NAME\""
fi

# 执行上传
echo "开始上传..."
cmd="python \"$SCRIPT_PATH\" \
  --model_path \"$MODEL_PATH\" \
  --model_name \"$MODEL_NAME\" \
  --access_token \"$MS_TOKEN\" \
  --username \"$USERNAME\" \
  --model_type \"$MODEL_TYPE\" \
  --license \"$LICENSE\" \
  --model_description \"$MODEL_DESCRIPTION\" \
  --model_tags \"$MODEL_TAGS\" \
  --commit_message \"$COMMIT_MESSAGE\" \
  $PRIVATE_FLAG"

# 添加中文名称如果存在
if [ -n "$CHINESE_NAME" ]; then
    cmd="$cmd --chinese_name \"$CHINESE_NAME\""
fi

# 执行命令
eval $cmd

# 检查上传结果
UPLOAD_STATUS=$?
if [ $UPLOAD_STATUS -eq 0 ]; then
    echo "上传完成，上传过程成功！"
    echo "您的模型已上传到："
    echo "https://modelscope.cn/models/$USERNAME/$MODEL_NAME"
    echo ""
    echo "在Python中，您可以使用以下代码加载模型："
    echo "-----------------------------------------"
    echo "from modelscope.models import Model"
    echo "model = Model.from_pretrained('$USERNAME/$MODEL_NAME')"
    echo "-----------------------------------------"
else
    echo "上传过程出错，请检查上面的错误信息。"
    exit 1
fi 