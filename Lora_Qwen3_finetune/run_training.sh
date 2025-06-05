#!/bin/bash
# Qwen3-14B LoRA微调一键训练脚本

echo "🚀 开始Qwen3-14B LoRA微调流程"
echo "=================================="

# 设置必要的环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "🌐 已设置HuggingFace镜像: $HF_ENDPOINT"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查CUDA是否可用
python3 -c "import torch; print('✅ CUDA可用' if torch.cuda.is_available() else '❌ CUDA不可用'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 检查是否存在数据集文件
if [ ! -f "dataset/alpaca_train.json" ]; then
    echo "❌ 错误: 找不到数据集文件 dataset/alpaca_train.json"
    echo "请将Alpaca格式的数据集文件放在 dataset/alpaca_train.json"
    exit 1
fi

echo ""
echo "步骤1: 安装依赖"
pip install -r requirements.txt

echo ""
echo "步骤2: 处理数据集"
python3 data_processor.py

# 检查数据处理是否成功
if [ ! -f "dataset/train/alpaca_train.json" ]; then
    echo "❌ 数据处理失败"
    exit 1
fi

echo ""
echo "步骤3: 开始训练"
python3 train.py

# 检查训练是否成功
if [ -d "model/finetuned/lora" ]; then
    echo ""
    echo "🎉 训练完成！"
    echo "模型已保存在: model/finetuned/lora"
    echo ""
    echo "现在可以运行以下命令测试模型:"
    echo "python3 test_model.py"
else
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi 