#!/bin/bash

echo "🚀 开始Qwen3模型微调效果评估"
echo "=================================="

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "🌐 已设置HuggingFace镜像: $HF_ENDPOINT"

# 检查必要文件
echo "🔍 检查必要文件..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查微调模型
if [ ! -d "model/finetuned/lora" ]; then
    echo "❌ 找不到微调模型目录: model/finetuned/lora"
    echo "请先运行 ./run_training.sh 完成模型微调"
    exit 1
fi

# 检查测试数据
if [ ! -f "dataset/test/alpaca_test.json" ]; then
    echo "❌ 找不到测试数据: dataset/test/alpaca_test.json"
    echo "请先运行数据处理脚本生成测试集"
    exit 1
fi

echo "✅ 文件检查完成"

# 解析命令行参数
SAMPLE_SIZE=300
TEST_DATA="dataset/test/alpaca_test.json"
CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --sample_size NUM    评估样本数量 (默认: 300)"
            echo "  --test_data PATH     测试数据文件路径"
            echo "  --checkpoint PATH    断点文件路径（恢复中断的评估）"
            echo "  -h, --help          显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p evaluation

echo "📊 评估配置:"
echo "  测试数据: $TEST_DATA"
echo "  样本数量: $SAMPLE_SIZE"
if [ ! -z "$CHECKPOINT" ]; then
    echo "  断点文件: $CHECKPOINT"
fi
echo ""

# 运行评估脚本
echo "🏃 开始模型评估..."
echo ""

if [ ! -z "$CHECKPOINT" ]; then
    python benchmark.py \
        --test_data "$TEST_DATA" \
        --sample_size "$SAMPLE_SIZE" \
        --checkpoint "$CHECKPOINT"
else
    python benchmark.py \
        --test_data "$TEST_DATA" \
        --sample_size "$SAMPLE_SIZE"
fi

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 评估完成！"
    echo ""
    echo "📁 输出文件:"
    echo "  📊 评估报告: evaluation/evaluation_report.md"
    echo "  📈 对比图表: evaluation/comparison.png"
    echo "  📋 详细数据: evaluation/evaluation_results.json"
    echo ""
    echo "💡 建议："
    echo "  1. 查看评估报告了解详细分析"
    echo "  2. 可视化图表直观对比两模型性能"
    echo "  3. 详细JSON数据可用于进一步分析"
else
    echo ""
    echo "❌ 评估失败，请检查错误信息"
    echo "💡 常见问题："
    echo "  1. 确保已安装所有依赖: pip install -r requirements.txt"
    echo "  2. 确保有足够的GPU内存"
    echo "  3. 检查测试数据格式是否正确"
fi 