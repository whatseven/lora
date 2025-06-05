# AI Report - Qwen3 LoRA微调项目

这是一个基于Qwen3-14B模型的LoRA微调项目，包含完整的训练、评估和模型管理功能。

## 🚀 项目特点

- **一键训练**: 使用LoRA技术对Qwen3-14B进行高效微调
- **智能评估**: 多维度模型性能评估（BLEU、ROUGE、BERTScore）
- **版本管理**: 自动化模型版本管理系统
- **完整文档**: 详细的使用指南和最佳实践

## 📁 项目结构

```
├── Lora_Qwen3_finetune/           # LoRA微调主项目
│   ├── train.py                   # 训练脚本
│   ├── benchmark.py               # 评估脚本
│   ├── model_manager.py           # 模型管理
│   ├── run_training.sh            # 一键训练
│   ├── run_benchmark.sh           # 一键评估
│   ├── 使用指南.md                # 完整使用文档
│   └── ...
├── Qwen3_(14B)_Reasoning_Conversational.ipynb  # Jupyter演示
├── lora_info.md                   # LoRA技术说明
└── README.md                      # 项目说明
```

## 🎯 快速开始

### 环境要求
- **硬件**: A100 32G GPU（或同等级别）
- **环境**: Python 3.8+, CUDA 11.8+
- **空间**: 至少50GB可用磁盘空间

### 一键运行
```bash
cd Lora_Qwen3_finetune

# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据集到 dataset/alpaca_train.json

# 3. 开始训练
./run_training.sh

# 4. 评估模型
./run_benchmark.sh
```

## 📖 详细文档

请查看 [Lora_Qwen3_finetune/使用指南.md](./Lora_Qwen3_finetune/使用指南.md) 获取完整的使用说明。

## 🛠️ 技术栈

- **基础模型**: Qwen3-14B
- **微调技术**: LoRA (Low-Rank Adaptation)
- **训练框架**: Unsloth, Transformers
- **评估指标**: BLEU-4, ROUGE-L, BERTScore

## 📊 性能特点

- **高效训练**: LoRA技术显著降低显存需求
- **快速收敛**: 优化的超参数配置
- **多维评估**: 全面的模型性能分析
- **版本管理**: 便于模型迭代和对比

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

---

**作者**: whatseven  
**邮箱**: 15812711506@163.com 