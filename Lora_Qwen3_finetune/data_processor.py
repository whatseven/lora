#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpaca数据集处理脚本
功能：自动分割训练/验证/测试集，转换为unsloth需要的格式
"""

import json
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset

def load_alpaca_data(file_path):
    """加载Alpaca格式数据"""
    print(f"正在加载数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"总数据量: {len(data)}")
    return data

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    分割数据集
    默认比例: 训练集80%, 验证集10%, 测试集10%
    """
    print(f"数据分割比例 - 训练集:{train_ratio}, 验证集:{val_ratio}, 测试集:{test_ratio}")
    
    # 首先分出训练集和临时集
    train_data, temp_data = train_test_split(
        data, 
        test_size=(1-train_ratio), 
        random_state=random_state
    )
    
    # 再从临时集中分出验证集和测试集
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1-val_size),
        random_state=random_state
    )
    
    print(f"分割结果 - 训练集:{len(train_data)}, 验证集:{len(val_data)}, 测试集:{len(test_data)}")
    return train_data, val_data, test_data

def save_split_data(train_data, val_data, test_data, output_dir):
    """保存分割后的数据"""
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/valid", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    
    # 保存训练集
    train_path = f"{output_dir}/train/alpaca_train.json"
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"训练集已保存: {train_path}")
    
    # 保存验证集
    val_path = f"{output_dir}/valid/alpaca_valid.json"
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"验证集已保存: {val_path}")
    
    # 保存测试集
    test_path = f"{output_dir}/test/alpaca_test.json"
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存: {test_path}")

def alpaca_to_conversation(example):
    """
    将单条Alpaca数据转换为conversation格式
    不使用系统提示词，直接处理instruction和input
    """
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']
    
    # 合并instruction和input
    if input_text.strip():
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction
    
    # 构造对话格式
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]
    
    return conversation

def process_for_training(data_list, tokenizer):
    """
    将数据列表处理为训练格式
    
    Args:
        data_list: Alpaca格式的数据列表
        tokenizer: 分词器
    
    Returns:
        Dataset: 可用于训练的数据集
    """
    print(f"正在处理 {len(data_list)} 条数据...")
    
    # 转换为conversation格式
    conversations = []
    for example in data_list:
        conversation = alpaca_to_conversation(example)
        conversations.append(conversation)
    
    # 应用chat_template
    formatted_texts = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
    )
    
    # 创建Dataset
    df = pd.DataFrame({"text": formatted_texts})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)
    
    print(f"数据处理完成，最终数据集大小: {len(dataset)}")
    return dataset

def preview_data(data_list, tokenizer, num_samples=2):
    """预览处理后的数据格式"""
    print(f"\n=== 数据预览 (显示前{num_samples}条) ===")
    
    for i in range(min(num_samples, len(data_list))):
        print(f"\n--- 样本 {i+1} ---")
        example = data_list[i]
        
        print("原始Alpaca格式:")
        print(f"  instruction: {example['instruction']}")
        print(f"  input: {example.get('input', '')}")
        print(f"  output: {example['output']}")
        
        # 转换为conversation
        conversation = alpaca_to_conversation(example)
        print(f"\nConversation格式:")
        for msg in conversation:
            print(f"  {msg['role']}: {msg['content']}")
        
        # 应用chat_template
        formatted_text = tokenizer.apply_chat_template([conversation], tokenize=False)[0]
        print(f"\n最终格式化文本:")
        print(formatted_text)
        print("-" * 60)

def main():
    """主函数：处理完整的数据处理流程"""
    
    # 配置路径
    input_file = "dataset/alpaca_train.json"  # 原始数据文件
    output_dir = "dataset"  # 输出目录
    
    print("=== Alpaca数据集处理开始 ===")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到数据文件 {input_file}")
        print("请确保将Alpaca数据集文件放在 dataset/alpaca_train.json")
        return
    
    # 1. 加载数据
    data = load_alpaca_data(input_file)
    
    # 2. 分割数据集
    train_data, val_data, test_data = split_dataset(data)
    
    # 3. 保存分割后的数据
    save_split_data(train_data, val_data, test_data, output_dir)
    
    # 4. 预览数据格式（需要先加载tokenizer）
    try:
        from unsloth import FastLanguageModel
        print("\n正在加载tokenizer进行数据预览...")
        
        # 注意：请确保运行前已设置 export HF_ENDPOINT=https://hf-mirror.com
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-14B-unsloth-bnb-4bit",  # 使用官方4bit版本
            max_seq_length=2048,
            load_in_4bit=True,
        )
        
        preview_data(train_data[:2], tokenizer, num_samples=2)
        
    except Exception as e:
        print(f"预览失败（这不影响数据处理）: {e}")
        print("如果需要预览，请先安装unsloth")
    
    print("\n=== 数据处理完成 ===")
    print("现在可以运行 train.py 开始训练")

if __name__ == "__main__":
    main() 