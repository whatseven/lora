#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-14B LoRA微调主训练脚本
使用unsloth框架进行高效训练
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from data_processor import process_for_training, load_alpaca_data

class Qwen3Trainer:
    def __init__(self, base_model_path=None, model_version=None):
        """
        初始化训练器
        
        Args:
            base_model_path: 基础模型路径，可以是预训练模型名称或之前微调的模型路径
            model_version: 模型版本标识，如果为None则自动生成时间戳版本
        """
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 设置基础模型路径
        if base_model_path is None:
            # 默认使用官方预训练模型
            self.base_model_path = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
            self.is_pretrained = True
        else:
            # 使用指定的模型路径（可能是之前微调的模型）
            self.base_model_path = base_model_path
            self.is_pretrained = False
        
        # 设置模型版本
        if model_version is None:
            # 自动生成时间戳版本
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.model_version = model_version
        
        # 配置路径
        self.cache_dir = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/pretrained"
        self.output_base_dir = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned"
        self.output_dir = os.path.join(self.output_base_dir, f"lora_v{self.model_version}")
        self.log_dir = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/logs"
        
        # 创建必要目录
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"🎯 训练器初始化完成")
        print(f"基础模型: {self.base_model_path}")
        print(f"模型版本: {self.model_version}")
        print(f"输出路径: {self.output_dir}")
    
    def setup_model(self, max_seq_length):
        """
        设置模型和分词器
        支持从预训练模型或之前微调的模型开始训练
        """
        print("=== 步骤1: 加载基础模型 ===")
        
        if self.is_pretrained:
            # 设置缓存目录环境变量
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            print("💡 请确保运行前已设置: export HF_ENDPOINT=https://hf-mirror.com")
            print(f"模型缓存目录: {self.cache_dir}")
            print("首次运行将自动下载模型，请耐心等待...")
        
        print(f"正在加载模型: {self.base_model_path}")
        
        # 加载模型和分词器
        if self.is_pretrained:
            # 从官方预训练模型加载
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
                cache_dir=self.cache_dir
            )
        else:
            # 从之前微调的模型加载
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
            )
        
        print("✅ 模型加载完成")
        return self.model, self.tokenizer
    
    def setup_lora(self, r, lora_alpha):
        """配置LoRA适配器"""
        print("=== 步骤2: 配置LoRA适配器 ===")
        print(f"LoRA rank: {r}, alpha: {lora_alpha}")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,                      # A100 32G可以用64
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,    # 通常等于rank
            lora_dropout=0,           # 0表示优化性能
            bias="none",              # "none"最优化
            use_gradient_checkpointing="unsloth",  # 节省30%内存
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("✅ LoRA配置完成")
        return self.model
    
    def load_datasets(self):
        """加载训练和验证数据集"""
        print("=== 步骤3: 加载数据集 ===")
        
        # 检查数据文件是否存在
        train_file = "dataset/train/alpaca_train.json"
        val_file = "dataset/valid/alpaca_valid.json"
        
        if not os.path.exists(train_file):
            print(f"错误: 找不到训练数据 {train_file}")
            print("请先运行 python data_processor.py 处理数据集")
            return None, None
        
        # 加载训练数据
        print("正在加载训练数据...")
        train_data = load_alpaca_data(train_file)
        train_dataset = process_for_training(train_data, self.tokenizer)
        
        # 加载验证数据（如果存在）
        val_dataset = None
        if os.path.exists(val_file):
            print("正在加载验证数据...")
            val_data = load_alpaca_data(val_file)
            val_dataset = process_for_training(val_data, self.tokenizer)
        
        print("✅ 数据集加载完成")
        return train_dataset, val_dataset
    
    def setup_training(self, train_dataset, val_dataset, 
                      num_epochs, batch_size, learning_rate):
        """配置训练参数"""
        print("=== 步骤4: 配置训练参数 ===")
        print(f"训练轮数: {num_epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,  # 有效batch size = 4*4=16
                warmup_steps=20,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=self.output_dir,
                logging_dir=self.log_dir,
                report_to="none",               # 可改为"wandb"
                save_strategy="epoch",
                save_steps=100,
                eval_strategy="epoch" if val_dataset else "no",
                eval_steps=50 if val_dataset else None,
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False if val_dataset else None,
            ),
        )
        
        print("✅ 训练配置完成")
        return self.trainer
    
    def show_gpu_stats(self):
        """显示GPU使用情况"""
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            print(f"GPU: {gpu_stats.name}")
            print(f"最大内存: {max_memory} GB")
            print(f"已用内存: {start_gpu_memory} GB")
            return start_gpu_memory, max_memory
        return 0, 0
    
    def train(self):
        """开始训练"""
        print("=== 步骤5: 开始训练 ===")
        
        # 显示训练前GPU状态
        start_memory, max_memory = self.show_gpu_stats()
        
        # 开始训练
        print("🚀 训练开始...")
        trainer_stats = self.trainer.train()
        
        # 显示训练后GPU状态
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            
            print(f"\n=== 训练完成统计 ===")
            print(f"训练时间: {trainer_stats.metrics['train_runtime']:.2f} 秒")
            print(f"训练时间: {round(trainer_stats.metrics['train_runtime']/60, 2)} 分钟")
            print(f"峰值内存使用: {used_memory} GB")
            print(f"训练额外内存: {used_memory_for_lora} GB")
            print(f"内存使用率: {used_percentage}%")
            print(f"训练内存使用率: {lora_percentage}%")
        
        print("✅ 训练完成")
        return trainer_stats
    
    def save_model(self):
        """保存训练后的模型"""
        print("=== 步骤6: 保存模型 ===")
        print(f"保存路径: {self.output_dir}")
        
        # 保存LoRA适配器
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 保存训练配置信息
        config_info = {
            "model_version": self.model_version,
            "base_model_path": self.base_model_path,
            "is_pretrained": self.is_pretrained,
            "save_time": datetime.now().isoformat(),
            "output_dir": self.output_dir
        }
        
        config_file = os.path.join(self.output_dir, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        
        print("✅ 模型保存完成")
        print(f"LoRA适配器已保存到: {self.output_dir}")
        print(f"训练配置已保存到: {config_file}")
        
        # 创建最新模型的软链接
        latest_link = os.path.join(self.output_base_dir, "latest")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(f"lora_v{self.model_version}", latest_link)
        print(f"✅ 最新模型链接已创建: {latest_link}")
    
    def test_model(self, test_prompt="请介绍一下锂电池的基本工作原理和主要应用领域"):
        """简单测试训练后的模型 - 修改为电池相关测试问题"""
        print("=== 步骤7: 模型测试 ===")
        print(f"测试问题: {test_prompt}")
        
        # 构造输入
        messages = [{"role": "user", "content": test_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        print("\n模型回答:")
        print("-" * 50)
        
        # 生成回复
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        print("-" * 50)
        print("✅ 测试完成")

def main():
    """主训练流程"""
    print("🚀 Qwen3-14B LoRA微调开始")
    print("=" * 60)
    
    # ===============================================
    # 🎯 模型路径配置 - 在这里修改基础模型路径
    # ===============================================
    
    # 基础模型配置
    # 选项1: 使用官方预训练模型（首次训练）
    BASE_MODEL_PATH = None  # 使用默认预训练模型
    
    # 选项2: 使用本地下载的预训练模型
    # BASE_MODEL_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/pretrained/Qwen3-14B"
    
    # 选项3: 使用之前微调的模型继续训练（增量训练）
    # BASE_MODEL_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora_v20241201_143000"
    # BASE_MODEL_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/latest"  # 使用最新版本
    
    # 模型版本（如果为None则自动生成时间戳）
    MODEL_VERSION = None  # 例如: "battery_v1" 或 None
    
    # ===============================================
    # 🎯 所有训练参数都在这里统一设置
    # ===============================================
    
    # 模型参数
    MAX_SEQ_LENGTH = 1024          # 序列长度 (快速验证用1024，正式训练用2048)
    
    # LoRA参数 
    LORA_RANK = 16                 # LoRA rank (快速验证用16，正式训练用64)
    LORA_ALPHA = 16                # LoRA alpha (通常等于rank)
    
    # 训练参数
    NUM_EPOCHS = 1                 # 训练轮数 (快速验证用1，正式训练用3)
    BATCH_SIZE = 2                 # 批次大小 (快速验证用2，正式训练用4)
    LEARNING_RATE = 2e-4           # 学习率
    
    print(f"📊 训练配置:")
    print(f"  基础模型: {BASE_MODEL_PATH or '官方预训练模型'}")
    print(f"  模型版本: {MODEL_VERSION or '自动时间戳'}")
    print(f"  序列长度: {MAX_SEQ_LENGTH}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print("=" * 60)
    
    # 初始化训练器
    trainer = Qwen3Trainer(
        base_model_path=BASE_MODEL_PATH,
        model_version=MODEL_VERSION
    )
    
    try:
        # 1. 设置模型
        model, tokenizer = trainer.setup_model(max_seq_length=MAX_SEQ_LENGTH)
        
        # 2. 配置LoRA
        model = trainer.setup_lora(r=LORA_RANK, lora_alpha=LORA_ALPHA)
        
        # 3. 加载数据集
        train_dataset, val_dataset = trainer.load_datasets()
        if train_dataset is None:
            return
        
        # 4. 配置训练
        trainer_obj = trainer.setup_training(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        # 5. 开始训练
        trainer_stats = trainer.train()
        
        # 6. 保存模型
        trainer.save_model()
        
        # 7. 测试模型
        trainer.test_model()
        
        print("\n🎉 训练流程全部完成！")
        print(f"模型已保存在: {trainer.output_dir}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("请检查错误信息并重试")
        return

if __name__ == "__main__":
    main() 