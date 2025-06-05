#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练后模型测试脚本
用于测试微调后的Qwen3模型
"""

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import os

class ModelTester:
    def __init__(self, model_path=None):
        """初始化测试器"""
        self.model_path = model_path or "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora"
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载训练后的模型"""
        print(f"正在加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"错误: 找不到模型文件 {self.model_path}")
            print("请先运行 python train.py 完成训练")
            return False
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            print("✅ 模型加载成功")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def chat(self, user_input, max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20):
        """与模型对话"""
        if self.model is None or self.tokenizer is None:
            print("请先加载模型")
            return
        
        # 构造输入
        messages = [{"role": "user", "content": user_input}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        print(f"\n用户: {user_input}")
        print("助手: ", end="", flush=True)
        
        # 生成回复
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return outputs
    
    def batch_test(self, test_questions=None):
        """批量测试多个问题"""
        if test_questions is None:
            test_questions = [
                "请介绍一下锂离子电池的工作原理",
                "无人机的飞行控制系统是如何工作的？",
                "电池管理系统(BMS)的主要功能有哪些？",
                "无人机在电池续航方面面临什么挑战？",
                "如何提高无人机电池的能量密度？",
                "请解释一下无人机的自动驾驶技术",
                "电池热管理系统的重要性是什么？",
                "无人机在物流配送中的应用前景如何？",
            ]
        
        print("=== 批量测试开始 ===")
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}/{len(test_questions)}")
            print('='*60)
            self.chat(question, max_new_tokens=256)
        
        print(f"\n{'='*60}")
        print("✅ 批量测试完成")
    
    def interactive_chat(self):
        """交互式对话"""
        print("=== 交互式对话模式 ===")
        print("输入 'exit' 或 'quit' 退出")
        print("输入 'clear' 清屏")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("再见！")
                    break
                
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if not user_input:
                    print("请输入问题")
                    continue
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")

def main():
    """主函数"""
    print("🚀 Qwen3模型测试器")
    print("=" * 40)
    
    # 初始化测试器
    tester = ModelTester()
    
    # 加载模型
    if not tester.load_model():
        return
    
    print("\n请选择测试模式:")
    print("1. 批量测试 (默认问题)")
    print("2. 自定义问题测试")
    print("3. 交互式对话")
    
    try:
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == '1' or choice == '':
            tester.batch_test()
        
        elif choice == '2':
            question = input("请输入测试问题: ").strip()
            if question:
                print("\n" + "="*60)
                tester.chat(question)
        
        elif choice == '3':
            tester.interactive_chat()
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n\n测试中断")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 