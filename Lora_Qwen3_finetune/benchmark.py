#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3模型微调效果评估脚本
对比原始模型vs微调模型的BLEU/ROUGE/BERTScore指标
"""

import json
import os
import random
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 模型库 - unsloth必须在transformers之前导入
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 评估指标库
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

class ModelBenchmark:
    def __init__(self, 
                 model_a_path=None,  # 模型A路径
                 model_b_path=None,  # 模型B路径
                 cache_dir="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/pretrained",
                 output_dir="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/evaluation",
                 enable_bertscore=False,  # BERTScore开关
                 batch_size=2):  # 批处理大小
        """
        初始化评估器 - 支持多种对比模式
        
        Args:
            model_a_path: 模型A路径（可以是预训练模型名称或本地路径）
            model_b_path: 模型B路径（可以是微调模型路径或其他模型路径）
            cache_dir: 缓存目录
            output_dir: 输出目录
        
        对比模式:
        1. 预训练 vs 微调: model_a_path=None, model_b_path=None (默认)
        2. 预训练 vs 指定微调: model_a_path=None, model_b_path="path/to/model"
        3. 两个微调版本对比: model_a_path="path/to/v1", model_b_path="path/to/v2"
        """
        
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        # 确定模型路径
        if model_a_path is None:
            # 默认使用官方预训练模型
            self.model_a_path = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
            self.model_a_is_pretrained = True
        else:
            self.model_a_path = model_a_path
            self.model_a_is_pretrained = not os.path.exists(model_a_path)  # 如果路径不存在，认为是模型名称
        
        if model_b_path is None:
            # 使用最新的微调模型
            self.model_b_path = self._get_latest_model_path()
            self.model_b_is_pretrained = False
        else:
            self.model_b_path = model_b_path
            self.model_b_is_pretrained = not os.path.exists(model_b_path)
        
        # 生成输出标识
        self.output_suffix = self._generate_output_suffix()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 模型和分词器
        self.model_a = None
        self.tokenizer_a = None
        self.model_b = None
        self.tokenizer_b = None
        
        # 性能优化配置
        self.enable_bertscore = enable_bertscore
        self.batch_size = batch_size
        
        # 评估指标
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
        print(f"🔧 评估器初始化完成")
        print(f"模型A: {self.model_a_path}")
        print(f"模型B: {self.model_b_path}")
        print(f"输出标识: {self.output_suffix}")
        print(f"输出目录: {self.output_dir}")
    
    def _get_latest_model_path(self):
        """获取最新的微调模型路径"""
        base_dir = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned"
        
        # 优先使用latest软链接
        latest_link = os.path.join(base_dir, "latest")
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            return latest_link
        
        # 如果没有latest链接，查找最新的lora_v*文件夹
        lora_dirs = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if item.startswith("lora_v") and os.path.isdir(os.path.join(base_dir, item)):
                    lora_dirs.append(item)
        
        if lora_dirs:
            # 按名称排序，取最新的
            lora_dirs.sort(reverse=True)
            return os.path.join(base_dir, lora_dirs[0])
        
        # 兼容旧版本路径
        old_path = os.path.join(base_dir, "lora")
        if os.path.exists(old_path):
            return old_path
        
        # 如果都没有找到，返回None
        return None
    
    def _generate_output_suffix(self):
        """生成输出文件的后缀标识"""
        def get_model_identifier(path, is_pretrained):
            if is_pretrained:
                return "pretrained"
            else:
                # 提取模型版本号
                basename = os.path.basename(path.rstrip('/'))
                if basename == "latest":
                    # 解析latest链接指向的实际目录
                    if os.path.islink(path):
                        target = os.readlink(path)
                        basename = target
                return basename
        
        model_a_id = get_model_identifier(self.model_a_path, self.model_a_is_pretrained)
        model_b_id = get_model_identifier(self.model_b_path, self.model_b_is_pretrained)
        
        return f"{model_a_id}_vs_{model_b_id}"
    
    def load_models(self):
        """加载两个对比模型"""
        print("\n=== 步骤1: 加载模型 ===")
        
        # 检查模型路径
        if not self.model_a_is_pretrained and not os.path.exists(self.model_a_path):
            print(f"❌ 找不到模型A: {self.model_a_path}")
            return False
        
        if not self.model_b_is_pretrained and not os.path.exists(self.model_b_path):
            print(f"❌ 找不到模型B: {self.model_b_path}")
            return False
        
        try:
            # 加载模型A
            print(f"正在加载模型A: {self.model_a_path}")
            
            if self.model_a_is_pretrained:
                # 从官方预训练模型加载
                self.model_a, self.tokenizer_a = FastLanguageModel.from_pretrained(
                    model_name=self.model_a_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    cache_dir=self.cache_dir
                )
            else:
                # 从本地路径加载
                self.model_a, self.tokenizer_a = FastLanguageModel.from_pretrained(
                    model_name=self.model_a_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )
            print("✅ 模型A加载完成")
            
            # 加载模型B
            print(f"正在加载模型B: {self.model_b_path}")
            
            if self.model_b_is_pretrained:
                # 从官方预训练模型加载
                self.model_b, self.tokenizer_b = FastLanguageModel.from_pretrained(
                    model_name=self.model_b_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    cache_dir=self.cache_dir
                )
            else:
                # 从本地路径加载
                self.model_b, self.tokenizer_b = FastLanguageModel.from_pretrained(
                    model_name=self.model_b_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )
            print("✅ 模型B加载完成")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def load_test_data(self, test_file, sample_size=300):
        """
        加载测试数据并采样
        
        Args:
            test_file: 测试数据文件路径
            sample_size: 采样数量，如果测试集小于此数量则使用全部
        
        Returns:
            list: 采样后的测试数据
        """
        print(f"\n=== 步骤2: 加载测试数据 ===")
        
        if not os.path.exists(test_file):
            print(f"❌ 找不到测试数据: {test_file}")
            return None
        
        # 加载数据
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"总测试数据量: {len(test_data)}")
        
        # 采样逻辑
        if len(test_data) <= sample_size:
            print(f"使用全部 {len(test_data)} 条数据进行评估")
            sampled_data = test_data
        else:
            print(f"随机采样 {sample_size} 条数据进行评估")
            random.seed(42)  # 确保可复现
            sampled_data = random.sample(test_data, sample_size)
        
        print(f"✅ 实际评估数据量: {len(sampled_data)}")
        return sampled_data
    
    def generate_batch_responses(self, model, tokenizer, batch_instructions, batch_inputs, max_new_tokens=512):
        """
        批量生成模型回复（优化版本）
        
        Args:
            model: 要使用的模型
            tokenizer: 对应的分词器
            batch_instructions: 指令列表
            batch_inputs: 输入文本列表
            max_new_tokens: 最大生成长度
            
        Returns:
            list: 生成的回复列表
        """
        try:
            # 构建批量提示
            batch_prompts = []
            for instruction, input_text in zip(batch_instructions, batch_inputs):
                if input_text.strip():
                    prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
                else:
                    prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"
                batch_prompts.append(prompt)
            
            # 批量编码
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=1024
            ).to(model.device)
            
            # 批量生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码批量结果
            responses = []
            for i, output in enumerate(outputs):
                # 移除输入部分
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                
                # 解码生成的部分
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                response = response.strip()
                
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"⚠️ 批量生成失败，回退到单个生成: {e}")
            # 回退到单个生成
            responses = []
            for instruction, input_text in zip(batch_instructions, batch_inputs):
                response = self.generate_response(model, tokenizer, instruction, input_text, max_new_tokens)
                responses.append(response)
            return responses

    def generate_response(self, model, tokenizer, instruction, input_text="", max_new_tokens=512):
        """
        生成模型回复
        
        Args:
            model: 模型
            tokenizer: 分词器
            instruction: 指令
            input_text: 输入文本
            max_new_tokens: 最大生成长度
        
        Returns:
            str: 生成的回复
        """
        # 构造输入
        if input_text.strip():
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction
        
        messages = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成回复
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return generated_text
    
    def calculate_metrics(self, reference, hypothesis):
        """
        计算评估指标
        
        Args:
            reference: 参考答案
            hypothesis: 生成答案
        
        Returns:
            dict: 包含各项指标的字典
        """
        # BLEU-4
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, 
                              weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=self.smoothing_function)
        
        # ROUGE-L
        rouge_scores = self.rouge_scorer.score(reference, hypothesis)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # BERTScore (可选，默认关闭以提升速度)
        if self.enable_bertscore:
            try:
                P, R, F1 = bert_score([hypothesis], [reference], 
                                    lang='zh', verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu')
                bert_score_f1 = F1.item()
            except Exception as e:
                # 如果BERTScore计算失败，输出错误信息并使用简化版本
                print(f"⚠️ BERTScore计算失败: {e}")
                bert_score_f1 = 0.0
        else:
            # BERTScore关闭，使用0.0作为占位符
            bert_score_f1 = 0.0
        
        return {
            'bleu_4': bleu_4,
            'rouge_l': rouge_l,
            'bert_score_f1': bert_score_f1
        }
    
    def evaluate_models(self, test_data, checkpoint_file=None):
        """
        评估两个模型的性能（批处理优化版）
        
        Args:
            test_data: 测试数据
            checkpoint_file: 断点续传文件
        
        Returns:
            dict: 评估结果
        """
        print(f"\n=== 步骤3: 模型评估（批处理模式）===")
        print(f"批处理大小: {self.batch_size}")
        print(f"BERTScore: {'启用' if self.enable_bertscore else '关闭'}")
        
        # 尝试加载断点
        start_idx = 0
        results = []
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"发现断点文件，正在恢复...")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data['results']
                start_idx = len(results)
            print(f"从第 {start_idx + 1} 条数据继续评估")
        
        # 评估进度
        total_batches = (len(test_data) - start_idx + self.batch_size - 1) // self.batch_size
        with tqdm(total=len(test_data) - start_idx, initial=0, 
                 desc="评估进度", unit="samples") as pbar:
            
            # 批处理循环
            for batch_start in range(start_idx, len(test_data), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(test_data))
                batch_data = test_data[batch_start:batch_end]
                
                try:
                    # 准备批次数据
                    batch_instructions = []
                    batch_inputs = []
                    batch_references = []
                    batch_indices = []
                    
                    for i, item in enumerate(batch_data):
                        batch_instructions.append(item['instruction'])
                        batch_inputs.append(item.get('input', ''))
                        batch_references.append(item['output'])
                        batch_indices.append(batch_start + i)
                    
                    # 批量生成模型A回复
                    model_a_responses = self.generate_batch_responses(
                        self.model_a, self.tokenizer_a,
                        batch_instructions, batch_inputs
                    )
                    
                    # 批量生成模型B回复
                    model_b_responses = self.generate_batch_responses(
                        self.model_b, self.tokenizer_b,
                        batch_instructions, batch_inputs
                    )
                    
                    # 处理批次结果
                    for i in range(len(batch_data)):
                        try:
                            # 计算指标
                            model_a_metrics = self.calculate_metrics(
                                batch_references[i], model_a_responses[i]
                            )
                            model_b_metrics = self.calculate_metrics(
                                batch_references[i], model_b_responses[i]
                            )
                            
                            # 保存结果
                            result = {
                                'index': batch_indices[i],
                                'instruction': batch_instructions[i],
                                'input': batch_inputs[i],
                                'reference': batch_references[i],
                                'model_a_response': model_a_responses[i],
                                'model_b_response': model_b_responses[i],
                                'model_a_metrics': model_a_metrics,
                                'model_b_metrics': model_b_metrics
                            }
                            results.append(result)
                            
                        except Exception as e:
                            print(f"\n⚠️  处理第 {batch_indices[i]} 条数据时出错: {e}")
                            continue
                    
                    # 更新进度
                    pbar.update(len(batch_data))
                    
                    # 定期保存断点（每10个样本）
                    if len(results) % 10 == 0:
                        self.save_checkpoint(results, checkpoint_file or 
                                           os.path.join(self.output_dir, 'checkpoint.json'))
                    
                    # GPU缓存清理（批次间）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n⚠️  批次 {batch_start}-{batch_end} 处理失败: {e}")
                    # 回退到单个处理
                    for i, item in enumerate(batch_data):
                        try:
                            idx = batch_start + i
                            model_a_response = self.generate_response(
                                self.model_a, self.tokenizer_a,
                                item['instruction'], item.get('input', '')
                            )
                            model_b_response = self.generate_response(
                                self.model_b, self.tokenizer_b,
                                item['instruction'], item.get('input', '')
                            )
                            
                            model_a_metrics = self.calculate_metrics(item['output'], model_a_response)
                            model_b_metrics = self.calculate_metrics(item['output'], model_b_response)
                            
                            result = {
                                'index': idx,
                                'instruction': item['instruction'],
                                'input': item.get('input', ''),
                                'reference': item['output'],
                                'model_a_response': model_a_response,
                                'model_b_response': model_b_response,
                                'model_a_metrics': model_a_metrics,
                                'model_b_metrics': model_b_metrics
                            }
                            results.append(result)
                            pbar.update(1)
                            
                        except Exception as e2:
                            print(f"\n⚠️  单个处理第 {idx} 条数据也失败: {e2}")
                            continue
        
        # 清理断点文件
        if checkpoint_file and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"✅ 评估完成，成功处理 {len(results)} 条数据")
        return results
    
    def save_checkpoint(self, results, checkpoint_file):
        """保存断点文件"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def calculate_summary_stats(self, results):
        """计算汇总统计信息"""
        print(f"\n=== 步骤4: 计算汇总统计 ===")
        
        # 提取指标
        model_a_bleu = [r['model_a_metrics']['bleu_4'] for r in results]
        model_a_rouge = [r['model_a_metrics']['rouge_l'] for r in results]
        model_a_bert = [r['model_a_metrics']['bert_score_f1'] for r in results]
        
        model_b_bleu = [r['model_b_metrics']['bleu_4'] for r in results]
        model_b_rouge = [r['model_b_metrics']['rouge_l'] for r in results]
        model_b_bert = [r['model_b_metrics']['bert_score_f1'] for r in results]
        
        # 计算平均值
        summary = {
            'model_a': {
                'bleu_4': np.mean(model_a_bleu),
                'rouge_l': np.mean(model_a_rouge),
                'bert_score_f1': np.mean(model_a_bert),
                'name': os.path.basename(self.model_a_path.rstrip('/'))
            },
            'model_b': {
                'bleu_4': np.mean(model_b_bleu),
                'rouge_l': np.mean(model_b_rouge),
                'bert_score_f1': np.mean(model_b_bert),
                'name': os.path.basename(self.model_b_path.rstrip('/'))
            }
        }
        
        # 计算差异（B相对于A的变化）
        summary['difference'] = {
            'bleu_4': summary['model_b']['bleu_4'] - summary['model_a']['bleu_4'],
            'rouge_l': summary['model_b']['rouge_l'] - summary['model_a']['rouge_l'],
            'bert_score_f1': summary['model_b']['bert_score_f1'] - summary['model_a']['bert_score_f1']
        }
        
        # 计算百分比变化
        def safe_percentage(diff, base):
            return (diff / base * 100) if base != 0 else 0.0
        
        summary['percentage_change'] = {
            'bleu_4': safe_percentage(summary['difference']['bleu_4'], summary['model_a']['bleu_4']),
            'rouge_l': safe_percentage(summary['difference']['rouge_l'], summary['model_a']['rouge_l']),
            'bert_score_f1': safe_percentage(summary['difference']['bert_score_f1'], summary['model_a']['bert_score_f1'])
        }
        
        print("✅ 统计信息计算完成")
        return summary
    
    def create_visualization(self, summary):
        """创建可视化图表"""
        print(f"\n=== 步骤5: 生成可视化图表 ===")
        
        metrics = ['BLEU-4', 'ROUGE-L', 'BERTScore-F1']
        model_a_scores = [
            summary['model_a']['bleu_4'],
            summary['model_a']['rouge_l'],
            summary['model_a']['bert_score_f1']
        ]
        model_b_scores = [
            summary['model_b']['bleu_4'],
            summary['model_b']['rouge_l'],
            summary['model_b']['bert_score_f1']
        ]
        
        # 创建对比图
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, model_a_scores, width, label=f'模型A ({summary["model_a"]["name"]})', alpha=0.8)
        bars2 = ax.bar(x + width/2, model_b_scores, width, label=f'模型B ({summary["model_b"]["name"]})', alpha=0.8)
        
        ax.set_ylabel('分数')
        ax.set_title('模型性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表（包含版本标识）
        chart_path = os.path.join(self.output_dir, f'comparison_{self.output_suffix}.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 对比图表已保存: {chart_path}")
        return chart_path
    
    def generate_report(self, summary, results, chart_path):
        """生成Markdown评估报告"""
        print(f"\n=== 步骤6: 生成评估报告 ===")
        
        report_content = f"""# 模型对比评估报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
评估样本数: {len(results)}

## 模型信息

- **模型A**: {summary['model_a']['name']} ({self.model_a_path})
- **模型B**: {summary['model_b']['name']} ({self.model_b_path})

## 模型对比

| 模型 | BLEU-4 | ROUGE-L | BERTScore-F1 |
|------|--------|---------|-------------|
| {summary['model_a']['name']} | {summary['model_a']['bleu_4']:.3f} | {summary['model_a']['rouge_l']:.3f} | {summary['model_a']['bert_score_f1']:.3f} |
| {summary['model_b']['name']} | **{summary['model_b']['bleu_4']:.3f}** | **{summary['model_b']['rouge_l']:.3f}** | **{summary['model_b']['bert_score_f1']:.3f}** |

## 关键结论

📊 **模型B相对于模型A的性能变化**：
- BLEU-4: {summary['difference']['bleu_4']:+.3f} ({summary['percentage_change']['bleu_4']:+.1f}%)
- ROUGE-L: {summary['difference']['rouge_l']:+.3f} ({summary['percentage_change']['rouge_l']:+.1f}%)
- BERTScore-F1: {summary['difference']['bert_score_f1']:+.3f} ({summary['percentage_change']['bert_score_f1']:+.1f}%)

## 性能分析

### BLEU-4 分析
BLEU-4衡量生成文本的流畅度和准确性。模型B相对于模型A得分{"提升" if summary['difference']['bleu_4'] > 0 else "下降"}了{abs(summary['difference']['bleu_4']):.3f}，
表明模型B{"有效改善" if summary['difference']['bleu_4'] > 0 else "可能影响"}了生成质量。

### ROUGE-L 分析
ROUGE-L衡量生成文本与参考答案的最长公共子序列匹配度。模型B相对于模型A得分{"提升" if summary['difference']['rouge_l'] > 0 else "下降"}了{abs(summary['difference']['rouge_l']):.3f}，
说明模型B{"更好地" if summary['difference']['rouge_l'] > 0 else "相比模型A在"}捕捉关键信息方面{"有改善" if summary['difference']['rouge_l'] > 0 else "有差异"}。

### BERTScore-F1 分析
BERTScore-F1衡量语义相似性，抗噪音能力更强。模型B相对于模型A得分{"提升" if summary['difference']['bert_score_f1'] > 0 else "下降"}了{abs(summary['difference']['bert_score_f1']):.3f}，
反映出模型B在语义理解能力方面{"有所提升" if summary['difference']['bert_score_f1'] > 0 else "有所不同"}。

## 数据文件

- 📊 [详细评估数据](evaluation_results_{self.output_suffix}.json)
- 📈 [性能对比图]({os.path.basename(chart_path)})

---
*此报告由 Qwen3 模型评估系统自动生成*
*对比标识: {self.output_suffix}*
"""
        
        # 保存报告（包含版本标识）
        report_path = os.path.join(self.output_dir, f'evaluation_report_{self.output_suffix}.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 评估报告已保存: {report_path}")
        return report_path
    
    def save_detailed_results(self, results, summary):
        """保存详细评估结果"""
        # 保存详细结果（包含版本标识）
        detailed_path = os.path.join(self.output_dir, f'evaluation_results_{self.output_suffix}.json')
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(results),
                'model_a': self.model_a_path,
                'model_b': self.model_b_path,
                'comparison_id': self.output_suffix
            },
            'summary': summary,
            'detailed_results': results
        }
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 详细结果已保存: {detailed_path}")
        return detailed_path

def main():
    """主评估流程"""
    print("🚀 Qwen3模型对比评估开始")
    print("=" * 60)
    
    # ===============================================
    # 🎯 模型对比配置 - 在这里修改要对比的模型
    # ===============================================
    
    # 模型配置选项：
    # 选项1: 预训练模型 vs 最新微调模型（默认）
    MODEL_A_PATH = None  # 使用默认预训练模型
    MODEL_B_PATH = None  # 使用最新微调模型
    
    # 选项2: 预训练模型 vs 指定微调模型
    # MODEL_A_PATH = None
    # MODEL_B_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora_v20241205_143000"
    
    # 选项3: 两个微调版本对比
    # MODEL_A_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora_v20241205_143000"
    # MODEL_B_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora_v20241205_150000"
    
    # 选项4: 使用latest链接
    # MODEL_A_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/latest"
    # MODEL_B_PATH = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora_v20241205_150000"
    
    print(f"📊 对比配置:")
    print(f"  模型A: {MODEL_A_PATH or '默认预训练模型'}")
    print(f"  模型B: {MODEL_B_PATH or '最新微调模型'}")
    print("=" * 60)
    
    # 解析命令行参数（优化版）
    import argparse
    parser = argparse.ArgumentParser(description='Qwen3模型对比评估')
    parser.add_argument('--test_data', type=str, 
                       default='dataset/test/alpaca_test.json',
                       help='测试数据文件路径')
    parser.add_argument('--sample_size', type=int, default=300,
                       help='评估样本数量')
    parser.add_argument('--checkpoint', type=str, 
                       help='断点文件路径（可选）')
    parser.add_argument('--enable_bertscore', action='store_true', default=False,
                       help='启用BERTScore评估（默认关闭以提升速度）')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批处理大小（默认2，减少GPU内存占用）')
    
    args = parser.parse_args()
    
    # 初始化评估器
    benchmark = ModelBenchmark(
        model_a_path=MODEL_A_PATH,
        model_b_path=MODEL_B_PATH,
        enable_bertscore=args.enable_bertscore,
        batch_size=args.batch_size
    )
    
    print(f"⚡ 性能优化配置:")
    print(f"  BERTScore: {'启用' if args.enable_bertscore else '关闭（加速模式）'}")
    print(f"  批处理大小: {args.batch_size}")
    print("=" * 60)
    
    try:
        # 1. 加载模型
        if not benchmark.load_models():
            return
        
        # 2. 加载测试数据
        test_data = benchmark.load_test_data(args.test_data, args.sample_size)
        if test_data is None:
            return
        
        # 3. 执行评估
        results = benchmark.evaluate_models(test_data, args.checkpoint)
        if not results:
            print("❌ 评估失败，没有获得有效结果")
            return
        
        # 4. 计算统计信息
        summary = benchmark.calculate_summary_stats(results)
        
        # 5. 生成可视化
        chart_path = benchmark.create_visualization(summary)
        
        # 6. 生成报告
        report_path = benchmark.generate_report(summary, results, chart_path)
        
        # 7. 保存详细结果
        detailed_path = benchmark.save_detailed_results(results, summary)
        
        # 8. 输出总结
        print(f"\n🎉 评估完成！")
        print(f"📊 评估报告: {report_path}")
        print(f"📈 对比图表: {chart_path}")
        print(f"📋 详细数据: {detailed_path}")
        
        # 显示关键结果
        print(f"\n📈 模型对比结果:")
        print(f"BLEU-4: {summary['difference']['bleu_4']:+.3f} ({summary['percentage_change']['bleu_4']:+.1f}%)")
        print(f"ROUGE-L: {summary['difference']['rouge_l']:+.3f} ({summary['percentage_change']['rouge_l']:+.1f}%)")
        print(f"BERTScore-F1: {summary['difference']['bert_score_f1']:+.3f} ({summary['percentage_change']['bert_score_f1']:+.1f}%)")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        return

if __name__ == "__main__":
    main() 