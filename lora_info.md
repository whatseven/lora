# LoRA数据处理分析与代码修改指南

## 问题分析

你提供的代码和unsloth示例有本质差异：

### 你的代码特点：
- **直接处理token层面**：手动分词、拼接input_ids、labels
- **使用传统SFT方法**：需要手动构造instruction模板和labels
- **自定义对话格式**：使用了特定的system prompt（甄嬛角色扮演）

### Unsloth示例特点：
- **文本层面处理**：先转换为conversation格式，再用chat_template
- **框架自动处理**：unsloth自动处理tokenization和labels
- **标准对话格式**：使用Qwen3的官方对话模板

## 格式对比

### 你的代码输出格式：
```
<|im_start|>system
现在你要扮演皇帝身边的女人--甄嬛<|im_end|>
<|im_start|>user
{instruction + input}<|im_end|>
<|im_start|>assistant
{output}
```

### Unsloth示例输出格式：
```
<|im_start|>user
{instruction + input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

## 核心差异

1. **数据处理层级不同**：
   - 你的代码：Alpaca → tokens (input_ids/labels)
   - Unsloth：Alpaca → conversation → formatted_text

2. **框架集成度不同**：
   - 你的代码：需要手动处理所有细节
   - Unsloth：框架自动处理，更简洁高效

## 修改方案

### 方案1：适配Unsloth框架（推荐）

```python
def convert_alpaca_to_conversation(example, system_prompt=None):
    """
    将Alpaca格式转换为conversation格式，适配unsloth
    
    Args:
        example: Alpaca格式的单条数据
        system_prompt: 可选的系统提示词
    
    Returns:
        list: conversation格式的对话
    """
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']
    
    # 构造用户输入
    if input_text.strip():
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction
    
    # 构造对话
    conversation = []
    
    # 如果有系统提示词，添加system role
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    
    conversation.extend([
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ])
    
    return conversation

def process_alpaca_dataset_for_unsloth(dataset_path, system_prompt=None):
    """
    处理整个Alpaca数据集用于unsloth训练
    
    Args:
        dataset_path: 数据集文件路径
        system_prompt: 可选的系统提示词
    
    Returns:
        list: 转换后的conversations列表
    """
    import json
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    for example in data:
        conversation = convert_alpaca_to_conversation(example, system_prompt)
        conversations.append(conversation)
    
    return conversations

# 使用示例
def prepare_dataset_with_custom_system_prompt():
    """使用自定义系统提示词准备数据集"""
    
    # 定义系统提示词（可选）
    system_prompt = "现在你要扮演皇帝身边的女人--甄嬛"
    
    # 转换数据集
    train_conversations = process_alpaca_dataset_for_unsloth(
        "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        system_prompt=system_prompt
    )
    
    # 应用对话模板
    formatted_texts = tokenizer.apply_chat_template(
        train_conversations,
        tokenize=False,
    )
    
    return formatted_texts

# 不使用系统提示词的版本
def prepare_dataset_standard():
    """标准版本，不使用系统提示词"""
    
    train_conversations = process_alpaca_dataset_for_unsloth(
        "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        system_prompt=None
    )
    
    formatted_texts = tokenizer.apply_chat_template(
        train_conversations,
        tokenize=False,
    )
    
    return formatted_texts
```

### 方案2：保持你的代码风格但适配Qwen3

```python
def process_func_for_qwen3(example, tokenizer, system_prompt=None, max_length=2048):
    """
    改进版的处理函数，适配Qwen3分词器
    
    Args:
        example: Alpaca格式数据
        tokenizer: Qwen3分词器
        system_prompt: 可选系统提示词
        max_length: 最大长度
    
    Returns:
        dict: 包含input_ids, attention_mask, labels的字典
    """
    # 构造用户输入
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']
    
    if input_text.strip():
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction
    
    # 构造完整的对话文本
    if system_prompt:
        full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    else:
        full_text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    # 分别编码instruction和response部分
    if system_prompt:
        instruction_part = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    else:
        instruction_part = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    response_part = f"{output}<|im_end|>"
    
    # 编码
    instruction_tokens = tokenizer(instruction_part, add_special_tokens=False)
    response_tokens = tokenizer(response_part, add_special_tokens=False)
    
    # 拼接
    input_ids = instruction_tokens["input_ids"] + response_tokens["input_ids"]
    attention_mask = instruction_tokens["attention_mask"] + response_tokens["attention_mask"]
    labels = [-100] * len(instruction_tokens["input_ids"]) + response_tokens["input_ids"]
    
    # 截断处理
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 批量处理数据集
def process_alpaca_dataset_token_level(dataset_path, tokenizer, system_prompt=None, max_length=2048):
    """
    在token层面处理Alpaca数据集
    
    Args:
        dataset_path: 数据集路径
        tokenizer: 分词器
        system_prompt: 系统提示词
        max_length: 最大长度
    
    Returns:
        Dataset: 处理后的数据集
    """
    import json
    from datasets import Dataset
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for example in data:
        processed_example = process_func_for_qwen3(
            example, 
            tokenizer, 
            system_prompt=system_prompt, 
            max_length=max_length
        )
        processed_data.append(processed_example)
    
    return Dataset.from_list(processed_data)
```

### 方案3：混合方案（最灵活）

```python
def flexible_alpaca_processor(dataset_path, tokenizer, processing_mode="unsloth", system_prompt=None, max_length=2048):
    """
    灵活的Alpaca数据处理器，支持多种处理模式
    
    Args:
        dataset_path: 数据集路径
        tokenizer: 分词器
        processing_mode: "unsloth" 或 "token_level"
        system_prompt: 系统提示词
        max_length: 最大长度
    
    Returns:
        相应格式的数据集
    """
    import json
    from datasets import Dataset
    import pandas as pd
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if processing_mode == "unsloth":
        # 使用unsloth方式处理
        conversations = []
        for example in data:
            conversation = convert_alpaca_to_conversation(example, system_prompt)
            conversations.append(conversation)
        
        # 应用对话模板
        formatted_texts = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
        )
        
        # 创建数据集
        data_df = pd.DataFrame({"text": formatted_texts})
        return Dataset.from_pandas(data_df)
    
    elif processing_mode == "token_level":
        # 使用token级别处理
        processed_data = []
        for example in data:
            processed_example = process_func_for_qwen3(
                example, 
                tokenizer, 
                system_prompt=system_prompt, 
                max_length=max_length
            )
            processed_data.append(processed_example)
        
        return Dataset.from_list(processed_data)
    
    else:
        raise ValueError("processing_mode must be 'unsloth' or 'token_level'")

# 使用示例
def main_processing_example():
    """主要处理示例"""
    from unsloth import FastLanguageModel
    
    # 加载模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # 方式1：使用unsloth方式（推荐）
    dataset_unsloth = flexible_alpaca_processor(
        dataset_path="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        tokenizer=tokenizer,
        processing_mode="unsloth",
        system_prompt="现在你要扮演皇帝身边的女人--甄嬛"  # 可选
    )
    
    # 方式2：使用token级别方式
    dataset_token_level = flexible_alpaca_processor(
        dataset_path="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        tokenizer=tokenizer,
        processing_mode="token_level",
        system_prompt="现在你要扮演皇帝身边的女人--甄嬛",
        max_length=2048
    )
    
    return dataset_unsloth, dataset_token_level
```

## 推荐方案

**强烈推荐使用方案1（unsloth方式）**，原因：

1. **更简洁**：unsloth框架自动处理复杂的tokenization
2. **更稳定**：减少手动处理token可能出现的错误
3. **更高效**：unsloth针对训练进行了优化
4. **更标准**：使用官方推荐的对话模板

## 完整集成代码

```python
# 完整的数据处理和训练流程
def complete_training_workflow():
    """完整的训练工作流程"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import pandas as pd
    from datasets import Dataset
    
    # 1. 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # 2. 配置LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 3. 处理数据（带自定义系统提示词）
    train_conversations = process_alpaca_dataset_for_unsloth(
        "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        system_prompt="现在你要扮演皇帝身边的女人--甄嬛"
    )
    
    # 4. 格式化数据
    formatted_texts = tokenizer.apply_chat_template(
        train_conversations,
        tokenize=False,
    )
    
    # 5. 创建数据集
    data_df = pd.DataFrame({"text": formatted_texts})
    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.shuffle(seed=3407)
    
    # 6. 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora",
            report_to="none",
        ),
    )
    
    # 7. 开始训练
    trainer.train()
    
    # 8. 保存模型
    model.save_pretrained("/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora")
    tokenizer.save_pretrained("/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = complete_training_workflow()
```

这样修改后，你的代码就能完美适配unsloth框架，同时保持你想要的角色扮演功能。

---

## 🔍 Unsloth数据处理详细分析

### Alpaca格式到对话格式的转换逻辑

#### 1. Alpaca原始格式
```json
{
    "instruction": "翻译以下中文到英文",
    "input": "你好，世界！",
    "output": "Hello, World!"
}
```

#### 2. Unsloth处理逻辑分析

**第一步：合并instruction和input**
```python
# unsloth的核心处理逻辑
if input_text.strip():
    user_content = f"{instruction}\n{input_text}"  # 用换行符连接
else:
    user_content = instruction  # 只有instruction的情况
```

**为什么要合并？**
- `instruction`：任务描述/指令
- `input`：具体的输入内容  
- 这两者共同构成了用户的完整请求
- 模型需要理解：要做什么(instruction) + 对什么操作(input)

**第二步：构造conversation格式**
```python
conversation = [
    {"role": "user", "content": "翻译以下中文到英文\n你好，世界！"},
    {"role": "assistant", "content": "Hello, World!"}
]
```

**第三步：应用chat_template**
```python
formatted_text = tokenizer.apply_chat_template(conversation, tokenize=False)
```

**最终输出格式：**
```
<|im_start|>user
翻译以下中文到英文
你好，世界！<|im_end|>
<|im_start|>assistant
Hello, World!<|im_end|>
```

### 为什么只剩下user和assistant？

这是**完全正确**的设计：

1. **instruction + input = user的完整请求**
   - 用户提供指令(instruction)和数据(input)
   - 这就是一个完整的user message

2. **output = assistant的回复**
   - 模型根据用户请求给出回答
   - 这就是assistant message

3. **符合对话的自然逻辑**
   - 真实对话中，用户会说："请帮我翻译这句话：你好世界"
   - 而不会分开说："我要翻译" + "你好世界"

### 完整的适配代码

```python
# ===== 完整的Unsloth适配代码 =====

import json
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

def alpaca_to_conversation(example, system_prompt=None):
    """
    将单条Alpaca数据转换为conversation格式
    
    Args:
        example: 单条Alpaca数据 {"instruction": "", "input": "", "output": ""}
        system_prompt: 可选的系统提示词
    
    Returns:
        list: conversation格式 [{"role": "user", "content": "..."}, ...]
    """
    instruction = example['instruction']
    input_text = example.get('input', '')  # 用get方法防止KeyError
    output = example['output']
    
    # 1. 合并instruction和input
    if input_text.strip():  # 如果input不为空
        user_content = f"{instruction}\n{input_text}"
    else:  # 如果input为空
        user_content = instruction  # 只有instruction的情况
    
    # 2. 构造conversation
    conversation = []
    
    # 可选：添加系统提示词
    if system_prompt:
        conversation.append({
            "role": "system", 
            "content": system_prompt
        })
    
    # 添加用户消息和助手回复
    conversation.extend([
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ])
    
    return conversation

def process_alpaca_dataset(dataset_path, tokenizer, system_prompt=None):
    """
    处理整个Alpaca数据集
    
    Args:
        dataset_path: 数据集文件路径
        tokenizer: Qwen3分词器
        system_prompt: 可选的系统提示词
    
    Returns:
        Dataset: 可用于训练的数据集
    """
    # 1. 加载原始数据
    print(f"正在加载数据集: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        alpaca_data = json.load(f)
    print(f"数据集大小: {len(alpaca_data)}")
    
    # 2. 转换为conversation格式
    print("正在转换为conversation格式...")
    conversations = []
    for example in alpaca_data:
        conversation = alpaca_to_conversation(example, system_prompt)
        conversations.append(conversation)
    
    # 3. 应用chat_template
    print("正在应用chat_template...")
    formatted_texts = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,  # 返回文本而不是token
    )
    
    # 4. 创建Dataset
    print("正在创建Dataset...")
    df = pd.DataFrame({"text": formatted_texts})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=3407)
    
    print(f"处理完成！最终数据集大小: {len(dataset)}")
    return dataset

def setup_model_and_training(dataset, output_dir, num_epochs=1, batch_size=4):
    """
    设置模型和训练配置
    
    Args:
        dataset: 处理好的数据集
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
    
    Returns:
        trainer: 配置好的训练器
    """
    # 1. 加载模型
    print("正在加载Qwen3-14B模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=2048,
        load_in_4bit=True,
        cache_dir="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/pretrained"
    )
    
    # 2. 配置LoRA
    print("正在配置LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # A100 32G可以使用较大的rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 3. 配置训练器
    print("正在配置训练器...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",  # 重要：指定文本字段
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            save_strategy="epoch",
            save_steps=100,
        ),
    )
    
    return trainer, model, tokenizer

# ===== 主要使用示例 =====

def main_training_pipeline():
    """主要训练流程"""
    
    # 步骤1：加载模型和分词器（用于数据处理）
    print("=== 步骤1: 加载分词器 ===")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # 步骤2：处理数据集
    print("\n=== 步骤2: 处理数据集 ===")
    dataset = process_alpaca_dataset(
        dataset_path="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        tokenizer=tokenizer,
        system_prompt="现在你要扮演皇帝身边的女人--甄嬛"  # 可选的系统提示词
    )
    
    # 步骤3：设置训练
    print("\n=== 步骤3: 设置训练 ===")
    trainer, model, tokenizer = setup_model_and_training(
        dataset=dataset,
        output_dir="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora",
        num_epochs=1,
        batch_size=4
    )
    
    # 步骤4：开始训练
    print("\n=== 步骤4: 开始训练 ===")
    trainer.train()
    
    # 步骤5：保存模型
    print("\n=== 步骤5: 保存模型 ===")
    model.save_pretrained("/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora")
    tokenizer.save_pretrained("/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned/lora")
    
    print("\n训练完成！")
    return model, tokenizer

# ===== 数据预览功能 =====

def preview_processed_data(dataset_path, tokenizer, system_prompt=None, num_samples=3):
    """
    预览处理后的数据格式
    
    Args:
        dataset_path: 数据集路径
        tokenizer: 分词器
        system_prompt: 系统提示词
        num_samples: 预览样本数
    """
    print("=== 数据处理预览 ===")
    
    # 加载原始数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        alpaca_data = json.load(f)
    
    for i in range(min(num_samples, len(alpaca_data))):
        print(f"\n--- 样本 {i+1} ---")
        example = alpaca_data[i]
        
        print("原始Alpaca格式:")
        print(f"  instruction: {example['instruction']}")
        print(f"  input: {example.get('input', '')}")
        print(f"  output: {example['output']}")
        
        # 转换为conversation
        conversation = alpaca_to_conversation(example, system_prompt)
        print(f"\nConversation格式:")
        for msg in conversation:
            print(f"  {msg['role']}: {msg['content']}")
        
        # 应用chat_template
        formatted_text = tokenizer.apply_chat_template([conversation], tokenize=False)[0]
        print(f"\n最终格式化文本:")
        print(formatted_text)
        print("-" * 50)

# 使用示例
if __name__ == "__main__":
    # 先预览数据处理效果
    from unsloth import FastLanguageModel
    _, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen3-14B", max_seq_length=2048, load_in_4bit=True)
    
    preview_processed_data(
        dataset_path="/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/dataset/train/alpaca_train.json",
        tokenizer=tokenizer,
        system_prompt="现在你要扮演皇帝身边的女人--甄嬛",
        num_samples=2
    )
    
    # 然后开始训练
    # main_training_pipeline()
```

### 关键要点总结

1. **instruction + input → user content**：这是标准做法，符合对话逻辑
2. **output → assistant content**：模型的回复
3. **system prompt**：可选的角色设定，会作为独立的system消息
4. **chat_template**：unsloth自动处理，生成标准的Qwen3对话格式
5. **完全兼容**：这种处理方式与你的需求完全兼容，还能享受unsloth的优化

这样处理后，你就能在保持角色扮演功能的同时，充分利用unsloth框架的所有优势！
