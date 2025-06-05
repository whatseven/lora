#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理工具
用于查看、管理和配置微调模型路径
"""

import os
import json
from datetime import datetime

class ModelManager:
    def __init__(self):
        self.base_dir = "/home/ubuntu/ZJQ/ai_report/Lora_Qwen3_finetune/model/finetuned"
        
    def list_models(self):
        """列出所有可用的微调模型"""
        print("🔍 查找可用的微调模型...")
        print("=" * 60)
        
        if not os.path.exists(self.base_dir):
            print("❌ 微调模型目录不存在")
            return []
        
        models = []
        
        # 查找所有lora_v*目录
        for item in os.listdir(self.base_dir):
            model_path = os.path.join(self.base_dir, item)
            if os.path.isdir(model_path) and (item.startswith("lora_v") or item == "lora"):
                # 检查是否包含必要文件
                adapter_file = os.path.join(model_path, "adapter_model.safetensors")
                config_file = os.path.join(model_path, "adapter_config.json")
                training_config = os.path.join(model_path, "training_config.json")
                
                if os.path.exists(adapter_file) and os.path.exists(config_file):
                    model_info = {
                        "name": item,
                        "path": model_path,
                        "adapter_size": self._get_file_size(adapter_file),
                        "created_time": self._get_creation_time(model_path)
                    }
                    
                    # 读取训练配置信息
                    if os.path.exists(training_config):
                        try:
                            with open(training_config, 'r', encoding='utf-8') as f:
                                train_info = json.load(f)
                                model_info.update({
                                    "base_model": train_info.get("base_model_path", "未知"),
                                    "save_time": train_info.get("save_time", "未知"),
                                    "is_pretrained": train_info.get("is_pretrained", True)
                                })
                        except:
                            pass
                    
                    models.append(model_info)
        
        # 检查latest链接
        latest_link = os.path.join(self.base_dir, "latest")
        latest_target = None
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            latest_target = os.readlink(latest_link)
        
        # 按时间排序
        models.sort(key=lambda x: x["created_time"], reverse=True)
        
        # 显示模型列表
        if models:
            print(f"找到 {len(models)} 个微调模型:")
            print()
            for i, model in enumerate(models, 1):
                is_latest = (latest_target and model["name"] == latest_target)
                latest_mark = " [LATEST]" if is_latest else ""
                
                print(f"{i}. {model['name']}{latest_mark}")
                print(f"   路径: {model['path']}")
                print(f"   大小: {model['adapter_size']}")
                print(f"   创建时间: {model['created_time']}")
                if "base_model" in model:
                    print(f"   基础模型: {model['base_model']}")
                    print(f"   训练类型: {'预训练模型' if model['is_pretrained'] else '增量训练'}")
                print()
        else:
            print("❌ 未找到可用的微调模型")
            print("请先运行训练脚本完成模型微调")
        
        print("=" * 60)
        return models
    
    def get_latest_model(self):
        """获取最新的模型路径"""
        latest_link = os.path.join(self.base_dir, "latest")
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            target = os.readlink(latest_link)
            return os.path.join(self.base_dir, target)
        return None
    
    def set_latest_model(self, model_name):
        """设置某个模型为最新模型"""
        model_path = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在: {model_name}")
            return False
        
        latest_link = os.path.join(self.base_dir, "latest")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        
        os.symlink(model_name, latest_link)
        print(f"✅ 已将 {model_name} 设置为最新模型")
        return True
    
    def show_training_template(self):
        """显示训练配置模板"""
        print("🛠️  训练配置模板")
        print("=" * 60)
        print("在 train.py 中修改以下配置:")
        print()
        print("# 1. 使用官方预训练模型（首次训练）")
        print("BASE_MODEL_PATH = None")
        print("MODEL_VERSION = None  # 自动生成时间戳")
        print()
        print("# 2. 使用之前微调的模型继续训练")
        print("BASE_MODEL_PATH = \"/path/to/previous/model\"")
        print("MODEL_VERSION = \"battery_v2\"  # 自定义版本号")
        print()
        print("示例：")
        
        models = self.list_models()
        if models:
            latest_model = models[0]  # 最新的模型
            print(f"BASE_MODEL_PATH = \"{latest_model['path']}\"")
            print("MODEL_VERSION = \"继续训练_v1\"")
        
        print()
        print("=" * 60)
    
    def _get_file_size(self, filepath):
        """获取文件大小的可读格式"""
        try:
            size = os.path.getsize(filepath)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "未知"
    
    def _get_creation_time(self, filepath):
        """获取文件创建时间"""
        try:
            timestamp = os.path.getctime(filepath)
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except:
            return "未知"

def main():
    """主函数"""
    manager = ModelManager()
    
    print("🚀 Qwen3 模型管理工具")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("1. 查看所有模型")
        print("2. 显示训练配置模板")
        print("3. 设置最新模型")
        print("4. 查看当前最新模型")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            manager.list_models()
        
        elif choice == "2":
            manager.show_training_template()
        
        elif choice == "3":
            models = manager.list_models()
            if models:
                print("请选择要设置为最新的模型:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model['name']}")
                
                try:
                    idx = int(input("请输入模型编号: ")) - 1
                    if 0 <= idx < len(models):
                        manager.set_latest_model(models[idx]["name"])
                    else:
                        print("❌ 无效的编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
        
        elif choice == "4":
            latest = manager.get_latest_model()
            if latest:
                print(f"✅ 当前最新模型: {latest}")
            else:
                print("❌ 未设置最新模型")
        
        elif choice == "5":
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效的选项，请重新选择")

if __name__ == "__main__":
    main() 