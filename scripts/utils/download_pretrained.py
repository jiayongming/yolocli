#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载YOLOv11预训练模型"""

from ultralytics import YOLO
import os

def download_pretrained_models():
    """下载预训练模型"""
    models_dir = "models/weights"
    os.makedirs(models_dir, exist_ok=True)
    
    # 要下载的模型列表
    model_names = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']
    
    for model_name in model_names:
        print(f"正在下载 {model_name}...")
        try:
            model = YOLO(model_name)  # 会自动下载
            # 移动到指定目录
            if os.path.exists(model_name):
                os.rename(model_name, os.path.join(models_dir, model_name))
                print(f"✓ {model_name} 下载完成")
            else:
                print(f"✗ {model_name} 下载失败")
        except Exception as e:
            print(f"✗ {model_name} 下载出错: {e}")
    
    print("\n预训练模型下载完成！")

if __name__ == "__main__":
    download_pretrained_models()
