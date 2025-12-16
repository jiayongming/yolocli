#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证数据集配置"""

import yaml
from pathlib import Path

def verify_dataset_config(yaml_file):
    """验证dataset.yaml配置"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    
    print("验证数据集配置...")
    print(f"数据集路径: {base_path}")
    print(f"类别数量: {config['nc']}")
    print(f"类别名称: {config['names']}")
    
    # 检查目录
    for split in ['train', 'val', 'test']:
        if split in config:
            img_dir = base_path / config[split]
            label_dir = base_path / 'labels' / split
            
            if img_dir.exists():
                img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
                label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
                print(f"\n{split}:")
                print(f"  图像: {img_count}")
                print(f"  标签: {label_count}")
                if img_count != label_count:
                    print(f"  ⚠ 警告: 图像和标签数量不匹配！")
            else:
                print(f"\n{split}: 目录不存在 {img_dir}")
    
    print("\n验证完成！")

if __name__ == "__main__":
    import sys
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.yaml"
    verify_dataset_config(yaml_file)
