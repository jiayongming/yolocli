#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计数据集信息"""

import os
from pathlib import Path
from collections import Counter
from PIL import Image

def analyze_dataset(data_dir):
    """分析数据集"""
    data_path = Path(data_dir)
    
    stats = {
        'total_images': 0,
        'categories': Counter(),
        'image_sizes': [],
        'formats': Counter()
    }
    
    for img_file in data_path.rglob('*.jpg'):
        stats['total_images'] += 1
        stats['formats']['jpg'] += 1
        
        # 获取类别（从目录名）
        category = img_file.parent.name
        stats['categories'][category] += 1
        
        # 获取图像尺寸
        try:
            img = Image.open(img_file)
            stats['image_sizes'].append(img.size)
        except Exception as e:
            print(f"无法读取 {img_file}: {e}")
    
    # 统计PNG
    for img_file in data_path.rglob('*.png'):
        stats['total_images'] += 1
        stats['formats']['png'] += 1
        category = img_file.parent.name
        stats['categories'][category] += 1
        try:
            img = Image.open(img_file)
            stats['image_sizes'].append(img.size)
        except Exception as e:
            print(f"无法读取 {img_file}: {e}")
    
    # 打印统计信息
    print("=" * 50)
    print("数据集统计信息")
    print("=" * 50)
    print(f"总图像数: {stats['total_images']}")
    print(f"\n类别分布:")
    for cat, count in stats['categories'].most_common():
        print(f"  {cat}: {count} ({count/stats['total_images']*100:.1f}%)")
    print(f"\n图像格式:")
    for fmt, count in stats['formats'].most_common():
        print(f"  {fmt}: {count}")
    
    if stats['image_sizes']:
        sizes = Counter(stats['image_sizes'])
        print(f"\n常见图像尺寸:")
        for size, count in sizes.most_common(5):
            print(f"  {size}: {count}")
    
    return stats

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    analyze_dataset(data_dir)
