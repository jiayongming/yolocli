#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""划分数据集为训练集、验证集、测试集"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(
    images_dir,
    labels_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """划分数据集"""
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # 收集所有图像-标签对
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    pairs = []
    for img_file in images_path.rglob('*.jpg'):
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            pairs.append((img_file, label_file))
    
    for img_file in images_path.rglob('*.png'):
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            pairs.append((img_file, label_file))
    
    # 打乱顺序
    random.shuffle(pairs)
    
    # 计算划分点
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分数据集
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    # 复制文件
    stats = {}
    for split_name, split_pairs in splits.items():
        count = 0
        for img_file, label_file in split_pairs:
            # 复制图像
            dst_img = os.path.join(output_dir, 'images', split_name, img_file.name)
            shutil.copy2(img_file, dst_img)
            
            # 复制标签
            dst_label = os.path.join(output_dir, 'labels', split_name, label_file.name)
            shutil.copy2(label_file, dst_label)
            
            count += 1
        
        stats[split_name] = count
        print(f"{split_name}: {count} 个样本")
    
    # 保存统计信息
    with open(os.path.join(output_dir, 'split_statistics.txt'), 'w') as f:
        f.write("数据集划分统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"训练集: {stats['train']} ({stats['train']/total*100:.1f}%)\n")
        f.write(f"验证集: {stats['val']} ({stats['val']/total*100:.1f}%)\n")
        f.write(f"测试集: {stats['test']} ({stats['test']/total*100:.1f}%)\n")
    
    print(f"\n数据集划分完成！总计 {total} 个样本")
    return stats

if __name__ == "__main__":
    import sys
    
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    labels_dir = sys.argv[2] if len(sys.argv) > 2 else "data/labels/raw"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/processed"
    
    split_dataset(images_dir, labels_dir, output_dir)
