#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据处理命令"""

import typer
from pathlib import Path
from typing import Optional
import yaml
import shutil
import random
from collections import defaultdict

from ..core.config import ConfigManager
from ..core.utils import ensure_dir, get_dataset_info, parse_ratio_string, find_files
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_dataset_info, print_section_header, print_table,
    create_progress_bar, print_key_value, console
)

app = typer.Typer(help="数据处理命令")


@app.command("split")
def split_dataset(
    images_dir: str = typer.Option(..., "--images", "-i", help="图像目录"),
    labels_dir: str = typer.Option(..., "--labels", "-l", help="标签目录"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    ratios: str = typer.Option("0.7:0.2:0.1", "--ratios", "-r", help="划分比例 (train:val:test)"),
    seed: int = typer.Option(42, "--seed", "-s", help="随机种子"),
):
    """划分数据集为训练集、验证集、测试集"""
    
    print_section_header("数据集划分")
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 验证输入目录
    if not images_path.exists():
        print_error(f"图像目录不存在: {images_dir}")
        raise typer.Exit(1)
    
    if not labels_path.exists():
        print_error(f"标签目录不存在: {labels_dir}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True)
    else:
        output_path = Path(output_dir)
    
    print_info(f"图像目录: {images_path}")
    print_info(f"标签目录: {labels_path}")
    print_info(f"输出目录: {output_path}")
    
    # 解析比例
    try:
        train_ratio, val_ratio, test_ratio = parse_ratio_string(ratios, 3)
        print_info(f"划分比例: 训练={train_ratio:.1%}, 验证={val_ratio:.1%}, 测试={test_ratio:.1%}")
    except ValueError as e:
        print_error(f"比例格式错误: {e}")
        raise typer.Exit(1)
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        ensure_dir(output_path / 'images' / split)
        ensure_dir(output_path / 'labels' / split)
    
    # 收集图像-标签对
    print_info("扫描图像和标签文件...")
    pairs = []
    
    for img_file in find_files(images_path, ['.jpg', '.jpeg', '.png']):
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            print_warning(f"标签文件缺失: {img_file.name}")
    
    if not pairs:
        print_error("未找到有效的图像-标签对")
        raise typer.Exit(1)
    
    print_info(f"找到 {len(pairs)} 个有效样本")
    
    # 打乱顺序
    random.shuffle(pairs)
    
    # 计算划分点
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    # 复制文件
    print_info("开始复制文件...")
    
    with create_progress_bar() as progress:
        task = progress.add_task("复制文件", total=total)
        
        stats = {}
        for split_name, split_pairs in splits.items():
            count = 0
            for img_file, label_file in split_pairs:
                # 复制图像
                dst_img = output_path / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # 复制标签
                dst_label = output_path / 'labels' / split_name / label_file.name
                shutil.copy2(label_file, dst_label)
                
                count += 1
                progress.advance(task)
            
            stats[split_name] = count
    
    # 打印统计信息
    console.print()
    columns = ["数据集", "样本数", "比例"]
    rows = [
        ["训练集", stats['train'], f"{stats['train']/total*100:.1f}%"],
        ["验证集", stats['val'], f"{stats['val']/total*100:.1f}%"],
        ["测试集", stats['test'], f"{stats['test']/total*100:.1f}%"],
        ["总计", total, "100.0%"],
    ]
    print_table("数据集划分结果", columns, rows, show_lines=True)
    
    # 保存统计信息
    stats_file = output_path / 'split_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据集划分统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"训练集: {stats['train']} ({stats['train']/total*100:.1f}%)\n")
        f.write(f"验证集: {stats['val']} ({stats['val']/total*100:.1f}%)\n")
        f.write(f"测试集: {stats['test']} ({stats['test']/total*100:.1f}%)\n")
        f.write(f"随机种子: {seed}\n")
    
    print_success(f"数据集划分完成！输出目录: {output_path}")


@app.command("generate-yaml")
def generate_yaml(
    data_path: Optional[str] = typer.Option(None, "--path", "-p", help="数据集路径"),
    classes_file: Optional[str] = typer.Option(None, "--classes", "-c", help="类别文件路径"),
    output: str = typer.Option("data/dataset.yaml", "--output", "-o", help="输出文件路径"),
    train_dir: str = typer.Option("images/train", "--train", help="训练集目录"),
    val_dir: str = typer.Option("images/val", "--val", help="验证集目录"),
    test_dir: str = typer.Option("images/test", "--test", help="测试集目录"),
):
    """生成dataset.yaml配置文件"""
    
    print_section_header("生成 dataset.yaml")
    
    # 确定数据集路径
    if data_path is None:
        config = ConfigManager()
        data_path = str(config.get_path('data_processed', absolute=True))
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        print_error(f"数据集路径不存在: {data_path}")
        raise typer.Exit(1)
    
    print_info(f"数据集路径: {data_path}")
    
    # 读取类别文件
    if classes_file is None:
        # 尝试在多个位置查找 classes.txt
        possible_locations = [
            data_path / 'classes.txt',
            data_path.parent / 'raw' / 'classes.txt',
            Path('data/raw/classes.txt'),
        ]
        
        classes_file = None
        for loc in possible_locations:
            if loc.exists():
                classes_file = loc
                break
        
        if classes_file is None:
            print_error("未找到 classes.txt 文件")
            print_info("请使用 --classes 参数指定类别文件")
            raise typer.Exit(1)
    else:
        classes_file = Path(classes_file)
        if not classes_file.exists():
            print_error(f"类别文件不存在: {classes_file}")
            raise typer.Exit(1)
    
    print_info(f"类别文件: {classes_file}")
    
    # 读取类别
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    if not classes:
        print_error("类别文件为空")
        raise typer.Exit(1)
    
    print_info(f"类别数量: {len(classes)}")
    print_info(f"类别: {', '.join(classes)}")
    
    # 验证目录存在
    train_path = data_path / train_dir
    val_path = data_path / val_dir
    test_path = data_path / test_dir
    
    if not train_path.exists():
        print_warning(f"训练集目录不存在: {train_path}")
    if not val_path.exists():
        print_warning(f"验证集目录不存在: {val_path}")
    
    # 生成YAML配置
    yaml_config = {
        'path': str(data_path),
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes),
    }
    
    # 保存YAML文件
    output_path = Path(output)
    ensure_dir(output_path.parent)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO 数据集配置文件\n")
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"dataset.yaml 已生成: {output_path}")
    
    # 显示生成的配置
    console.print("\n生成的配置内容:")
    console.print("─" * 50)
    print_key_value("path", yaml_config['path'])
    print_key_value("train", yaml_config['train'])
    print_key_value("val", yaml_config['val'])
    print_key_value("test", yaml_config['test'])
    print_key_value("nc", yaml_config['nc'])
    print_key_value("names", ", ".join(yaml_config['names'].values()))


@app.command("verify")
def verify_dataset(
    data_path: Optional[str] = typer.Option(None, "--path", "-p", help="数据集路径"),
):
    """验证数据集完整性"""
    
    print_section_header("数据集验证")
    
    # 确定数据集路径
    if data_path is None:
        config = ConfigManager()
        data_path = config.get_path('data_processed', absolute=True)
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        print_error(f"数据集路径不存在: {data_path}")
        raise typer.Exit(1)
    
    print_info(f"数据集路径: {data_path}")
    
    # 获取数据集信息
    info = get_dataset_info(data_path)
    
    # 打印统计信息
    print_dataset_info(info)
    
    # 验证图像-标签对应关系
    issues = []
    
    for split in ['train', 'val', 'test']:
        img_dir = data_path / 'images' / split
        label_dir = data_path / 'labels' / split
        
        if not img_dir.exists():
            continue
        
        # 检查每个图像是否有对应的标签
        for img_file in find_files(img_dir):
            label_file = label_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                issues.append(f"{split}: 缺少标签 - {img_file.name}")
        
        # 检查标签文件格式
        if label_dir.exists():
            for label_file in label_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) < 5:
                                issues.append(f"{split}: 标签格式错误 - {label_file.name}:{line_num}")
                except Exception as e:
                    issues.append(f"{split}: 无法读取标签 - {label_file.name}: {e}")
    
    # 显示验证结果
    console.print()
    if issues:
        print_warning(f"发现 {len(issues)} 个问题:")
        for issue in issues[:20]:  # 最多显示20个
            print_error(f"  • {issue}")
        if len(issues) > 20:
            print_warning(f"  ... 还有 {len(issues) - 20} 个问题")
    else:
        print_success("数据集验证通过，未发现问题")


@app.command("stats")
def dataset_stats(
    data_path: Optional[str] = typer.Option(None, "--path", "-p", help="数据集路径"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="显示详细统计"),
):
    """数据集统计分析"""
    
    print_section_header("数据集统计")
    
    # 确定数据集路径
    if data_path is None:
        config = ConfigManager()
        data_path = config.get_path('data_processed', absolute=True)
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        print_error(f"数据集路径不存在: {data_path}")
        raise typer.Exit(1)
    
    print_info(f"数据集路径: {data_path}")
    
    # 获取基本信息
    info = get_dataset_info(data_path)
    print_dataset_info(info)
    
    if detailed:
        # 统计类别分布
        print_section_header("类别分布统计")
        
        class_counts = defaultdict(int)
        bbox_counts = 0
        
        for split in ['train', 'val', 'test']:
            label_dir = data_path / 'labels' / split
            if not label_dir.exists():
                continue
            
            for label_file in label_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                                bbox_counts += 1
                except Exception:
                    pass
        
        if class_counts:
            columns = ["类别ID", "数量", "比例"]
            rows = []
            total = sum(class_counts.values())
            
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                rows.append([
                    class_id,
                    count,
                    f"{count/total*100:.1f}%"
                ])
            
            print_table("类别分布", columns, rows, show_lines=True)
            print_info(f"总边界框数量: {bbox_counts}")
        else:
            print_warning("未找到标注数据")


if __name__ == "__main__":
    app()
