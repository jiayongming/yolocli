#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据处理命令"""

import typer
from pathlib import Path
from typing import Optional, List
import yaml
import shutil
import random
from collections import defaultdict

from ..core.config import ConfigManager
from ..core.utils import (
    ensure_dir, get_dataset_info, parse_ratio_string, find_files,
    TaskType, validate_task_type
)
from ..core.deduplicator import ImageDeduplicator
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_dataset_info, print_section_header, print_table,
    create_progress_bar, print_key_value, console
)
from ..converters.labelstudio import LabelStudioClient, LabelStudioConverter

app = typer.Typer(help="数据处理命令")


def _validate_segment_label(label_file: Path) -> bool:
    """
    验证分割标签文件格式
    
    Args:
        label_file: 标签文件路径
        
    Returns:
        bool: 标签格式是否有效
    """
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 分割标签至少需要：class_id + 至少3个点（6个坐标值）
                if len(parts) < 7:  # 1 (class) + 6 (3个点的坐标)
                    return False
                # 检查是否有偶数个坐标值（x,y配对）
                if (len(parts) - 1) % 2 != 0:
                    return False
                # 验证所有值都是有效数字
                try:
                    int(parts[0])  # class_id
                    for val in parts[1:]:
                        coord = float(val)
                        if coord < 0 or coord > 1:
                            return False
                except ValueError:
                    return False
        return True
    except Exception:
        return False


def _validate_detect_label(label_file: Path) -> bool:
    """
    验证检测标签文件格式
    
    Args:
        label_file: 标签文件路径
        
    Returns:
        bool: 标签格式是否有效
    """
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 检测标签需要：class_id + 4个坐标值
                if len(parts) != 5:
                    return False
                # 验证所有值都是有效数字
                try:
                    int(parts[0])  # class_id
                    for val in parts[1:]:
                        coord = float(val)
                        if coord < 0 or coord > 1:
                            return False
                except ValueError:
                    return False
        return True
    except Exception:
        return False


def _validate_pose_label(label_file: Path, expected_kpt_count: Optional[int] = None) -> bool:
    """
    验证姿势估计标签文件格式
    
    格式: class_id x y w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    每个关键点需要 3 个值 (x, y, visibility)
    
    Args:
        label_file: 标签文件路径
        expected_kpt_count: 期望的关键点数量（可选）
        
    Returns:
        bool: 标签格式是否有效
    """
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 最少：class_id + bbox(4) + 至少1个关键点(3)
                if len(parts) < 8:
                    return False
                
                # 检查关键点部分是否为3的倍数
                kpt_values = len(parts) - 5  # 减去 class_id 和 bbox
                if kpt_values % 3 != 0:
                    return False
                
                # 如果指定了关键点数量，验证是否匹配
                if expected_kpt_count is not None:
                    if kpt_values // 3 != expected_kpt_count:
                        return False
                
                # 验证数值有效性
                try:
                    int(parts[0])  # class_id
                    for val in parts[1:5]:  # bbox
                        coord = float(val)
                        if coord < 0 or coord > 1:
                            return False
                    
                    # 验证关键点
                    for i in range(5, len(parts), 3):
                        kp_x = float(parts[i])
                        kp_y = float(parts[i+1])
                        
                        # visibility可以是整数0/1/2或浮点数（会被转换）
                        try:
                            kp_v_raw = float(parts[i+2])
                            # 检查是否为标准格式（整数0/1/2）
                            if kp_v_raw.is_integer():
                                kp_v = int(kp_v_raw)
                                if kp_v not in [0, 1, 2]:
                                    return False
                            # 如果是浮点数（如0.967），也接受（但会在后续提示用户修复）
                            elif 0 <= kp_v_raw <= 1:
                                pass  # 接受浮点数形式的confidence
                            else:
                                return False
                        except ValueError:
                            return False
                        
                        if kp_x < 0 or kp_x > 1 or kp_y < 0 or kp_y > 1:
                            return False
                except (ValueError, IndexError):
                    return False
        return True
    except Exception:
        return False


@app.command("split")
def split_dataset(
    images_dir: Optional[str] = typer.Option(None, "--images", "-i", help="图像目录 (检测/分割任务)"),
    labels_dir: Optional[str] = typer.Option(None, "--labels", "-l", help="标签目录 (检测/分割任务)"),
    source_dir: Optional[str] = typer.Option(None, "--source", "-s", help="源目录 (分类任务，已按类别组织)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    ratios: Optional[str] = typer.Option(None, "--ratios", "-r", help="划分比例 (train:val:test，如: 0.7:0.2:0.1)"),
    counts: Optional[str] = typer.Option(None, "--counts", "-c", help="划分样本数 (train:val:test，如: 100:30:10)"),
    seed: int = typer.Option(42, "--seed", help="随机种子"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
    create_empty_labels: bool = typer.Option(False, "--create-empty-labels/--no-empty-labels", help="为缺失标签的图片创建空标签（负样本）"),
    deduplicate: bool = typer.Option(False, "--deduplicate", "-d", help="拆分前去除重复图片（推荐，避免数据泄露）"),
    dedup_mode: str = typer.Option("exact", "--dedup-mode", help="去重模式: exact=完全相同(快), similar=相似图片(慢)"),
    similarity_threshold: int = typer.Option(8, "--similarity-threshold", help="相似度阈值(0-64, 仅similar模式), 越小越严格, 推荐:5-10"),
):
    """划分数据集为训练集、验证集、测试集
    
    支持两种使用方式:
    1. 检测/分割任务: --images 和 --labels
    2. 分类任务: --source
    
    支持两种划分方式（二选一）:
    1. 按比例划分: --ratios 0.7:0.2:0.1 (默认)
    2. 按样本数划分: --counts 100:30:10
    
    对于检测任务，可以使用 --create-empty-labels 将无标签图片作为负样本
    
    示例:
    \b
      # 按比例划分（传统方式）
      yolo-cli data split --images data/raw/images --labels data/raw/labels --ratios 0.7:0.2:0.1
      
    \b
      # 按样本数划分（新功能）
      yolo-cli data split --images data/raw/images --labels data/raw/labels --counts 100:30:10
      
    \b
      # 分类任务按样本数划分
      yolo-cli data split --source data/raw/images --task classify --counts 200:50:20
    """
    
    print_section_header("数据集划分")
    
    # 验证任务类型
    task = validate_task_type(task)
    print_info(f"任务类型: {task}")
    
    # 验证划分参数：ratios 和 counts 只能选一个
    if ratios and counts:
        print_error("--ratios 和 --counts 不能同时使用，请选择其中一种划分方式")
        raise typer.Exit(1)
    
    # 如果都没有指定，使用默认比例
    if not ratios and not counts:
        ratios = "0.7:0.2:0.1"
        print_info("使用默认划分比例: 0.7:0.2:0.1")
    
    # 确定划分方式
    split_mode = "counts" if counts else "ratios"
    split_param = counts if counts else ratios
    
    # 根据任务类型验证参数
    if task == 'classify':
        if not source_dir:
            print_error("分类任务需要指定 --source 参数")
            raise typer.Exit(1)
        if images_dir or labels_dir:
            print_warning("分类任务不需要 --images 和 --labels 参数，将被忽略")
        if create_empty_labels:
            print_warning("分类任务不支持 --create-empty-labels 参数，将被忽略")
        return _split_classify_dataset(source_dir, output_dir, split_param, seed, split_mode, deduplicate, dedup_mode, similarity_threshold)
    else:
        if not images_dir or not labels_dir:
            print_error("检测/分割任务需要指定 --images 和 --labels 参数")
            raise typer.Exit(1)
        if source_dir:
            print_warning("检测/分割任务不需要 --source 参数，将被忽略")
        return _split_detect_segment_dataset(images_dir, labels_dir, output_dir, split_param, seed, task, create_empty_labels, split_mode, deduplicate, dedup_mode, similarity_threshold)


def _split_detect_segment_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: Optional[str],
    split_param: str,
    seed: int,
    task: str,
    create_empty_labels: bool = False,
    split_mode: str = "ratios",
    deduplicate: bool = False,
    dedup_mode: str = "exact",
    similarity_threshold: int = 8,
):
    """检测/分割任务的数据集划分
    
    Args:
        split_mode: "ratios" 按比例划分, "counts" 按样本数划分
        deduplicate: 是否在拆分前去重
        dedup_mode: 去重模式 ("exact" 或 "similar")
        similarity_threshold: 相似度阈值（仅 similar 模式）
    """
    
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
    
    # 解析划分参数
    train_count = val_count = test_count = None
    train_ratio = val_ratio = test_ratio = None
    
    if split_mode == "ratios":
        # 按比例划分
        try:
            train_ratio, val_ratio, test_ratio = parse_ratio_string(split_param, 3)
            print_info(f"划分方式: 按比例")
            print_info(f"划分比例: 训练={train_ratio:.1%}, 验证={val_ratio:.1%}, 测试={test_ratio:.1%}")
        except ValueError as e:
            print_error(f"比例格式错误: {e}")
            raise typer.Exit(1)
    else:
        # 按样本数划分
        try:
            parts = split_param.split(':')
            if len(parts) != 3:
                raise ValueError("必须提供3个数值（train:val:test）")
            train_count = int(parts[0])
            val_count = int(parts[1])
            test_count = int(parts[2])
            if train_count < 0 or val_count < 0 or test_count < 0:
                raise ValueError("样本数不能为负数")
            print_info(f"划分方式: 按样本数")
            print_info(f"目标样本数: 训练={train_count}, 验证={val_count}, 测试={test_count}")
        except ValueError as e:
            print_error(f"样本数格式错误: {e}")
            raise typer.Exit(1)
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        ensure_dir(output_path / 'images' / split)
        ensure_dir(output_path / 'labels' / split)
    
    # 数据去重（如果启用）
    removed_images = set()
    if deduplicate:
        print_section_header("数据去重")
        
        # 显示去重模式
        mode_desc = "完全相同检测 (MD5)" if dedup_mode == "exact" else f"相似图片检测 (感知哈希, 阈值={similarity_threshold})"
        print_info(f"去重模式: {mode_desc}")
        
        try:
            deduplicator = ImageDeduplicator(mode=dedup_mode, similarity_threshold=similarity_threshold)
        except ImportError as e:
            print_error(str(e))
            print_warning("将跳过去重步骤")
            console.print()
            deduplicator = None
        
        if deduplicator:
            # 收集所有图像文件
            print_info("收集图像文件...")
            image_files = list(find_files(images_path, ['.jpg', '.jpeg', '.png']))
            original_count = len(image_files)
            print_info(f"找到 {original_count} 张图片")
            
            # 查找重复
            if dedup_mode == "exact":
                print_info("扫描完全相同的图片...")
            else:
                print_info(f"扫描相似图片（阈值≤{similarity_threshold}）...")
            
            duplicates_map = deduplicator.find_duplicates(image_files)
            
            if duplicates_map:
                # 统计
                total_duplicates = sum(len(files) - 1 for files in duplicates_map.values())
                dup_type = "重复" if dedup_mode == "exact" else "相似"
                print_warning(f"发现 {len(duplicates_map)} 组{dup_type}图片，共 {total_duplicates} 个文件")
                
                # 删除重复
                removed_files = deduplicator.remove_duplicates(
                    duplicates_map, 
                    labels_dir=labels_path
                )
                removed_images = set(removed_files)
                
                # 生成报告
                report_path = output_path / 'deduplication_report.json'
                deduplicator.generate_report(duplicates_map, report_path, original_count)
                
                stats = deduplicator.get_statistics()
                print_success(f"✓ 已删除 {stats['removed_count']} 个{dup_type}文件")
                print_info(f"  节省空间: {stats['space_saved_mb']} MB")
                print_info(f"  去重报告: {report_path}")
                print_info(f"  去重后剩余: {original_count - stats['removed_count']} 张图片")
            else:
                dup_type = "重复" if dedup_mode == "exact" else "相似"
                print_success(f"✓ 未发现{dup_type}图片")
            
            console.print()
    
    # 收集图像-标签对
    print_info("扫描图像和标签文件...")
    pairs = []
    missing_labels = []
    created_labels = []
    
    for img_file in find_files(images_path, ['.jpg', '.jpeg', '.png']):
        # 跳过已删除的重复图片
        if img_file in removed_images:
            continue
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            # 验证标签格式
            if task == 'segment':
                if _validate_segment_label(label_file):
                    pairs.append((img_file, label_file, False))  # False = 非负样本
                else:
                    print_warning(f"分割标签格式无效: {img_file.name}")
            elif task == 'pose':
                if _validate_pose_label(label_file):
                    pairs.append((img_file, label_file, False))  # False = 非负样本
                else:
                    # 提供详细的错误信息
                    try:
                        with open(label_file, 'r') as f:
                            line = f.readline().strip()
                            parts = line.split() if line else []
                            if len(parts) < 5:
                                print_warning(f"Pose标签格式无效: {img_file.name} (缺少边界框数据，需要至少5个字段)")
                            elif len(parts) < 8:
                                print_warning(f"Pose标签格式无效: {img_file.name} (缺少关键点数据，只有 {len(parts)} 个字段，至少需要8个)")
                            elif (len(parts) - 5) % 3 != 0:
                                kpt_values = len(parts) - 5
                                print_warning(f"Pose标签格式无效: {img_file.name} (关键点数据不完整，有 {kpt_values} 个值，应为3的倍数)")
                            else:
                                print_warning(f"Pose标签格式无效: {img_file.name} (坐标值超出范围或其他格式错误)")
                    except:
                        print_warning(f"Pose标签格式无效: {img_file.name}")
            else:
                pairs.append((img_file, label_file, False))  # False = 非负样本
        else:
            if create_empty_labels:
                # 创建空标签文件（负样本）
                try:
                    label_file.parent.mkdir(parents=True, exist_ok=True)
                    label_file.touch()  # 创建空文件
                    pairs.append((img_file, label_file, True))  # True = 负样本
                    created_labels.append(img_file.name)
                except Exception as e:
                    print_warning(f"无法创建标签文件 {label_file.name}: {e}")
                    missing_labels.append(img_file.name)
            else:
                print_warning(f"标签文件缺失: {img_file.name}")
                missing_labels.append(img_file.name)
    
    if not pairs:
        print_error("未找到有效的图像-标签对")
        raise typer.Exit(1)
    
    # 统计正负样本
    positive_count = sum(1 for _, _, is_negative in pairs if not is_negative)
    negative_count = sum(1 for _, _, is_negative in pairs if is_negative)
    
    print_info(f"找到 {len(pairs)} 个有效样本")
    if positive_count > 0:
        print_info(f"  正样本（有标注）: {positive_count}")
    if negative_count > 0:
        print_success(f"  负样本（无标注）: {negative_count} - 已创建空标签文件")
    if missing_labels and not create_empty_labels:
        print_warning(f"  跳过的图片（缺失标签）: {len(missing_labels)}")
        print_info(f"    提示: 使用 --create-empty-labels 将这些图片作为负样本")
    
    # 打乱顺序
    random.shuffle(pairs)
    
    # 计算划分点
    total = len(pairs)
    
    if split_mode == "ratios":
        # 按比例划分
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
    else:
        # 按样本数划分
        total_requested = train_count + val_count + test_count
        if total_requested > total:
            print_warning(f"请求的总样本数({total_requested})大于可用样本数({total})")
            print_info(f"将自动调整为使用所有 {total} 个样本，保持比例不变")
            # 按比例缩放
            scale = total / total_requested
            train_end = int(train_count * scale)
            val_end = train_end + int(val_count * scale)
        else:
            train_end = train_count
            val_end = train_end + val_count
            if total_requested < total:
                print_info(f"总样本数({total}) > 请求样本数({total_requested}), 将随机抽取 {total_requested} 个样本")
                # 只使用前 total_requested 个样本
                pairs = pairs[:total_requested]
                total = total_requested
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    # 复制文件
    print_info("开始复制文件...")
    
    with create_progress_bar() as progress:
        task_id = progress.add_task("复制文件", total=total)
        
        stats = {}
        split_negative_counts = {}
        for split_name, split_pairs in splits.items():
            count = 0
            negative_count_split = 0
            for img_file, label_file, is_negative in split_pairs:
                # 复制图像
                dst_img = output_path / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # 复制标签
                dst_label = output_path / 'labels' / split_name / label_file.name
                shutil.copy2(label_file, dst_label)
                
                count += 1
                if is_negative:
                    negative_count_split += 1
                progress.advance(task_id)
            
            stats[split_name] = count
            split_negative_counts[split_name] = negative_count_split
    
    # 打印统计信息
    console.print()
    if negative_count > 0:
        columns = ["数据集", "样本数", "正样本", "负样本", "比例"]
        rows = [
            ["训练集", stats['train'], stats['train'] - split_negative_counts['train'], 
             split_negative_counts['train'], f"{stats['train']/total*100:.1f}%"],
            ["验证集", stats['val'], stats['val'] - split_negative_counts['val'], 
             split_negative_counts['val'], f"{stats['val']/total*100:.1f}%"],
            ["测试集", stats['test'], stats['test'] - split_negative_counts['test'], 
             split_negative_counts['test'], f"{stats['test']/total*100:.1f}%"],
            ["总计", total, positive_count, negative_count, "100.0%"],
        ]
    else:
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
        f.write(f"划分方式: {'按样本数' if split_mode == 'counts' else '按比例'}\n")
        if split_mode == "counts":
            f.write(f"目标样本数: 训练={train_count}, 验证={val_count}, 测试={test_count}\n")
        else:
            f.write(f"划分比例: 训练={train_ratio:.1%}, 验证={val_ratio:.1%}, 测试={test_ratio:.1%}\n")
        f.write(f"总样本数: {total}\n")
        if negative_count > 0:
            f.write(f"  正样本（有标注）: {positive_count}\n")
            f.write(f"  负样本（无标注）: {negative_count}\n")
        f.write(f"训练集: {stats['train']} ({stats['train']/total*100:.1f}%)\n")
        if negative_count > 0:
            f.write(f"  正样本: {stats['train'] - split_negative_counts['train']}\n")
            f.write(f"  负样本: {split_negative_counts['train']}\n")
        f.write(f"验证集: {stats['val']} ({stats['val']/total*100:.1f}%)\n")
        if negative_count > 0:
            f.write(f"  正样本: {stats['val'] - split_negative_counts['val']}\n")
            f.write(f"  负样本: {split_negative_counts['val']}\n")
        f.write(f"测试集: {stats['test']} ({stats['test']/total*100:.1f}%)\n")
        if negative_count > 0:
            f.write(f"  正样本: {stats['test'] - split_negative_counts['test']}\n")
            f.write(f"  负样本: {split_negative_counts['test']}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"创建空标签: {'是' if create_empty_labels else '否'}\n")
    
    # 自动生成/补全 dataset.yaml
    _auto_generate_dataset_yaml_for_split(
        source_dir=images_path.parent,
        output_dir=output_path,
        task=task
    )
    
    print_success(f"数据集划分完成！输出目录: {output_path}")


def _split_classify_dataset(
    source_dir: str,
    output_dir: Optional[str],
    split_param: str,
    seed: int,
    split_mode: str = "ratios",
    deduplicate: bool = False,
    dedup_mode: str = "exact",
    similarity_threshold: int = 8,
):
    """分类任务的数据集划分
    
    Args:
        split_mode: "ratios" 按比例划分, "counts" 按样本数划分
        deduplicate: 是否在拆分前去重
        dedup_mode: 去重模式 ("exact" 或 "similar")
        similarity_threshold: 相似度阈值（仅 similar 模式）
    """
    
    source_path = Path(source_dir)
    
    # 验证源目录
    if not source_path.exists():
        print_error(f"源目录不存在: {source_dir}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True)
    else:
        output_path = Path(output_dir)
    
    print_info(f"源目录: {source_path}")
    print_info(f"输出目录: {output_path}")
    
    # 获取所有类别
    classes = [d.name for d in source_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    classes.sort()
    
    if not classes:
        print_error("源目录中没有找到类别子目录")
        raise typer.Exit(1)
    
    print_info(f"类别数量: {len(classes)}")
    print_info(f"类别: {', '.join(classes)}")
    
    # 解析划分参数
    train_count = val_count = test_count = None
    train_ratio = val_ratio = test_ratio = None
    
    if split_mode == "ratios":
        # 按比例划分
        try:
            train_ratio, val_ratio, test_ratio = parse_ratio_string(split_param, 3)
            print_info(f"划分方式: 按比例")
            print_info(f"划分比例: 训练={train_ratio:.1%}, 验证={val_ratio:.1%}, 测试={test_ratio:.1%}")
        except ValueError as e:
            print_error(f"比例格式错误: {e}")
            raise typer.Exit(1)
    else:
        # 按样本数划分（针对整个数据集）
        try:
            parts = split_param.split(':')
            if len(parts) != 3:
                raise ValueError("必须提供3个数值（train:val:test）")
            train_count = int(parts[0])
            val_count = int(parts[1])
            test_count = int(parts[2])
            if train_count < 0 or val_count < 0 or test_count < 0:
                raise ValueError("样本数不能为负数")
            print_info(f"划分方式: 按样本数（总样本数）")
            print_info(f"目标样本数: 训练={train_count}, 验证={val_count}, 测试={test_count}")
        except ValueError as e:
            print_error(f"样本数格式错误: {e}")
            raise typer.Exit(1)
    
    # 设置随机种子
    random.seed(seed)
    
    # 数据去重（如果启用）
    removed_images = set()
    if deduplicate:
        print_section_header("数据去重")
        
        # 显示去重模式
        mode_desc = "完全相同检测 (MD5)" if dedup_mode == "exact" else f"相似图片检测 (感知哈希, 阈值={similarity_threshold})"
        print_info(f"去重模式: {mode_desc}")
        
        try:
            deduplicator = ImageDeduplicator(mode=dedup_mode, similarity_threshold=similarity_threshold)
        except ImportError as e:
            print_error(str(e))
            print_warning("将跳过去重步骤")
            console.print()
            deduplicator = None
        
        if deduplicator:
            # 收集所有图像文件
            print_info("收集图像文件...")
            all_image_files = []
            for class_name in classes:
                class_dir = source_path / class_name
                images = list(class_dir.glob('*'))
                images = [img for img in images if img.is_file() and not img.name.startswith('.')]
                all_image_files.extend(images)
            
            original_count = len(all_image_files)
            print_info(f"找到 {original_count} 张图片")
            
            # 查找重复
            if dedup_mode == "exact":
                print_info("扫描完全相同的图片...")
            else:
                print_info(f"扫描相似图片（阈值≤{similarity_threshold}）...")
            
            duplicates_map = deduplicator.find_duplicates(all_image_files)
            
            if duplicates_map:
                # 统计
                total_duplicates = sum(len(files) - 1 for files in duplicates_map.values())
                dup_type = "重复" if dedup_mode == "exact" else "相似"
                print_warning(f"发现 {len(duplicates_map)} 组{dup_type}图片，共 {total_duplicates} 个文件")
                
                # 删除重复（分类任务没有单独的labels目录）
                removed_files = deduplicator.remove_duplicates(
                    duplicates_map, 
                    labels_dir=None
                )
                removed_images = set(removed_files)
                
                # 生成报告
                report_path = output_path / 'deduplication_report.json'
                deduplicator.generate_report(duplicates_map, report_path, original_count)
                
                stats_dedup = deduplicator.get_statistics()
                print_success(f"✓ 已删除 {stats_dedup['removed_count']} 个{dup_type}文件")
                print_info(f"  节省空间: {stats_dedup['space_saved_mb']} MB")
                print_info(f"  去重报告: {report_path}")
                print_info(f"  去重后剩余: {original_count - stats_dedup['removed_count']} 张图片")
            else:
                dup_type = "重复" if dedup_mode == "exact" else "相似"
                print_success(f"✓ 未发现{dup_type}图片")
            
            console.print()
    
    # 创建输出目录结构 - 统一使用 images/ 目录
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            ensure_dir(output_path / 'images' / split / class_name)
    
    # 统计
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    
    print_info("开始划分数据...")
    
    # 如果是按样本数划分，需要先收集所有图片并计算总数
    if split_mode == "counts":
        # 收集所有图片（跳过去重时删除的）
        all_images = []
        for class_name in classes:
            class_dir = source_path / class_name
            images = list(class_dir.glob('*'))
            images = [img for img in images if img.is_file() and not img.name.startswith('.')]
            for img in images:
                if img not in removed_images:  # 跳过已删除的重复图片
                    all_images.append((img, class_name))
        
        total_available = len(all_images)
        total_requested = train_count + val_count + test_count
        
        print_info(f"总可用样本数: {total_available}")
        
        if total_requested > total_available:
            print_warning(f"请求的总样本数({total_requested})大于可用样本数({total_available})")
            print_info(f"将自动调整为使用所有 {total_available} 个样本，保持比例不变")
            # 按比例缩放
            scale = total_available / total_requested
            actual_train = int(train_count * scale)
            actual_val = int(val_count * scale)
            actual_test = total_available - actual_train - actual_val
        else:
            actual_train = train_count
            actual_val = val_count
            actual_test = test_count
            if total_requested < total_available:
                print_info(f"将从 {total_available} 个样本中随机抽取 {total_requested} 个")
        
        # 打乱所有图片
        random.shuffle(all_images)
        
        # 按样本数划分
        train_images = all_images[:actual_train]
        val_images = all_images[actual_train:actual_train + actual_val]
        test_images = all_images[actual_train + actual_val:actual_train + actual_val + actual_test]
        
        # 复制文件
        for img_path, class_name in train_images:
            dst_path = output_path / 'images' / 'train' / class_name / img_path.name
            shutil.copy2(img_path, dst_path)
            stats[class_name]['train'] += 1
            total_stats['train'] += 1
        
        for img_path, class_name in val_images:
            dst_path = output_path / 'images' / 'val' / class_name / img_path.name
            shutil.copy2(img_path, dst_path)
            stats[class_name]['val'] += 1
            total_stats['val'] += 1
        
        for img_path, class_name in test_images:
            dst_path = output_path / 'images' / 'test' / class_name / img_path.name
            shutil.copy2(img_path, dst_path)
            stats[class_name]['test'] += 1
            total_stats['test'] += 1
    
    else:
        # 按比例划分（原有逻辑）
        # 处理每个类别
        for class_name in classes:
            class_dir = source_path / class_name
            images = list(class_dir.glob('*'))
            images = [img for img in images if img.is_file() and not img.name.startswith('.')]
            # 跳过去重时删除的图片
            images = [img for img in images if img not in removed_images]
            
            # 打乱顺序
            random.shuffle(images)
            
            # 计算划分点
            total = len(images)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            # 划分数据
            splits_data = {
                'train': images[:train_end],
                'val': images[train_end:val_end],
                'test': images[val_end:]
            }
            
            # 复制文件
            for split_name, split_images in splits_data.items():
                for img_path in split_images:
                    dst_path = output_path / 'images' / split_name / class_name / img_path.name
                    shutil.copy2(img_path, dst_path)
                    stats[class_name][split_name] += 1
                    total_stats[split_name] += 1
    
    # 打印统计信息
    console.print()
    columns = ["数据集", "样本数", "比例"]
    total_samples = sum(total_stats.values())
    rows = [
        ["训练集", total_stats['train'], f"{total_stats['train']/total_samples*100:.1f}%"],
        ["验证集", total_stats['val'], f"{total_stats['val']/total_samples*100:.1f}%"],
        ["测试集", total_stats['test'], f"{total_stats['test']/total_samples*100:.1f}%"],
        ["总计", total_samples, "100.0%"],
    ]
    print_table("数据集划分结果", columns, rows, show_lines=True)
    
    # 显示每个类别的统计
    console.print()
    print_info("各类别分布:")
    for class_name in classes:
        total_class = sum(stats[class_name].values())
        print_info(f"  {class_name}: train={stats[class_name]['train']}, val={stats[class_name]['val']}, test={stats[class_name]['test']} (总计{total_class})")
    
    # 保存类别文件
    classes_file = output_path / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    # 保存统计信息
    stats_file = output_path / 'split_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("分类数据集划分统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"划分方式: {'按样本数' if split_mode == 'counts' else '按比例'}\n")
        if split_mode == "counts":
            f.write(f"目标样本数: 训练={train_count}, 验证={val_count}, 测试={test_count}\n")
        else:
            f.write(f"划分比例: 训练={train_ratio:.1%}, 验证={val_ratio:.1%}, 测试={test_ratio:.1%}\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"类别数量: {len(classes)}\n")
        f.write(f"训练集: {total_stats['train']} ({total_stats['train']/total_samples*100:.1f}%)\n")
        f.write(f"验证集: {total_stats['val']} ({total_stats['val']/total_samples*100:.1f}%)\n")
        f.write(f"测试集: {total_stats['test']} ({total_stats['test']/total_samples*100:.1f}%)\n")
        f.write(f"\n各类别详细统计:\n")
        for class_name in classes:
            total_class = sum(stats[class_name].values())
            f.write(f"  {class_name}: train={stats[class_name]['train']}, val={stats[class_name]['val']}, test={stats[class_name]['test']} (总计{total_class})\n")
        f.write(f"随机种子: {seed}\n")
    
    print_success(f"✓ 类别列表已保存: {classes_file}")
    print_success(f"✓ 统计信息已保存: {stats_file}")
    
    # 自动生成/补全 dataset.yaml
    _auto_generate_dataset_yaml_for_split(
        source_dir=source_path,
        output_dir=output_path,
        task='classify'
    )
    
    print_success(f"✓ 分类数据集划分完成！输出目录: {output_path}")


def _auto_generate_dataset_yaml_for_split(source_dir: Path, output_dir: Path, task: str):
    """数据拆分后自动生成/补全 dataset.yaml
    
    Args:
        source_dir: 源数据目录（可能包含 dataset.yaml）
        output_dir: 输出目录（拆分后的数据集目录）
        task: 任务类型
    """
    print_info("\n自动生成 dataset.yaml...")
    
    # 读取标签信息的优先级：
    # 1. 源目录的 dataset.yaml
    # 2. classes.txt
    
    label_config = {}
    source_found = False
    
    # 优先级1: 源目录的 dataset.yaml
    source_yaml = source_dir / 'dataset.yaml'
    if source_yaml.exists():
        try:
            with open(source_yaml, 'r', encoding='utf-8') as f:
                source_data = yaml.safe_load(f)
            
            # 提取标签信息
            if source_data:
                if 'nc' in source_data:
                    label_config['nc'] = source_data['nc']
                if 'names' in source_data:
                    label_config['names'] = source_data['names']
                if 'kpt_shape' in source_data:
                    label_config['kpt_shape'] = source_data['kpt_shape']
                if 'keypoint_names' in source_data:
                    label_config['keypoint_names'] = source_data['keypoint_names']
                if 'flip_idx' in source_data:
                    label_config['flip_idx'] = source_data['flip_idx']
                
                if label_config:
                    print_info(f"✓ 从源目录 dataset.yaml 读取标签信息")
                    source_found = True
        except Exception as e:
            print_warning(f"读取源 dataset.yaml 失败: {e}")
    
    # 优先级2: 尝试从 classes.txt 读取
    if not source_found:
        classes_file = source_dir / 'classes.txt'
        if not classes_file.exists():
            classes_file = output_dir / 'classes.txt'
        
        if classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                label_config['nc'] = len(classes)
                label_config['names'] = {i: name for i, name in enumerate(classes)}
                print_info(f"✓ 从 classes.txt 读取类别信息")
                source_found = True
            except Exception as e:
                print_warning(f"读取 classes.txt 失败: {e}")
    
    if not source_found or not label_config:
        print_warning("未找到标签信息，跳过生成 dataset.yaml")
        return
    
    # 补全路径信息
    yaml_config = {
        'path': str(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
    }
    yaml_config.update(label_config)
    
    # 保存完整的 dataset.yaml
    dataset_yaml_path = output_dir / 'dataset.yaml'
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO Dataset Configuration (Complete)\n")
        f.write("# 此文件在数据拆分后自动生成，包含完整的路径和标签信息\n\n")
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"✓ dataset.yaml 已生成: {dataset_yaml_path}")
    
    # 显示关键信息
    print_info(f"  类别数: {yaml_config.get('nc', 'N/A')}")
    if task == 'pose' and 'keypoint_names' in yaml_config:
        print_info(f"  关键点: {yaml_config['keypoint_names']}")


@app.command("generate-yaml")
def generate_yaml(
    data_path: Optional[str] = typer.Option(None, "--path", "-p", help="数据集路径"),
    classes_file: Optional[str] = typer.Option(None, "--classes", "-c", help="类别文件路径"),
    output: str = typer.Option("data/processed/dataset.yaml", "--output", "-o", help="输出文件路径"),
    train_dir: Optional[str] = typer.Option(None, "--train", help="训练集目录 (默认根据任务类型自动设置)"),
    val_dir: Optional[str] = typer.Option(None, "--val", help="验证集目录 (默认根据任务类型自动设置)"),
    test_dir: Optional[str] = typer.Option(None, "--test", help="测试集目录 (默认根据任务类型自动设置)"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
):
    """生成dataset.yaml配置文件"""
    
    print_section_header("生成 dataset.yaml")
    
    # 验证任务类型
    task = validate_task_type(task)
    print_info(f"任务类型: {task}")
    
    # 根据任务类型设置默认目录（统一使用 images/ 目录）
    # 检查是否为 None 或未传递（typer.Option 对象）
    if train_dir is None or not isinstance(train_dir, str):
        train_dir = 'images/train'
    if val_dir is None or not isinstance(val_dir, str):
        val_dir = 'images/val'
    if test_dir is None or not isinstance(test_dir, str):
        test_dir = 'images/test'
    
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
    
    # Pose任务需要额外的配置
    if task == 'pose':
        # 优先级1: 从已有的 dataset.yaml 读取标签信息
        existing_yaml = data_path / 'dataset.yaml'
        existing_label_config = None
        if existing_yaml.exists() and existing_yaml != Path(output):
            try:
                with open(existing_yaml, 'r', encoding='utf-8') as f:
                    existing_data = yaml.safe_load(f)
                if existing_data and 'kpt_shape' in existing_data:
                    existing_label_config = {
                        'kpt_shape': existing_data.get('kpt_shape'),
                        'keypoint_names': existing_data.get('keypoint_names'),
                        'flip_idx': existing_data.get('flip_idx'),
                    }
                    yaml_config.update(existing_label_config)
                    print_info(f"✓ 从已有 dataset.yaml 读取 Pose 配置")
                    print_info(f"  关键点名称: {existing_label_config.get('keypoint_names')}")
            except Exception as e:
                print_warning(f"读取已有 dataset.yaml 失败: {e}")
        
        # 优先级2: 从标签文件中检测关键点数量
        if not existing_label_config:
            kpt_count = 17  # 默认
            detected_kpt_count = None
            
            # 多种路径查找策略，确保能找到标签文件
            label_files = []
            possible_label_paths = [
                data_path / 'labels' / 'train',  # 相对路径: data_path/labels/train
                data_path / 'labels',             # 相对路径: data_path/labels
                train_path.parent.parent / 'labels' / 'train',  # 从 images/train 推断
                train_path.parent.parent / 'labels',            # 从 images 推断
            ]
            
            for label_path in possible_label_paths:
                if label_path.exists() and label_path.is_dir():
                    found_files = list(label_path.glob('*.txt'))
                    if found_files:
                        label_files = found_files
                        print_info(f"找到标签文件目录: {label_path}")
                        break
            
            if label_files:
                # 尝试多个文件，因为第一个可能是空的
                for label_file in label_files[:10]:  # 检查前10个文件
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                parts = line.split()
                            # YOLO Pose 格式: class_id x y w h kp1_x kp1_y kp1_v ...
                            # 总列数 = 1 (class) + 4 (bbox) + N*3 (keypoints)
                            if len(parts) > 5:
                                kpt_data_count = len(parts) - 5
                                if kpt_data_count % 3 == 0:
                                    detected_kpt_count = kpt_data_count // 3
                                    kpt_count = detected_kpt_count
                                    print_info(f"从标签文件检测到 {kpt_count} 个关键点")
                                    break
                            
                            if detected_kpt_count:
                                break
                    except Exception as e:
                        continue
            else:
                print_warning("未找到标签文件，使用默认配置")
        
        yaml_config['kpt_shape'] = [kpt_count, 3]
        
        # 根据关键点数量设置 flip_idx 和关键点名称
        if kpt_count == 17:
            # COCO 17 关键点的对称索引
            yaml_config['flip_idx'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            yaml_config['keypoint_names'] = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
        elif kpt_count == 4:
            # 4个关键点（start, end, center, pointer）
            # 假设没有对称关系，使用原始顺序
            yaml_config['flip_idx'] = [0, 1, 2, 3]
            yaml_config['keypoint_names'] = ['start', 'end', 'center', 'pointer']
        else:
            # 其他数量，使用原始顺序
            yaml_config['flip_idx'] = list(range(kpt_count))
            yaml_config['keypoint_names'] = [f'kp_{i}' for i in range(kpt_count)]
        
        if detected_kpt_count:
            print_info(f"已添加 Pose 任务配置: kpt_shape=[{kpt_count}, 3] (检测到 {kpt_count} 个关键点)")
        else:
            print_info(f"已添加 Pose 任务配置: kpt_shape=[{kpt_count}, 3] (默认 COCO 17关键点)")
        
        # 显示关键点名称
        if 'keypoint_names' in yaml_config:
            print_info(f"关键点名称: {yaml_config['keypoint_names']}")
    
    # 保存YAML文件
    output_path = Path(output)
    ensure_dir(output_path.parent)
    
    # 调试：在保存前检查 yaml_config
    import sys
    if task == 'pose' and 'keypoint_names' not in yaml_config:
        print_warning("警告: yaml_config 中缺少 keypoint_names 字段！", file=sys.stderr)
    
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
    if task == 'pose':
        print_key_value("kpt_shape", str(yaml_config['kpt_shape']))
        print_key_value("keypoint_names", ", ".join(yaml_config['keypoint_names']))
        print_key_value("flip_idx", "已配置")


@app.command("verify")
def verify_dataset(
    data_path: Optional[str] = typer.Option(None, "--path", "-p", help="数据集路径"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
):
    """验证数据集完整性"""
    
    print_section_header("数据集验证")
    
    # 验证任务类型
    task = validate_task_type(task)
    print_info(f"任务类型: {task}")
    
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
    
    # 获取并打印数据集信息
    if task == 'classify':
        # 分类任务：统计按类别组织的图像（统一使用 images/ 目录）
        info = {}
        for split in ['train', 'val', 'test']:
            split_dir = data_path / 'images' / split
            count = 0
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith('.'):
                        count += len(list(find_files(class_dir)))
            info[f'{split}_images'] = count
            info[f'{split}_labels'] = count  # 分类任务图像=标签
    else:
        # 检测/分割任务：从 images/ 和 labels/ 目录统计
        info = get_dataset_info(data_path)
    
    # 打印统计信息
    print_dataset_info(info)
    
    # 验证图像-标签对应关系
    issues = []
    
    if task == 'classify':
        # 分类任务验证（统一使用 images/ 目录）
        all_classes = set()
        split_classes = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = data_path / 'images' / split
            
            if not split_dir.exists():
                issues.append(f"缺少 images/{split} 目录")
                continue
            
            # 获取类别目录
            classes = [d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            split_classes[split] = set(classes)
            all_classes.update(classes)
            
            if not classes:
                issues.append(f"images/{split}: 没有找到类别子目录")
                continue
            
            # 验证每个类别是否有图像
            for class_name in classes:
                class_dir = split_dir / class_name
                images = list(find_files(class_dir))
                if not images:
                    issues.append(f"images/{split}/{class_name}: 类别目录为空")
        
        # 检查类别一致性
        if len(split_classes) > 1:
            reference_classes = split_classes.get('train', set())
            for split, classes in split_classes.items():
                if split == 'train':
                    continue
                missing = reference_classes - classes
                extra = classes - reference_classes
                if missing:
                    issues.append(f"{split}: 缺少类别 {missing}")
                if extra:
                    issues.append(f"{split}: 多余类别 {extra}")
        
        # 显示类别信息
        if all_classes:
            console.print()
            print_info(f"检测到 {len(all_classes)} 个类别:")
            for cls in sorted(all_classes):
                print_info(f"  • {cls}")
    
    else:
        # 检测/分割任务验证
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
                        # 根据任务类型验证标签格式
                        if task == 'segment':
                            if not _validate_segment_label(label_file):
                                issues.append(f"{split}: 分割标签格式错误 - {label_file.name}")
                        elif task == 'detect':
                            if not _validate_detect_label(label_file):
                                issues.append(f"{split}: 检测标签格式错误 - {label_file.name}")
                        elif task == 'pose':
                            # 尝试从 dataset.yaml 读取 kpt_shape
                            kpt_count = None
                            dataset_yaml = data_path / 'dataset.yaml'
                            if dataset_yaml.exists():
                                with open(dataset_yaml, 'r', encoding='utf-8') as f:
                                    yaml_data = yaml.safe_load(f)
                                    kpt_shape = yaml_data.get('kpt_shape')
                                    if kpt_shape:
                                        kpt_count = kpt_shape[0]
                            
                            if not _validate_pose_label(label_file, kpt_count):
                                issues.append(f"{split}: Pose标签格式错误 - {label_file.name}")
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



@app.command("prepare-classify")
def prepare_classify(
    images_dir: str = typer.Option(..., "--images", "-i", help="图像目录"),
    labels_dir: str = typer.Option(..., "--labels", "-l", help="标签目录"),
    classes_file: str = typer.Option(..., "--classes", "-c", help="类别文件路径"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    ratios: str = typer.Option("0.7:0.2:0.1", "--ratios", "-r", help="划分比例 (train:val:test)"),
    seed: int = typer.Option(42, "--seed", "-s", help="随机种子"),
):
    """为分类任务准备数据集（从标签文件组织为目录结构）"""
    
    print_section_header("准备分类数据集")
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    classes_path = Path(classes_file)
    
    # 验证输入
    if not images_path.exists():
        print_error(f"图像目录不存在: {images_dir}")
        raise typer.Exit(1)
    
    if not labels_path.exists():
        print_error(f"标签目录不存在: {labels_dir}")
        raise typer.Exit(1)
    
    if not classes_path.exists():
        print_error(f"类别文件不存在: {classes_file}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True) / 'classify'
    else:
        output_path = Path(output_dir)
    
    print_info(f"图像目录: {images_path}")
    print_info(f"标签目录: {labels_path}")
    print_info(f"类别文件: {classes_path}")
    print_info(f"输出目录: {output_path}")
    
    # 读取类别
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    print_info(f"类别数量: {len(classes)}")
    print_info(f"类别: {', '.join(classes)}")
    
    # 解析比例
    try:
        train_ratio, val_ratio, test_ratio = parse_ratio_string(ratios, 3)
        print_info(f"划分比例: 训练={train_ratio:.1%}, 验证={val_ratio:.1%}, 测试={test_ratio:.1%}")
    except ValueError as e:
        print_error(f"比例格式错误: {e}")
        raise typer.Exit(1)
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            ensure_dir(output_path / split / class_name)
    
    # 收集图像及其标签
    print_info("扫描图像和标签文件...")
    samples = []
    
    for img_file in find_files(images_path, ['.jpg', '.jpeg', '.png']):
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            try:
                # 读取标签文件（分类任务：每个文件只包含一个类别ID）
                with open(label_file, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line)
                        if 0 <= class_id < len(classes):
                            samples.append((img_file, class_id))
                        else:
                            print_warning(f"类别ID超出范围: {img_file.name} (class_id={class_id})")
            except Exception as e:
                print_warning(f"无法读取标签: {label_file.name} - {e}")
        else:
            print_warning(f"标签文件缺失: {img_file.name}")
    
    if not samples:
        print_error("未找到有效的图像-标签对")
        raise typer.Exit(1)
    
    print_info(f"找到 {len(samples)} 个有效样本")
    
    # 打乱顺序
    random.shuffle(samples)
    
    # 计算划分点
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits_data = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    # 复制文件到对应的类别目录
    print_info("开始组织文件...")
    
    with create_progress_bar() as progress:
        task = progress.add_task("组织文件", total=total)
        
        stats = {split: defaultdict(int) for split in ['train', 'val', 'test']}
        
        for split_name, split_samples in splits_data.items():
            for img_file, class_id in split_samples:
                class_name = classes[class_id]
                
                # 复制图像到对应类别目录
                dst_img = output_path / split_name / class_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                stats[split_name][class_name] += 1
                progress.advance(task)
    
    # 打印统计信息
    console.print()
    for split_name in ['train', 'val', 'test']:
        split_stats = stats[split_name]
        total_split = sum(split_stats.values())
        
        if total_split > 0:
            console.print(f"\n[bold cyan]{split_name.upper()}集统计:[/bold cyan]")
            for class_name in classes:
                count = split_stats.get(class_name, 0)
                if count > 0:
                    print_info(f"  {class_name}: {count} 张")
            print_info(f"  总计: {total_split} 张")
    
    print_success(f"\n分类数据集准备完成！输出目录: {output_path}")
    print_info(f"数据集结构：")
    print_info(f"  {output_path}/")
    print_info(f"    ├── train/")
    for class_name in classes[:3]:  # 显示前3个类别
        print_info(f"    │   ├── {class_name}/")
    if len(classes) > 3:
        print_info(f"    │   └── ...")
    print_info(f"    ├── val/")
    print_info(f"    └── test/")


def _print_positive_negative_stats_classify(data_path: Path, positive_classes: List[str]):
    """
    统计并打印分类任务的正负样本数量
    
    Args:
        data_path: 数据集路径
        positive_classes: 正类列表
    """
    if not positive_classes:
        console.print()
        print_warning("⚠️  未选择正类，无法进行正负样本统计")
        print_info("提示: 在多选列表中，使用空格键选择一个或多个正类，然后按回车确认")
        print_info("或者在命令行模式中使用 --positive-classes 参数指定正类")
        console.print()
        return
    
    split_stats = {}
    class_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for split in ['train', 'val', 'test']:
        split_dir = data_path / 'images' / split
        
        if not split_dir.exists():
            continue
        
        positive_count = 0
        negative_count = 0
        
        # 遍历所有类别目录
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            # 统计该类别的图片数量
            images = list(find_files(class_dir))
            count = len(images)
            class_stats[class_name][split] = count
            
            # 判断是正类还是负类
            if class_name in positive_classes:
                positive_count += count
            else:
                negative_count += count
        
        total = positive_count + negative_count
        
        if total > 0:
            split_stats[split] = {
                'positive': positive_count,
                'negative': negative_count,
                'total': total,
            }
    
    # 打印统计表格
    if split_stats:
        console.print()
        
        # 显示正类信息
        print_info(f"正类定义: {', '.join(positive_classes)}")
        console.print()
        
        # 分集统计
        columns = ["数据集", "正类样本", "负类样本", "总样本", "正类比例"]
        rows = []
        
        total_positive = 0
        total_negative = 0
        total_samples = 0
        
        for split in ['train', 'val', 'test']:
            if split in split_stats:
                stats = split_stats[split]
                rows.append([
                    split.upper(),
                    stats['positive'],
                    stats['negative'],
                    stats['total'],
                    f"{stats['positive']/stats['total']*100:.1f}%"
                ])
                total_positive += stats['positive']
                total_negative += stats['negative']
                total_samples += stats['total']
        
        # 添加总计行
        if total_samples > 0:
            rows.append([
                "总计",
                total_positive,
                total_negative,
                total_samples,
                f"{total_positive/total_samples*100:.1f}%"
            ])
        
        print_table("正负样本分布", columns, rows, show_lines=True)
        
        # 打印详细的类别分布
        console.print()
        print_info("各类别详细分布:")
        
        # 按正负分组显示
        console.print("\n[bold cyan]正类:[/bold cyan]")
        for class_name in sorted(positive_classes):
            if class_name in class_stats:
                stats = class_stats[class_name]
                total_class = sum(stats.values())
                print_info(f"  {class_name}: train={stats['train']}, val={stats['val']}, test={stats['test']} (总计 {total_class})")
        
        console.print("\n[bold yellow]负类:[/bold yellow]")
        negative_classes = [c for c in class_stats.keys() if c not in positive_classes]
        for class_name in sorted(negative_classes):
            stats = class_stats[class_name]
            total_class = sum(stats.values())
            print_info(f"  {class_name}: train={stats['train']}, val={stats['val']}, test={stats['test']} (总计 {total_class})")
        
        # 样本说明
        console.print()
        print_info("样本说明:")
        print_info(f"  • 正类: {', '.join(positive_classes)}")
        print_info(f"  • 负类: 除正类外的所有类别")
    else:
        print_warning("未找到有效的数据集")


def _print_positive_negative_stats(data_path: Path):
    """
    统计并打印正负样本数量（检测/分割任务）
    
    正样本：标签文件存在且非空（包含至少一个标注）
    负样本：标签文件不存在或为空
    
    Args:
        data_path: 数据集路径
    """
    split_stats = {}
    
    for split in ['train', 'val', 'test']:
        img_dir = data_path / 'images' / split
        label_dir = data_path / 'labels' / split
        
        if not img_dir.exists():
            continue
        
        positive_count = 0
        negative_count = 0
        positive_with_objects = 0  # 有标注对象的图片数
        total_objects = 0  # 总标注对象数
        
        # 遍历所有图像
        for img_file in find_files(img_dir):
            label_file = label_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                # 检查标签文件是否非空
                try:
                    with open(label_file, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    
                    if lines:
                        # 标签文件存在且有内容 -> 正样本
                        positive_count += 1
                        positive_with_objects += 1
                        total_objects += len(lines)
                    else:
                        # 标签文件存在但为空 -> 负样本
                        negative_count += 1
                except Exception:
                    # 无法读取标签文件 -> 视为负样本
                    negative_count += 1
            else:
                # 标签文件不存在 -> 负样本
                negative_count += 1
        
        total = positive_count + negative_count
        
        if total > 0:
            split_stats[split] = {
                'positive': positive_count,
                'negative': negative_count,
                'total': total,
                'total_objects': total_objects,
                'avg_objects': total_objects / positive_with_objects if positive_with_objects > 0 else 0
            }
    
    # 打印统计表格
    if split_stats:
        console.print()
        
        # 分集统计
        columns = ["数据集", "正样本", "负样本", "总样本", "正样本比例"]
        rows = []
        
        total_positive = 0
        total_negative = 0
        total_samples = 0
        
        for split in ['train', 'val', 'test']:
            if split in split_stats:
                stats = split_stats[split]
                rows.append([
                    split.upper(),
                    stats['positive'],
                    stats['negative'],
                    stats['total'],
                    f"{stats['positive']/stats['total']*100:.1f}%"
                ])
                total_positive += stats['positive']
                total_negative += stats['negative']
                total_samples += stats['total']
        
        # 添加总计行
        if total_samples > 0:
            rows.append([
                "总计",
                total_positive,
                total_negative,
                total_samples,
                f"{total_positive/total_samples*100:.1f}%"
            ])
        
        print_table("正负样本分布", columns, rows, show_lines=True)
        
        # 打印详细信息
        console.print()
        print_info("样本说明:")
        print_info("  • 正样本: 包含标注对象的图像（标签文件存在且非空）")
        print_info("  • 负样本: 不包含标注对象的图像（标签文件不存在或为空）")
        
        # 打印每个集的平均标注数
        console.print()
        print_info("平均标注对象数:")
        for split in ['train', 'val', 'test']:
            if split in split_stats:
                stats = split_stats[split]
                print_info(f"  {split.upper()}: {stats['avg_objects']:.2f} 个对象/图像 (总计 {stats['total_objects']} 个对象)")
    else:
        print_warning("未找到有效的数据集")


@app.command("stats")
def dataset_stats(
    data_path: Optional[str] = typer.Option(None, "--path", "-p", help="数据集路径"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="显示详细统计"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
    positive_classes: Optional[str] = typer.Option(None, "--positive-classes", help="正类列表（逗号分隔，仅用于分类任务）"),
):
    """数据集统计分析
    
    示例:
      # 检测任务统计（含正负样本）
      python yolo_cli.py data stats --path data/processed --detailed --task detect
      
      # 分类任务统计（指定正类）
      python yolo_cli.py data stats --path data/processed --detailed --task classify --positive-classes "normal,good"
    """
    
    print_section_header("数据集统计")
    
    # 规范化参数（处理从代码直接调用的情况）
    # 当函数被其他代码直接调用时，typer 的默认值可能是 OptionInfo 对象
    from typer.models import OptionInfo
    if isinstance(data_path, OptionInfo):
        data_path = None
    if isinstance(detailed, OptionInfo):
        detailed = False
    if isinstance(task, OptionInfo):
        task = "detect"
    if isinstance(positive_classes, OptionInfo):
        positive_classes = None
    
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
    print_info(f"任务类型: {task}")
    
    # 检测任务类型（如果是分类，检查是否有分类结构）
    is_classify = task == 'classify'
    if not is_classify:
        # 自动检测：检查是否是分类数据集结构
        train_dir = data_path / 'images' / 'train'
        if train_dir.exists():
            subdirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            # 如果train目录下有子目录，且没有labels目录，可能是分类任务
            if subdirs and not (data_path / 'labels' / 'train').exists():
                is_classify = True
                print_info("检测到分类数据集结构")
    
    # 获取基本信息
    info = get_dataset_info(data_path)
    print_dataset_info(info)
    
    if detailed:
        # 统计正负样本
        if is_classify:
            # 分类任务：需要指定正类
            if positive_classes and isinstance(positive_classes, str):
                print_section_header("正负样本统计")
                positive_class_list = [c.strip() for c in positive_classes.split(',') if c.strip()]
                _print_positive_negative_stats_classify(data_path, positive_class_list)
        else:
            # 检测/分割任务
            print_section_header("正负样本统计")
            _print_positive_negative_stats(data_path)
        
        # 统计类别分布
        print_section_header("类别分布统计")
        
        if is_classify:
            # 分类任务：统计每个类别的图片数量
            for split in ['train', 'val', 'test']:
                split_dir = data_path / 'images' / split
                if not split_dir.exists():
                    continue
                
                class_counts = {}
                classes = [d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                
                for class_name in sorted(classes):
                    class_dir = split_dir / class_name
                    images = list(find_files(class_dir))
                    class_counts[class_name] = len(images)
                
                if class_counts:
                    console.print(f"\n[bold]{split.upper()} 集:[/bold]")
                    columns = ["类别", "图片数量", "比例"]
                    rows = []
                    total = sum(class_counts.values())
                    
                    for class_name in sorted(class_counts.keys()):
                        count = class_counts[class_name]
                        rows.append([
                            class_name,
                            count,
                            f"{count/total*100:.1f}%"
                        ])
                    
                    print_table(f"{split} 类别分布", columns, rows, show_lines=True)
                    print_info(f"总图片数: {total}, 类别数: {len(class_counts)}")
        else:
            # 检测/分割任务：统计边界框
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


@app.command("convert-labelstudio")
def convert_labelstudio(
    input_file: str = typer.Option(..., "--input", "-i", help="Label Studio导出文件 (JSON/CSV)"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    token: str = typer.Option(..., "--token", "-t", help="API访问令牌"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录 (默认: data/raw)"),
    task: str = typer.Option("detect", "--task", help="任务类型 (detect/classify)"),
    format_type: str = typer.Option("auto", "--format", "-f", help="输入格式 (auto/json/csv)"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip", help="跳过已下载的图片"),
    max_workers: int = typer.Option(4, "--max-workers", "-w", help="并发下载线程数"),
    include_negative: bool = typer.Option(True, "--include-negative/--no-negative", help="包含无标注图片作为负样本（检测任务）"),
    max_tasks: Optional[int] = typer.Option(None, "--max-tasks", "-m", help="限制下载的最大任务数（用于测试或部分下载）"),
    task_ids: Optional[str] = typer.Option(None, "--task-ids", help="指定要下载的任务ID列表（逗号分隔，如: 100,200,300）"),
    task_range: Optional[List[int]] = typer.Option(None, "--task-range", help="指定任务ID范围 (start end)"),
    filter_labels: Optional[str] = typer.Option(None, "--filter-labels", help="只下载包含指定标签的任务（逗号分隔，如: person,car）"),
):
    """从Label Studio导出数据转换为YOLO格式
    
    对于检测任务，无标注的图片将作为负样本被下载并创建空标签文件。
    负样本有助于减少误报，提高模型鲁棒性（推荐包含10-20%负样本）。
    
    部分数据集下载选项:
        --max-tasks: 限制下载数量（如：--max-tasks 100 只下载前100个任务）
        --task-ids: 指定任务ID（如：--task-ids 100,200,300）
        --task-range: 指定ID范围（如：--task-range 100 500）
        --filter-labels: 按标签筛选（如：--filter-labels person,car）
    
    示例:
        # 只下载前50个任务（快速测试）
        python yolo_cli.py data convert-labelstudio -i export.json --max-tasks 50
        
        # 下载特定任务
        python yolo_cli.py data convert-labelstudio -i export.json --task-ids 100,200,300
        
        # 下载ID范围的任务
        python yolo_cli.py data convert-labelstudio -i export.json --task-range 100 500
        
        # 只下载包含特定标签的任务
        python yolo_cli.py data convert-labelstudio -i export.json --filter-labels person,car
    """
    
    print_section_header("Label Studio 数据转换")
    
    # 验证任务类型
    task = validate_task_type(task)
    print_info(f"任务类型: {task}")
    
    # 验证输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        print_error(f"输入文件不存在: {input_file}")
        raise typer.Exit(1)
    
    print_info(f"输入文件: {input_path}")
    
    # 确定输出目录（默认为 data/raw）
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_raw', absolute=True)
    else:
        output_path = Path(output_dir)
    
    print_info(f"输出目录: {output_path}")
    
    # 初始化Label Studio客户端
    print_info(f"连接到 Label Studio: {url}")
    client = LabelStudioClient(url, token)
    
    # 测试连接
    success, message = client.test_connection()
    if not success:
        print_error(f"连接失败: {message}")
        raise typer.Exit(1)
    
    print_success(f"✓ {message}")
    
    # 检测文件格式
    if format_type == "auto":
        format_type = LabelStudioConverter.detect_format(input_path)
    
    print_info(f"文件格式: {format_type.upper()}")
    
    # 解析数据
    print_info("解析标注数据...")
    try:
        if format_type == "json":
            parsed_data = LabelStudioConverter.parse_json(input_path, include_negative=include_negative)
        else:
            parsed_data = LabelStudioConverter.parse_csv(input_path, include_negative=include_negative)
        
        if not parsed_data:
            print_error("未找到有效的标注数据")
            raise typer.Exit(1)
        
        print_success(f"✓ 解析完成：找到 {len(parsed_data)} 个任务")
        
        # 应用筛选条件
        original_count = len(parsed_data)
        
        # 1. 按任务ID列表筛选
        if task_ids:
            id_list = [int(tid.strip()) for tid in task_ids.split(',')]
            id_set = set(id_list)
            parsed_data = [item for item in parsed_data if item.get('task_id') in id_set]
            print_info(f"按任务ID筛选: {len(id_list)} 个指定ID，匹配到 {len(parsed_data)} 个任务")
        
        # 2. 按任务ID范围筛选
        elif task_range:
            if len(task_range) != 2:
                print_error("任务ID范围需要两个参数: --task-range <起始ID> <结束ID>")
                raise typer.Exit(1)
            start_id = task_range[0]
            end_id = task_range[1]
            if start_id > end_id:
                print_error(f"任务ID范围错误: 起始ID ({start_id}) 大于结束ID ({end_id})")
                raise typer.Exit(1)
            
            # 调试：显示任务ID范围
            task_ids_in_data = [item.get('task_id', 0) for item in parsed_data if item.get('task_id')]
            if task_ids_in_data:
                min_id = min(task_ids_in_data)
                max_id = max(task_ids_in_data)
                print_info(f"数据集中的任务ID范围: {min_id} - {max_id}")
                print_info(f"筛选ID范围: {start_id} - {end_id}")
            
            parsed_data = [item for item in parsed_data if start_id <= item.get('task_id', 0) <= end_id]
            print_info(f"按任务ID范围筛选: {start_id}-{end_id}，匹配到 {len(parsed_data)} 个任务")
        
        # 3. 按标签筛选
        if filter_labels:
            label_list = [label.strip() for label in filter_labels.split(',')]
            label_set = set(label_list)
            
            def has_matching_label(item):
                """检查任务是否包含指定的标签"""
                for ann in item.get('annotations', []):
                    item_labels = ann.get('labels', [])
                    if any(label in label_set for label in item_labels):
                        return True
                # 对于分类任务
                if item.get('category') in label_set:
                    return True
                return False
            
            parsed_data = [item for item in parsed_data if has_matching_label(item)]
            print_info(f"按标签筛选: {', '.join(label_list)}，匹配到 {len(parsed_data)} 个任务")
        
        # 4. 限制最大任务数
        if max_tasks and max_tasks < len(parsed_data):
            max_tasks_int = int(max_tasks)
            parsed_data = parsed_data[:max_tasks_int]
            print_info(f"限制任务数: 取前 {max_tasks_int} 个任务")
        
        if not parsed_data:
            print_warning("应用筛选条件后没有匹配的任务")
            print_info(f"原始任务数: {original_count}")
            raise typer.Exit(1)
        
        if original_count > len(parsed_data):
            print_success(f"✓ 筛选后: {len(parsed_data)}/{original_count} 个任务将被处理")
        
        # 统计正负样本
        positive_count = sum(1 for item in parsed_data if not item.get('is_negative', False))
        negative_count = sum(1 for item in parsed_data if item.get('is_negative', False))
        
        if task == 'detect':
            print_info(f"  正样本（有标注）: {positive_count}")
            if include_negative:
                print_info(f"  负样本（无标注）: {negative_count}")
                if negative_count > 0:
                    print_info(f"  负样本比例: {negative_count/len(parsed_data)*100:.1f}%")
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"解析失败: {str(e)}")
        raise typer.Exit(1)
    
    # 构建类别映射
    print_info("构建类别映射...")
    
    # 调试：检查数据结构
    if parsed_data:
        first_item = parsed_data[0]
        if first_item.get('annotations'):
            first_ann = first_item['annotations'][0]
            print_info(f"  样本标注类型: {first_ann.get('type', 'unknown')}")
            if first_ann.get('labels'):
                print_info(f"  样本标签: {first_ann.get('labels')}")
            if first_ann.get('keypoints'):
                kp_count = len(first_ann['keypoints'])
                print_info(f"  关键点数量: {kp_count}")
        else:
            print_warning("  第一个样本没有标注数据")
    
    # 检测是否为 Pose 格式（需要在类别映射之前检测）
    if task == 'detect' and parsed_data:
        # 检查第一个有标注的样本是否包含 pose 类型
        for item in parsed_data:
            if item['annotations']:
                if item['annotations'][0].get('type') == 'pose':
                    task = 'pose'  # 更新任务类型
                    print_info("检测到 Pose 格式标注，任务类型已更新为 'pose'")
                break
    
    class_mapping = LabelStudioConverter.build_class_mapping(parsed_data, task)
    
    if not class_mapping:
        print_error("未找到任何类别")
        print_info("提示: 请检查 Label Studio 导出数据是否包含有效的标注")
        if task == 'pose':
            print_info("      对于 Pose 任务，请确保导出数据包含 KeyPointLabels 标注")
        raise typer.Exit(1)
    
    print_success(f"✓ 找到 {len(class_mapping)} 个类别: {', '.join(class_mapping.keys())}")
    
    # 创建输出目录结构
    images_dir = output_path / 'images'
    if task in ['detect', 'pose']:
        labels_dir = output_path / 'labels'
        ensure_dir(images_dir)
        ensure_dir(labels_dir)
    else:  # classify
        labels_dir = None
        ensure_dir(images_dir)
        # 为每个类别创建目录
        for class_name in class_mapping.keys():
            ensure_dir(images_dir / class_name)
    
    # 准备下载列表
    print_info("准备下载图片...")
    if task in ['detect', 'pose']:
        download_list = LabelStudioConverter.prepare_download_list(parsed_data, images_dir)
    else:  # classify - 直接下载到类别目录
        download_list = []
        for item in parsed_data:
            category = item.get('category')
            if category and category in class_mapping:
                ls_path = item['image_path']
                filename = item['filename']
                local_path = images_dir / category / filename
                download_list.append((ls_path, local_path))
    
    print_info(f"共需下载 {len(download_list)} 张图片")
    if skip_existing:
        print_info("断点续传已启用，已存在的文件将被跳过")
    
    # 批量下载图片
    print_info(f"开始下载图片 (并发数: {max_workers})...")
    
    download_stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    with create_progress_bar() as progress:
        download_task = progress.add_task("下载图片", total=len(download_list))
        
        def progress_callback(current, total, status, filename):
            progress.update(download_task, advance=1)
        
        download_stats = client.download_images_batch(
            download_list,
            skip_existing=skip_existing,
            max_workers=max_workers,
            progress_callback=progress_callback
        )
    
    console.print()
    
    # 打印下载统计
    columns = ["状态", "数量"]
    rows = [
        ["✓ 已下载", download_stats["downloaded"]],
        ["⊙ 已跳过", download_stats["skipped"]],
        ["✗ 失败", download_stats["failed"]],
        ["总计", len(download_list)],
    ]
    print_table("下载统计", columns, rows, show_lines=True)
    
    if download_stats["failed"] > 0:
        print_warning(f"有 {download_stats['failed']} 张图片下载失败")
    
    # 生成YOLO格式标签/组织文件
    print_info("生成YOLO格式数据...")
    
    if task in ['detect', 'pose']:
        # 检测/姿势任务：生成标签文件
        generated_count = 0
        negative_count = 0
        
        with create_progress_bar() as progress:
            label_task = progress.add_task("生成标签", total=len(parsed_data))
            
            for item in parsed_data:
                filename = item['filename']
                image_path = images_dir / filename
                
                # 只为成功下载的图片生成标签
                if not image_path.exists():
                    progress.update(label_task, advance=1)
                    continue
                
                label_path = labels_dir / f"{Path(filename).stem}.txt"
                
                # 写入标签（负样本创建空文件）
                with open(label_path, 'w', encoding='utf-8') as f:
                    for ann in item['annotations']:
                        labels = ann.get('labels', [])
                        if not labels:
                            continue
                        
                        # 获取类别ID
                        class_name = labels[0]
                        class_id = class_mapping.get(class_name, 0)
                        
                        # 检查是否为 Pose 格式
                        if ann.get('type') == 'pose':
                            # Pose 格式：class_id x y w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
                            # 转换边界框坐标
                            x_center, y_center, w, h = LabelStudioConverter.convert_bbox_to_yolo(
                                ann['x'], ann['y'], ann['width'], ann['height']
                            )
                            
                            # 开始写入：class_id + bbox
                            line_parts = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{w:.6f}", f"{h:.6f}"]
                            
                            # 添加关键点
                            keypoints = ann.get('keypoints', [])
                            for kp in keypoints:
                                kp_x = kp['x'] / 100.0  # Label Studio 使用百分比坐标
                                kp_y = kp['y'] / 100.0
                                kp_v = kp['visibility']
                                line_parts.extend([f"{kp_x:.6f}", f"{kp_y:.6f}", str(kp_v)])
                            
                            f.write(" ".join(line_parts) + "\n")
                        else:
                            # 普通检测格式
                            # 转换坐标
                            x_center, y_center, w, h = LabelStudioConverter.convert_bbox_to_yolo(
                                ann['x'], ann['y'], ann['width'], ann['height']
                            )
                            
                            # 写入YOLO格式
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
                if item['annotations']:
                    generated_count += 1
                else:
                    negative_count += 1
                
                progress.update(label_task, advance=1)
        
        console.print()
        print_success(f"✓ 生成了 {generated_count} 个标签文件（正样本）")
        if negative_count > 0:
            print_success(f"✓ 创建了 {negative_count} 个空标签文件（负样本）")
            print_info(f"  负样本有助于减少误报，提高模型鲁棒性")
    
    else:  # classify
        # 分类任务：统计每个类别的图片数量
        organized_count = 0
        class_stats = defaultdict(int)
        
        for item in parsed_data:
            category = item['category']
            if not category:
                continue
            
            filename = item['filename']
            file_path = images_dir / category / filename
            
            if file_path.exists():
                organized_count += 1
                class_stats[category] += 1
        
        console.print()
        print_success(f"✓ 分类图片已按类别组织: {organized_count} 个文件")
        
        # 显示每个类别的统计
        for class_name in sorted(class_stats.keys()):
            print_info(f"  {class_name}: {class_stats[class_name]} 张")
    
    # 保存classes.txt（按 class ID 顺序，而不是字母顺序）
    classes_file = output_path / 'classes.txt'
    # 反转映射: {class_id: class_name}
    id_to_name = {idx: name for name, idx in class_mapping.items()}
    with open(classes_file, 'w', encoding='utf-8') as f:
        for idx in sorted(id_to_name.keys()):
            f.write(f"{id_to_name[idx]}\n")
    
    print_success(f"✓ 保存类别列表: {classes_file}")
    
    # 生成 dataset.yaml 的标签部分（不包含路径信息，等待数据拆分后补全）
    # 构建类别字典 {class_id: class_name}（保持与 class_mapping 相同的顺序）
    classes_dict = {idx: name for name, idx in class_mapping.items()}
    
    yaml_config = {
        'nc': len(classes_dict),
        'names': classes_dict,
    }
    
    # 提取关键点信息（如果是 pose 任务）
    if task == 'pose' and parsed_data:
        keypoint_names = None
        # 从第一个有关键点的样本中提取关键点顺序
        for item in parsed_data:
            for ann in item.get('annotations', []):
                if ann.get('type') == 'pose' and ann.get('keypoints'):
                    keypoint_names = [kp.get('label', f'kp_{i}') for i, kp in enumerate(ann['keypoints'])]
                    break
            if keypoint_names:
                break
        
        if keypoint_names:
            num_kpts = len(keypoint_names)
            yaml_config['kpt_shape'] = [num_kpts, 3]
            yaml_config['keypoint_names'] = keypoint_names
            
            # 设置 flip_idx
            if num_kpts == 17:
                yaml_config['flip_idx'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            elif num_kpts == 4:
                yaml_config['flip_idx'] = [0, 1, 2, 3]
            else:
                yaml_config['flip_idx'] = list(range(num_kpts))
    
    dataset_yaml_path = output_path / 'dataset.yaml'
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO Dataset Configuration (Labels Only)\n")
        f.write("# 此文件由 Label Studio 导出自动生成\n")
        f.write("# 数据集拆分后会自动补全 path, train, val, test 路径信息\n\n")
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"✓ 生成 dataset.yaml (标签部分): {dataset_yaml_path}")
    print_info("  数据集拆分后会自动补全路径信息")
    
    # 保存转换日志
    log_file = output_path / 'convert_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Label Studio 数据转换日志\n")
        f.write("=" * 50 + "\n")
        f.write(f"输入文件: {input_path}\n")
        f.write(f"Label Studio URL: {url}\n")
        f.write(f"任务类型: {task}\n")
        f.write(f"类别数量: {len(class_mapping)}\n")
        f.write(f"类别列表: {', '.join(class_mapping.keys())}\n")
        f.write(f"\n下载统计:\n")
        f.write(f"  已下载: {download_stats['downloaded']}\n")
        f.write(f"  已跳过: {download_stats['skipped']}\n")
        f.write(f"  失败: {download_stats['failed']}\n")
        f.write(f"  总计: {len(download_list)}\n")
        if task in ['detect', 'pose']:
            f.write(f"\n标签生成:\n")
            f.write(f"  正样本（有标注）: {generated_count}\n")
            f.write(f"  负样本（无标注）: {negative_count}\n")
            f.write(f"  包含负样本: {'是' if include_negative else '否'}\n")
            if task == 'pose':
                f.write(f"  格式: YOLO Pose (bbox + keypoints)\n")
        else:
            f.write(f"\n文件组织:\n")
            f.write(f"  组织数量: {organized_count}\n")
    
    # 显示最终统计
    console.print()
    print_section_header("转换完成")
    
    columns = ["项目", "值"]
    rows = [
        ["输出目录", str(output_path)],
        ["任务类型", task],
        ["类别数量", len(class_mapping)],
        ["图片总数", len(download_list)],
        ["成功下载", download_stats['downloaded']],
        ["跳过下载", download_stats['skipped']],
    ]
    
    if task in ['detect', 'pose']:
        rows.append(["标签文件", generated_count])
        if task == 'pose':
            rows.append(["标签格式", "YOLO Pose"])
    
    print_table("转换摘要", columns, rows, show_lines=True)
    
    # 显示后续步骤提示
    console.print()
    print_section_header("后续步骤")
    print_info("数据已转换为YOLO格式，保存在 data/raw 目录")
    print_info("接下来可以使用以下命令继续处理:")
    console.print()
    
    if task in ['detect', 'pose']:
        console.print("  [bold cyan]1. 划分数据集:[/bold cyan]")
        console.print(f"     python yolo_cli.py data split \\")
        console.print(f"       --images {output_path}/images \\")
        console.print(f"       --labels {output_path}/labels \\")
        console.print(f"       --output data/processed \\")
        console.print(f"       --ratios 0.7:0.2:0.1 \\")
        console.print(f"       --task {task}")
    else:
        console.print("  [bold cyan]1. 划分分类数据集:[/bold cyan]")
        console.print(f"     python yolo_cli.py data split \\")
        console.print(f"       --source {output_path}/images \\")
        console.print(f"       --task classify \\")
        console.print(f"       --ratios 0.7:0.2:0.1")
    
    console.print()
    console.print("  [bold cyan]2. 生成配置文件:[/bold cyan]")
    console.print(f"     python yolo_cli.py data generate-yaml \\")
    console.print(f"       --path data/processed \\")
    console.print(f"       --classes {output_path}/classes.txt \\")
    console.print(f"       --task {task}")
    
    if task == 'pose':
        console.print()
        console.print("  [bold yellow]💡 Pose 任务说明:[/bold yellow]")
        console.print("     - 数据已转换为 YOLO Pose 格式")
        console.print("     - 每行包含: class_id bbox keypoints")
        console.print("     - generate-yaml 会自动添加 kpt_shape 配置")
    
    console.print()
    console.print("  [bold cyan]3. 开始训练:[/bold cyan]")
    console.print(f"     python yolo_cli.py train --data data/processed/dataset.yaml")
    
    console.print()
    print_success("转换完成！")


if __name__ == "__main__":
    app()
