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
from ..core.label_scaler import LabelScaler
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


@app.command("merge")
def merge_datasets(
    datasets: Optional[str] = typer.Option(None, "--datasets", "-d", help="数据集路径列表（逗号分隔），如: path1,path2,path3。如不指定则进入交互式选择模式"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
    handle_duplicates: str = typer.Option("skip", "--duplicates", help="重复文件处理: skip=跳过, rename=重命名, error=报错"),
    deduplicate: bool = typer.Option(False, "--deduplicate", help="合并后去除完全相同的图片"),
):
    """合并多个数据集
    
    支持合并不同标签的数据集，自动处理类别ID重映射。
    
    示例:
    \b
      # 交互式选择数据集合并（从datasets目录）
      yolo-cli data merge
      
    \b
      # 手动指定数据集路径合并
      yolo-cli data merge \\
        --datasets data/dataset1,data/dataset2 \\
        --output data/merged \\
        --task detect
      
    \b
      # 合并三个数据集并去重
      yolo-cli data merge \\
        --datasets data/ds1,data/ds2,data/ds3 \\
        --output data/merged \\
        --deduplicate
    """
    
    print_section_header("合并数据集")
    
    # 验证任务类型
    task = validate_task_type(task)
    print_info(f"任务类型: {task}")
    
    # 如果没有指定数据集参数，进入交互式选择模式
    if datasets is None:
        from ..ui.prompts import select_multiple, confirm_action
        
        # 获取 datasets 目录
        config = ConfigManager()
        datasets_root = config.project_root / 'datasets'
        
        if not datasets_root.exists():
            print_error(f"datasets 目录不存在: {datasets_root}")
            print_info("请先创建 datasets 目录并在其中放置数据集")
            raise typer.Exit(1)
        
        # 扫描 datasets 目录下的所有子目录
        available_datasets = []
        for item in sorted(datasets_root.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                # 检查是否是有效的数据集目录（包含 images 目录或数据集配置文件）
                has_images = (item / 'images').exists() or \
                             (item / 'train').exists() or \
                             (item / 'val').exists() or \
                             (item / 'test').exists()
                has_config = (item / 'data.yaml').exists() or \
                            (item / 'dataset.yaml').exists() or \
                            (item / 'classes.txt').exists()
                
                if has_images or has_config:
                    available_datasets.append(item)
        
        if not available_datasets:
            print_error(f"在 {datasets_root} 目录下没有找到任何数据集")
            print_info("数据集目录应包含以下任一结构:")
            print_info("  • images/ 目录")
            print_info("  • train/val/test 目录")
            print_info("  • data.yaml 或 dataset.yaml 配置文件")
            raise typer.Exit(1)
        
        print_info(f"发现 {len(available_datasets)} 个数据集")
        console.print()
        
        # 让用户多选数据集
        choices = [f"{ds.name} ({ds})" for ds in available_datasets]
        selected_choices = select_multiple(
            "请选择要合并的数据集 (空格选择，回车确认):",
            choices
        )
        
        if not selected_choices:
            print_warning("未选择任何数据集")
            raise typer.Exit(0)
        
        if len(selected_choices) < 2:
            print_error("至少需要选择2个数据集进行合并")
            raise typer.Exit(1)
        
        # 提取选中的数据集路径
        dataset_paths = []
        for choice in selected_choices:
            # 从 "name (path)" 格式中提取路径
            ds_path_str = choice.split('(')[1].rstrip(')')
            dataset_paths.append(Path(ds_path_str))
        
        print_info(f"已选择 {len(dataset_paths)} 个数据集进行合并")
    else:
        # 解析数据集路径（手动指定模式）
        dataset_paths = [Path(p.strip()) for p in datasets.split(',')]
        
        if len(dataset_paths) < 2:
            print_error("至少需要指定2个数据集进行合并")
            raise typer.Exit(1)
        
        print_info(f"待合并数据集数量: {len(dataset_paths)}")
    
    # 验证所有数据集路径存在
    for i, ds_path in enumerate(dataset_paths, 1):
        if not ds_path.exists():
            print_error(f"数据集 {i} 不存在: {ds_path}")
            raise typer.Exit(1)
        print_info(f"  {i}. {ds_path}")
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True) / 'merged'
    else:
        output_path = Path(output_dir)
    
    print_info(f"输出目录: {output_path}")
    
    # 调用合并函数
    return _merge_datasets_impl(
        dataset_paths=dataset_paths,
        output_path=output_path,
        task=task,
        handle_duplicates=handle_duplicates,
        deduplicate=deduplicate
    )


@app.command("convert-format")
def convert_dataset_format(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="数据集路径（包含data.yaml的目录）"),
    source_format: str = typer.Option(..., "--from", "-f", help="源格式 (detect/segment/pose)"),
    target_format: str = typer.Option(..., "--to", "-t", help="目标格式 (detect/segment/pose)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    bbox_expand: float = typer.Option(0.0, "--bbox-expand", help="边界框扩展比例 (0.0-0.5，用于segment→detect)"),
    keep_confidence: bool = typer.Option(False, "--keep-confidence", help="保留置信度信息（如果有）"),
    preserve_structure: bool = typer.Option(True, "--preserve-structure", help="保留原始的train/val/test分割"),
):
    """转换数据集标注格式
    
    支持多种格式之间的转换，并提供丰富的自定义参数。
    
    示例:
    \b
      # 分割→检测（计算外接矩形）
      yolo-cli data convert-format \\
        --dataset data/segment_dataset \\
        --from segment --to detect \\
        --output data/detect_dataset
      
    \b
      # 分割→检测（扩展边界框10%）
      yolo-cli data convert-format \\
        --dataset data/segment_dataset \\
        --from segment --to detect \\
        --bbox-expand 0.1 \\
        --output data/detect_expanded
      
    \b
      # Pose→检测（只保留边界框）
      yolo-cli data convert-format \\
        --dataset data/pose_dataset \\
        --from pose --to detect \\
        --output data/detect_dataset
      
    \b
      # 检测→分割（矩形作为多边形）
      yolo-cli data convert-format \\
        --dataset data/detect_dataset \\
        --from detect --to segment \\
        --output data/segment_dataset
    """
    
    print_section_header("数据集格式转换")
    
    # 验证格式
    valid_formats = ['detect', 'segment', 'pose']
    if source_format not in valid_formats:
        print_error(f"不支持的源格式: {source_format}")
        print_info(f"支持的格式: {', '.join(valid_formats)}")
        raise typer.Exit(1)
    
    if target_format not in valid_formats:
        print_error(f"不支持的目标格式: {target_format}")
        print_info(f"支持的格式: {', '.join(valid_formats)}")
        raise typer.Exit(1)
    
    if source_format == target_format:
        print_error("源格式和目标格式相同，无需转换")
        raise typer.Exit(1)
    
    # 验证bbox扩展比例
    if bbox_expand < 0 or bbox_expand > 0.5:
        print_error("边界框扩展比例必须在 0.0-0.5 之间")
        raise typer.Exit(1)
    
    # 解析路径
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True) / f'converted_{target_format}'
    else:
        output_path = Path(output_dir)
    
    # 调用转换函数
    return _convert_format_impl(
        dataset_path=dataset_path,
        output_path=output_path,
        source_format=source_format,
        target_format=target_format,
        bbox_expand=bbox_expand,
        keep_confidence=keep_confidence,
        preserve_structure=preserve_structure
    )


@app.command("deduplicate")
def deduplicate_dataset(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="数据集路径（包含train/val/test的目录）"),
    mode: str = typer.Option("hash", "--mode", "-m", help="去重模式: hash (哈希), perceptual (感知哈希), both (两者结合)"),
    action: str = typer.Option("report", "--action", "-a", help="处理方式: report (仅报告), delete (删除), move (移动到duplicates目录)"),
    priority: str = typer.Option("train>val>test", "--priority", "-p", help="保留优先级: train>val>test 或 val>train>test"),
    threshold: float = typer.Option(0.95, "--threshold", "-t", help="相似度阈值 (0.0-1.0，仅用于感知哈希)"),
    cross_split: bool = typer.Option(True, "--cross-split/--within-split", help="是否跨集合去重（train/val/test之间）"),
):
    """对已拆分的数据集进行去重
    
    检测并处理训练集、验证集、测试集中的重复图片。
    
    示例:
    \b
      # 仅生成去重报告
      yolo-cli data deduplicate \\
        --dataset data/processed \\
        --mode hash \\
        --action report
      
    \b
      # 删除重复图片（保留train优先）
      yolo-cli data deduplicate \\
        --dataset data/processed \\
        --mode hash \\
        --action delete \\
        --priority "train>val>test"
      
    \b
      # 移动重复图片到duplicates目录
      yolo-cli data deduplicate \\
        --dataset data/processed \\
        --mode perceptual \\
        --action move \\
        --threshold 0.95
      
    \b
      # 只在各个集合内部去重，不跨集合
      yolo-cli data deduplicate \\
        --dataset data/processed \\
        --mode hash \\
        --action delete \\
        --within-split
    """
    
    print_section_header("数据集去重")
    
    # 验证参数
    valid_modes = ['hash', 'perceptual', 'both']
    if mode not in valid_modes:
        print_error(f"无效的去重模式: {mode}")
        print_info(f"可用模式: {', '.join(valid_modes)}")
        raise typer.Exit(1)
    
    valid_actions = ['report', 'delete', 'move']
    if action not in valid_actions:
        print_error(f"无效的处理方式: {action}")
        print_info(f"可用方式: {', '.join(valid_actions)}")
        raise typer.Exit(1)
    
    # 解析数据集路径
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    # 解析优先级
    priority_list = [p.strip() for p in priority.split('>')]
    valid_splits = ['train', 'val', 'test', 'valid']
    for split in priority_list:
        if split not in valid_splits:
            print_error(f"无效的集合名称: {split}")
            print_info(f"可用名称: {', '.join(valid_splits)}")
            raise typer.Exit(1)
    
    # 调用去重函数
    return _deduplicate_dataset_impl(
        dataset_path=dataset_path,
        mode=mode,
        action=action,
        priority_list=priority_list,
        threshold=threshold,
        cross_split=cross_split
    )


@app.command("merge-labels")
def merge_labels(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="数据集路径（包含data.yaml的目录）"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    mapping: Optional[str] = typer.Option(None, "--mapping", "-m", help="映射规则，格式: 'source1,source2:target;source3:target2'"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
):
    """合并多个类别标签为一个
    
    将多个相似或相关的类别合并为一个类别，简化模型训练。
    
    示例:
    \b
      # 合并车辆类别
      yolo-cli data merge-labels \\
        --dataset data/processed \\
        --mapping "car,truck,bus:vehicle" \\
        --output data/merged_vehicle
      
    \b
      # 多个合并规则
      yolo-cli data merge-labels \\
        --dataset data/processed \\
        --mapping "car,truck,bus:vehicle;cat,dog:pet;apple,banana:fruit" \\
        --output data/simplified
      
    \b
      # 交互式配置映射（不指定mapping参数）
      yolo-cli data merge-labels \\
        --dataset data/processed \\
        --output data/merged
    """
    
    print_section_header("合并类别标签")
    
    # 验证任务类型
    task = validate_task_type(task)
    
    # 解析数据集路径
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True) / 'merged_labels'
    else:
        output_path = Path(output_dir)
    
    # 调用合并函数
    return _merge_labels_impl(
        dataset_path=dataset_path,
        output_path=output_path,
        mapping_str=mapping,
        task=task
    )


@app.command("filter")
def filter_dataset(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="数据集路径（包含data.yaml的目录）"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    include_labels: Optional[str] = typer.Option(None, "--include", "-i", help="包含的标签列表（逗号分隔），如: person,car,dog"),
    exclude_labels: Optional[str] = typer.Option(None, "--exclude", "-e", help="排除的标签列表（逗号分隔），如: background,other"),
    keep_negative: bool = typer.Option(True, "--keep-negative", help="保留没有任何标注的图片（负样本）"),
    limit: Optional[str] = typer.Option(None, "--limit", "-l", help="限制每个集合的样本数量，格式: train:val:test，如: 100:30:10 或 all:50:20"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
):
    """按标签过滤数据集
    
    根据指定的标签包含/排除条件，从现有数据集中筛选样本生成新的数据集。
    自动重映射类别ID，保持连续性。可选限制每个集合的样本数量。
    
    示例:
    \b
      # 只保留特定类别
      yolo-cli data filter \\
        --dataset data/processed \\
        --include person,car \\
        --output data/filtered_person_car
      
    \b
      # 排除特定类别
      yolo-cli data filter \\
        --dataset data/processed \\
        --exclude background,other \\
        --output data/cleaned
      
    \b
      # 保留特定类别，不保留负样本
      yolo-cli data filter \\
        --dataset data/processed \\
        --include cat,dog \\
        --keep-negative False \\
        --output data/pets_only
      
    \b
      # 过滤并限制样本数量（train:100张, val:30张, test:10张）
      yolo-cli data filter \\
        --dataset data/processed \\
        --include person,car \\
        --limit 100:30:10 \\
        --output data/small_sample
      
    \b
      # 保留所有训练集，限制验证集和测试集
      yolo-cli data filter \\
        --dataset data/processed \\
        --include cat,dog \\
        --limit all:50:20 \\
        --output data/pets_limited
    """
    
    print_section_header("按标签过滤数据集")
    
    # 验证参数
    if include_labels is None and exclude_labels is None:
        print_error("必须指定 --include 或 --exclude 参数之一")
        raise typer.Exit(1)
    
    if include_labels is not None and exclude_labels is not None:
        print_error("--include 和 --exclude 不能同时使用")
        raise typer.Exit(1)
    
    # 验证任务类型
    task = validate_task_type(task)
    
    # 解析数据集路径
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_path = config.get_path('data_processed', absolute=True) / 'filtered'
    else:
        output_path = Path(output_dir)
    
    # 解析样本数量限制
    limit_dict = {}  # {split: count or 'all'}
    if limit:
        try:
            parts = limit.split(':')
            if len(parts) != 3:
                print_error("--limit 格式错误，应为 train:val:test，如: 100:30:10")
                raise typer.Exit(1)
            
            for split_name, count_str in zip(['train', 'val', 'test'], parts):
                count_str = count_str.strip().lower()
                if count_str == 'all' or count_str == '*':
                    limit_dict[split_name] = 'all'
                else:
                    try:
                        count = int(count_str)
                        if count <= 0:
                            print_error(f"样本数量必须大于0: {count}")
                            raise typer.Exit(1)
                        limit_dict[split_name] = count
                    except ValueError:
                        print_error(f"无效的样本数量: {count_str}")
                        raise typer.Exit(1)
            
            print_info(f"样本数量限制: train={limit_dict.get('train', 'all')}, val={limit_dict.get('val', 'all')}, test={limit_dict.get('test', 'all')}")
        except Exception as e:
            if not isinstance(e, typer.Exit):
                print_error(f"解析 --limit 参数失败: {e}")
                raise typer.Exit(1)
            raise
    
    # 调用过滤函数
    return _filter_dataset_impl(
        dataset_path=dataset_path,
        output_path=output_path,
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        keep_negative=keep_negative,
        limit_dict=limit_dict if limit else None,
        task=task
    )


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
    
    # 读取类别名称映射（用于显示类别名称）
    class_names = {}  # {class_id: class_name}
    yaml_file = None
    for yaml_name in ['data.yaml', 'dataset.yaml']:
        yaml_path = data_path / yaml_name
        if yaml_path.exists():
            yaml_file = yaml_path
            break
    
    if yaml_file:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and 'names' in yaml_config:
                    names_data = yaml_config['names']
                    if isinstance(names_data, dict):
                        # {0: 'class1', 1: 'class2'}
                        class_names = names_data
                    elif isinstance(names_data, list):
                        # ['class1', 'class2']
                        class_names = {i: name for i, name in enumerate(names_data)}
        except Exception as e:
            print_warning(f"无法读取类别名称: {e}")
    
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
            # 检测/分割任务：统计边界框和样本数
            class_counts = defaultdict(int)  # 标注数量
            class_image_counts = defaultdict(set)  # 样本数（包含该类别的图片）
            bbox_counts = 0
            
            for split in ['train', 'val', 'test']:
                label_dir = data_path / 'labels' / split
                if not label_dir.exists():
                    continue
                
                for label_file in label_dir.glob('*.txt'):
                    try:
                        image_name = label_file.stem  # 图片文件名（不含扩展名）
                        with open(label_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    class_image_counts[class_id].add(image_name)  # 记录图片
                                    bbox_counts += 1
                    except Exception:
                        pass
            
            if class_counts:
                # 根据是否有类别名称决定列数
                if class_names:
                    columns = ["类别ID", "类别名称", "标注数", "标注占比", "样本数", "样本占比"]
                else:
                    columns = ["类别ID", "标注数", "标注占比", "样本数", "样本占比"]
                
                rows = []
                total_annotations = sum(class_counts.values())
                total_images = len(set().union(*class_image_counts.values()))  # 去重后的总图片数
                
                for class_id in sorted(class_counts.keys()):
                    annotation_count = class_counts[class_id]
                    image_count = len(class_image_counts[class_id])
                    
                    annotation_ratio = f"{annotation_count/total_annotations*100:.1f}%"
                    image_ratio = f"{image_count/total_images*100:.1f}%"
                    
                    if class_names:
                        class_name = class_names.get(class_id, f"未知_{class_id}")
                        rows.append([
                            class_id,
                            class_name,
                            annotation_count,
                            annotation_ratio,
                            image_count,
                            image_ratio
                        ])
                    else:
                        rows.append([
                            class_id,
                            annotation_count,
                            annotation_ratio,
                            image_count,
                            image_ratio
                        ])
                
                print_table("类别分布", columns, rows, show_lines=True)
                print_info(f"总标注数量: {bbox_counts}")
                print_info(f"包含标注的图片数: {total_images}")
                
                # 显示类别总数
                if class_names:
                    print_info(f"类别总数: {len(class_names)}")
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


def _detect_label_format(label_file: Path) -> str:
    """
    自动检测标签文件格式
    
    Args:
        label_file: 标签文件路径
    
    Returns:
        str: 'detect', 'segment', 'pose', 或 'unknown'
    """
    if not label_file.exists():
        return 'unknown'
    
    try:
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                return 'unknown'
            
            # 分析第一行
            first_line = lines[0]
            parts = first_line.split()
            
            if len(parts) < 5:
                return 'unknown'
            
            # 检测格式
            # Segment: class_id + 至少6个坐标值（3个点，每个点x,y）
            if len(parts) >= 7:
                # 进一步判断是 segment 还是 pose
                # Segment: 所有值应该在0-1之间（归一化坐标）
                # Pose: 固定格式 class_id x y w h [kp_x kp_y kp_v ...]
                
                # 如果是5个值 + 3的倍数，可能是pose
                if len(parts) > 5 and (len(parts) - 5) % 3 == 0:
                    # 检查第2-5个值是否是bbox格式（通常<1）
                    try:
                        bbox_vals = [float(parts[i]) for i in range(1, 5)]
                        if all(0 <= v <= 1 for v in bbox_vals):
                            return 'pose'
                    except:
                        pass
                
                # 否则认为是segment（多边形点）
                return 'segment'
            
            # Detect: class_id x_center y_center width height (正好5个值)
            elif len(parts) == 5:
                return 'detect'
            
            return 'unknown'
    except Exception:
        return 'unknown'


def _convert_segment_to_detect(parts: List[str], bbox_expand: float = 0.0) -> str:
    """
    将分割标注转换为检测标注（计算外接矩形）
    
    Args:
        parts: 标签行分割后的部分 [class_id, x1, y1, x2, y2, ...]
        bbox_expand: 边界框扩展比例 (0.0-0.5)
    
    Returns:
        str: 检测格式的标签行 "class_id x_center y_center width height"
    """
    try:
        class_id = parts[0]
        
        # 提取所有坐标点
        coords = [float(x) for x in parts[1:]]
        
        # 分离x和y坐标
        x_coords = coords[0::2]
        y_coords = coords[1::2]
        
        # 计算边界框
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # 转换为YOLO格式（中心点 + 宽高）
        width = x_max - x_min
        height = y_max - y_min
        
        # 应用扩展（如果需要）
        if bbox_expand > 0:
            width *= (1 + bbox_expand)
            height *= (1 + bbox_expand)
            # 确保不超出图像边界（归一化坐标 0-1）
            width = min(width, 1.0)
            height = min(height, 1.0)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # 确保中心点在合理范围内
        x_center = max(width/2, min(1 - width/2, x_center))
        y_center = max(height/2, min(1 - height/2, y_center))
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    except Exception as e:
        # 转换失败，返回原始行
        return ' '.join(parts)


def _convert_pose_to_detect(parts: List[str], bbox_expand: float = 0.0) -> str:
    """
    将姿态估计标注转换为检测标注（只保留bbox部分）
    
    Args:
        parts: 标签行分割后的部分 [class_id, x, y, w, h, kp_x, kp_y, kp_v, ...]
        bbox_expand: 边界框扩展比例 (0.0-0.5)
    
    Returns:
        str: 检测格式的标签行 "class_id x_center y_center width height"
    """
    try:
        # Pose格式前5个值就是detect格式
        if len(parts) >= 5:
            if bbox_expand > 0:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3]) * (1 + bbox_expand)
                height = float(parts[4]) * (1 + bbox_expand)
                
                # 确保不超出边界
                width = min(width, 1.0)
                height = min(height, 1.0)
                x_center = max(width/2, min(1 - width/2, x_center))
                y_center = max(height/2, min(1 - height/2, y_center))
                
                return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            else:
                return ' '.join(parts[:5])
        return ' '.join(parts)
    except Exception:
        return ' '.join(parts)


def _convert_detect_to_segment(parts: List[str]) -> str:
    """
    将检测标注转换为分割标注（矩形转为4点多边形）
    
    Args:
        parts: 标签行分割后的部分 [class_id, x_center, y_center, width, height]
    
    Returns:
        str: 分割格式的标签行 "class_id x1 y1 x2 y2 x3 y3 x4 y4"
    """
    try:
        if len(parts) < 5:
            return ' '.join(parts)
        
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # 计算矩形四个角点（顺时针）
        x1 = x_center - width / 2  # 左上
        y1 = y_center - height / 2
        x2 = x_center + width / 2  # 右上
        y2 = y_center - height / 2
        x3 = x_center + width / 2  # 右下
        y3 = y_center + height / 2
        x4 = x_center - width / 2  # 左下
        y4 = y_center + height / 2
        
        # 确保坐标在0-1范围内
        points = [x1, y1, x2, y2, x3, y3, x4, y4]
        points = [max(0, min(1, p)) for p in points]
        
        points_str = ' '.join([f"{p:.6f}" for p in points])
        return f"{class_id} {points_str}"
    except Exception:
        return ' '.join(parts)


def _convert_detect_to_pose(parts: List[str], num_keypoints: int = 17) -> str:
    """
    将检测标注转换为Pose标注（添加默认关键点）
    
    Args:
        parts: 标签行分割后的部分 [class_id, x_center, y_center, width, height]
        num_keypoints: 关键点数量
    
    Returns:
        str: Pose格式的标签行 "class_id x y w h kp_x kp_y kp_v ..."
    """
    try:
        if len(parts) < 5:
            return ' '.join(parts)
        
        # 保留原始bbox
        result = ' '.join(parts[:5])
        
        # 添加默认关键点（全部设为不可见，坐标为0）
        for _ in range(num_keypoints):
            result += " 0 0 0"  # x=0, y=0, visibility=0 (未标注)
        
        return result
    except Exception:
        return ' '.join(parts)


def _convert_segment_to_pose(parts: List[str], num_keypoints: int = 17) -> str:
    """
    将分割标注转换为Pose标注（计算bbox + 默认关键点）
    
    Args:
        parts: 标签行分割后的部分 [class_id, x1, y1, x2, y2, ...]
        num_keypoints: 关键点数量
    
    Returns:
        str: Pose格式的标签行 "class_id x y w h kp_x kp_y kp_v ..."
    """
    try:
        # 先转为detect格式
        detect_line = _convert_segment_to_detect(parts, 0.0)
        detect_parts = detect_line.split()
        
        # 再转为pose格式
        return _convert_detect_to_pose(detect_parts, num_keypoints)
    except Exception:
        return ' '.join(parts)


def _convert_pose_to_segment(parts: List[str]) -> str:
    """
    将Pose标注转换为分割标注（使用bbox作为矩形）
    
    Args:
        parts: 标签行分割后的部分 [class_id, x, y, w, h, kp_x, kp_y, kp_v, ...]
    
    Returns:
        str: 分割格式的标签行 "class_id x1 y1 x2 y2 x3 y3 x4 y4"
    """
    try:
        if len(parts) < 5:
            return ' '.join(parts)
        
        # 提取bbox部分
        detect_parts = parts[:5]
        
        # 转为segment格式
        return _convert_detect_to_segment(detect_parts)
    except Exception:
        return ' '.join(parts)


def _merge_datasets_impl(
    dataset_paths: List[Path],
    output_path: Path,
    task: str,
    handle_duplicates: str = 'skip',
    deduplicate: bool = False
):
    """合并数据集的实现函数
    
    Args:
        dataset_paths: 数据集路径列表
        output_path: 输出路径
        task: 目标任务类型（合并后的格式）
        handle_duplicates: 重复文件处理方式
        deduplicate: 是否去重
    """
    
    # 1. 收集所有数据集的类别信息和任务类型
    print_section_header("分析数据集")
    
    dataset_classes = []  # [(dataset_idx, classes_dict, dataset_path, detected_task)]
    all_class_names = []
    dataset_task_types = {}  # {dataset_idx: task_type}
    
    for idx, ds_path in enumerate(dataset_paths):
        print_info(f"\n分析数据集 {idx + 1}: {ds_path.name}")
        # 尝试从 data.yaml 或 dataset.yaml 读取类别
        data_yaml = ds_path / 'data.yaml'
        dataset_yaml = ds_path / 'dataset.yaml'
        classes_file = ds_path / 'classes.txt'
        
        classes_dict = {}
        yaml_file = None
        
        # 优先尝试 data.yaml，然后是 dataset.yaml
        if data_yaml.exists():
            yaml_file = data_yaml
        elif dataset_yaml.exists():
            yaml_file = dataset_yaml
        
        if yaml_file:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data and 'names' in yaml_data:
                        classes_dict = yaml_data['names']
                        # 确保是 {id: name} 格式
                        if isinstance(classes_dict, list):
                            classes_dict = {i: name for i, name in enumerate(classes_dict)}
            except Exception as e:
                print_warning(f"读取 {yaml_file} 失败: {e}")
        
        elif classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                    classes_dict = {i: name for i, name in enumerate(classes)}
            except Exception as e:
                print_warning(f"读取 {classes_file} 失败: {e}")
        
        if not classes_dict:
            print_error(f"数据集 {idx + 1} 未找到类别信息: {ds_path}")
            print_warning(f"请确保以下任一文件存在且包含 'names' 字段:")
            print_warning(f"  • {ds_path}/data.yaml")
            print_warning(f"  • {ds_path}/dataset.yaml")
            print_warning(f"  • {ds_path}/classes.txt")
            raise typer.Exit(1)
        
        # 自动检测任务类型
        detected_task = 'unknown'
        
        # 尝试找一个标签文件进行格式检测
        for split in ['train', 'val', 'test']:
            label_dir = ds_path / 'labels' / split
            if not label_dir.exists():
                label_dir = ds_path / split / 'labels'
            
            if label_dir.exists():
                # 找第一个标签文件
                label_files = list(label_dir.glob('*.txt'))
                if label_files:
                    detected_task = _detect_label_format(label_files[0])
                    break
        
        dataset_task_types[idx] = detected_task
        dataset_classes.append((idx, classes_dict, ds_path, detected_task))
        
        # 收集类别名称
        class_names = list(classes_dict.values())
        all_class_names.extend(class_names)
        
        # 显示信息
        task_emoji = {
            'detect': '📦',
            'segment': '🎨', 
            'pose': '🤸',
            'unknown': '❓'
        }
        print_info(f"  类别数: {len(classes_dict)}")
        print_info(f"  类别: {', '.join(class_names)}")
        print_info(f"  {task_emoji.get(detected_task, '❓')} 检测到的任务类型: {detected_task}")
    
    # 检查是否需要格式转换
    console.print()
    print_section_header("检查格式兼容性")
    
    unique_task_types = set(dataset_task_types.values()) - {'unknown'}
    needs_conversion = len(unique_task_types) > 1 or (unique_task_types and task not in unique_task_types)
    
    if needs_conversion:
        print_warning(f"\n⚠️  检测到混合任务类型:")
        for idx, detected in dataset_task_types.items():
            if detected != 'unknown':
                print_warning(f"   数据集 {idx + 1}: {detected}")
        
        print_info(f"\n目标任务类型: {task}")
        print_info("将自动转换标签格式:")
        
        conversion_count = 0
        for idx, detected in dataset_task_types.items():
            if detected != task and detected != 'unknown':
                conversion_count += 1
                if detected == 'segment' and task == 'detect':
                    print_info(f"   数据集 {idx + 1}: segment → detect (多边形→外接矩形)")
                elif detected == 'pose' and task == 'detect':
                    print_info(f"   数据集 {idx + 1}: pose → detect (保留边界框，丢弃关键点)")
                elif detected == 'detect' and task == 'segment':
                    print_warning(f"   数据集 {idx + 1}: detect → segment (⚠️  无法精确转换，将使用矩形作为掩码)")
                else:
                    print_warning(f"   数据集 {idx + 1}: {detected} → {task} (可能无法完美转换)")
        
        if conversion_count > 0:
            console.print()
            print_info("💡 转换说明:")
            print_info("   • segment→detect: 从多边形计算最小外接矩形，精度不损失")
            print_info("   • pose→detect: 保留边界框，丢弃关键点信息")
            print_info("   • detect→segment: 不建议，矩形作为分割掩码效果较差")
    else:
        print_success(f"✓ 所有数据集任务类型一致: {task}")
    
    console.print()
    
    # 2. 构建统一的类别映射
    print_section_header("构建统一类别映射")
    
    # 去重并保持顺序
    unique_classes = []
    seen = set()
    for name in all_class_names:
        if name not in seen:
            unique_classes.append(name)
            seen.add(name)
    
    # 创建新的类别ID映射
    merged_classes = {i: name for i, name in enumerate(unique_classes)}
    print_info(f"合并后总类别数: {len(merged_classes)}")
    print_info(f"类别列表: {', '.join(unique_classes)}")
    
    # 为每个数据集创建类别ID重映射表
    class_remapping = {}  # {dataset_idx: {old_id: new_id}}
    
    for idx, old_classes, _, _ in dataset_classes:
        remapping = {}
        for old_id, class_name in old_classes.items():
            # 找到新的类别ID
            new_id = next(i for i, name in merged_classes.items() if name == class_name)
            remapping[old_id] = new_id
        
        class_remapping[idx] = remapping
        
        # 显示映射关系
        if remapping and any(old_id != new_id for old_id, new_id in remapping.items()):
            print_info(f"数据集 {idx + 1} 类别ID重映射:")
            for old_id, new_id in remapping.items():
                if old_id != new_id:
                    class_name = old_classes[old_id]
                    print_info(f"  {class_name}: {old_id} → {new_id}")
    
    console.print()
    
    # 3. 创建输出目录结构
    for split in ['train', 'val', 'test']:
        ensure_dir(output_path / 'images' / split)
        if task != 'classify':
            ensure_dir(output_path / 'labels' / split)
    
    # 4. 合并数据集
    print_section_header("合并数据文件")
    
    total_files = 0
    skipped_files = 0
    renamed_files = 0
    file_registry = set()  # 记录已复制的文件名
    split_counts = {'train': 0, 'val': 0, 'test': 0}  # 记录各split的文件数
    
    converted_labels_count = 0
    
    for ds_idx, old_classes, ds_path, detected_task in dataset_classes:
        print_info(f"处理数据集 {ds_idx + 1}: {ds_path.name}")
        
        remapping = class_remapping[ds_idx]
        source_task = dataset_task_types[ds_idx]
        
        for split in ['train', 'val', 'test']:
            # 尝试两种目录结构：
            # 结构1: dataset/images/train/, dataset/labels/train/
            # 结构2: dataset/train/images/, dataset/train/labels/
            src_img_dir = ds_path / 'images' / split
            src_label_dir = ds_path / 'labels' / split
            
            # 如果结构1不存在，尝试结构2
            if not src_img_dir.exists():
                src_img_dir = ds_path / split / 'images'
                src_label_dir = ds_path / split / 'labels'
            
            # 处理 val/valid 的别名
            if not src_img_dir.exists() and split == 'val':
                # 尝试 valid 作为 val 的别名
                src_img_dir = ds_path / 'images' / 'valid'
                src_label_dir = ds_path / 'labels' / 'valid'
                
                if not src_img_dir.exists():
                    src_img_dir = ds_path / 'valid' / 'images'
                    src_label_dir = ds_path / 'valid' / 'labels'
            
            if not src_img_dir.exists():
                continue
            
            # 获取所有图片文件
            image_files = list(find_files(src_img_dir, ['.jpg', '.jpeg', '.png']))
            
            if not image_files:
                continue
            
            print_info(f"  {split}: {len(image_files)} 张图片")
            
            for img_file in image_files:
                # 处理重复文件名
                dst_filename = img_file.name
                
                if dst_filename in file_registry:
                    if handle_duplicates == 'skip':
                        skipped_files += 1
                        continue
                    elif handle_duplicates == 'rename':
                        # 重命名: image.jpg → image_ds2.jpg
                        stem = img_file.stem
                        suffix = img_file.suffix
                        dst_filename = f"{stem}_ds{ds_idx + 1}{suffix}"
                        renamed_files += 1
                    elif handle_duplicates == 'error':
                        print_error(f"重复文件名: {dst_filename}")
                        raise typer.Exit(1)
                
                # 复制图片
                dst_img = output_path / 'images' / split / dst_filename
                shutil.copy2(img_file, dst_img)
                file_registry.add(dst_filename)
                total_files += 1
                split_counts[split] += 1
                
                # 处理标签文件
                if task != 'classify':
                    label_file = src_label_dir / f"{img_file.stem}.txt"
                    
                    if label_file.exists():
                        # 读取并更新标签文件
                        dst_label = output_path / 'labels' / split / f"{Path(dst_filename).stem}.txt"
                        
                        with open(label_file, 'r') as f_in, open(dst_label, 'w') as f_out:
                            for line in f_in:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                parts = line.split()
                                if len(parts) >= 1:
                                    # 更新类别ID
                                    old_class_id = int(parts[0])
                                    new_class_id = remapping.get(old_class_id, old_class_id)
                                    parts[0] = str(new_class_id)
                                    
                                    # 格式转换（如果需要）
                                    if source_task != task and source_task != 'unknown':
                                        # segment → detect
                                        if source_task == 'segment' and task == 'detect':
                                            if len(parts) >= 7:  # 确保是分割格式
                                                line = _convert_segment_to_detect(parts)
                                                converted_labels_count += 1
                                            else:
                                                line = ' '.join(parts)
                                        # pose → detect
                                        elif source_task == 'pose' and task == 'detect':
                                            if len(parts) > 5:  # 确保是pose格式
                                                line = _convert_pose_to_detect(parts)
                                                converted_labels_count += 1
                                            else:
                                                line = ' '.join(parts)
                                        else:
                                            # 其他转换暂不支持，保持原样
                                            line = ' '.join(parts)
                                    else:
                                        line = ' '.join(parts)
                                    
                                    f_out.write(line + '\n')
    
    # 5. 统计信息
    console.print()
    print_section_header("合并统计")
    
    print_info("各集合文件数:")
    print_info(f"  训练集 (train): {split_counts['train']} 张")
    print_info(f"  验证集 (val):   {split_counts['val']} 张")
    print_info(f"  测试集 (test):  {split_counts['test']} 张")
    print_info(f"  总计:           {total_files} 张")
    
    # 验证总数是否一致
    split_sum = sum(split_counts.values())
    if split_sum != total_files:
        print_warning(f"⚠️  内部统计不一致: 分集合总和({split_sum}) ≠ 总计({total_files})")
    
    if skipped_files > 0:
        console.print()
        print_warning(f"跳过重复文件: {skipped_files} 张")
    if renamed_files > 0:
        print_info(f"重命名文件: {renamed_files} 张")
    if converted_labels_count > 0:
        console.print()
        print_success(f"✓ 格式转换: {converted_labels_count} 个标注已转换为 {task} 格式")
    
    # 6. 保存类别信息
    classes_file = output_path / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in unique_classes:
            f.write(f"{class_name}\n")
    
    print_success(f"✓ 类别列表已保存: {classes_file}")
    
    # 7. 生成 dataset.yaml
    yaml_config = {
        'path': str(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': merged_classes,
        'nc': len(merged_classes),
    }
    
    dataset_yaml_path = output_path / 'dataset.yaml'
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO 合并数据集配置文件\n")
        f.write(f"# 合并自 {len(dataset_paths)} 个数据集\n\n")
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"✓ dataset.yaml 已生成: {dataset_yaml_path}")
    
    # 7.5 验证实际文件数
    console.print()
    print_section_header("验证数据集")
    
    actual_counts = {'train': 0, 'val': 0, 'test': 0}
    for split in ['train', 'val', 'test']:
        img_dir = output_path / 'images' / split
        if img_dir.exists():
            actual_counts[split] = len(list(find_files(img_dir, ['.jpg', '.jpeg', '.png'])))
    
    actual_total = sum(actual_counts.values())
    
    print_info("实际生成文件数:")
    print_info(f"  训练集 (train): {actual_counts['train']} 张")
    print_info(f"  验证集 (val):   {actual_counts['val']} 张")
    print_info(f"  测试集 (test):  {actual_counts['test']} 张")
    print_info(f"  总计:           {actual_total} 张")
    
    # 对比统计数据
    if actual_total == total_files:
        print_success("✓ 文件数验证通过")
    else:
        print_warning(f"⚠️  文件数不一致: 实际({actual_total}) vs 统计({total_files})")
    
    # 8. 可选：去重
    if deduplicate:
        console.print()
        print_section_header("数据去重")
        print_info("对合并后的数据集进行去重...")
        
        from ..core.deduplicator import ImageDeduplicator
        
        deduplicator = ImageDeduplicator()
        
        for split in ['train', 'val', 'test']:
            img_dir = output_path / 'images' / split
            label_dir = output_path / 'labels' / split
            
            if not img_dir.exists():
                continue
            
            image_files = list(find_files(img_dir, ['.jpg', '.jpeg', '.png']))
            
            if image_files:
                print_info(f"检查 {split} 集...")
                duplicates_map = deduplicator.find_duplicates(image_files)
                
                if duplicates_map:
                    total_dup = sum(len(files) - 1 for files in duplicates_map.values())
                    print_warning(f"  发现 {total_dup} 个重复文件")
                    
                    removed = deduplicator.remove_duplicates(
                        duplicates_map,
                        labels_dir=label_dir if task != 'classify' else None
                    )
                    
                    print_success(f"  ✓ 已删除 {len(removed)} 个重复文件")
                else:
                    print_success(f"  ✓ 未发现重复")
    
    # 9. 最终统计
    console.print()
    print_section_header("合并完成")
    
    final_stats = {}
    for split in ['train', 'val', 'test']:
        img_dir = output_path / 'images' / split
        if img_dir.exists():
            count = len(list(find_files(img_dir, ['.jpg', '.jpeg', '.png'])))
            final_stats[split] = count
    
    total_final = sum(final_stats.values())
    
    columns = ["数据集", "图片数量"]
    rows = [
        ["训练集", final_stats.get('train', 0)],
        ["验证集", final_stats.get('val', 0)],
        ["测试集", final_stats.get('test', 0)],
        ["总计", total_final],
    ]
    print_table("合并后数据集统计", columns, rows, show_lines=True)
    
    print_success(f"数据集合并完成！输出目录: {output_path}")


@app.command()
def scale_labels(
    dataset_dir: str = typer.Option(..., "--dataset", "-d", help="数据集目录"),
    output_dir: str = typer.Option(..., "--output", "-o", help="输出目录"),
    scale: float = typer.Option(..., "--scale", "-s", help="缩放比例 (>0)，如0.8表示缩小到80%，1.2表示放大到120%"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/pose)"),
    splits: Optional[str] = typer.Option(None, "--splits", help="处理的子集，逗号分隔 (如: train,val)"),
    classes: Optional[str] = typer.Option(None, "--classes", help="处理的类别ID，逗号分隔 (如: 0,2,5)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览模式，不实际修改文件"),
):
    """批量调整标注框大小（保持中心点不变）
    
    适用场景：
    - 标注框过大/过小导致模型效果不佳
    - 需要统一调整标注风格
    
    示例:
    \b
      # 缩小所有标注到80%
      yolo-cli data scale-labels \\
        --dataset datasets/original \\
        --output datasets/scaled_0.8 \\
        --scale 0.8 \\
        --task detect
    
    \b
      # 放大训练集标注到120%，只处理类别0和1
      yolo-cli data scale-labels \\
        --dataset datasets/original \\
        --output datasets/scaled \\
        --scale 1.2 \\
        --splits train \\
        --classes 0,1
    """
    
    print_section_header("批量调整标注大小")
    
    # 1. 验证参数
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    if not dataset_path.exists():
        print_error(f"数据集目录不存在: {dataset_path}")
        raise typer.Exit(1)
    
    if scale <= 0:
        print_error(f"缩放比例必须 > 0，当前值: {scale}")
        raise typer.Exit(1)
    
    if scale < 0.1 or scale > 2.0:
        print_warning(f"缩放比例 {scale} 超出推荐范围 [0.1, 2.0]，可能导致标注质量问题")
        if not typer.confirm("确认继续？"):
            raise typer.Exit(0)
    
    # 验证任务类型
    task = validate_task_type(task)
    
    # 解析子集
    split_list = None
    if splits:
        split_list = [s.strip() for s in splits.split(',')]
        for split in split_list:
            if split not in ['train', 'val', 'valid', 'test']:
                print_error(f"无效的子集名称: {split}")
                raise typer.Exit(1)
    
    # 解析类别
    target_classes = None
    if classes:
        try:
            target_classes = set(int(c.strip()) for c in classes.split(','))
        except ValueError:
            print_error(f"无效的类别ID格式: {classes}")
            raise typer.Exit(1)
    
    # 显示配置
    console.print()
    print_info("📋 配置信息:")
    print_info(f"  数据集: {dataset_path}")
    print_info(f"  输出: {output_path}")
    print_info(f"  任务类型: {task}")
    print_info(f"  缩放比例: {scale} ({'缩小' if scale < 1 else '放大' if scale > 1 else '不变'})")
    print_info(f"  处理子集: {', '.join(split_list) if split_list else '全部'}")
    print_info(f"  处理类别: {', '.join(map(str, sorted(target_classes))) if target_classes else '全部'}")
    print_info(f"  模式: {'预览' if dry_run else '正式'}")
    console.print()
    
    if dry_run:
        print_warning("⚠️  预览模式：将显示处理结果但不会实际修改文件")
        console.print()
    
    # 2. 初始化缩放器
    scaler = LabelScaler()
    
    # 3. 检测目录结构并确定要处理的子集
    # 结构1: dataset/images/split/, dataset/labels/split/
    # 结构2: dataset/split/images/, dataset/split/labels/
    use_structure_1 = False
    
    if not split_list:
        split_list = []
        for split in ['train', 'val', 'valid', 'test']:
            # 先检查结构1
            if (dataset_path / 'images' / split).exists():
                split_list.append(split)
                use_structure_1 = True
            # 再检查结构2
            elif (dataset_path / split / 'images').exists():
                split_list.append(split)
    else:
        # 用户指定了子集，检测第一个子集的结构
        for split in split_list:
            if (dataset_path / 'images' / split).exists():
                use_structure_1 = True
                break
    
    if not split_list:
        print_error("未找到任何数据子集（train/val/test）")
        raise typer.Exit(1)
    
    structure_type = "结构1 (images/split)" if use_structure_1 else "结构2 (split/images)"
    print_info(f"找到 {len(split_list)} 个子集: {', '.join(split_list)}")
    print_info(f"检测到目录结构: {structure_type}")
    console.print()
    
    # 4. 创建输出目录结构（保持与输入相同）
    if not dry_run:
        if use_structure_1:
            # 结构1: dataset/images/split/, dataset/labels/split/
            ensure_dir(output_path / 'images')
            ensure_dir(output_path / 'labels')
            for split in split_list:
                ensure_dir(output_path / 'images' / split)
                ensure_dir(output_path / 'labels' / split)
        else:
            # 结构2: dataset/split/images/, dataset/split/labels/
            for split in split_list:
                ensure_dir(output_path / split / 'images')
                ensure_dir(output_path / split / 'labels')
    
    # 5. 处理每个子集
    print_section_header("处理标注文件")
    split_stats = {}
    
    for split in split_list:
        print_info(f"处理子集: {split}")
        
        # 确定目录结构
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        if not img_dir.exists():
            img_dir = dataset_path / split / 'images'
            label_dir = dataset_path / split / 'labels'
        
        if not img_dir.exists():
            print_warning(f"  跳过：未找到图片目录")
            continue
        
        # 获取所有图片文件
        image_files = list(find_files(img_dir, ['.jpg', '.jpeg', '.png', '.bmp']))
        
        if not image_files:
            print_warning(f"  跳过：未找到图片文件")
            continue
        
        print_info(f"  找到 {len(image_files)} 张图片")
        
        # 处理每个标注文件
        processed_count = 0
        annotation_count = 0
        
        with create_progress_bar() as progress:
            task_id = progress.add_task(f"  处理 {split}", total=len(image_files))
            
            for img_file in image_files:
                # 标注文件路径
                label_file = label_dir / f"{img_file.stem}.txt"
                
                # 确定输出路径（根据目录结构）
                if use_structure_1:
                    # 结构1: dataset/images/split/
                    dst_img = output_path / 'images' / split / img_file.name
                    output_label = output_path / 'labels' / split / f"{img_file.stem}.txt"
                else:
                    # 结构2: dataset/split/images/
                    dst_img = output_path / split / 'images' / img_file.name
                    output_label = output_path / split / 'labels' / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    # 没有标注，只复制图片
                    if not dry_run:
                        shutil.copy2(img_file, dst_img)
                    progress.update(task_id, advance=1)
                    continue
                
                # 处理标注文件
                if not dry_run:
                    count = scaler.process_label_file(
                        label_file,
                        output_label,
                        scale,
                        task,
                        target_classes
                    )
                    annotation_count += count
                    
                    # 复制图片
                    shutil.copy2(img_file, dst_img)
                else:
                    # 预览模式：只统计
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            lines = [line.strip() for line in f if line.strip()]
                            annotation_count += len(lines)
                
                processed_count += 1
                progress.update(task_id, advance=1)
        
        split_stats[split] = {
            'files': processed_count,
            'annotations': annotation_count
        }
        
        print_success(f"  ✓ 完成：{processed_count} 文件，{annotation_count} 标注")
    
    # 6. 生成配置文件
    console.print()
    print_section_header("生成配置文件")
    
    if not dry_run:
        # 读取原始 YAML 获取类别和其他配置
        yaml_data = None
        for yaml_name in ['data.yaml', 'dataset.yaml']:
            yaml_file = dataset_path / yaml_name
            if yaml_file.exists():
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                break
        
        if yaml_data:
            # 生成新的 dataset.yaml
            # 使用相对路径（相对于当前工作目录）
            new_yaml_config = {
                'path': str(output_path),
            }
            
            # 添加子集路径（根据实际存在的子集和目录结构）
            for split in split_list:
                # 检查目录是否存在
                if use_structure_1:
                    split_img_dir = output_path / 'images' / split
                else:
                    split_img_dir = output_path / split / 'images'
                
                if split_img_dir.exists():
                    # 使用相对路径
                    if use_structure_1:
                        # 结构1: images/train
                        yaml_key = 'val' if split == 'valid' else split
                        new_yaml_config[yaml_key] = f'images/{split}'
                    else:
                        # 结构2: train/images
                        if split == 'valid':
                            new_yaml_config['val'] = 'valid/images'
                        else:
                            new_yaml_config[split] = f'{split}/images'
            
            # 复制类别信息
            if 'names' in yaml_data:
                new_yaml_config['names'] = yaml_data['names']
            if 'nc' in yaml_data:
                new_yaml_config['nc'] = yaml_data['nc']
            
            # 对于姿态估计任务，复制关键点配置
            if task == 'pose':
                if 'kpt_shape' in yaml_data:
                    new_yaml_config['kpt_shape'] = yaml_data['kpt_shape']
                if 'flip_idx' in yaml_data:
                    new_yaml_config['flip_idx'] = yaml_data['flip_idx']
            
            # 写入新的 dataset.yaml
            output_yaml = output_path / 'dataset.yaml'
            with open(output_yaml, 'w', encoding='utf-8') as f:
                f.write("# YOLO 数据集配置文件\n")
                f.write(f"# 由 scale-labels 命令生成\n")
                f.write(f"# 缩放比例: {scale}\n\n")
                yaml.dump(new_yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print_success(f"✓ 已生成: dataset.yaml")
        
        # 复制 classes.txt
        classes_file = dataset_path / 'classes.txt'
        if classes_file.exists():
            shutil.copy2(classes_file, output_path / 'classes.txt')
            print_success(f"✓ 已复制: classes.txt")
    
    # 7. 生成报告
    if not dry_run:
        console.print()
        print_section_header("生成处理报告")
        
        scaler.generate_report(
            output_path,
            str(dataset_path),
            str(output_path),
            scale,
            task,
            split_list,
            list(target_classes) if target_classes else None,
            split_stats
        )
        
        report_file = output_path / 'adjustment_report.txt'
        print_success(f"✓ 报告已生成: {report_file}")
    
    # 8. 显示统计信息
    console.print()
    print_section_header("处理完成" if not dry_run else "预览结果")
    
    stats = scaler.get_statistics()
    
    if dry_run:
        # 预览模式统计
        total_files = sum(s['files'] for s in split_stats.values())
        total_annotations = sum(s['annotations'] for s in split_stats.values())
        
        print_info(f"总文件数: {total_files}")
        print_info(f"总标注数: {total_annotations}")
    else:
        # 正式模式统计
        print_info(f"总文件数: {stats['total_files']}")
        print_info(f"处理的文件数: {stats['processed_files']}")
        print_info(f"总标注数: {stats['total_annotations']}")
        print_info(f"缩放的标注数: {stats['scaled_annotations']}")
        
        if target_classes:
            print_info(f"跳过的标注数: {stats['skipped_annotations']} (非目标类别)")
    
    console.print()
    
    # 子集统计表
    columns = ["子集", "文件数", "标注数"]
    rows = []
    for split_name, split_stat in split_stats.items():
        rows.append([split_name, split_stat['files'], split_stat['annotations']])
    
    total_files = sum(s['files'] for s in split_stats.values())
    total_annotations = sum(s['annotations'] for s in split_stats.values())
    rows.append(["总计", total_files, total_annotations])
    
    print_table("处理统计", columns, rows, show_lines=True)
    
    # 警告信息
    if not dry_run and stats['warnings']:
        console.print()
        print_warning(f"⚠️  发现 {len(stats['warnings'])} 个警告")
        for warning in stats['warnings'][:5]:
            print_warning(f"  • {warning}")
        if len(stats['warnings']) > 5:
            print_warning(f"  ... 还有 {len(stats['warnings']) - 5} 个警告，详见报告文件")
    
    console.print()
    
    if dry_run:
        print_success("✓ 预览完成！使用 --no-dry-run 执行实际处理")
    else:
        print_success(f"✓ 标注调整完成！输出目录: {output_path}")
        
        # 后续步骤提示
        console.print()
        print_section_header("后续步骤")
        print_info("1. 检查调整后的标注是否符合预期")
        print_info("2. 使用新数据集训练模型并对比效果")
        console.print(f"   python yolo_cli.py train --data {output_path / 'data.yaml'}")


def _filter_dataset_impl(
    dataset_path: Path,
    output_path: Path,
    include_labels: Optional[str],
    exclude_labels: Optional[str],
    keep_negative: bool,
    limit_dict: Optional[dict],
    task: str
):
    """过滤数据集的实现函数
    
    Args:
        dataset_path: 数据集路径
        output_path: 输出路径
        include_labels: 包含的标签列表（逗号分隔）
        exclude_labels: 排除的标签列表（逗号分隔）
        keep_negative: 是否保留负样本
        limit_dict: 样本数量限制 {split: count or 'all'}
        task: 任务类型
    """
    
    # 1. 读取数据集配置
    print_section_header("读取数据集配置")
    
    # 检查输入是文件还是目录
    if dataset_path.is_file() and dataset_path.suffix in ['.yaml', '.yml']:
        # 输入的就是 yaml 文件
        data_yaml = dataset_path
        dataset_path = dataset_path.parent  # 更新为目录路径
        print_info(f"使用配置文件: {data_yaml.name}")
    elif dataset_path.is_dir():
        # 输入的是目录，在目录中查找 yaml 文件
        yaml_files = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
        data_yaml = None
        for yaml_file in yaml_files:
            if yaml_file.name in ['data.yaml', 'dataset.yaml']:
                data_yaml = yaml_file
                break
        
        if data_yaml is None and yaml_files:
            data_yaml = yaml_files[0]
        
        if data_yaml is None:
            print_error(f"在 {dataset_path} 中未找到 data.yaml 或 dataset.yaml")
            raise typer.Exit(1)
        
        print_info(f"找到配置文件: {data_yaml.name}")
    else:
        print_error(f"路径不存在或不是有效的文件/目录: {dataset_path}")
        raise typer.Exit(1)
    
    # 读取配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取类别信息
    if 'names' not in config:
        print_error("数据集配置中缺少 'names' 字段")
        raise typer.Exit(1)
    
    if isinstance(config['names'], dict):
        original_classes = config['names']  # {id: name}
    elif isinstance(config['names'], list):
        original_classes = {i: name for i, name in enumerate(config['names'])}
    else:
        print_error("数据集配置中的 'names' 字段格式不正确")
        raise typer.Exit(1)
    
    print_info(f"原始类别数: {len(original_classes)}")
    print_info(f"类别列表: {', '.join(original_classes.values())}")
    
    # 2. 确定要保留的类别
    print_section_header("确定过滤条件")
    
    if include_labels is not None:
        # 包含模式
        include_set = set(label.strip() for label in include_labels.split(','))
        print_info(f"包含模式: 只保留 {', '.join(include_set)}")
        
        # 检查类别是否存在
        invalid_labels = include_set - set(original_classes.values())
        if invalid_labels:
            print_warning(f"以下标签在原数据集中不存在: {', '.join(invalid_labels)}")
        
        # 创建过滤后的类别映射
        filtered_classes = {}
        class_remapping = {}  # {old_id: new_id}
        new_id = 0
        
        for old_id, class_name in original_classes.items():
            if class_name in include_set:
                filtered_classes[new_id] = class_name
                class_remapping[old_id] = new_id
                new_id += 1
    
    else:
        # 排除模式
        exclude_set = set(label.strip() for label in exclude_labels.split(','))
        print_info(f"排除模式: 移除 {', '.join(exclude_set)}")
        
        # 检查类别是否存在
        invalid_labels = exclude_set - set(original_classes.values())
        if invalid_labels:
            print_warning(f"以下标签在原数据集中不存在: {', '.join(invalid_labels)}")
        
        # 创建过滤后的类别映射
        filtered_classes = {}
        class_remapping = {}
        new_id = 0
        
        for old_id, class_name in original_classes.items():
            if class_name not in exclude_set:
                filtered_classes[new_id] = class_name
                class_remapping[old_id] = new_id
                new_id += 1
    
    if not filtered_classes:
        print_error("过滤后没有剩余类别！")
        raise typer.Exit(1)
    
    print_info(f"过滤后类别数: {len(filtered_classes)}")
    print_info(f"过滤后类别: {', '.join(filtered_classes.values())}")
    
    # 显示类别ID映射
    if any(old_id != new_id for old_id, new_id in class_remapping.items()):
        console.print()
        print_info("类别ID重映射:")
        for old_id, new_id in class_remapping.items():
            if old_id != new_id:
                print_info(f"  {original_classes[old_id]}: {old_id} → {new_id}")
    
    console.print()
    
    # 3. 创建输出目录
    for split in ['train', 'val', 'test']:
        ensure_dir(output_path / 'images' / split)
        if task != 'classify':
            ensure_dir(output_path / 'labels' / split)
    
    # 4. 过滤数据集
    print_section_header("过滤数据文件")
    
    total_images = 0
    kept_images = 0
    negative_images = 0
    filtered_annotations = 0
    total_annotations = 0
    
    for split in ['train', 'val', 'test']:
        # 查找图片目录
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        # 尝试另一种目录结构
        if not img_dir.exists():
            img_dir = dataset_path / split / 'images'
            label_dir = dataset_path / split / 'labels'
        
        # 处理 val/valid 别名
        if not img_dir.exists() and split == 'val':
            # 尝试 valid 作为 val 的别名
            img_dir = dataset_path / 'images' / 'valid'
            label_dir = dataset_path / 'labels' / 'valid'
            
            if not img_dir.exists():
                img_dir = dataset_path / 'valid' / 'images'
                label_dir = dataset_path / 'valid' / 'labels'
        
        if not img_dir.exists():
            continue
        
        # 获取所有图片
        image_files = list(find_files(img_dir, ['.jpg', '.jpeg', '.png']))
        if not image_files:
            continue
        
        total_images += len(image_files)
        split_kept = 0
        split_negative = 0
        
        # 检查该集合的样本数量限制
        split_limit = None
        if limit_dict and split in limit_dict:
            limit_value = limit_dict[split]
            if limit_value != 'all':
                split_limit = limit_value
                print_info(f"处理 {split} 数据集: {len(image_files)} 张图片 (限制: {split_limit})")
            else:
                print_info(f"处理 {split} 数据集: {len(image_files)} 张图片 (不限制)")
        else:
            print_info(f"处理 {split} 数据集: {len(image_files)} 张图片")
        
        progress = create_progress_bar()
        task_id = progress.add_task(f"[cyan]{split}", total=len(image_files))
        
        with progress:
            for img_file in image_files:
                progress.update(task_id, advance=1)
                
                # 检查是否达到样本数量限制
                if split_limit is not None and split_kept >= split_limit:
                    # 达到限制，跳过后续图片
                    continue
                
                # 读取标签文件
                label_file = label_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    # 负样本（没有标注）
                    if keep_negative:
                        # 复制图片
                        dst_img = output_path / 'images' / split / img_file.name
                        shutil.copy2(img_file, dst_img)
                        
                        # 创建空标签文件
                        if task != 'classify':
                            dst_label = output_path / 'labels' / split / f"{img_file.stem}.txt"
                            dst_label.touch()
                        
                        kept_images += 1
                        split_kept += 1
                        negative_images += 1
                    continue
                
                # 读取并过滤标签
                filtered_lines = []
                
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if not parts:
                            continue
                        
                        total_annotations += 1
                        
                        # 获取类别ID
                        try:
                            old_class_id = int(parts[0])
                        except ValueError:
                            continue
                        
                        # 检查是否在保留的类别中
                        if old_class_id in class_remapping:
                            # 重映射类别ID
                            new_class_id = class_remapping[old_class_id]
                            parts[0] = str(new_class_id)
                            filtered_lines.append(' '.join(parts))
                            filtered_annotations += 1
                
                # 如果有保留的标注，或者是负样本且需要保留
                if filtered_lines or (not filtered_lines and keep_negative):
                    # 复制图片
                    dst_img = output_path / 'images' / split / img_file.name
                    shutil.copy2(img_file, dst_img)
                    
                    # 写入过滤后的标签
                    if task != 'classify':
                        dst_label = output_path / 'labels' / split / f"{img_file.stem}.txt"
                        with open(dst_label, 'w') as f:
                            f.write('\n'.join(filtered_lines))
                            if filtered_lines:
                                f.write('\n')
                    
                    kept_images += 1
                    split_kept += 1
                    
                    if not filtered_lines:
                        negative_images += 1
        
        print_info(f"  保留: {split_kept} 张图片")
    
    # 5. 生成新的 data.yaml
    print_section_header("生成配置文件")
    
    new_config = {
        'path': '.',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(filtered_classes),
        'names': filtered_classes
    }
    
    # 如果是 pose 任务，保留关键点配置
    if task == 'pose' and 'kpt_shape' in config:
        new_config['kpt_shape'] = config['kpt_shape']
        if 'flip_idx' in config:
            new_config['flip_idx'] = config['flip_idx']
        if 'keypoint_names' in config:
            new_config['keypoint_names'] = config['keypoint_names']
    
    output_yaml = output_path / 'data.yaml'
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"配置文件已生成: {output_yaml}")
    
    # 6. 打印统计信息
    console.print()
    print_section_header("过滤统计")
    
    print_key_value("原始图片数", total_images)
    print_key_value("保留图片数", kept_images)
    print_key_value("保留比例", f"{kept_images/total_images*100:.1f}%" if total_images > 0 else "0%")
    print_key_value("负样本数", negative_images)
    
    console.print()
    print_key_value("原始标注数", total_annotations)
    print_key_value("保留标注数", filtered_annotations)
    print_key_value("保留比例", f"{filtered_annotations/total_annotations*100:.1f}%" if total_annotations > 0 else "0%")
    
    console.print()
    print_key_value("原始类别数", len(original_classes))
    print_key_value("过滤后类别数", len(filtered_classes))
    
    console.print()
    print_success(f"✓ 数据集过滤完成！输出目录: {output_path}")
    
    # 后续步骤提示
    console.print()
    print_section_header("后续步骤")
    print_info("1. 验证过滤后的数据集")
    print_info(f"   python yolo_cli.py data verify --path {output_path}")
    print_info("2. 查看数据统计")
    print_info(f"   python yolo_cli.py data stats --path {output_path} --detailed")
    print_info("3. 使用过滤后的数据集训练")
    print_info(f"   python yolo_cli.py train start --data {output_path}/data.yaml")


def _convert_format_impl(
    dataset_path: Path,
    output_path: Path,
    source_format: str,
    target_format: str,
    bbox_expand: float,
    keep_confidence: bool,
    preserve_structure: bool
):
    """格式转换的实现函数"""
    
    # 1. 读取数据集配置
    print_section_header("读取数据集配置")
    
    # 检查输入是文件还是目录
    if dataset_path.is_file() and dataset_path.suffix in ['.yaml', '.yml']:
        data_yaml = dataset_path
        dataset_path = dataset_path.parent
        print_info(f"使用配置文件: {data_yaml.name}")
    elif dataset_path.is_dir():
        yaml_files = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
        data_yaml = None
        for yaml_file in yaml_files:
            if yaml_file.name in ['data.yaml', 'dataset.yaml']:
                data_yaml = yaml_file
                break
        
        if data_yaml is None and yaml_files:
            data_yaml = yaml_files[0]
        
        if data_yaml is None:
            print_error(f"在 {dataset_path} 中未找到 data.yaml 或 dataset.yaml")
            raise typer.Exit(1)
        
        print_info(f"找到配置文件: {data_yaml.name}")
    else:
        print_error(f"路径不存在或不是有效的文件/目录: {dataset_path}")
        raise typer.Exit(1)
    
    # 读取配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取类别信息
    if 'names' not in config:
        print_error("数据集配置中缺少 'names' 字段")
        raise typer.Exit(1)
    
    if isinstance(config['names'], dict):
        classes = config['names']
    elif isinstance(config['names'], list):
        classes = {i: name for i, name in enumerate(config['names'])}
    else:
        print_error("数据集配置中的 'names' 字段格式不正确")
        raise typer.Exit(1)
    
    print_info(f"类别数: {len(classes)}")
    print_info(f"类别列表: {', '.join(classes.values())}")
    
    # 2. 确定转换策略
    console.print()
    print_section_header("转换配置")
    
    conversion_name = f"{source_format} → {target_format}"
    print_info(f"转换方向: {conversion_name}")
    
    if bbox_expand > 0:
        print_info(f"边界框扩展: {bbox_expand*100:.1f}%")
    
    # 显示转换说明
    warnings = []
    if source_format == 'segment' and target_format == 'detect':
        print_info("✓ 转换质量: 优秀（无精度损失）")
        print_info("  方法: 计算多边形的最小外接矩形")
    elif source_format == 'pose' and target_format == 'detect':
        print_info("✓ 转换质量: 良好（保留边界框）")
        print_info("  方法: 提取前5个值（bbox），丢弃关键点")
    elif source_format == 'detect' and target_format == 'segment':
        print_warning("⚠️  转换质量: 一般（精度较低）")
        print_warning("  方法: 将矩形框转为4点多边形")
        warnings.append("检测→分割转换精度有限，仅适合特殊场景")
    elif source_format == 'detect' and target_format == 'pose':
        print_warning("⚠️  转换质量: 一般（无关键点信息）")
        print_warning("  方法: 保留bbox，添加默认关键点（visibility=0）")
        warnings.append("检测→Pose转换无法生成真实关键点，需要后续标注")
    elif source_format == 'segment' and target_format == 'pose':
        print_warning("⚠️  转换质量: 一般（无关键点信息）")
        print_warning("  方法: 从多边形计算bbox，添加默认关键点")
        warnings.append("分割→Pose转换无法生成真实关键点，需要后续标注")
    elif source_format == 'pose' and target_format == 'segment':
        print_warning("⚠️  转换质量: 一般（使用bbox作为矩形）")
        print_warning("  方法: 将bbox转为4点多边形，丢弃关键点")
        warnings.append("Pose→分割转换丢失关键点信息，仅保留bbox")
    
    # 3. 创建输出目录
    console.print()
    for split in ['train', 'val', 'test']:
        ensure_dir(output_path / 'images' / split)
        ensure_dir(output_path / 'labels' / split)
    
    # 4. 转换数据集
    print_section_header("转换标注文件")
    
    total_images = 0
    converted_labels = 0
    skipped_labels = 0
    
    for split in ['train', 'val', 'test']:
        # 查找目录
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        # 尝试另一种结构
        if not img_dir.exists():
            img_dir = dataset_path / split / 'images'
            label_dir = dataset_path / split / 'labels'
        
        # 处理 val/valid别名
        if not img_dir.exists() and split == 'val':
            img_dir = dataset_path / 'images' / 'valid'
            label_dir = dataset_path / 'labels' / 'valid'
            if not img_dir.exists():
                img_dir = dataset_path / 'valid' / 'images'
                label_dir = dataset_path / 'valid' / 'labels'
        
        if not img_dir.exists():
            continue
        
        # 获取所有图片
        image_files = list(find_files(img_dir, ['.jpg', '.jpeg', '.png']))
        if not image_files:
            continue
        
        total_images += len(image_files)
        print_info(f"处理 {split}: {len(image_files)} 张图片")
        
        progress = create_progress_bar()
        task_id = progress.add_task(f"[cyan]{split}", total=len(image_files))
        
        with progress:
            for img_file in image_files:
                progress.update(task_id, advance=1)
                
                # 复制图片
                dst_img = output_path / 'images' / split / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # 处理标签
                label_file = label_dir / f"{img_file.stem}.txt"
                dst_label = output_path / 'labels' / split / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    # 创建空标签
                    dst_label.touch()
                    skipped_labels += 1
                    continue
                
                # 转换标签
                with open(label_file, 'r') as f_in, open(dst_label, 'w') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if not parts:
                            continue
                        
                        # 根据转换方向调用相应函数
                        if source_format == 'segment' and target_format == 'detect':
                            converted_line = _convert_segment_to_detect(parts, bbox_expand)
                        elif source_format == 'pose' and target_format == 'detect':
                            converted_line = _convert_pose_to_detect(parts, bbox_expand)
                        elif source_format == 'detect' and target_format == 'segment':
                            converted_line = _convert_detect_to_segment(parts)
                        elif source_format == 'detect' and target_format == 'pose':
                            # 获取关键点数量（如果有配置）
                            num_kpts = 17  # 默认COCO
                            if 'kpt_shape' in config:
                                kpt_shape = config['kpt_shape']
                                num_kpts = kpt_shape[0] if isinstance(kpt_shape, list) else kpt_shape
                            converted_line = _convert_detect_to_pose(parts, num_kpts)
                        elif source_format == 'segment' and target_format == 'pose':
                            num_kpts = 17
                            if 'kpt_shape' in config:
                                kpt_shape = config['kpt_shape']
                                num_kpts = kpt_shape[0] if isinstance(kpt_shape, list) else kpt_shape
                            converted_line = _convert_segment_to_pose(parts, num_kpts)
                        elif source_format == 'pose' and target_format == 'segment':
                            converted_line = _convert_pose_to_segment(parts)
                        else:
                            converted_line = ' '.join(parts)
                        
                        f_out.write(converted_line + '\n')
                        converted_labels += 1
    
    # 5. 生成新的 data.yaml
    console.print()
    print_section_header("生成配置文件")
    
    new_config = {
        'path': '.',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    # 如果是pose任务，添加关键点配置
    if target_format == 'pose':
        if 'kpt_shape' in config:
            new_config['kpt_shape'] = config['kpt_shape']
        else:
            new_config['kpt_shape'] = [17, 3]  # 默认COCO
        
        if 'flip_idx' in config:
            new_config['flip_idx'] = config['flip_idx']
        
        if 'keypoint_names' in config:
            new_config['keypoint_names'] = config['keypoint_names']
    
    output_yaml = output_path / 'data.yaml'
    with open(output_yaml, 'w', encoding='utf-8') as f:
        f.write(f"# YOLO {target_format.upper()} 数据集配置\n")
        f.write(f"# 从 {source_format} 格式转换而来\n\n")
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"配置文件已生成: {output_yaml}")
    
    # 6. 显示统计信息
    console.print()
    print_section_header("转换完成")
    
    print_key_value("总图片数", total_images)
    print_key_value("转换标注数", converted_labels)
    print_key_value("空标签数", skipped_labels)
    print_key_value("转换方向", conversion_name)
    if bbox_expand > 0:
        print_key_value("边界框扩展", f"{bbox_expand*100:.1f}%")
    
    # 显示警告
    if warnings:
        console.print()
        print_warning("⚠️  注意事项:")
        for warning in warnings:
            print_warning(f"   • {warning}")
    
    console.print()
    print_success(f"✓ 转换完成！输出目录: {output_path}")
    
    # 后续步骤
    console.print()
    print_section_header("后续步骤")
    print_info("1. 验证转换后的数据集")
    print_info(f"   python yolo_cli.py data verify --path {output_path}")
    print_info("2. 查看数据统计")
    print_info(f"   python yolo_cli.py data stats --path {output_path} --detailed --task {target_format}")
    print_info("3. 使用FiftyOne可视化检查")
    print_info(f"   python yolo_cli.py interactive-mode → fiftyone → load")
    print_info("4. 开始训练")
    print_info(f"   python yolo_cli.py train start --data {output_path}/data.yaml")


def _deduplicate_dataset_impl(
    dataset_path: Path,
    mode: str,
    action: str,
    priority_list: list,
    threshold: float,
    cross_split: bool
):
    """数据集去重的实现函数"""
    
    from ..core.deduplicator import ImageDeduplicator
    
    # 1. 扫描所有图片
    print_section_header("扫描数据集")
    
    split_images = {}  # {split_name: [image_paths]}
    split_labels = {}  # {split_name: [label_paths]}
    
    # 检查可能的目录结构
    for split in ['train', 'val', 'valid', 'test']:
        images = []
        labels = []
        
        # 尝试 images/split 结构
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        # 尝试 split/images 结构
        if not img_dir.exists():
            img_dir = dataset_path / split / 'images'
            label_dir = dataset_path / split / 'labels'
        
        # 尝试直接 split 目录
        if not img_dir.exists():
            img_dir = dataset_path / split
            label_dir = dataset_path / split.replace('images', 'labels')
        
        if img_dir.exists():
            # 收集图片
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.extend(list(img_dir.glob(f'*{ext}')))
                images.extend(list(img_dir.glob(f'*{ext.upper()}')))
            
            # 收集对应的标签文件
            if label_dir.exists():
                for img_file in images:
                    label_file = label_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        labels.append(label_file)
                    else:
                        labels.append(None)  # 标记为无标签
            
            if images:
                # 统一 valid 为 val
                split_name = 'val' if split == 'valid' else split
                split_images[split_name] = images
                split_labels[split_name] = labels
                print_info(f"{split_name}: {len(images)} 张图片")
    
    if not split_images:
        print_error("未找到任何图片")
        print_info("支持的目录结构:")
        print_info("  • images/train, images/val, images/test")
        print_info("  • train/images, val/images, test/images")
        print_info("  • train/, val/, test/")
        raise typer.Exit(1)
    
    total_images = sum(len(imgs) for imgs in split_images.values())
    print_info(f"\n总图片数: {total_images}")
    console.print()
    
    # 2. 检测重复
    print_section_header("检测重复图片")
    
    print_info(f"去重模式: {mode}")
    print_info(f"相似度阈值: {threshold}")
    if cross_split:
        print_info("跨集合去重: 是（train/val/test 之间）")
    else:
        print_info("跨集合去重: 否（仅在各集合内部）")
    console.print()
    
    # 构建图片到集合的映射
    img_to_split = {}
    all_images = []
    
    if cross_split:
        # 跨集合去重：所有图片放在一起
        for split_name, images in split_images.items():
            for img in images:
                img_to_split[str(img)] = split_name
                all_images.append(img)
        
        print_info("正在计算图片哈希值...")
        
        if mode == 'hash':
            # 使用 MD5 哈希（完全相同检测）
            deduplicator = ImageDeduplicator(mode='exact')
            duplicates = deduplicator.find_duplicates(all_images)
        elif mode == 'perceptual':
            # 使用感知哈希（相似图片检测）
            # 将阈值从 0-1 转换为汉明距离（0-64）
            hamming_threshold = int((1 - threshold) * 64)
            deduplicator = ImageDeduplicator(mode='similar', similarity_threshold=hamming_threshold)
            duplicates = deduplicator.find_duplicates(all_images)
        else:  # both
            # 先用哈希检测完全相同的
            deduplicator_hash = ImageDeduplicator(mode='exact')
            hash_dups = deduplicator_hash.find_duplicates(all_images)
            
            # 再用感知哈希检测相似的
            hamming_threshold = int((1 - threshold) * 64)
            deduplicator_perceptual = ImageDeduplicator(mode='similar', similarity_threshold=hamming_threshold)
            perceptual_dups = deduplicator_perceptual.find_duplicates(all_images)
            
            # 合并结果
            duplicates = hash_dups.copy()
            for key, files in perceptual_dups.items():
                if key not in duplicates:
                    duplicates[key] = files
                else:
                    # 合并文件列表
                    existing = set(duplicates[key])
                    for f in files:
                        if f not in existing:
                            duplicates[key].append(f)
    else:
        # 仅在各集合内部去重
        duplicates = {}
        for split_name, images in split_images.items():
            print_info(f"正在检测 {split_name} 集合...")
            
            if mode == 'hash':
                deduplicator = ImageDeduplicator(mode='exact')
                split_dups = deduplicator.find_duplicates(images)
            elif mode == 'perceptual':
                hamming_threshold = int((1 - threshold) * 64)
                deduplicator = ImageDeduplicator(mode='similar', similarity_threshold=hamming_threshold)
                split_dups = deduplicator.find_duplicates(images)
            else:  # both
                # 哈希去重
                deduplicator_hash = ImageDeduplicator(mode='exact')
                hash_dups = deduplicator_hash.find_duplicates(images)
                
                # 感知哈希去重
                hamming_threshold = int((1 - threshold) * 64)
                deduplicator_perceptual = ImageDeduplicator(mode='similar', similarity_threshold=hamming_threshold)
                perceptual_dups = deduplicator_perceptual.find_duplicates(images)
                
                split_dups = hash_dups.copy()
                for key, files in perceptual_dups.items():
                    if key not in split_dups:
                        split_dups[key] = files
                    else:
                        existing = set(split_dups[key])
                        for f in files:
                            if f not in existing:
                                split_dups[key].append(f)
            
            duplicates.update(split_dups)
            for img in images:
                img_to_split[str(img)] = split_name
    
    # 3. 分析重复结果
    console.print()
    print_section_header("重复分析")
    
    duplicate_groups = []
    total_duplicates = 0
    
    for key, files in duplicates.items():
        if len(files) > 1:
            duplicate_groups.append(files)
            total_duplicates += len(files) - 1  # 保留一个，其余都是重复
    
    if not duplicate_groups:
        print_success("✓ 未发现重复图片！")
        return
    
    print_warning(f"发现 {len(duplicate_groups)} 组重复图片")
    print_warning(f"重复图片总数: {total_duplicates} 张")
    console.print()
    
    # 显示详细的重复信息
    print_info("重复详情:")
    for i, group in enumerate(duplicate_groups[:10], 1):  # 只显示前10组
        splits_info = {}
        for img_path in group:
            split = img_to_split.get(str(img_path), 'unknown')
            if split not in splits_info:
                splits_info[split] = []
            splits_info[split].append(Path(img_path).name)
        
        console.print(f"\n  [bold]组 {i}:[/bold] {len(group)} 张重复")
        for split, names in splits_info.items():
            console.print(f"    {split}: {', '.join(names[:3])}" + 
                         (f" ... (+{len(names)-3})" if len(names) > 3 else ""))
    
    if len(duplicate_groups) > 10:
        console.print(f"\n  ... 还有 {len(duplicate_groups) - 10} 组重复未显示")
    
    console.print()
    
    # 4. 根据优先级决定要保留/删除的文件
    files_to_remove = []  # [(img_path, label_path, split_name)]
    files_to_keep = []    # [(img_path, split_name)]
    
    for group in duplicate_groups:
        # 按优先级排序
        sorted_group = sorted(group, key=lambda x: (
            priority_list.index(img_to_split.get(str(x), 'unknown')) 
            if img_to_split.get(str(x), 'unknown') in priority_list 
            else 999,
            str(x)  # 相同优先级时按路径排序
        ))
        
        # 保留第一个，删除其余
        keep_file = sorted_group[0]
        keep_split = img_to_split.get(str(keep_file), 'unknown')
        files_to_keep.append((keep_file, keep_split))
        
        for dup_file in sorted_group[1:]:
            dup_split = img_to_split.get(str(dup_file), 'unknown')
            # 找到对应的标签文件
            split_idx = split_images[dup_split].index(Path(dup_file))
            label_file = split_labels[dup_split][split_idx] if split_idx < len(split_labels[dup_split]) else None
            
            files_to_remove.append((Path(dup_file), label_file, dup_split))
    
    # 5. 执行操作
    if action == 'report':
        print_section_header("去重报告（仅报告模式）")
        
        print_info(f"保留优先级: {' > '.join(priority_list)}")
        console.print()
        
        # 按集合统计
        remove_by_split = {}
        for img_path, label_path, split in files_to_remove:
            if split not in remove_by_split:
                remove_by_split[split] = []
            remove_by_split[split].append((img_path, label_path))
        
        print_info("将要删除的文件:")
        for split in ['train', 'val', 'test']:
            if split in remove_by_split:
                count = len(remove_by_split[split])
                print_warning(f"  {split}: {count} 张图片 + 标签")
        
        console.print()
        print_info("💡 使用 --action delete 删除重复文件")
        print_info("💡 使用 --action move 移动重复文件到 duplicates 目录")
    
    elif action == 'delete':
        from ..ui.prompts import confirm_action
        
        print_section_header("删除重复文件")
        
        print_warning(f"即将删除 {len(files_to_remove)} 张重复图片及其标签")
        console.print()
        
        if not confirm_action("确认删除?", default=False):
            print_info("已取消删除操作")
            return
        
        deleted_count = 0
        progress = create_progress_bar()
        task_id = progress.add_task("[red]删除中", total=len(files_to_remove))
        
        with progress:
            for img_path, label_path, split in files_to_remove:
                try:
                    # 删除图片
                    if img_path.exists():
                        img_path.unlink()
                        deleted_count += 1
                    
                    # 删除标签
                    if label_path and label_path.exists():
                        label_path.unlink()
                    
                    progress.update(task_id, advance=1)
                except Exception as e:
                    print_error(f"删除失败 {img_path.name}: {e}")
        
        console.print()
        print_success(f"✓ 已删除 {deleted_count} 张重复图片")
        
    elif action == 'move':
        print_section_header("移动重复文件")
        
        # 创建 duplicates 目录
        duplicates_dir = dataset_path / 'duplicates'
        ensure_dir(duplicates_dir)
        
        for split in split_images.keys():
            ensure_dir(duplicates_dir / split / 'images')
            ensure_dir(duplicates_dir / split / 'labels')
        
        moved_count = 0
        progress = create_progress_bar()
        task_id = progress.add_task("[yellow]移动中", total=len(files_to_remove))
        
        with progress:
            for img_path, label_path, split in files_to_remove:
                try:
                    # 移动图片
                    dst_img = duplicates_dir / split / 'images' / img_path.name
                    if img_path.exists():
                        shutil.move(str(img_path), str(dst_img))
                        moved_count += 1
                    
                    # 移动标签
                    if label_path and label_path.exists():
                        dst_label = duplicates_dir / split / 'labels' / label_path.name
                        shutil.move(str(label_path), str(dst_label))
                    
                    progress.update(task_id, advance=1)
                except Exception as e:
                    print_error(f"移动失败 {img_path.name}: {e}")
        
        console.print()
        print_success(f"✓ 已移动 {moved_count} 张重复图片到: {duplicates_dir}")
    
    # 6. 显示最终统计
    console.print()
    print_section_header("去重统计")
    
    print_key_value("原始图片总数", total_images)
    print_key_value("重复组数", len(duplicate_groups))
    print_key_value("重复图片数", total_duplicates)
    
    if action != 'report':
        remaining = total_images - len(files_to_remove)
        print_key_value("剩余图片数", remaining)
        print_key_value("去重率", f"{len(files_to_remove)/total_images*100:.1f}%")


def _merge_labels_impl(
    dataset_path: Path,
    output_path: Path,
    mapping_str: Optional[str],
    task: str
):
    """合并类别标签的实现函数"""
    
    # 1. 读取数据集配置
    print_section_header("读取数据集配置")
    
    # 检查输入是文件还是目录
    if dataset_path.is_file() and dataset_path.suffix in ['.yaml', '.yml']:
        data_yaml = dataset_path
        dataset_path = dataset_path.parent
        print_info(f"使用配置文件: {data_yaml.name}")
    elif dataset_path.is_dir():
        yaml_files = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
        data_yaml = None
        for yaml_file in yaml_files:
            if yaml_file.name in ['data.yaml', 'dataset.yaml']:
                data_yaml = yaml_file
                break
        
        if data_yaml is None and yaml_files:
            data_yaml = yaml_files[0]
        
        if data_yaml is None:
            print_error(f"在 {dataset_path} 中未找到 data.yaml 或 dataset.yaml")
            raise typer.Exit(1)
        
        print_info(f"找到配置文件: {data_yaml.name}")
    else:
        print_error(f"路径不存在或不是有效的文件/目录: {dataset_path}")
        raise typer.Exit(1)
    
    # 读取配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取类别信息
    if 'names' not in config:
        print_error("数据集配置中缺少 'names' 字段")
        raise typer.Exit(1)
    
    if isinstance(config['names'], dict):
        original_classes = config['names']  # {id: name}
    elif isinstance(config['names'], list):
        original_classes = {i: name for i, name in enumerate(config['names'])}
    else:
        print_error("数据集配置中的 'names' 字段格式不正确")
        raise typer.Exit(1)
    
    print_info(f"原始类别数: {len(original_classes)}")
    print_info(f"类别列表: {', '.join(original_classes.values())}")
    
    # 2. 解析或配置映射规则
    console.print()
    print_section_header("配置合并规则")
    
    label_mapping = {}  # {old_name: new_name}
    
    if mapping_str:
        # 解析命令行映射规则
        # 格式: "source1,source2:target;source3:target2"
        try:
            rules = mapping_str.split(';')
            for rule in rules:
                if ':' not in rule:
                    print_error(f"映射规则格式错误: {rule}")
                    print_info("正确格式: 'source1,source2:target'")
                    raise typer.Exit(1)
                
                sources, target = rule.split(':', 1)
                source_labels = [s.strip() for s in sources.split(',')]
                target_label = target.strip()
                
                # 验证源标签存在
                for src in source_labels:
                    if src not in original_classes.values():
                        print_error(f"源标签不存在: {src}")
                        raise typer.Exit(1)
                    label_mapping[src] = target_label
                
                print_info(f"合并规则: {', '.join(source_labels)} → {target_label}")
        
        except ValueError:
            print_error("映射规则格式错误")
            print_info("格式: 'source1,source2:target;source3,source4:target2'")
            raise typer.Exit(1)
    else:
        # 交互式配置
        from ..ui.prompts import confirm_action, input_text, select_multiple
        
        print_info("💡 交互式配置类别合并规则")
        print_info("   可以将多个类别合并为一个")
        console.print()
        
        # 显示所有类别
        class_list = list(original_classes.values())
        print_info(f"可用类别: {', '.join(class_list)}")
        console.print()
        
        # 循环添加合并规则
        while True:
            console.print()
            if not confirm_action("添加一个合并规则?", default=True):
                break
            
            # 选择要合并的源类别（多选）
            selected_sources = select_multiple(
                "选择要合并的源类别 (空格选择，回车确认):",
                class_list
            )
            
            if not selected_sources:
                print_warning("未选择任何类别")
                continue
            
            if len(selected_sources) < 2:
                print_warning("至少需要选择2个类别进行合并")
                if not confirm_action("重新选择?", default=True):
                    continue
            
            # 输入目标类别名称
            target_label = input_text(
                "输入合并后的类别名称:",
                default=selected_sources[0]
            )
            
            if not target_label:
                print_warning("未输入目标类别名称")
                continue
            
            # 保存映射规则
            for src in selected_sources:
                label_mapping[src] = target_label
            
            print_success(f"✓ 已添加: {', '.join(selected_sources)} → {target_label}")
        
        if not label_mapping:
            print_warning("未配置任何合并规则")
            raise typer.Exit(0)
    
    # 3. 构建新的类别列表
    console.print()
    print_section_header("生成新类别列表")
    
    # 收集所有类别（合并后）
    new_class_names = set()
    for class_name in original_classes.values():
        if class_name in label_mapping:
            new_class_names.add(label_mapping[class_name])
        else:
            new_class_names.add(class_name)
    
    # 排序并分配ID
    new_class_names = sorted(list(new_class_names))
    new_classes = {i: name for i, name in enumerate(new_class_names)}
    
    print_info(f"合并后类别数: {len(new_classes)}")
    print_info(f"类别列表: {', '.join(new_classes.values())}")
    
    # 创建ID映射表
    class_id_mapping = {}  # {old_id: new_id}
    for old_id, old_name in original_classes.items():
        # 确定新名称
        new_name = label_mapping.get(old_name, old_name)
        # 找到新ID
        new_id = next(i for i, name in new_classes.items() if name == new_name)
        class_id_mapping[old_id] = new_id
    
    # 显示映射关系
    console.print()
    print_info("类别ID映射:")
    for old_id, old_name in original_classes.items():
        new_id = class_id_mapping[old_id]
        new_name = new_classes[new_id]
        if old_name != new_name or old_id != new_id:
            if old_name == new_name:
                print_info(f"  {old_name}: ID {old_id} → {new_id}")
            else:
                print_info(f"  {old_name} → {new_name}: ID {old_id} → {new_id}")
    
    console.print()
    
    # 4. 创建输出目录
    for split in ['train', 'val', 'test']:
        ensure_dir(output_path / 'images' / split)
        if task != 'classify':
            ensure_dir(output_path / 'labels' / split)
    
    # 5. 处理数据集
    print_section_header("处理数据文件")
    
    total_images = 0
    processed_labels = 0
    merged_annotations = 0
    
    for split in ['train', 'val', 'test']:
        # 查找图片目录
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        # 尝试另一种目录结构
        if not img_dir.exists():
            img_dir = dataset_path / split / 'images'
            label_dir = dataset_path / split / 'labels'
        
        # 处理 val/valid 别名
        if not img_dir.exists() and split == 'val':
            img_dir = dataset_path / 'images' / 'valid'
            label_dir = dataset_path / 'labels' / 'valid'
            if not img_dir.exists():
                img_dir = dataset_path / 'valid' / 'images'
                label_dir = dataset_path / 'valid' / 'labels'
        
        if not img_dir.exists():
            continue
        
        # 获取所有图片
        image_files = list(find_files(img_dir, ['.jpg', '.jpeg', '.png']))
        if not image_files:
            continue
        
        total_images += len(image_files)
        print_info(f"处理 {split} 数据集: {len(image_files)} 张图片")
        
        progress = create_progress_bar()
        task_id = progress.add_task(f"[cyan]{split}", total=len(image_files))
        
        with progress:
            for img_file in image_files:
                progress.update(task_id, advance=1)
                
                # 复制图片
                dst_img = output_path / 'images' / split / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # 处理标签文件
                if task != 'classify':
                    label_file = label_dir / f"{img_file.stem}.txt"
                    dst_label = output_path / 'labels' / split / f"{img_file.stem}.txt"
                    
                    if not label_file.exists():
                        # 创建空标签
                        dst_label.touch()
                        continue
                    
                    # 读取并更新标签
                    label_lines = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) >= 1:
                                try:
                                    old_class_id = int(parts[0])
                                    # 映射到新的类别ID
                                    new_class_id = class_id_mapping.get(old_class_id, old_class_id)
                                    
                                    # 检查是否发生了合并
                                    old_name = original_classes.get(old_class_id, '')
                                    new_name = new_classes.get(new_class_id, '')
                                    if old_name != new_name:
                                        merged_annotations += 1
                                    
                                    parts[0] = str(new_class_id)
                                    label_lines.append(' '.join(parts))
                                    processed_labels += 1
                                except (ValueError, KeyError):
                                    # 保持原样
                                    label_lines.append(' '.join(parts))
                    
                    # 写入新标签文件
                    with open(dst_label, 'w') as f:
                        f.write('\n'.join(label_lines))
                        if label_lines:
                            f.write('\n')
    
    # 6. 生成新的 data.yaml
    console.print()
    print_section_header("生成配置文件")
    
    new_config = {
        'path': '.',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(new_classes),
        'names': new_classes
    }
    
    # 如果是 pose 任务，保留关键点配置
    if task == 'pose' and 'kpt_shape' in config:
        new_config['kpt_shape'] = config['kpt_shape']
        if 'flip_idx' in config:
            new_config['flip_idx'] = config['flip_idx']
        if 'keypoint_names' in config:
            new_config['keypoint_names'] = config['keypoint_names']
    
    output_yaml = output_path / 'data.yaml'
    with open(output_yaml, 'w', encoding='utf-8') as f:
        f.write("# YOLO 数据集配置文件\n")
        f.write("# 类别已合并\n\n")
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"配置文件已生成: {output_yaml}")
    
    # 7. 显示统计信息
    console.print()
    print_section_header("合并统计")
    
    print_key_value("总图片数", total_images)
    print_key_value("处理标注数", processed_labels)
    print_key_value("合并标注数", merged_annotations)
    
    console.print()
    print_key_value("原始类别数", len(original_classes))
    print_key_value("合并后类别数", len(new_classes))
    print_key_value("减少类别", len(original_classes) - len(new_classes))
    
    # 显示合并详情
    console.print()
    print_info("合并详情:")
    merge_groups = {}
    for old_name, new_name in label_mapping.items():
        if new_name not in merge_groups:
            merge_groups[new_name] = []
        merge_groups[new_name].append(old_name)
    
    for new_name, old_names in merge_groups.items():
        if len(old_names) > 1:
            print_info(f"  {new_name} ← {', '.join(old_names)}")
    
    console.print()
    print_success(f"✓ 类别合并完成！输出目录: {output_path}")
    
    # 后续步骤
    console.print()
    print_section_header("后续步骤")
    print_info("1. 验证合并后的数据集")
    print_info(f"   python yolo_cli.py data verify --path {output_path}")
    print_info("2. 查看数据统计")
    print_info(f"   python yolo_cli.py data stats --path {output_path} --detailed")
    print_info("3. 使用FiftyOne可视化检查")
    print_info(f"   python yolo_cli.py interactive-mode → fiftyone → load")
    print_info("4. 使用合并后的数据集训练")
    print_info(f"   python yolo_cli.py train start --data {output_path}/data.yaml")


if __name__ == "__main__":
    app()
