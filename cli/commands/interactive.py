#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式模式"""

import typer
from pathlib import Path

from ..ui.prompts import (
    select_main_menu, select_model_operation, select_data_operation,
    select_train_operation, select_detect_operation, select_validate_operation,
    select_validation_split, select_option, select_multiple,
    select_yolo_version, select_task_type, select_model_size, select_device,
    select_augmentation_preset, select_optimizer, select_export_formats,
    build_training_config, input_text, input_path, input_number,
    confirm_action,
    select_labelstudio_operation, select_labelstudio_project, input_labelstudio_config,
    select_fiftyone_operation, select_fiftyone_dataset,
    select_task_type_for_predict
)
from ..ui.display import (
    print_logo, clear_screen, print_section_header,
    print_success, print_error, print_info, print_warning,
    console
)
from ..core.config import ConfigManager
from ..core.version import YOLOVersionManager

# 导入命令函数
from . import model, data, train, detect, validate

app = typer.Typer(help="交互式模式")


def run_model_operations():
    """模型管理操作"""
    while True:
        operation = select_model_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'download':
                # 下载模型
                print_section_header("下载预训练模型")
                
                version = select_yolo_version()
                task_type = select_task_type()
                size_str = select_model_size()
                
                sizes = [size_str] if size_str != 'all' else None
                download_all = (size_str == 'all')
                
                if confirm_action(f"确认下载 {version} {task_type} {size_str if not download_all else '所有'} 模型?"):
                    from ..commands.model import download
                    download(version=version, size=sizes, task=task_type, all=download_all, output_dir=None)
            
            elif operation == 'export':
                # 导出模型
                print_section_header("导出模型")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                formats = select_export_formats()
                
                if formats and confirm_action(f"确认导出为 {', '.join(formats)} 格式?"):
                    from ..commands.model import export
                    export(model=model_path, formats=formats, imgsz=640, device='auto', output_dir=None)
            
            elif operation == 'list':
                # 列出模型
                from ..commands.model import list_models
                list_models(directory=None, version=None)
            
            console.print()
            if not confirm_action("继续模型操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def check_and_clear_directory(directory: str) -> bool:
    """检查目录是否为空，如果不为空询问是否清空
    
    Args:
        directory: 目录路径
        
    Returns:
        bool: True 表示可以继续（目录为空或已清空），False 表示用户取消
    """
    from pathlib import Path
    import shutil
    
    dir_path = Path(directory)
    if not dir_path.exists():
        return True
    
    # 检查目录内容（忽略隐藏文件）
    contents = [f for f in dir_path.iterdir() if not f.name.startswith('.')]
    
    if not contents:
        return True
    
    # 目录不为空，询问用户
    console.print()
    print_warning(f"{directory} 目录不为空，包含以下内容:")
    for item in contents[:10]:
        print_info(f"  • {item.name}")
    if len(contents) > 10:
        print_info(f"  ... 还有 {len(contents) - 10} 个文件/目录")
    
    console.print()
    if not confirm_action("是否清空该目录并继续?", default=False):
        print_warning("已取消操作")
        return False
    
    # 清空目录
    print_info("正在清空目录...")
    for item in contents:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print_error(f"删除 {item.name} 失败: {e}")
            return False
    
    print_success("✓ 目录已清空")
    console.print()
    return True


def run_data_operations():
    """数据处理操作"""
    while True:
        operation = select_data_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'convert-labelstudio':
                # 转换Label Studio数据
                print_section_header("转换Label Studio数据")
                
                input_file = input_path("Label Studio导出文件 (JSON/CSV):", default="labelstudioexport/project.json", must_exist=False)
                url = input_text("Label Studio URL:", default="http://localhost:8080")
                token = input_text("访问令牌 (支持Refresh Token):")
                task_type = select_task_type()
                max_workers = input_number("并发下载线程数:", default=4, min_value=1, max_value=16)
                
                # 对于检测和Pose任务，询问是否包含负样本
                include_negative = True
                if task_type in ['detect', 'pose']:
                    console.print()
                    print_info("💡 负样本说明：")
                    print_info("   - 无标注的图片可作为负样本训练")
                    print_info("   - 有助于减少误报，提高模型鲁棒性")
                    print_info("   - 推荐包含10-20%的负样本")
                    console.print()
                    include_negative = confirm_action("包含无标注图片作为负样本?", default=True)
                
                # 数据筛选选项 🆕
                console.print()
                print_info("📊 部分数据集下载选项（可选）：")
                print_info("   - 限制下载数量（适合快速测试）")
                print_info("   - 指定任务ID或范围")
                print_info("   - 按标签筛选")
                console.print()
                
                use_filters = confirm_action("是否使用数据筛选?", default=False)
                
                max_tasks = None
                task_ids = None
                task_range = None
                filter_labels = None
                
                if use_filters:
                    console.print()
                    filter_type = select_option(
                        "选择筛选方式:",
                        [
                            "限制下载数量 (如: 前50个任务)",
                            "指定任务ID列表 (如: 100,200,300)",
                            "指定任务ID范围 (如: 100-500)",
                            "按标签筛选 (如: person,car)",
                        ]
                    )
                    
                    if "限制下载数量" in filter_type:
                        max_tasks = input_number("最大下载任务数:", default=50, min_value=1)
                        print_info(f"将下载前 {max_tasks} 个任务")
                    elif "任务ID列表" in filter_type:
                        task_ids = input_text("任务ID列表（逗号分隔）:", default="100,200,300")
                        print_info(f"将下载指定ID的任务: {task_ids}")
                    elif "任务ID范围" in filter_type:
                        from ..ui.prompts import input_task_range
                        start_id, end_id = input_task_range()
                        task_range = [start_id, end_id]  # 改为列表
                        print_info(f"将下载ID范围内的任务: {start_id}-{end_id}")
                    elif "按标签筛选" in filter_type:
                        filter_labels = input_text("标签列表（逗号分隔）:", default="person,car")
                        print_info(f"将下载包含这些标签的任务: {filter_labels}")
                
                if confirm_action(f"确认转换 {task_type} 数据?"):
                    from ..commands.data import convert_labelstudio
                    convert_labelstudio(
                        input_file=input_file,
                        url=url,
                        token=token,
                        output_dir=None,
                        task=task_type,
                        format_type='auto',
                        skip_existing=True,
                        max_workers=int(max_workers) if max_workers else 4,
                        include_negative=include_negative,
                        max_tasks=int(max_tasks) if max_tasks else None,
                        task_ids=task_ids,
                        task_range=task_range,
                        filter_labels=filter_labels
                    )
            
            elif operation == 'split':
                # 划分数据集
                print_section_header("划分数据集")
                
                task_type = select_task_type()
                
                # 配置输出目录
                output_dir = input_path("输出目录:", default="data/processed", must_exist=False)
                
                # 选择划分方式
                console.print()
                print_info("📊 数据集划分方式：")
                print_info("   • 按比例划分：使用全部数据，按比例自动计算样本数")
                print_info("   • 按样本数划分：从数据集中抽取固定数量的样本")
                console.print()
                
                split_mode = select_option(
                    "选择划分方式:",
                    choices=[
                        "按比例划分 (推荐用于常规训练)",
                        "按样本数划分 (推荐用于快速实验)",
                    ]
                )
                
                use_counts = "样本数" in split_mode
                
                if task_type == 'classify':
                    # 分类任务
                    source_dir = input_path("源目录 (已按类别组织):", default="data/raw/images", must_exist=False)
                    
                    # 询问是否去重（分类任务）
                    console.print()
                    print_info("🧹 数据去重：")
                    print_info("   - 检测并删除重复/相似的图片")
                    print_info("   - 避免训练/验证/测试集之间的数据泄露")
                    print_info("   - 推荐在拆分前进行去重")
                    console.print()
                    deduplicate_classify = confirm_action("是否在拆分前去除重复图片?", default=True)
                    
                    # 如果启用去重，询问去重模式
                    dedup_mode_classify = "exact"
                    similarity_threshold_classify = 8
                    if deduplicate_classify:
                        console.print()
                        print_info("📋 去重模式：")
                        print_info("   • 完全相同：只删除文件内容完全相同的图片（快速，推荐）")
                        print_info("   • 相似图片：删除视觉上相似的图片（较慢，需要imagehash库）")
                        console.print()
                        
                        mode_choice = select_option(
                            "选择去重模式:",
                            choices=[
                                "完全相同检测 (推荐，快速)",
                                "相似图片检测 (检测不同压缩质量、轻微编辑的图片)",
                            ]
                        )
                        
                        if "相似" in mode_choice:
                            dedup_mode_classify = "similar"
                            console.print()
                            print_info("💡 相似度阈值说明：")
                            print_info("   • 0-5:  几乎相同（不同压缩质量）")
                            print_info("   • 6-10: 很相似（轻微编辑、裁剪）")
                            print_info("   • 11-15: 相似（明显编辑）")
                            print_info("   • 推荐值：8（平衡准确性和召回率）")
                            console.print()
                            similarity_threshold_classify = input_number(
                                "相似度阈值 (0-64):", 
                                default=8, 
                                min_value=0, 
                                max_value=64
                            )
                    
                    if use_counts:
                        console.print()
                        print_info("💡 按样本数划分说明：")
                        print_info("   - 输入格式：train:val:test (如: 200:50:30)")
                        print_info("   - 从所有类别中随机抽取指定总数的样本")
                        console.print()
                        counts = input_text("样本数 (train:val:test):", default="100:30:10")
                        
                        console.print()
                        print_info(f"将划分数据到: {output_dir}")
                        
                        # 检查输出目录
                        if not check_and_clear_directory(output_dir):
                            continue
                        
                        if confirm_action("确认划分分类数据集?"):
                            from ..commands.data import split_dataset
                            split_dataset(
                                images_dir=None,
                                labels_dir=None,
                                source_dir=source_dir,
                                output_dir=output_dir,
                                ratios=None,
                                counts=counts,
                                seed=42,
                                task=task_type,
                                deduplicate=deduplicate_classify,
                                dedup_mode=dedup_mode_classify,
                                similarity_threshold=similarity_threshold_classify
                            )
                    else:
                        ratios = input_text("划分比例 (train:val:test):", default="0.7:0.2:0.1")
                        
                        console.print()
                        print_info(f"将划分数据到: {output_dir}")
                        
                        # 检查输出目录
                        if not check_and_clear_directory(output_dir):
                            continue
                        
                        if confirm_action("确认划分分类数据集?"):
                            from ..commands.data import split_dataset
                            split_dataset(
                                images_dir=None,
                                labels_dir=None,
                                source_dir=source_dir,
                                output_dir=output_dir,
                                ratios=ratios,
                                counts=None,
                                seed=42,
                                task=task_type,
                                deduplicate=deduplicate_classify,
                                dedup_mode=dedup_mode_classify,
                                similarity_threshold=similarity_threshold_classify
                            )
                else:
                    # 检测/分割任务
                    images_dir = input_path("图像目录:", default="data/raw/images", must_exist=False)
                    labels_dir = input_path("标签目录:", default="data/raw/labels", must_exist=False)
                    
                    if use_counts:
                        console.print()
                        print_info("💡 按样本数划分说明：")
                        print_info("   - 输入格式：train:val:test (如: 100:30:10)")
                        print_info("   - 从所有样本中随机抽取指定数量")
                        print_info("   - 适合快速原型验证和数据增量实验")
                        console.print()
                        counts = input_text("样本数 (train:val:test):", default="100:30:10")
                    else:
                        ratios = input_text("划分比例 (train:val:test):", default="0.7:0.2:0.1")
                    
                    # 询问是否为缺失标签创建空文件
                    console.print()
                    print_info("💡 负样本处理：")
                    print_info("   - 如果有图片缺失标签文件，可创建空标签作为负样本")
                    print_info("   - 负样本有助于减少误报，提高模型鲁棒性")
                    console.print()
                    create_empty = confirm_action("为缺失标签的图片创建空标签文件?", default=False)
                    
                    # 询问是否去重
                    console.print()
                    print_info("🧹 数据去重：")
                    print_info("   - 检测并删除重复/相似的图片")
                    print_info("   - 避免训练/验证/测试集之间的数据泄露")
                    print_info("   - 推荐在拆分前进行去重")
                    console.print()
                    deduplicate = confirm_action("是否在拆分前去除重复图片?", default=True)
                    
                    # 如果启用去重，询问去重模式
                    dedup_mode = "exact"
                    similarity_threshold = 8
                    if deduplicate:
                        console.print()
                        print_info("📋 去重模式：")
                        print_info("   • 完全相同：只删除文件内容完全相同的图片（快速，推荐）")
                        print_info("   • 相似图片：删除视觉上相似的图片（较慢，需要imagehash库）")
                        console.print()
                        
                        mode_choice = select_option(
                            "选择去重模式:",
                            choices=[
                                "完全相同检测 (推荐，快速)",
                                "相似图片检测 (检测不同压缩质量、轻微编辑的图片)",
                            ]
                        )
                        
                        if "相似" in mode_choice:
                            dedup_mode = "similar"
                            console.print()
                            print_info("💡 相似度阈值说明：")
                            print_info("   • 0-5:  几乎相同（不同压缩质量）")
                            print_info("   • 6-10: 很相似（轻微编辑、裁剪）")
                            print_info("   • 11-15: 相似（明显编辑）")
                            print_info("   • 推荐值：8（平衡准确性和召回率）")
                            console.print()
                            similarity_threshold = input_number(
                                "相似度阈值 (0-64):", 
                                default=8, 
                                min_value=0, 
                                max_value=64
                            )
                    
                    console.print()
                    print_info(f"将划分数据到: {output_dir}")
                    
                    # 检查输出目录
                    if not check_and_clear_directory(output_dir):
                        continue
                    
                    if confirm_action("确认划分数据集?"):
                        from ..commands.data import split_dataset
                        split_dataset(
                            images_dir=images_dir,
                            labels_dir=labels_dir,
                            source_dir=None,
                            output_dir=output_dir,
                            ratios=ratios if not use_counts else None,
                            counts=counts if use_counts else None,
                            seed=42,
                            task=task_type,
                            create_empty_labels=create_empty,
                            deduplicate=deduplicate,
                            dedup_mode=dedup_mode,
                            similarity_threshold=similarity_threshold
                        )
            
            elif operation == 'merge':
                # 合并数据集
                print_section_header("合并数据集")
                
                console.print()
                print_info("📦 数据集合并：")
                print_info("   - 合并多个数据集为一个")
                print_info("   - 自动处理类别ID重映射")
                print_info("   - 保留原始train/val/test分割")
                print_info("   - 适合合并不同标签的数据集")
                console.print()
                
                # 选择任务类型
                task_type = select_task_type()
                
                # 扫描 datasets 目录
                from pathlib import Path
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                datasets_root = config.project_root / 'datasets'
                
                if not datasets_root.exists():
                    print_error(f"datasets 目录不存在: {datasets_root}")
                    print_info("请先创建 datasets 目录并在其中放置数据集")
                    continue
                
                # 扫描所有有效的数据集
                available_datasets = []
                for item in sorted(datasets_root.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        # 检查是否是有效的数据集目录
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
                    continue
                
                print_info(f"发现 {len(available_datasets)} 个数据集")
                console.print()
                
                # 操作说明
                print_info("💡 操作提示:")
                print_info("   • 使用 ↑↓ 方向键移动光标")
                print_info("   • 使用 空格键 选择/取消数据集")
                print_info("   • 使用 回车键 确认选择")
                print_info("   • 至少需要选择 2 个数据集进行合并")
                console.print()
                
                # 让用户多选数据集（循环直到选择至少2个）
                choices = [f"{ds.name} ({ds})" for ds in available_datasets]
                selected_choices = None
                
                while not selected_choices or len(selected_choices) < 2:
                    selected_choices = select_multiple(
                        "请选择要合并的数据集:",
                        choices
                    )
                    
                    if not selected_choices:
                        console.print()
                        print_warning("⚠️  您还没有选择任何数据集！")
                        print_info("提示: 使用空格键选择数据集，然后按回车确认")
                        console.print()
                        
                        if not confirm_action("重新选择?", default=True):
                            print_info("操作已取消")
                            break
                        console.print()
                    elif len(selected_choices) < 2:
                        console.print()
                        print_warning(f"⚠️  合并操作至少需要选择 2 个数据集，您只选择了 {len(selected_choices)} 个")
                        console.print()
                        
                        if not confirm_action("重新选择?", default=True):
                            print_info("操作已取消")
                            break
                        console.print()
                
                if not selected_choices or len(selected_choices) < 2:
                    continue
                
                # 提取选中的数据集路径
                dataset_paths = []
                for choice in selected_choices:
                    # 从 "name (path)" 格式中提取路径
                    ds_path_str = choice.split('(')[1].rstrip(')')
                    dataset_paths.append(ds_path_str)
                
                # 构建逗号分隔的路径字符串（用于传递给 merge_datasets 命令）
                datasets_input = ','.join(dataset_paths)
                
                print_info(f"已选择 {len(dataset_paths)} 个数据集进行合并")
                console.print()
                
                # 输出目录
                output_dir = input_path(
                    "输出目录:",
                    default="datasets/merged",
                    must_exist=False
                )
                if not output_dir:
                    print_warning("操作已取消")
                    continue
                
                # 重复文件处理
                console.print()
                print_info("🔄 重复文件处理：")
                print_info("   • 跳过：如果文件名重复，跳过后续的文件")
                print_info("   • 重命名：自动为重复文件名添加后缀")
                print_info("   • 报错：遇到重复文件名时停止并报错")
                console.print()
                
                duplicate_choice = select_option(
                    "选择重复文件处理方式:",
                    choices=[
                        "跳过 (推荐，保留第一个)",
                        "重命名 (保留所有，自动添加后缀)",
                        "报错 (遇到重复时停止)",
                    ]
                )
                
                if "跳过" in duplicate_choice:
                    handle_duplicates = "skip"
                elif "重命名" in duplicate_choice:
                    handle_duplicates = "rename"
                else:
                    handle_duplicates = "error"
                
                # 询问是否去重
                console.print()
                print_info("🧹 数据去重（可选）：")
                print_info("   - 合并后删除完全相同的图片")
                print_info("   - 使用MD5哈希检测")
                console.print()
                deduplicate = confirm_action("是否在合并后去除完全相同的图片?", default=False)
                
                # 显示配置摘要
                console.print()
                print_section_header("配置摘要")
                print_info(f"任务类型: {task_type}")
                print_info(f"数据集列表: {datasets_input}")
                print_info(f"输出目录: {output_dir}")
                print_info(f"重复处理: {handle_duplicates}")
                print_info(f"去重: {'是' if deduplicate else '否'}")
                console.print()
                
                # 检查输出目录
                if not check_and_clear_directory(output_dir):
                    continue
                
                if confirm_action("确认合并数据集?"):
                    from ..commands.data import merge_datasets
                    merge_datasets(
                        datasets=datasets_input,
                        output_dir=output_dir,
                        task=task_type,
                        handle_duplicates=handle_duplicates,
                        deduplicate=deduplicate
                    )
            
            elif operation == 'filter':
                # 按标签过滤数据集
                print_section_header("按标签过滤数据集")
                
                console.print()
                print_info("🏷️  标签过滤功能：")
                print_info("   - 从数据集中提取特定类别")
                print_info("   - 排除不需要的类别")
                print_info("   - 自动重映射类别ID")
                print_info("   - 可选保留负样本（无标注图片）")
                console.print()
                
                # 选择任务类型
                task_type = select_task_type()
                
                # 选择数据集
                from pathlib import Path
                import yaml
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                datasets_root = config.project_root / 'datasets'
                
                if not datasets_root.exists():
                    print_error(f"datasets 目录不存在: {datasets_root}")
                    print_info("请先创建 datasets 目录并在其中放置数据集")
                    continue
                
                # 扫描 datasets 目录
                available_datasets = []
                for item in sorted(datasets_root.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        has_images = (item / 'images').exists() or (item / 'train').exists() or (item / 'val').exists() or (item / 'valid').exists() or (item / 'test').exists()
                        has_config = (item / 'data.yaml').exists() or (item / 'dataset.yaml').exists() or (item / 'classes.txt').exists()
                        if has_images or has_config:
                            available_datasets.append(item)
                
                if not available_datasets:
                    print_error(f"在 {datasets_root} 目录下没有找到任何数据集")
                    continue
                
                print_info(f"📁 发现 {len(available_datasets)} 个数据集")
                console.print()
                
                # 让用户单选数据集
                choices = [f"{ds.name}" for ds in available_datasets]
                selected_name = select_option(
                    "请选择要过滤的数据集:",
                    choices
                )
                
                # 找到对应的路径
                dataset_path_obj = None
                for ds in available_datasets:
                    if ds.name == selected_name:
                        dataset_path_obj = ds
                        break
                
                if dataset_path_obj is None:
                    print_error("未能找到选中的数据集")
                    continue
                
                dataset_path = str(dataset_path_obj)
                print_info(f"已选择数据集: {dataset_path_obj.name}")
                console.print()
                
                # 读取数据集类别列表
                yaml_file = None
                for yaml_name in ['data.yaml', 'dataset.yaml']:
                    yaml_path = dataset_path_obj / yaml_name
                    if yaml_path.exists():
                        yaml_file = yaml_path
                        break
                
                if not yaml_file:
                    print_error(f"在 {dataset_path} 中未找到 data.yaml 或 dataset.yaml")
                    continue
                
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                    
                    if not yaml_config or 'names' not in yaml_config:
                        print_error("配置文件中缺少 'names' 字段")
                        continue
                    
                    # 解析类别名称
                    names_data = yaml_config['names']
                    if isinstance(names_data, dict):
                        class_names = list(names_data.values())
                    elif isinstance(names_data, list):
                        class_names = names_data
                    else:
                        print_error("配置文件中的 'names' 字段格式不正确")
                        continue
                    
                    if not class_names:
                        print_error("数据集中没有类别")
                        continue
                    
                    console.print()
                    print_info(f"📋 数据集包含 {len(class_names)} 个类别:")
                    print_info(f"   {', '.join(class_names)}")
                    
                except Exception as e:
                    print_error(f"读取配置文件失败: {e}")
                    continue
                
                # 选择过滤模式
                console.print()
                filter_mode = select_option(
                    "选择过滤模式:",
                    [
                        "包含模式 - 只保留指定类别",
                        "排除模式 - 移除指定类别",
                    ]
                )
                
                is_include_mode = "包含" in filter_mode
                
                # 从类别列表中多选
                console.print()
                print_info("💡 操作提示:")
                print_info("   • 使用 ↑↓ 方向键移动光标")
                print_info("   • 使用 空格键 选择/取消类别")
                print_info("   • 使用 回车键 确认选择")
                console.print()
                
                selected_labels = None
                
                if is_include_mode:
                    # 包含模式：选择要保留的类别
                    while not selected_labels:
                        selected_labels = select_multiple(
                            "请选择要保留的类别:",
                            class_names
                        )
                        
                        if not selected_labels:
                            console.print()
                            print_warning("⚠️  您还没有选择任何类别！")
                            print_info("提示: 使用空格键选择类别，然后按回车确认")
                            console.print()
                            
                            if not confirm_action("重新选择?", default=True):
                                print_info("操作已取消")
                                break
                            console.print()
                    
                    if not selected_labels:
                        continue
                    
                    labels_input = ','.join(selected_labels)
                    include_labels = labels_input
                    exclude_labels = None
                else:
                    # 排除模式：选择要排除的类别
                    while not selected_labels:
                        selected_labels = select_multiple(
                            "请选择要排除的类别:",
                            class_names
                        )
                        
                        if not selected_labels:
                            console.print()
                            print_warning("⚠️  您还没有选择任何类别！")
                            print_info("提示: 使用空格键选择类别，然后按回车确认")
                            console.print()
                            
                            if not confirm_action("重新选择?", default=True):
                                print_info("操作已取消")
                                break
                            console.print()
                    
                    if not selected_labels:
                        continue
                    
                    labels_input = ','.join(selected_labels)
                    include_labels = None
                    exclude_labels = labels_input
                
                # 是否保留负样本
                console.print()
                print_info("📊 负样本处理：")
                print_info("   - 负样本：没有任何标注的图片")
                print_info("   - 保留负样本有助于减少误报")
                console.print()
                keep_negative = confirm_action("保留负样本（无标注图片）?", default=True)
                
                # 输出目录
                output_dir = input_path(
                    "输出目录:",
                    default="data/filtered",
                    must_exist=False
                )
                if not output_dir:
                    print_warning("操作已取消")
                    continue
                
                # 是否限制样本数量
                console.print()
                print_info("📊 样本数量限制：")
                print_info("   - 可以限制每个集合保留的样本数量")
                print_info("   - 格式: train:val:test，如 100:30:10")
                print_info("   - 使用 'all' 表示不限制，如 all:50:20")
                console.print()
                
                limit_samples = confirm_action("是否限制样本数量?", default=False)
                limit_str = None
                
                if limit_samples:
                    console.print()
                    print_info("💡 输入格式说明:")
                    print_info("   • 100:30:10  - train保留100张, val保留30张, test保留10张")
                    print_info("   • all:50:20  - train不限制, val保留50张, test保留20张")
                    print_info("   • 200:all:all - train保留200张, val和test不限制")
                    console.print()
                    
                    limit_str = input_text(
                        "样本数量限制 (train:val:test):",
                        default="100:30:10"
                    )
                    
                    if not limit_str:
                        print_warning("未设置样本数量限制，将保留所有样本")
                
                # 显示配置摘要
                console.print()
                print_section_header("配置摘要")
                print_info(f"数据集路径: {dataset_path}")
                print_info(f"任务类型: {task_type}")
                if is_include_mode:
                    print_info(f"过滤模式: 包含模式")
                    print_info(f"保留类别: {labels_input}")
                else:
                    print_info(f"过滤模式: 排除模式")
                    print_info(f"排除类别: {labels_input}")
                print_info(f"保留负样本: {'是' if keep_negative else '否'}")
                if limit_str:
                    print_info(f"样本数量限制: {limit_str}")
                else:
                    print_info(f"样本数量限制: 无（保留所有样本）")
                print_info(f"输出目录: {output_dir}")
                console.print()
                
                # 检查输出目录
                if not check_and_clear_directory(output_dir):
                    continue
                
                if confirm_action("确认过滤数据集?"):
                    from ..commands.data import filter_dataset
                    filter_dataset(
                        dataset_path=dataset_path,
                        output_dir=output_dir,
                        include_labels=include_labels,
                        exclude_labels=exclude_labels,
                        keep_negative=keep_negative,
                        limit=limit_str,
                        task=task_type
                    )
            
            elif operation == 'merge-labels':
                # 合并多个类别标签
                print_section_header("合并类别标签")
                
                console.print()
                print_info("🏷️  标签合并功能：")
                print_info("   - 将多个类别合并为一个")
                print_info("   - 简化类别数量")
                print_info("   - 自动重映射类别ID")
                print_info("   - 适合二分类、粗粒度分类等场景")
                console.print()
                
                print_info("💡 使用场景示例:")
                print_info("   • car,truck,bus → vehicle (车辆)")
                print_info("   • cat,dog,rabbit → pet (宠物)")
                print_info("   • apple,banana,orange → fruit (水果)")
                console.print()
                
                # 选择任务类型
                task_type = select_task_type()
                
                # 选择数据集
                from pathlib import Path
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                datasets_root = config.project_root / 'datasets'
                
                if not datasets_root.exists():
                    print_error(f"datasets 目录不存在: {datasets_root}")
                    print_info("请先创建 datasets 目录并在其中放置数据集")
                    continue
                
                # 扫描 datasets 目录
                available_datasets = []
                for item in sorted(datasets_root.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        has_images = (item / 'images').exists() or (item / 'train').exists() or (item / 'val').exists() or (item / 'valid').exists() or (item / 'test').exists()
                        has_config = (item / 'data.yaml').exists() or (item / 'dataset.yaml').exists() or (item / 'classes.txt').exists()
                        if has_images or has_config:
                            available_datasets.append(item)
                
                if not available_datasets:
                    print_error(f"在 {datasets_root} 目录下没有找到任何数据集")
                    continue
                
                print_info(f"📁 发现 {len(available_datasets)} 个数据集")
                console.print()
                
                # 让用户单选数据集
                choices = [f"{ds.name}" for ds in available_datasets]
                selected_name = select_option(
                    "请选择要合并标签的数据集:",
                    choices
                )
                
                # 找到对应的路径
                dataset_path_obj = None
                for ds in available_datasets:
                    if ds.name == selected_name:
                        dataset_path_obj = ds
                        break
                
                if dataset_path_obj is None:
                    print_error("未能找到选中的数据集")
                    continue
                
                dataset_path = str(dataset_path_obj)
                print_info(f"已选择数据集: {dataset_path_obj.name}")
                console.print()
                
                # 输出目录
                output_dir = input_path(
                    "输出目录:",
                    default="data/merged_labels",
                    must_exist=False
                )
                if not output_dir:
                    print_warning("操作已取消")
                    continue
                
                console.print()
                print_info("接下来将进入交互式配置合并规则")
                print_info("您可以多次添加合并规则，每次选择多个类别合并为一个")
                console.print()
                
                if not confirm_action("开始配置?"):
                    continue
                
                # 检查输出目录
                if not check_and_clear_directory(output_dir):
                    continue
                
                # 调用合并函数（交互式配置）
                from ..commands.data import merge_labels
                merge_labels(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    mapping=None,  # 交互式配置
                    task=task_type
                )
            
            elif operation == 'deduplicate':
                # 数据集去重
                print_section_header("数据集去重")
                
                console.print()
                print_info("🔍 数据集去重功能：")
                print_info("   - 检测已拆分数据集中的重复图片")
                print_info("   - 支持哈希和感知哈希两种模式")
                print_info("   - 可跨集合或仅集合内部去重")
                print_info("   - 自动处理对应的标签文件")
                console.print()
                
                print_info("💡 使用场景:")
                print_info("   • 清理训练/验证/测试集中的重复数据")
                print_info("   • 避免数据泄漏（train和val之间有重复）")
                print_info("   • 减少存储空间和训练时间")
                console.print()
                
                # 选择数据集
                from pathlib import Path
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                datasets_root = config.project_root / 'datasets'
                
                if not datasets_root.exists():
                    print_error(f"datasets 目录不存在: {datasets_root}")
                    print_info("请先创建 datasets 目录并在其中放置数据集")
                    continue
                
                # 扫描 datasets 目录
                available_datasets = []
                for item in sorted(datasets_root.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        has_images = (item / 'images').exists() or (item / 'train').exists() or (item / 'val').exists() or (item / 'valid').exists() or (item / 'test').exists()
                        has_config = (item / 'data.yaml').exists() or (item / 'dataset.yaml').exists() or (item / 'classes.txt').exists()
                        if has_images or has_config:
                            available_datasets.append(item)
                
                if not available_datasets:
                    print_error(f"在 {datasets_root} 目录下没有找到任何数据集")
                    continue
                
                print_info(f"📁 发现 {len(available_datasets)} 个数据集")
                console.print()
                
                # 让用户单选数据集
                choices = [f"{ds.name}" for ds in available_datasets]
                selected_name = select_option(
                    "请选择要去重的数据集:",
                    choices
                )
                
                # 找到对应的路径
                dataset_path_obj = None
                for ds in available_datasets:
                    if ds.name == selected_name:
                        dataset_path_obj = ds
                        break
                
                if dataset_path_obj is None:
                    print_error("未能找到选中的数据集")
                    continue
                
                dataset_path = str(dataset_path_obj)
                print_info(f"已选择数据集: {dataset_path_obj.name}")
                console.print()
                
                # 选择去重模式
                console.print()
                mode_choice = select_option(
                    "选择去重模式:",
                    [
                        "哈希去重 - 完全相同的图片（推荐，快速）",
                        "感知哈希 - 相似的图片（较慢，更全面）",
                        "组合模式 - 两者结合（最全面，最慢）",
                    ]
                )
                
                if "哈希去重" in mode_choice:
                    mode = "hash"
                elif "感知哈希" in mode_choice:
                    mode = "perceptual"
                else:
                    mode = "both"
                
                # 相似度阈值（仅感知哈希需要）
                threshold = 0.95
                if mode in ['perceptual', 'both']:
                    console.print()
                    threshold = input_number(
                        "相似度阈值 (0.0-1.0):",
                        default=0.95,
                        min_value=0.0,
                        max_value=1.0
                    )
                
                # 跨集合去重
                console.print()
                cross_split = confirm_action(
                    "是否跨集合去重 (检测train/val/test之间的重复)?",
                    default=True
                )
                
                # 保留优先级
                console.print()
                if cross_split:
                    priority_choice = select_option(
                        "选择保留优先级（当跨集合有重复时）:",
                        [
                            "train > val > test（优先保留训练集）",
                            "val > train > test（优先保留验证集）",
                            "test > val > train（优先保留测试集）",
                        ]
                    )
                    
                    if "train > val > test" in priority_choice:
                        priority = "train>val>test"
                    elif "val > train > test" in priority_choice:
                        priority = "val>train>test"
                    else:
                        priority = "test>val>train"
                else:
                    priority = "train>val>test"  # 默认值，不影响集合内部去重
                
                # 处理方式
                console.print()
                action_choice = select_option(
                    "选择处理方式:",
                    [
                        "仅报告 - 不删除，只查看重复情况",
                        "删除 - 直接删除重复文件",
                        "移动 - 移动到duplicates目录保留",
                    ]
                )
                
                if "仅报告" in action_choice:
                    action = "report"
                elif "删除" in action_choice:
                    action = "delete"
                else:
                    action = "move"
                
                console.print()
                
                # 调用去重函数
                from ..commands.data import deduplicate_dataset
                deduplicate_dataset(
                    dataset_path=dataset_path,
                    mode=mode,
                    action=action,
                    priority=priority,
                    threshold=threshold,
                    cross_split=cross_split
                )
            
            elif operation == 'convert-format':
                # 转换标注格式
                print_section_header("转换标注格式")
                
                console.print()
                print_info("🔄 格式转换功能：")
                print_info("   - segment ← → detect: 分割 ← → 检测")
                print_info("   - pose ← → detect: 姿态 ← → 检测")
                print_info("   - pose ← → segment: 姿态 ← → 分割")
                print_info("   - 自动处理类别映射")
                print_info("   - 支持边界框扩展等高级参数")
                console.print()
                
                # 选择数据集
                from pathlib import Path
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                datasets_root = config.project_root / 'datasets'
                
                if not datasets_root.exists():
                    print_error(f"datasets 目录不存在: {datasets_root}")
                    print_info("请先创建 datasets 目录并在其中放置数据集")
                    continue
                
                # 扫描 datasets 目录
                available_datasets = []
                for item in sorted(datasets_root.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        has_images = (item / 'images').exists() or (item / 'train').exists() or (item / 'val').exists() or (item / 'valid').exists() or (item / 'test').exists()
                        has_config = (item / 'data.yaml').exists() or (item / 'dataset.yaml').exists() or (item / 'classes.txt').exists()
                        if has_images or has_config:
                            available_datasets.append(item)
                
                if not available_datasets:
                    print_error(f"在 {datasets_root} 目录下没有找到任何数据集")
                    continue
                
                print_info(f"📁 发现 {len(available_datasets)} 个数据集")
                console.print()
                
                # 让用户单选数据集
                choices = [f"{ds.name}" for ds in available_datasets]
                selected_name = select_option(
                    "请选择要转换格式的数据集:",
                    choices
                )
                
                # 找到对应的路径
                dataset_path_obj = None
                for ds in available_datasets:
                    if ds.name == selected_name:
                        dataset_path_obj = ds
                        break
                
                if dataset_path_obj is None:
                    print_error("未能找到选中的数据集")
                    continue
                
                dataset_path = str(dataset_path_obj)
                print_info(f"已选择数据集: {dataset_path_obj.name}")
                console.print()
                
                # 选择源格式
                console.print()
                source_format = select_option(
                    "源格式（当前数据集的格式）:",
                    [
                        "detect - 目标检测",
                        "segment - 实例分割",
                        "pose - 姿态估计",
                    ]
                )
                source_format = source_format.split(' ')[0]
                
                # 选择目标格式
                target_format = select_option(
                    "目标格式（要转换成的格式）:",
                    [
                        "detect - 目标检测",
                        "segment - 实例分割",
                        "pose - 姿态估计",
                    ]
                )
                target_format = target_format.split(' ')[0]
                
                if source_format == target_format:
                    print_error("源格式和目标格式相同，无需转换")
                    continue
                
                # 边界框扩展（仅对转换到detect有效）
                bbox_expand = 0.0
                if target_format == 'detect':
                    console.print()
                    print_info("📏 边界框扩展（可选）:")
                    print_info("   - 扩展边界框以包含更多上下文")
                    print_info("   - 范围: 0-50% (0=不扩展)")
                    print_info("   - 推荐: 5-10% 用于一般场景")
                    console.print()
                    if confirm_action("是否扩展边界框?", default=False):
                        expand_pct = input_number(
                            "扩展比例 (%):",
                            default=10.0,
                            min_value=0.0,
                            max_value=50.0
                        )
                        bbox_expand = expand_pct / 100.0
                
                # 输出目录
                output_dir = input_path(
                    "输出目录:",
                    default=f"data/converted_{target_format}",
                    must_exist=False
                )
                if not output_dir:
                    print_warning("操作已取消")
                    continue
                
                # 显示配置摘要
                console.print()
                print_section_header("配置摘要")
                print_info(f"数据集路径: {dataset_path}")
                print_info(f"转换方向: {source_format} → {target_format}")
                if bbox_expand > 0:
                    print_info(f"边界框扩展: {bbox_expand*100:.1f}%")
                print_info(f"输出目录: {output_dir}")
                
                # 转换质量说明
                console.print()
                if source_format == 'segment' and target_format == 'detect':
                    print_info("✓ 转换质量: 优秀（无精度损失）")
                elif source_format == 'pose' and target_format == 'detect':
                    print_info("✓ 转换质量: 良好（保留边界框）")
                else:
                    print_warning("⚠️  转换质量: 一般（可能损失信息）")
                
                console.print()
                
                # 检查输出目录
                if not check_and_clear_directory(output_dir):
                    continue
                
                if confirm_action("确认转换格式?"):
                    from ..commands.data import convert_dataset_format
                    convert_dataset_format(
                        dataset_path=dataset_path,
                        source_format=source_format,
                        target_format=target_format,
                        output_dir=output_dir,
                        bbox_expand=bbox_expand,
                        keep_confidence=False,
                        preserve_structure=True
                    )
            
            elif operation == 'scale-labels':
                # 批量调整标注大小
                print_section_header("批量调整标注大小")
                
                console.print()
                print_info("📐 标注缩放功能：")
                print_info("   - 调整标注框大小，保持中心点不变")
                print_info("   - 适用场景：标注框过大/过小导致模型效果不佳")
                print_info("   - 缩放比例：<1表示缩小，>1表示放大，=1不变")
                console.print()
                
                # 选择数据集
                from pathlib import Path
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                datasets_root = config.project_root / 'datasets'
                
                if not datasets_root.exists():
                    print_error(f"datasets 目录不存在: {datasets_root}")
                    print_info("请先创建 datasets 目录并在其中放置数据集")
                    continue
                
                # 扫描 datasets 目录
                available_datasets = []
                for item in sorted(datasets_root.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        has_images = (item / 'images').exists() or (item / 'train').exists() or (item / 'val').exists() or (item / 'valid').exists() or (item / 'test').exists()
                        has_config = (item / 'data.yaml').exists() or (item / 'dataset.yaml').exists() or (item / 'classes.txt').exists()
                        if has_images or has_config:
                            available_datasets.append(item)
                
                if not available_datasets:
                    print_error(f"在 {datasets_root} 目录下没有找到任何数据集")
                    continue
                
                print_info(f"📁 发现 {len(available_datasets)} 个数据集")
                console.print()
                
                # 让用户单选数据集
                choices = [f"{ds.name}" for ds in available_datasets]
                selected_name = select_option(
                    "请选择要调整标注的数据集:",
                    choices
                )
                
                # 找到对应的路径
                dataset_path_obj = None
                for ds in available_datasets:
                    if ds.name == selected_name:
                        dataset_path_obj = ds
                        break
                
                if dataset_path_obj is None:
                    print_error("未能找到选中的数据集")
                    continue
                
                dataset_dir = str(dataset_path_obj)
                print_info(f"已选择数据集: {dataset_path_obj.name}")
                console.print()
                
                # 智能生成默认输出目录
                dataset_path = Path(dataset_dir)
                parent_dir = dataset_path.parent
                default_output = parent_dir / f"{dataset_path.name}_scaled"
                output_dir = input_path("输出目录:", default=str(default_output), must_exist=False)
                if not output_dir:
                    print_warning("操作已取消")
                    continue
                
                # 选择任务类型
                task_type = select_task_type()
                
                # 输入缩放比例
                console.print()
                print_info("💡 缩放比例说明：")
                print_info("   • 0.8 = 缩小到80%（推荐用于标注框过大）")
                print_info("   • 1.0 = 保持不变")
                print_info("   • 1.2 = 放大到120%（推荐用于标注框过小）")
                console.print()
                
                scale_factor = input_number(
                    "缩放比例:", 
                    default=0.8, 
                    min_value=0.1, 
                    max_value=2.0
                )
                
                # 询问是否选择特定子集（多选）
                console.print()
                if confirm_action("是否只处理特定子集?", default=False):
                    print_info("💡 使用空格键选择/取消选择，回车确认")
                    available_splits = ['train', 'val', 'test']
                    selected_splits = select_multiple(
                        "选择要处理的子集:",
                        choices=available_splits
                    )
                    if selected_splits:
                        splits = ','.join(selected_splits)
                    else:
                        print_warning("未选择任何子集，将处理全部")
                        splits = None
                else:
                    splits = None
                
                # 询问是否只处理特定类别
                console.print()
                if confirm_action("是否只处理特定类别?", default=False):
                    # 尝试读取类别信息
                    from pathlib import Path as PathLib
                    import yaml
                    dataset_path_obj = PathLib(dataset_dir)
                    class_names = []
                    
                    # 尝试从 data.yaml 或 dataset.yaml 读取类别
                    for yaml_name in ['data.yaml', 'dataset.yaml']:
                        yaml_file = dataset_path_obj / yaml_name
                        if yaml_file.exists():
                            try:
                                with open(yaml_file, 'r', encoding='utf-8') as f:
                                    yaml_data = yaml.safe_load(f)
                                    if yaml_data and 'names' in yaml_data:
                                        names = yaml_data['names']
                                        if isinstance(names, list):
                                            class_names = names
                                        elif isinstance(names, dict):
                                            class_names = [names[i] for i in sorted(names.keys())]
                                        break
                            except Exception:
                                pass
                    
                    if class_names:
                        # 有类别信息，使用多选
                        print_info(f"检测到 {len(class_names)} 个类别")
                        print_info("💡 使用空格键选择/取消选择，回车确认")
                        console.print()
                        
                        # 构建选项列表（ID + 名称）
                        class_choices = [f"{i} - {name}" for i, name in enumerate(class_names)]
                        selected_classes = select_multiple(
                            "选择要处理的类别:",
                            choices=class_choices
                        )
                        
                        if selected_classes:
                            # 提取类别ID
                            class_ids = [choice.split(' - ')[0] for choice in selected_classes]
                            classes = ','.join(class_ids)
                        else:
                            print_warning("未选择任何类别，将处理全部")
                            classes = None
                    else:
                        # 没有类别信息，手动输入
                        print_warning("未找到类别信息，请手动输入")
                        classes = input_text(
                            "类别ID列表（逗号分隔，如: 0,1,2）:", 
                            default="0"
                        )
                else:
                    classes = None
                
                # 显示配置摘要
                console.print()
                print_section_header("配置摘要")
                print_info(f"数据集: {dataset_dir}")
                print_info(f"输出: {output_dir}")
                print_info(f"任务类型: {task_type}")
                print_info(f"缩放比例: {scale_factor} ({'缩小' if scale_factor < 1 else '放大' if scale_factor > 1 else '不变'})")
                print_info(f"处理子集: {splits or '全部'}")
                print_info(f"处理类别: {classes or '全部'}")
                console.print()
                
                # 检查输出目录
                if not check_and_clear_directory(output_dir):
                    continue
                
                if confirm_action("确认调整标注?"):
                    from ..commands.data import scale_labels
                    scale_labels(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        scale=scale_factor,
                        task=task_type,
                        splits=splits,
                        classes=classes,
                        dry_run=False
                    )
            
            elif operation == 'generate-yaml':
                # 生成dataset.yaml
                print_section_header("生成 dataset.yaml")
                
                task_type = select_task_type()
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                output = input_text("输出文件:", default="data/processed/dataset.yaml")
                
                if confirm_action("确认生成配置文件?"):
                    from ..commands.data import generate_yaml
                    generate_yaml(
                        data_path=data_path,
                        classes_file=None,
                        output=output,
                        train_dir=None,
                        val_dir=None,
                        test_dir=None,
                        task=task_type
                    )
            
            elif operation == 'verify':
                # 验证数据集
                print_section_header("验证数据集")
                
                task_type = select_task_type()
                
                # 不传递 data_path 参数，让 verify_dataset 从 datasets 目录中选择
                from ..commands.data import verify_dataset
                verify_dataset(data_path=None, task=task_type)
            
            elif operation == 'stats':
                # 数据统计
                print_section_header("数据统计")
                
                task_type = select_task_type()
                detailed = confirm_action("显示详细统计 (含正负样本统计)?", default=True)
                
                # 如果是分类任务且需要详细统计，让用户输入正类
                positive_classes_str = None
                if task_type == 'classify' and detailed:
                    console.print()
                    print_info("💡 正负样本统计说明（分类任务）：")
                    print_info("   - 输入一个或多个类别名称作为「正类」（逗号分隔）")
                    print_info("   - 其余类别将自动归为「负类」")
                    print_info("   - 适用于异常检测、二分类等场景")
                    console.print()
                    
                    if confirm_action("是否指定正类进行正负样本统计?", default=False):
                        positive_classes_str = input_text(
                            "请输入正类名称 (逗号分隔，如: normal,good):",
                            default=""
                        )
                        if positive_classes_str:
                            print_success(f"✓ 已指定正类: {positive_classes_str}")
                
                # 不传递 data_path 参数，让 dataset_stats 从 datasets 目录中选择
                from ..commands.data import dataset_stats
                dataset_stats(data_path=None, detailed=detailed, task=task_type, positive_classes=positive_classes_str)
            
            console.print()
            if not confirm_action("继续数据操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def run_train_operations():
    """训练操作"""
    while True:
        operation = select_train_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'start':
                # 开始训练
                print_section_header("配置训练")
                
                config = build_training_config()
                
                # 数据集路径
                data_path = input_path("数据集配置文件:", default="data/processed/dataset.yaml", must_exist=True)
                
                # 模型名称
                model_name = YOLOVersionManager.get_model_name(config['version'], config['model_size'])
                
                print_info(f"训练配置:")
                print_info(f"  模型: {model_name}")
                print_info(f"  任务类型: {config['task']}")
                print_info(f"  数据集: {data_path}")
                print_info(f"  训练轮数: {config['epochs']}")
                print_info(f"  批次大小: {config['batch']}")
                print_info(f"  图像尺寸: {config['imgsz']}")
                print_info(f"  设备: {config['device']}")
                print_info(f"  数据增强: {config['augmentation']}")
                
                if confirm_action("确认开始训练?"):
                    from ..commands.train import start_training
                    start_training(
                        model=model_name,
                        data=data_path,
                        task=config['task'],
                        epochs=config['epochs'],
                        batch=config['batch'],
                        imgsz=config['imgsz'],
                        device=config['device'],
                        project=None,
                        name=None,
                        augmentation=config['augmentation'],
                        optimizer=config.get('optimizer_type', 'auto'),
                        freeze=config.get('freeze'),
                        patience=config['patience'],
                        save_period=config['save_period'],
                        resume=False,
                        pretrained=True,
                    )
            
            elif operation == 'resume':
                # 恢复训练
                print_section_header("恢复训练")
                
                checkpoint = input_path("检查点路径 (留空自动查找):", default="results/training/last.pt", must_exist=False)
                
                if confirm_action("确认恢复训练?"):
                    from ..commands.train import resume_training
                    resume_training(checkpoint=checkpoint if checkpoint else None, project=None, name=None)
            
            elif operation == 'config':
                # 生成配置
                print_section_header("生成训练配置")
                
                output = input_text("输出文件:", default="train_config.yaml")
                
                from ..commands.train import generate_config
                generate_config(output=output, profile=None)
            
            console.print()
            if not confirm_action("继续训练操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def run_quick_train():
    """一键训练操作"""
    print_section_header("⚡ 一键训练")
    
    try:
        print_info("一键训练会自动完成以下步骤:")
        print_info("  1. 数据集划分")
        print_info("  2. 生成配置文件")
        print_info("  3. 验证数据集")
        print_info("  4. 数据统计")
        print_info("  5. 检查/下载模型")
        print_info("  6. 开始训练")
        console.print()
        
        # 步骤1: 选择数据集划分方式 (前移到第一步)
        console.print()
        print_section_header("步骤1: 数据集划分配置")
        print_info("📊 数据集划分方式：")
        print_info("   • 按比例划分：使用全部数据，按比例自动计算样本数")
        print_info("   • 按样本数划分：从数据集中抽取固定数量的样本")
        console.print()
        
        split_mode = select_option(
            "选择划分方式:",
            choices=[
                "按比例划分 (推荐用于常规训练)",
                "按样本数划分 (推荐用于快速实验)",
            ]
        )
        
        use_counts = "样本数" in split_mode
        
        if use_counts:
            console.print()
            print_info("💡 按样本数划分说明：")
            print_info("   - 输入格式：train:val:test (如: 100:30:10)")
            print_info("   - 从所有样本中随机抽取指定数量")
            print_info("   - 适合快速原型验证和数据增量实验")
            console.print()
            counts = input_text("样本数 (train:val:test):", default="100:30:10")
            ratios = None
        else:
            ratios = input_text("划分比例 (train:val:test):", default="0.7:0.2:0.1")
            counts = None
        
        # 步骤2: 训练配置 (包含任务类型选择)
        console.print()
        print_section_header("步骤2: 训练参数配置")
        config = build_training_config()
        task_type = config['task']
        
        # 步骤3: 数据路径配置
        console.print()
        print_section_header("步骤3: 数据路径配置")
        # 根据任务类型获取不同的数据路径
        if task_type == 'classify':
            images_dir = input_path("图像目录 (按类别组织):", default="data/raw/images", must_exist=False)
            labels_dir = images_dir  # 分类任务图像和标签目录相同
        else:
            images_dir = input_path("图像目录:", default="data/raw/images", must_exist=False)
            labels_dir = input_path("标签目录:", default="data/raw/labels", must_exist=False)
        
        # 步骤4: 高级选项
        console.print()
        print_section_header("步骤4: 高级选项")
        skip_verify = not confirm_action("验证数据集?", default=True)
        skip_stats = not confirm_action("统计数据分布?", default=True)
        
        console.print()
        print_info("配置摘要:")
        print_info(f"  任务类型: {task_type}")
        print_info(f"  图像目录: {images_dir}")
        print_info(f"  标签目录: {labels_dir}")
        print_info(f"  YOLO版本: {config['version']}")
        print_info(f"  模型大小: {config['model_size']}")
        print_info(f"  训练轮数: {config['epochs']}")
        print_info(f"  批次大小: {config['batch']}")
        print_info(f"  图像尺寸: {config['imgsz']}")
        print_info(f"  设备: {config['device']}")
        print_info(f"  数据增强: {config['augmentation']}")
        print_info(f"  优化器: {config.get('optimizer_type', 'auto')}")
        if use_counts:
            print_info(f"  📊 数据集划分: 按样本数 ({counts})")
        else:
            print_info(f"  📊 数据集划分: 按比例 ({ratios})")
        
        console.print()
        if not confirm_action("确认开始一键训练?", default=True):
            print_warning("已取消")
            return
        
        # 执行一键训练
        from ..commands.quick import quick_train
        
        # 准备训练参数
        train_kwargs = {
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'classes_file': None,
            'task': task_type,
            'model_version': config['version'],
            'model_size': config['model_size'],
            'epochs': config['epochs'],
            'batch': config['batch'],
            'imgsz': config['imgsz'],
            'device': config['device'],
            'augmentation': config['augmentation'],
            'optimizer': config.get('optimizer_type', 'auto'),
            'freeze': config.get('freeze'),
            'ratios': ratios,
            'counts': counts,
            'skip_verify': skip_verify,
            'skip_stats': skip_stats,
            'project': None,
            'name': None,
            'patience': config.get('patience', 50),
            'save_period': config.get('save_period', 10),
        }
        
        # 添加高级配置（如果有）
        if config.get('augmentation_custom'):
            train_kwargs['augmentation_custom'] = config['augmentation_custom']
        if config.get('optimizer'):
            train_kwargs['optimizer_config'] = config['optimizer']
        if config.get('loss_weights'):
            train_kwargs['loss_weights'] = config['loss_weights']
        
        quick_train(**train_kwargs)
        
    except Exception as e:
        print_error(f"一键训练失败: {e}")


def run_detect_operations():
    """检测操作"""
    while True:
        operation = select_detect_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'image':
                # 单张图片检测
                print_section_header("单张图片检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                image_path = input_path("图片路径:", default="test.jpg", must_exist=False)
                
                # 选择任务类型
                task_type = select_task_type()
                
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                console.print()
                print_info(f"任务类型: {task_type}")
                print_info(f"模型: {model_path}")
                print_info(f"图片: {image_path}")
                console.print()
                
                if confirm_action("确认检测?"):
                    from ..commands.predict import predict_image
                    predict_image(model=model_path, image=image_path, task=task_type,
                               conf=conf, iou=0.45, output=None, save_txt=True, 
                               save_json=True, show=False, device='auto', top_k=5)
            
            elif operation == 'batch':
                # 批量检测
                print_section_header("批量检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                source_path = input_path("图片目录或视频:", default="test_images/", must_exist=False)
                
                # 选择任务类型
                task_type = select_task_type()
                
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                console.print()
                print_info(f"任务类型: {task_type}")
                print_info(f"模型: {model_path}")
                print_info(f"源: {source_path}")
                console.print()
                
                if confirm_action("确认批量检测?"):
                    from ..commands.predict import detect_batch
                    detect_batch(model=model_path, source=source_path, task=task_type, 
                               conf=conf, iou=0.45, output=None, save_txt=True, 
                               save_json=True, device='auto', batch=1)
            
            elif operation == 'video':
                # 视频检测
                print_section_header("视频检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                video_path = input_path("视频路径:", default="test_video.mp4", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                show = confirm_action("实时显示?", default=False)
                
                if confirm_action("确认检测?"):
                    from ..commands.predict import detect_video
                    detect_video(model=model_path, video=video_path, conf=conf, iou=0.45,
                               output=None, save_txt=False, show=show, device='auto')
            
            console.print()
            if not confirm_action("继续检测操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def run_labelstudio_operations():
    """Label Studio数据管理操作"""
    from ..converters.labelstudio import LabelStudioClient
    from ..core.config import ConfigManager
    import json
    from pathlib import Path
    from datetime import datetime
    
    # 获取配置管理器
    config_mgr = ConfigManager()
    
    # 读取Label Studio配置
    ls_config = config_mgr.config.get('labelstudio', {})
    default_url = ls_config.get('url', 'http://localhost:8080')
    default_token = ls_config.get('token', '')
    
    # 初始化客户端（稍后配置）
    client = None
    
    while True:
        operation = select_labelstudio_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'config':
                # 配置Label Studio连接
                print_section_header("配置Label Studio连接")
                
                ls_user_config = input_labelstudio_config()
                default_url = ls_user_config['url']
                default_token = ls_user_config['token']
                
                # 测试连接
                print_info("正在测试连接...")
                test_client = LabelStudioClient(url=default_url, token=default_token)
                success, msg = test_client.test_connection()
                
                if success:
                    print_success(f"✓ {msg}")
                    client = test_client
                    
                    # 询问是否保存到配置文件
                    if confirm_action("是否保存到配置文件?", default=False):
                        config_mgr.config['labelstudio'] = {
                            'url': default_url,
                            'token': default_token,
                            'auto_refresh': True,
                            'token_type': 'auto'
                        }
                        config_mgr.save()
                        print_success("✓ 配置已保存")
                else:
                    print_error(f"✗ {msg}")
                    continue
            
            elif operation == 'predict':
                # 使用本地模型预测任务
                print_section_header("Label Studio 任务预测")
                
                # 确保有连接配置
                if not default_url or not default_token:
                    print_warning("请先配置Label Studio连接")
                    continue
                
                # 输入模型路径
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                if not Path(model_path).exists():
                    print_error(f"模型不存在: {model_path}")
                    continue
                
                # 输入项目ID
                project_id = int(input_number("Label Studio项目ID:", min_value=1))
                
                # 选择任务类型
                from ..ui.prompts import select_task_type_for_predict
                task_type = select_task_type_for_predict()
                
                if task_type is None:
                    print_info("ℹ 将从模型自动推断任务类型（推荐）")
                
                # 选择任务筛选方式
                from ..ui.prompts import select_task_filter_mode, input_task_ids, input_task_range
                filter_mode = select_task_filter_mode()
                
                task_ids = None
                task_range = None
                unlabeled = False
                
                if filter_mode == 'ids':
                    task_ids = input_task_ids()
                    filter_desc = f"任务ID: {', '.join(map(str, task_ids[:5]))}" + ('...' if len(task_ids) > 5 else '')
                elif filter_mode == 'range':
                    task_range = input_task_range()
                    filter_desc = f"ID范围: {task_range[0]}-{task_range[1]}"
                else:  # unlabeled
                    unlabeled = True
                    filter_desc = "所有未标注任务"
                    print_info("ℹ 将预测项目中所有未标注的任务")
                
                # 配置预测参数
                console.print()
                print_info("配置预测参数:")
                conf = float(input_text("置信度阈值:", default="0.25"))
                iou = float(input_text("IOU阈值:", default="0.45"))
                device_choice = select_device()
                max_workers = int(input_number("最大并发数:", default=4, min_value=1, max_value=16))
                
                # 显示预测配置
                console.print()
                print_info("预测配置:")
                print_info(f"  模型: {model_path}")
                print_info(f"  Label Studio项目: {project_id}")
                print_info(f"  任务类型: {task_type or '自动推断'}")
                print_info(f"  任务筛选: {filter_desc}")
                print_info(f"  置信度阈值: {conf}")
                print_info(f"  IOU阈值: {iou}")
                print_info(f"  设备: {device_choice}")
                print_info(f"  并发数: {max_workers}")
                console.print()
                
                if not confirm_action("确认开始预测?", default=True):
                    continue
                
                # 执行预测
                try:
                    from ..integrations.labelstudio_uploader import LabelStudioUploader
                    
                    uploader = LabelStudioUploader(default_url, default_token, project_id, task_type=task_type or 'detect')
                    
                    print_info("\n连接到 Label Studio...")
                    if not uploader.test_connection():
                        print_error("连接失败，请检查URL和API密钥")
                        continue
                    
                    # 执行预测
                    print_info("\n开始预测...")
                    success, failed = uploader.predict_tasks_with_yolo(
                        model_path=Path(model_path),
                        task_ids=task_ids,
                        task_range=task_range,
                        unlabeled=unlabeled,
                        task_type=task_type,
                        conf=conf,
                        iou=iou,
                        device=device_choice,
                        max_workers=max_workers
                    )
                    
                    print_section_header("预测完成")
                    print_success(f"成功: {success} 个任务")
                    if failed > 0:
                        print_error(f"失败: {failed} 个任务")
                
                except Exception as e:
                    print_error(f"预测失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            elif operation == 'batch-annotate':
                # 批量打标签
                print_section_header("Label Studio 批量打标签")
                
                # 确保有连接配置
                if not default_url or not default_token:
                    print_warning("请先配置Label Studio连接")
                    continue
                
                # 输入项目ID
                project_id = int(input_number("Label Studio项目ID:", min_value=1))
                
                # 连接并获取项目配置
                from ..integrations.labelstudio_uploader import LabelStudioUploader
                
                try:
                    uploader = LabelStudioUploader(
                        url=default_url,
                        api_key=default_token,
                        project_id=project_id,
                        task_type='detect'  # 默认值，batch-annotate会自动解析项目配置
                    )
                    
                    # 测试连接
                    print_info("\n连接到 Label Studio...")
                    if not uploader.test_connection():
                        print_error("连接失败，请检查URL和Token")
                        continue
                    print_success("✓ 连接成功")
                    
                    # 解析项目标签配置
                    print_info("\n解析项目标签模板...")
                    labeling_config = uploader.parse_labeling_config()
                    
                    if not labeling_config:
                        print_error("无法解析项目标签配置，请确保项目已配置标签模板")
                        continue
                    
                    print_success(f"✓ 找到 {len(labeling_config)} 种标注类型")
                    
                    # 选择标注类型
                    from ..ui.prompts import select_annotation_type
                    annotation_type = select_annotation_type()
                    
                    # 检查项目是否支持该类型
                    config_key = f"{annotation_type}labels"
                    if config_key not in labeling_config:
                        print_error(f"项目不支持 {annotation_type} 标注类型")
                        print_info(f"可用类型: {', '.join(labeling_config.keys())}")
                        continue
                    
                    # 选择目标类型（annotation或prediction）
                    from ..ui.prompts import select_target_type
                    target_type = select_target_type()
                    
                    # 选择任务筛选方式
                    from ..ui.prompts import select_task_filter_mode, input_task_ids, input_task_range
                    filter_mode = select_task_filter_mode()
                    
                    task_ids = None
                    task_range = None
                    unlabeled = False
                    
                    if filter_mode == 'ids':
                        task_ids = input_task_ids()
                        filter_desc = f"任务ID: {', '.join(map(str, task_ids[:5]))}" + ('...' if len(task_ids) > 5 else '')
                    elif filter_mode == 'range':
                        task_range = input_task_range()
                        filter_desc = f"ID范围: {task_range[0]}-{task_range[1]}"
                    else:  # unlabeled
                        unlabeled = True
                        filter_desc = "所有未标注任务"
                    
                    # 构建task_filter
                    task_filter = {
                        'mode': filter_mode,
                        'task_ids': task_ids,
                        'task_range': task_range,
                        'unlabeled': unlabeled
                    }
                    
                    # 预先获取tasks，检查是否有已标注的
                    print_info("\n获取任务列表...")
                    
                    # 使用 uploader 的内部方法获取tasks
                    tasks = uploader._get_tasks_by_filter(task_filter)
                    
                    if not tasks:
                        print_warning("没有找到匹配的任务")
                        continue
                    
                    print_success(f"✓ 找到 {len(tasks)} 个任务")
                    
                    # 检查是否有已标注的
                    has_annotations = any(len(t.get('annotations', [])) > 0 for t in tasks)
                    
                    # 选择合并策略
                    from ..ui.prompts import select_merge_mode
                    merge_mode = select_merge_mode(has_existing_annotations=has_annotations)
                    
                    # 交互式输入标签
                    console.print()
                    annotation_data = None
                    
                    if annotation_type == 'rectangle':
                        from ..ui.prompts import input_rectangle_annotation
                        available_labels = labeling_config[config_key]['labels']
                        rect_data = input_rectangle_annotation(available_labels)
                        
                        # 转换为Label Studio格式
                        from_name = labeling_config[config_key]['from_name']
                        to_name = labeling_config[config_key]['to_name']
                        
                        # 坐标转换：归一化 -> 百分比
                        x = (rect_data['center_x'] - rect_data['width'] / 2) * 100
                        y = (rect_data['center_y'] - rect_data['height'] / 2) * 100
                        w = rect_data['width'] * 100
                        h = rect_data['height'] * 100
                        
                        annotation_data = {
                            'type': 'rectanglelabels',
                            'from_name': from_name,
                            'to_name': to_name,
                            'value': {
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'rotation': 0,
                                'rectanglelabels': [rect_data['label']]
                            }
                        }
                    
                    elif annotation_type == 'keypoint':
                        from ..ui.prompts import input_keypoint_annotations
                        keypoint_labels = labeling_config[config_key]['labels']
                        kp_data = input_keypoint_annotations(keypoint_labels)
                        
                        # 转换为Label Studio格式
                        from_name = labeling_config[config_key]['from_name']
                        to_name = labeling_config[config_key]['to_name']
                        
                        # 每个关键点是一个单独的result
                        annotation_data = []
                        for kp in kp_data:
                            kp_result = {
                                'type': 'keypointlabels',
                                'from_name': from_name,
                                'to_name': to_name,
                                'value': {
                                    'x': kp['x'] * 100,  # 归一化 -> 百分比
                                    'y': kp['y'] * 100,
                                    'width': 0.5,  # 关键点默认宽度
                                    'keypointlabels': [kp['label']]
                                }
                            }
                            if not kp['visible']:
                                kp_result['value']['hidden'] = True
                            annotation_data.append(kp_result)
                    
                    # 预览标注内容
                    console.print()
                    print_info("=" * 60)
                    print_info("标注内容预览")
                    print_info("=" * 60)
                    print_info(f"标注类型: {annotation_type}")
                    print_info(f"目标类型: {target_type}")
                    print_info(f"合并模式: {merge_mode}")
                    print_info(f"将影响: {len(tasks)} 个tasks ({filter_desc})")
                    console.print()
                    
                    import json
                    if isinstance(annotation_data, list):
                        print_info(f"将添加 {len(annotation_data)} 个标注:")
                        for i, data in enumerate(annotation_data[:3], 1):  # 只显示前3个
                            print(json.dumps(data, indent=2, ensure_ascii=False))
                            if i < min(len(annotation_data), 3):
                                console.print()
                        if len(annotation_data) > 3:
                            print_info(f"... 还有 {len(annotation_data) - 3} 个标注")
                    else:
                        print(json.dumps(annotation_data, indent=2, ensure_ascii=False))
                    
                    print_info("=" * 60)
                    console.print()
                    
                    # 确认执行
                    if not confirm_action("确认批量创建？", default=True):
                        print_info("已取消")
                        continue
                    
                    # 执行批量标注
                    console.print()
                    print_info("开始批量创建...")
                    
                    stats = uploader.batch_annotate_tasks(
                        annotation_data=annotation_data,
                        target_type=target_type,
                        task_filter=task_filter,
                        merge_mode=merge_mode,
                        dry_run=False,
                        max_workers=4
                    )
                    
                    # 显示统计
                    console.print()
                    print_section_header("批量标注完成")
                    print_success(f"✓ 成功: {stats['success']} 个任务")
                    if stats['failed'] > 0:
                        print_error(f"✗ 失败: {stats['failed']} 个任务")
                    if stats.get('skipped', 0) > 0:
                        print_info(f"ℹ 跳过: {stats['skipped']} 个任务")
                    
                except Exception as e:
                    print_error(f"批量标注失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            elif operation == 'audit':
                # 审计标注质量
                print_section_header("Label Studio 标注审计")
                
                # 确保有连接配置
                if not default_url or not default_token:
                    print_warning("请先配置Label Studio连接")
                    continue
                
                # 输入项目ID
                project_id = int(input_number("Label Studio项目ID:", min_value=1))
                
                # 配置审计参数
                console.print()
                
                # 询问是否抽样审计
                sample_audit = confirm_action("是否进行抽样审计?（否则审计全部任务）", default=False)
                max_tasks = None
                if sample_audit:
                    max_tasks = int(input_number("抽样任务数量:", default=500, min_value=1))
                
                show_details = confirm_action("显示异常任务的详细信息?", default=True)
                max_samples = int(input_number("每种异常类型显示的最大样本数:", default=10, min_value=1, max_value=100))
                
                # 询问是否导出报告
                export_report = confirm_action("导出审计报告到文件?", default=False)
                output_file = None
                if export_report:
                    from datetime import datetime
                    default_filename = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    output_file = input_text("报告文件名（将保存到results/audit/）:", default=default_filename)
                
                # 显示审计配置
                console.print()
                print_info("审计配置:")
                print_info(f"  项目ID: {project_id}")
                print_info(f"  审计模式: {'抽样审计' if max_tasks else '全部审计'}")
                if max_tasks:
                    print_info(f"  抽样数量: {max_tasks} 个任务")
                print_info(f"  显示详情: {'是' if show_details else '否'}")
                print_info(f"  样本数量: {max_samples}")
                if output_file:
                    print_info(f"  导出报告: results/audit/{Path(output_file).name}")
                console.print()
                
                if not confirm_action("确认开始审计?", default=True):
                    continue
                
                # 执行审计
                try:
                    from ..integrations.labelstudio_uploader import LabelStudioUploader
                    
                    uploader = LabelStudioUploader(default_url, default_token, project_id)
                    
                    print_info("\n连接到 Label Studio...")
                    if not uploader.test_connection():
                        print_error("连接失败，请检查URL和API密钥")
                        continue
                    
                    # 执行审计
                    print_info("\n开始审计...")
                    audit_report = uploader.audit_annotations(
                        show_details=show_details,
                        max_samples=max_samples,
                        max_tasks=max_tasks
                    )
                    
                    # 导出报告
                    if output_file and audit_report:
                        try:
                            import json
                            
                            # 创建 results/audit 目录
                            audit_dir = config_mgr.project_root / 'results' / 'audit'
                            audit_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 如果用户输入的是绝对路径，直接使用
                            # 否则，保存到 results/audit 目录
                            output_path = Path(output_file)
                            if not output_path.is_absolute():
                                # 只取文件名，放到 results/audit 目录
                                output_path = audit_dir / output_path.name
                            
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(audit_report, f, indent=2, ensure_ascii=False)
                            
                            # 验证文件是否存在
                            if output_path.exists():
                                file_size = output_path.stat().st_size
                                # 显示相对路径
                                try:
                                    rel_path = output_path.relative_to(config_mgr.project_root)
                                    print_success(f"\n✅ 报告已导出到: {rel_path}")
                                except ValueError:
                                    # 如果无法计算相对路径，显示绝对路径
                                    print_success(f"\n✅ 报告已导出到: {output_path}")
                                print_info(f"   文件大小: {file_size:,} 字节")
                            else:
                                print_error(f"\n✗ 报告文件未创建: {output_path}")
                        except Exception as e:
                            print_error(f"\n✗ 导出报告失败: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    print_section_header("审计完成")
                
                except Exception as e:
                    print_error(f"审计失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            elif operation == 'upload':
                # 上传数据集到Label Studio
                print_section_header("上传数据集到Label Studio")
                
                # 确保有连接配置
                if not default_url or not default_token:
                    print_warning("请先配置Label Studio连接")
                    continue
                
                # 选择数据集
                datasets_base = config_mgr.project_root / 'datasets'
                
                # 获取可用数据集
                available_datasets = []
                if datasets_base.exists():
                    available_datasets = [d.name for d in datasets_base.iterdir() if d.is_dir()]
                
                if not available_datasets:
                    print_warning("datasets目录下没有找到数据集")
                    print_info("您可以先将数据集复制到datasets目录，或输入自定义路径")
                
                # 选择或输入数据集路径
                if available_datasets:
                    print_info(f"找到 {len(available_datasets)} 个数据集:")
                    for ds in available_datasets:
                        print_info(f"  • {ds}")
                    console.print()
                    
                    use_existing = confirm_action("使用datasets目录下的数据集?", default=True)
                    
                    if use_existing:
                        choices = [f"{ds}" for ds in available_datasets]
                        choices.append("back - 返回")
                        dataset_choice = select_option("选择数据集:", choices)
                        
                        if dataset_choice == "back":
                            continue
                        
                        dataset_path = datasets_base / dataset_choice
                    else:
                        dataset_path = Path(input_path("数据集路径:", must_exist=True))
                else:
                    dataset_path = Path(input_path("数据集路径:", must_exist=True))
                
                # 输入项目ID
                project_id = int(input_number("Label Studio项目ID:", min_value=1))
                
                # 选择任务类型
                console.print()
                print_info("💡 任务类型说明：")
                print_info("   - 目标检测(detect): 矩形框标注")
                print_info("   - 分割(segment): 多边形标注")  
                print_info("   - 姿势估计(pose): 关键点标注")
                console.print()
                task_type = select_task_type()
                
                # 选择要上传的数据集分割
                split_choices = [
                    "all - 全部 (train, val, test)",
                    "train - 仅训练集",
                    "val - 仅验证集",
                    "test - 仅测试集",
                    "custom - 自定义选择",
                ]
                split_choice = select_option("选择要上传的数据集分割:", split_choices)
                
                splits = []
                if split_choice.startswith("all"):
                    splits = ['train', 'val', 'test']
                elif split_choice.startswith("train"):
                    splits = ['train']
                elif split_choice.startswith("val"):
                    splits = ['val']
                elif split_choice.startswith("test"):
                    splits = ['test']
                elif split_choice.startswith("custom"):
                    # 多选
                    if confirm_action("上传train?", default=True):
                        splits.append('train')
                    if confirm_action("上传val?", default=True):
                        splits.append('val')
                    if confirm_action("上传test?", default=False):
                        splits.append('test')
                
                if not splits:
                    print_warning("未选择任何数据集分割")
                    continue
                
                # 配置并发数
                console.print()
                print_info("💡 并发上传说明：")
                print_info("   - 并发数越大，上传速度越快")
                print_info("   - 推荐值：4-8，根据网络情况调整")
                print_info("   - 过大可能导致网络拥塞或API限流")
                console.print()
                max_workers = int(input_number("最大并发数:", default=4, min_value=1, max_value=16))
                
                # 高级选项
                console.print()
                print_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                show_advanced = confirm_action("显示高级选项（断点续传、重试等）?", default=False)
                
                # 默认值
                force = False
                no_resume = False
                skip_server_check = False
                retry_times = 3
                
                if show_advanced:
                    console.print()
                    print_info("⚙️  高级选项：")
                    console.print()
                    
                    # 检查是否有进度记录
                    from ..core.upload_progress import UploadProgressTracker
                    temp_tracker = UploadProgressTracker(
                        project_id=project_id,
                        dataset_path=dataset_path,
                        url=default_url
                    )
                    progress_info = temp_tracker.get_progress_info()
                    
                    if progress_info:
                        # 有进度记录，显示并提供选择
                        print_warning("⚠️  检测到之前的上传进度：")
                        print_info(f"   • 数据集：{progress_info['dataset_name']}")
                        print_info(f"   • 项目ID：{progress_info['project_id']}")
                        print_info(f"   • 已上传：{progress_info['uploaded_count']} 个文件")
                        print_info(f"   • 失败：{progress_info['failed_count']} 个文件")
                        print_info(f"   • 最后更新：{progress_info['last_updated']}")
                        console.print()
                        
                        resume_choices = [
                            "continue - 继续上传（从断点恢复）✨ 推荐",
                            "restart - 重新开始（清除进度，但检查服务器避免重复）",
                            "force - 强制全部重传（会创建重复任务）⚠️"
                        ]
                        resume_choice = select_option("选择操作:", resume_choices)
                        
                        if resume_choice.startswith("restart"):
                            no_resume = True
                        elif resume_choice.startswith("force"):
                            force = True
                        # continue 使用默认值
                    else:
                        # 没有进度记录
                        print_info("✨ 自动启用断点续传功能")
                        enable_resume = confirm_action("启用断点续传?", default=True)
                        if not enable_resume:
                            no_resume = True
                    
                    console.print()
                    
                    # 服务器检查选项（如果不是 force 模式）
                    if not force:
                        # 获取项目任务数
                        task_count = None
                        try:
                            from ..integrations.labelstudio_uploader import LabelStudioUploader
                            temp_uploader = LabelStudioUploader(default_url, default_token, project_id)
                            task_count = temp_uploader.get_project_task_count()
                        except:
                            pass
                        
                        if task_count and task_count >= 5000:
                            print_warning(f"⚠️  项目任务数较大（{task_count} 个），服务器检查可能较慢")
                            skip_server_check = not confirm_action(
                                "检查服务器重复（避免重复任务）?",
                                default=False
                            )
                        else:
                            skip_server_check = not confirm_action(
                                "检查服务器重复（避免重复任务）?",
                                default=True
                            )
                    
                    console.print()
                    
                    # 重试次数
                    retry_times = int(input_number(
                        "网络失败重试次数:",
                        default=3,
                        min_value=0,
                        max_value=10
                    ))
                    
                    console.print()
                    print_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                
                # 询问是否配置标注模板
                console.print()
                print_info("💡 标注模板配置说明：")
                print_info("   - 自动从数据集配置读取类别信息")
                print_info("   - 生成Label Studio标注界面")
                print_info("   - 支持矩形框和多边形标注")
                console.print()
                setup_config = confirm_action("是否配置Label Studio标注模板?", default=True)
                
                # 询问是否上传后验证
                verify = confirm_action("上传后是否验证结果?", default=True)
                
                # 显示上传信息
                console.print()
                print_info("上传配置:")
                print_info(f"  数据集: {dataset_path}")
                print_info(f"  Label Studio: {default_url}")
                print_info(f"  项目ID: {project_id}")
                print_info(f"  任务类型: {task_type}")
                print_info(f"  数据集分割: {', '.join(splits)}")
                print_info(f"  并发数: {max_workers}")
                print_info(f"  重试次数: {retry_times}")
                if force:
                    print_warning(f"  模式: ⚠️ 强制重传（会创建重复任务）")
                elif no_resume:
                    print_info(f"  模式: 🔄 重新开始（清除本地缓存）")
                elif skip_server_check:
                    print_info(f"  模式: ⚡ 快速模式（跳过服务器检查）")
                else:
                    print_info(f"  模式: ✨ 断点续传（自动跳过已上传）")
                print_info(f"  配置标注模板: {'是' if setup_config else '否'}")
                console.print()
                
                if not confirm_action("确认开始上传?", default=True):
                    continue
                
                # 执行上传
                from ..integrations.labelstudio_uploader import LabelStudioUploader
                
                try:
                    uploader = LabelStudioUploader(default_url, default_token, project_id, task_type=task_type)
                    
                    # 测试连接
                    print_info("测试连接...")
                    if not uploader.test_connection():
                        print_error("连接失败，请检查URL和API密钥")
                        continue
                    
                    # 加载数据集配置
                    print_info("加载数据集配置...")
                    uploader.load_dataset_config(dataset_path)
                    
                    # 配置标注模板
                    if setup_config:
                        print_section_header("配置标注模板")
                        if uploader.setup_project_config(task_type=task_type):
                            print_success("✓ 标注模板配置完成")
                        else:
                            print_warning("标注模板配置失败，将继续上传")
                        console.print()
                    
                    # 上传数据集
                    print_section_header("上传数据集")
                    total_uploaded, total_failed = uploader.upload_tasks(
                        dataset_path=dataset_path,
                        splits=splits,
                        max_images=None,
                        max_workers=max_workers,
                        force=force,
                        no_resume=no_resume,
                        skip_server_check=skip_server_check,
                        retry_times=retry_times
                    )
                    
                    # 显示结果
                    console.print()
                    print_section_header("上传完成")
                    print_success(f"成功: {total_uploaded} 个任务")
                    if total_failed > 0:
                        print_error(f"失败: {total_failed} 个任务")
                    
                    # 验证上传结果
                    if verify and total_uploaded > 0:
                        console.print()
                        print_section_header("验证上传结果")
                        uploader.verify_uploaded_tasks(num_samples=5)
                    
                except KeyboardInterrupt:
                    # 用户中断，不显示错误堆栈
                    console.print()
                    print_warning("⚠️  上传已被用户中断")
                    print_info("💾 进度已保存，下次运行时将自动继续")
                    # 不继续执行后续操作
                    
                except Exception as e:
                    print_error(f"上传失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            elif operation == 'list':
                # 列出所有项目
                print_section_header("Label Studio项目列表")
                
                # 确保有客户端
                if client is None:
                    if not default_url or not default_token:
                        print_warning("请先配置Label Studio连接")
                        continue
                    
                    client = LabelStudioClient(url=default_url, token=default_token)
                    success, msg = client.test_connection()
                    if not success:
                        print_error(f"连接失败: {msg}")
                        print_info("请先使用 'config' 命令配置连接")
                        continue
                
                # 获取项目列表
                print_info("正在获取项目列表...")
                success, projects, error = client.list_projects()
                
                if not success:
                    print_error(f"获取失败: {error}")
                    continue
                
                if not projects:
                    print_warning("没有找到任何项目")
                    continue
                
                # 显示项目列表
                from ..ui.display import print_table
                table_data = []
                for proj in projects:
                    table_data.append([
                        str(proj['id']),
                        proj['title'],
                        str(proj['task_number']),
                        proj.get('description', '')[:50]
                    ])
                
                print_table(
                    "Label Studio项目",
                    ["ID", "项目名称", "任务数", "描述"],
                    table_data
                )
            
            elif operation == 'fetch':
                # 获取项目数据
                print_section_header("获取Label Studio项目数据")
                
                # 确保有客户端
                if client is None:
                    if not default_url or not default_token:
                        print_warning("请先配置Label Studio连接")
                        continue
                    
                    client = LabelStudioClient(url=default_url, token=default_token)
                    success, msg = client.test_connection()
                    if not success:
                        print_error(f"连接失败: {msg}")
                        continue
                
                # 获取项目列表供选择
                print_info("正在获取项目列表...")
                success, projects, error = client.list_projects()
                
                if not success:
                    print_error(f"获取失败: {error}")
                    continue
                
                if not projects:
                    print_warning("没有找到任何项目")
                    continue
                
                # 选择项目
                selected_project = select_labelstudio_project(projects)
                if selected_project is None:
                    continue
                
                project_id = selected_project['id']
                project_title = selected_project['title']
                
                print_info(f"已选择项目: {project_title} (ID: {project_id})")
                print_info(f"任务数: {selected_project['task_number']}")
                
                # 选择任务类型
                task_type = select_task_type()
                
                # 配置输出路径
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_export_dir = f"labelstudioexport/project_{project_id}_{timestamp}"
                output_dir = input_path("输出目录:", default=default_export_dir, must_exist=False)
                
                # 配置下载参数
                max_workers = input_number("并发下载线程数:", default=4, min_value=1, max_value=16)
                
                # 对于检测和Pose任务，询问是否包含负样本
                include_negative = True
                if task_type in ['detect', 'pose']:
                    console.print()
                    print_info("💡 负样本说明：")
                    print_info("   - 无标注的图片可作为负样本训练")
                    print_info("   - 有助于减少误报，提高模型鲁棒性")
                    print_info("   - 推荐包含10-20%的负样本")
                    console.print()
                    include_negative = confirm_action("包含无标注图片作为负样本?", default=True)
                
                # 数据筛选选项 🆕
                console.print()
                print_info("📊 部分数据集下载选项（可选）：")
                print_info("   - 限制下载数量（适合快速测试）")
                print_info("   - 指定任务ID或范围")
                print_info("   - 按标签筛选")
                console.print()
                
                use_filters = confirm_action("是否使用数据筛选?", default=False)
                
                max_tasks = None
                task_ids = None
                task_range = None
                filter_labels = None
                
                if use_filters:
                    console.print()
                    filter_type = select_option(
                        "选择筛选方式:",
                        [
                            "限制下载数量 (如: 前50个任务)",
                            "指定任务ID列表 (如: 100,200,300)",
                            "指定任务ID范围 (如: 100-500)",
                            "按标签筛选 (如: person,car)",
                        ]
                    )
                    
                    if "限制下载数量" in filter_type:
                        max_tasks = input_number("最大下载任务数:", default=50, min_value=1)
                        print_info(f"将下载前 {max_tasks} 个任务")
                    elif "任务ID列表" in filter_type:
                        task_ids = input_text("任务ID列表（逗号分隔）:", default="100,200,300")
                        print_info(f"将下载指定ID的任务: {task_ids}")
                    elif "任务ID范围" in filter_type:
                        from ..ui.prompts import input_task_range
                        start_id, end_id = input_task_range()
                        task_range = (start_id, end_id)  # 保存为元组
                        print_info(f"将下载ID范围内的任务: {start_id}-{end_id}")
                    elif "按标签筛选" in filter_type:
                        filter_labels = input_text("标签列表（逗号分隔）:", default="person,car")
                        print_info(f"将下载包含这些标签的任务: {filter_labels}")
                
                console.print()
                print_info("将执行以下操作:")
                print_info(f"  1. 导出项目 {project_id} 的标注数据")
                if use_filters:
                    print_info(f"  2. 应用数据筛选条件")
                    print_info(f"  3. 下载筛选后的图片 (并发数: {int(max_workers)})")
                    print_info(f"  4. 转换为YOLO {task_type} 格式")
                    print_info(f"  5. 保存到: {output_dir}")
                else:
                    print_info(f"  2. 下载所有图片 (并发数: {int(max_workers)})")
                    print_info(f"  3. 转换为YOLO {task_type} 格式")
                    print_info(f"  4. 保存到: {output_dir}")
                
                console.print()
                
                if not confirm_action("确认开始?", default=True):
                    continue
                
                # 创建输出目录
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # 1. 导出标注数据
                print_section_header("步骤1: 导出标注数据")
                print_info("正在导出...")
                success, data, error = client.export_project(project_id, 'JSON')
                
                if not success:
                    print_error(f"导出失败: {error}")
                    continue
                
                # 保存导出的JSON
                export_json_path = output_path / f"project_{project_id}_export.json"
                with open(export_json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print_success(f"✓ 导出完成: {len(data)} 个任务")
                print_info(f"  JSON已保存: {export_json_path}")
                
                # 2. 应用筛选并下载图片
                if use_filters:
                    print_section_header("步骤2: 应用数据筛选")
                else:
                    print_section_header("步骤2: 下载图片")
                
                images_dir = output_path / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # 准备下载列表
                from ..converters.labelstudio import LabelStudioConverter
                converter = LabelStudioConverter()
                
                # 加载原始 JSON 数据
                with open(export_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                parsed_data = converter.parse_json(export_json_path, include_negative=True)
                
                # 应用筛选条件（在下载之前）
                if use_filters:
                    original_count = len(parsed_data)
                    print_info(f"原始任务数: {original_count}")
                    
                    # 创建 filename 到原始 task 的映射
                    filename_to_task = {}
                    for task in original_data:
                        image_path = task.get('data', {}).get('image', '')
                        if image_path:
                            filename = Path(image_path).name
                            filename_to_task[filename] = task
                    
                    # 创建 task_id 到原始 task 的映射（用于 ID 筛选）
                    task_id_to_task = {task.get('id'): task for task in original_data}
                    
                    # 1. 按任务ID列表筛选
                    if task_ids:
                        id_list = [int(tid.strip()) for tid in task_ids.split(',')]
                        id_set = set(id_list)
                        # 从原始数据中筛选
                        filtered_tasks = [task for task in original_data if task.get('id') in id_set]
                        # 更新 parsed_data
                        filtered_filenames = {Path(t.get('data', {}).get('image', '')).name for t in filtered_tasks}
                        parsed_data = [item for item in parsed_data if item.get('filename') in filtered_filenames]
                        print_info(f"按任务ID筛选: {len(id_list)} 个指定ID，匹配到 {len(parsed_data)} 个任务")
                    
                    # 2. 按任务ID范围筛选
                    elif task_range:
                        start_id, end_id = task_range  # 直接使用元组
                        # 从原始数据中筛选
                        filtered_tasks = [task for task in original_data if start_id <= task.get('id', 0) <= end_id]
                        # 更新 parsed_data
                        filtered_filenames = {Path(t.get('data', {}).get('image', '')).name for t in filtered_tasks}
                        parsed_data = [item for item in parsed_data if item.get('filename') in filtered_filenames]
                        print_info(f"按任务ID范围筛选: {start_id}-{end_id}，匹配到 {len(parsed_data)} 个任务")
                    
                    # 3. 按标签筛选
                    if filter_labels:
                        label_list = [label.strip() for label in filter_labels.split(',')]
                        label_set = set(label_list)
                        
                        def has_matching_label(item):
                            for ann in item.get('annotations', []):
                                item_labels = ann.get('labels', [])
                                if any(label in label_set for label in item_labels):
                                    return True
                            if item.get('category') in label_set:
                                return True
                            return False
                        
                        parsed_data = [item for item in parsed_data if has_matching_label(item)]
                        print_info(f"按标签筛选: {', '.join(label_list)}，匹配到 {len(parsed_data)} 个任务")
                    
                    # 4. 限制最大任务数
                    if max_tasks and max_tasks < len(parsed_data):
                        parsed_data = parsed_data[:int(max_tasks)]
                        print_info(f"限制任务数: 取前 {int(max_tasks)} 个任务")
                    
                    print_success(f"✓ 筛选后: {len(parsed_data)}/{original_count} 个任务将被处理")
                    
                    # 保存筛选后的原始 JSON 数据（用于后续转换）
                    filtered_filenames = {item.get('filename') for item in parsed_data}
                    filtered_original_data = [
                        task for task in original_data 
                        if Path(task.get('data', {}).get('image', '')).name in filtered_filenames
                    ]
                    
                    filtered_json_path = output_path / f"project_{project_id}_filtered.json"
                    with open(filtered_json_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_original_data, f, ensure_ascii=False, indent=2)
                    print_info(f"  筛选后的数据已保存: {filtered_json_path}")
                    
                    # 更新导出文件路径为筛选后的文件
                    export_json_path = filtered_json_path
                    console.print()
                
                print_info(f"正在下载图片到: {images_dir}")
                download_list = converter.prepare_download_list(parsed_data, images_dir)
                
                print_info(f"共 {len(download_list)} 张图片需要下载")
                
                # 批量下载（使用进度条）
                if use_filters:
                    print_section_header("步骤3: 下载筛选后的图片")
                
                from ..ui.display import create_progress_bar
                
                with create_progress_bar() as progress:
                    task_id = progress.add_task("下载图片", total=len(download_list))
                    
                    def progress_callback(current, total, status, filename):
                        progress.update(task_id, advance=1)
                    
                    # 批量下载
                    stats = client.download_images_batch(
                        image_list=download_list,
                        skip_existing=True,
                        max_workers=int(max_workers),
                        progress_callback=progress_callback
                    )
                
                print_success(f"✓ 下载完成")
                print_info(f"  新下载: {stats['downloaded']}")
                print_info(f"  已跳过: {stats['skipped']}")
                if stats['failed'] > 0:
                    print_warning(f"  失败: {stats['failed']}")
                
                # 4/3. 转换为YOLO格式
                step_num = 4 if use_filters else 3
                print_section_header(f"步骤{step_num}: 转换为YOLO格式")
                
                # 调用现有的转换命令
                from ..commands.data import convert_labelstudio
                
                try:
                    # 如果使用了筛选，数据已经被筛选并保存为新文件，不需要再传筛选参数
                    convert_labelstudio(
                        input_file=str(export_json_path),
                        url=default_url,
                        token=default_token,
                        output_dir=str(output_path),
                        task=task_type,
                        format_type='json',
                        skip_existing=True,
                        max_workers=int(max_workers),
                        include_negative=include_negative,
                        max_tasks=None,  # 已在步骤2筛选，不需要再限制
                        task_ids=None,
                        task_range=None,
                        filter_labels=None
                    )
                    
                    print_success("✓ 转换完成")
                    
                    # 询问是否进行数据处理
                    console.print()
                    if confirm_action("是否进行数据处理 (划分、验证、统计)?", default=True):
                        print_section_header("数据处理")
                        
                        import shutil
                        import os
                        
                        # 目标目录
                        raw_data_dir = Path("data/raw")
                        raw_data_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 检查 data/raw 目录是否为空
                        raw_contents = list(raw_data_dir.iterdir())
                        # 过滤掉 .gitkeep 等隐藏文件
                        raw_contents = [f for f in raw_contents if not f.name.startswith('.')]
                        
                        if raw_contents:
                            console.print()
                            print_warning(f"data/raw 目录不为空，包含以下内容:")
                            for item in raw_contents[:10]:  # 最多显示10个
                                print_info(f"  • {item.name}")
                            if len(raw_contents) > 10:
                                print_info(f"  ... 还有 {len(raw_contents) - 10} 个文件/目录")
                            
                            console.print()
                            if not confirm_action("是否清空 data/raw 目录并移动新数据?", default=False):
                                print_warning("取消数据处理")
                                continue
                            
                            # 清空目录（保留 .gitkeep）
                            print_info("正在清空 data/raw 目录...")
                            for item in raw_contents:
                                if item.is_dir():
                                    shutil.rmtree(item)
                                else:
                                    item.unlink()
                            print_success("✓ 目录已清空")
                        
                        # 移动所有内容到 data/raw
                        print_info(f"正在将数据移动到 data/raw 目录...")
                        
                        moved_count = 0
                        for item in output_path.iterdir():
                            # 跳过隐藏文件
                            if item.name.startswith('.'):
                                continue
                            # 保留 dataset.yaml，但跳过其他 yaml/json 文件
                            if item.name != 'dataset.yaml' and (item.name.endswith('.json') or item.name.endswith('.yaml')):
                                continue
                            
                            dest_path = raw_data_dir / item.name
                            
                            # 如果目标已存在，先删除
                            if dest_path.exists():
                                if dest_path.is_dir():
                                    shutil.rmtree(dest_path)
                                else:
                                    dest_path.unlink()
                            
                            # 移动文件/目录
                            shutil.move(str(item), str(dest_path))
                            moved_count += 1
                            print_info(f"  ✓ {item.name} → data/raw/{item.name}")
                        
                        if moved_count > 0:
                            print_success(f"✓ 已移动 {moved_count} 个文件/目录到 data/raw")
                            
                            console.print()
                            print_info("即将进入数据处理菜单...")
                            console.print()
                            
                            # 跳转到数据处理操作
                            run_data_operations()
                        else:
                            print_warning("没有找到可移动的数据文件")
                    
                    else:
                        # 询问是否使用FiftyOne查看
                        console.print()
                        if confirm_action("是否使用FiftyOne查看数据集?", default=True):
                            # 查找生成的dataset.yaml
                            dataset_yaml = output_path / "dataset.yaml"
                            if dataset_yaml.exists():
                                # 加载到FiftyOne
                                from ..integrations.fiftyone_manager import FiftyOneManager
                                fo_mgr = FiftyOneManager()
                                
                                available, error = fo_mgr.ensure_fiftyone()
                                if available:
                                    print_info("正在将数据集复制到 datasets 目录...")
                                    print_info("正在加载数据集到FiftyOne...")
                                    success, ds_name, error = fo_mgr.load_yolo_dataset(
                                        data_yaml_path=str(dataset_yaml),
                                        dataset_name=f"ls_project_{project_id}",
                                        persistent=True,
                                        copy_to_datasets=True
                                    )
                                    
                                    if success:
                                        print_success(f"✓ 数据集已加载: {ds_name}")
                                        print_info(f"  数据集位置: datasets/{ds_name}/")
                                        
                                        if confirm_action("启动FiftyOne可视化?", default=True):
                                            print_info("正在启动FiftyOne App...")
                                            print_info("提示: 按 Ctrl+C 可以关闭可视化并继续")
                                            fo_mgr.launch_app(dataset_name=ds_name, auto_open=True)
                                    else:
                                        print_error(f"加载数据集失败: {error}")
                                else:
                                    print_warning(f"FiftyOne不可用: {error}")
                            else:
                                print_warning("未找到dataset.yaml文件")
                    
                except Exception as e:
                    print_error(f"转换失败: {e}")
            
            console.print()
            if not confirm_action("继续Label Studio操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            import traceback
            traceback.print_exc()
            if not confirm_action("继续?", default=True):
                break


def run_fiftyone_operations():
    """FiftyOne数据集可视化和管理操作"""
    from ..integrations.fiftyone_manager import FiftyOneManager
    
    fo_mgr = FiftyOneManager()
    
    # 检查FiftyOne是否可用
    available, error = fo_mgr.ensure_fiftyone()
    if not available:
        print_error(error)
        print_info("安装FiftyOne: pip install fiftyone")
        return
    
    while True:
        operation = select_fiftyone_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'load':
                # 加载数据集
                print_section_header("加载YOLO数据集到FiftyOne")
                
                yaml_path = input_path("dataset.yaml路径:", default="data/processed/dataset.yaml", must_exist=True)
                dataset_name = input_text("数据集名称 (留空自动生成):", default="")
                
                # 选择任务类型
                task_type = select_task_type_for_predict()
                
                copy_dataset = confirm_action("是否将数据集复制到 datasets 目录统一管理?", default=True)
                
                if not dataset_name:
                    dataset_name = None
                
                if copy_dataset:
                    print_info("正在将数据集复制到 datasets 目录...")
                print_info("正在加载数据集...")
                success, ds_name, error = fo_mgr.load_yolo_dataset(
                    data_yaml_path=yaml_path,
                    dataset_name=dataset_name,
                    persistent=True,
                    copy_to_datasets=copy_dataset,
                    task_type=task_type
                )
                
                if success:
                    print_success(f"✓ 数据集已加载: {ds_name}")
                    if copy_dataset:
                        print_info(f"  数据集位置: datasets/{ds_name}/")
                    
                    if confirm_action("立即启动可视化?", default=True):
                        print_info("正在启动FiftyOne App...")
                        fo_mgr.launch_app(dataset_name=ds_name, auto_open=True)
                else:
                    print_error(f"加载失败: {error}")
            
            elif operation == 'load_predictions':
                # 加载预测结果创建数据集
                print_section_header("加载YOLO预测结果")
                
                images_dir = input_path("图片目录:", must_exist=True)
                predictions_dir = input_path("预测结果目录（包含labels文件夹或txt文件）:", must_exist=True)
                dataset_name = input_text("数据集名称:", default="predictions")
                
                # 选择任务类型
                task_type = select_task_type_for_predict()
                
                # 检查是否有 classes.txt
                classes_file = Path(predictions_dir) / 'classes.txt'
                if classes_file.exists():
                    print_success(f"✓ 找到类别文件: {classes_file}")
                    classes = None  # 自动从文件读取
                    classes_str = ""
                else:
                    print_warning("未找到 classes.txt，请手动输入类别列表")
                    classes_str = input_text("类别列表（逗号分隔）:", default="class1,class2")
                    classes = [c.strip() for c in classes_str.split(',') if c.strip()]
                
                conf_threshold = float(input_text("置信度阈值:", default="0.25"))
                
                print_info("正在创建预测数据集...")
                success, ds_name, error = fo_mgr.load_predictions_dataset(
                    images_dir=images_dir,
                    predictions_dir=predictions_dir,
                    classes=classes,
                    dataset_name=dataset_name,
                    conf_threshold=conf_threshold,
                    persistent=True,
                    task_type=task_type
                )
                
                if success:
                    print_success(f"✓ 预测数据集已创建: {ds_name}")
                    
                    if confirm_action("立即启动可视化?", default=True):
                        print_info("正在启动FiftyOne App...")
                        fo_mgr.launch_app(dataset_name=ds_name, auto_open=True)
                else:
                    print_error(f"创建失败: {error}")
            
            elif operation == 'add_predictions':
                # 添加预测结果到现有数据集
                print_section_header("添加预测结果到数据集")
                
                # 获取数据集列表
                success, datasets, error = fo_mgr.list_datasets()
                
                if not success:
                    print_error(f"获取数据集列表失败: {error}")
                    continue
                
                if not datasets:
                    print_warning("没有可用的数据集")
                    print_info("请先使用 'load' 命令加载数据集")
                    continue
                
                # 选择数据集
                dataset_name = select_fiftyone_dataset(list(datasets))
                if dataset_name is None:
                    continue
                
                predictions_dir = input_path("预测结果目录（包含labels文件夹或txt文件）:", must_exist=True)
                
                # 选择任务类型
                task_type = select_task_type_for_predict()
                
                # 检查是否有 classes.txt
                classes_file = Path(predictions_dir) / 'classes.txt'
                if classes_file.exists():
                    print_success(f"✓ 找到类别文件: {classes_file}")
                    classes = None  # 自动从文件读取
                else:
                    print_warning("未找到 classes.txt，将尝试从数据集的 ground_truth 中获取类别")
                    if not confirm_action("是否手动输入类别列表?", default=False):
                        classes = None  # 自动从数据集获取
                    else:
                        classes_str = input_text("类别列表（逗号分隔）:", default="class1,class2")
                        classes = [c.strip() for c in classes_str.split(',') if c.strip()]
                
                field_name = input_text("预测结果字段名:", default="predictions")
                conf_threshold = float(input_text("置信度阈值:", default="0.0"))
                
                print_info(f"正在添加预测结果到数据集 '{dataset_name}'...")
                success, stats, error = fo_mgr.add_predictions_to_dataset(
                    dataset_name=dataset_name,
                    predictions_dir=predictions_dir,
                    classes=classes,
                    field_name=field_name,
                    conf_threshold=conf_threshold,
                    task_type=task_type
                )
                
                if success:
                    print_success(f"✓ 预测结果已添加")
                    print_info(f"  总样本数: {stats['total_samples']}")
                    print_info(f"  更新样本数: {stats['updated_samples']}")
                    print_info(f"  总预测数: {stats['total_predictions']}")
                    if stats['skipped_low_conf'] > 0:
                        print_info(f"  跳过低置信度: {stats['skipped_low_conf']}")
                    
                    if confirm_action("立即启动可视化查看?", default=True):
                        print_info("正在启动FiftyOne App...")
                        print_info(f"💡 提示: 在App中可以对比 'ground_truth' 和 '{field_name}'")
                        fo_mgr.launch_app(dataset_name=dataset_name, auto_open=True)
                else:
                    print_error(f"添加失败: {error}")
            
            elif operation == 'launch':
                # 启动可视化
                print_section_header("启动FiftyOne可视化")
                
                # 获取数据集列表
                success, datasets, error = fo_mgr.list_datasets()
                
                if not success:
                    print_error(f"获取数据集列表失败: {error}")
                    continue
                
                if not datasets:
                    print_warning("没有可用的数据集")
                    print_info("请先使用 'load' 命令加载数据集")
                    continue
                
                # 选择数据集
                dataset_name = select_fiftyone_dataset(list(datasets))
                if dataset_name is None:
                    continue
                
                print_info(f"正在启动FiftyOne App (数据集: {dataset_name})...")
                print_info("提示: 浏览器会自动打开，按 Ctrl+C 可以关闭")
                
                success, error = fo_mgr.launch_app(dataset_name=dataset_name, auto_open=True)
                
                if not success:
                    print_error(f"启动失败: {error}")
            
            elif operation == 'list':
                # 列出所有数据集
                print_section_header("FiftyOne数据集列表")
                
                success, datasets, error = fo_mgr.list_datasets()
                
                if not success:
                    print_error(f"获取失败: {error}")
                    continue
                
                if not datasets:
                    print_warning("没有找到任何数据集")
                    continue
                
                print_info(f"共 {len(datasets)} 个数据集:")
                for ds in datasets:
                    console.print(f"  • {ds}")
            
            elif operation == 'info':
                # 查看数据集信息
                print_section_header("数据集详细信息")
                
                # 获取数据集列表
                success, datasets, error = fo_mgr.list_datasets()
                
                if not success:
                    print_error(f"获取数据集列表失败: {error}")
                    continue
                
                if not datasets:
                    print_warning("没有可用的数据集")
                    continue
                
                # 选择数据集
                dataset_name = select_fiftyone_dataset(list(datasets))
                if dataset_name is None:
                    continue
                
                print_info(f"正在获取数据集信息: {dataset_name}")
                success, info, error = fo_mgr.get_dataset_info(dataset_name)
                
                if not success:
                    print_error(f"获取失败: {error}")
                    continue
                
                # 显示信息
                from ..ui.display import print_key_value
                
                console.print()
                print_key_value("数据集名称", info['name'])
                print_key_value("样本总数", info['total_samples'])
                print_key_value("媒体类型", info['media_type'])
                print_key_value("持久化", "是" if info['persistent'] else "否")
                
                if 'splits' in info:
                    console.print()
                    print_info("数据集划分:")
                    for split, count in info['splits'].items():
                        print_key_value(f"  {split}", count)
                
                if 'classes' in info:
                    console.print()
                    print_key_value("类别数", info['num_classes'])
                    print_info("类别列表:")
                    for cls in info['classes']:
                        console.print(f"  • {cls}")
                    
                    if 'class_counts' in info:
                        console.print()
                        print_info("类别分布:")
                        for cls, count in info['class_counts'].items():
                            print_key_value(f"  {cls}", count)
            
            elif operation == 'delete':
                # 删除数据集
                print_section_header("删除数据集")
                
                # 获取数据集列表
                success, datasets, error = fo_mgr.list_datasets()
                
                if not success:
                    print_error(f"获取数据集列表失败: {error}")
                    continue
                
                if not datasets:
                    print_warning("没有可用的数据集")
                    continue
                
                # 选择数据集（允许"删除全部"选项）
                dataset_name = select_fiftyone_dataset(list(datasets), allow_delete_all=True)
                if dataset_name is None:
                    continue
                
                # 处理删除全部
                if dataset_name == "__DELETE_ALL__":
                    print_warning(f"⚠️  即将删除所有 {len(datasets)} 个数据集!")
                    print_info("数据集列表:")
                    for ds in datasets:
                        print_info(f"  - {ds}")
                    console.print()
                    print_info("注意: 这不会删除原始图片和标签文件")
                    console.print()
                    
                    if confirm_action("确认删除全部数据集?", default=False):
                        success, deleted_count, error = fo_mgr.delete_all_datasets()
                        
                        if success:
                            print_success(f"✓ 已成功删除 {deleted_count} 个数据集")
                        else:
                            if deleted_count > 0:
                                print_warning(f"⚠️  已删除 {deleted_count} 个数据集，但有错误: {error}")
                            else:
                                print_error(f"删除失败: {error}")
                else:
                    # 删除单个数据集
                    print_warning(f"即将删除数据集: {dataset_name}")
                    print_info("注意: 这不会删除原始图片和标签文件")
                    
                    if confirm_action("确认删除?", default=False):
                        success, error = fo_mgr.delete_dataset(dataset_name)
                        
                        if success:
                            print_success(f"✓ 数据集已删除: {dataset_name}")
                        else:
                            print_error(f"删除失败: {error}")
            
            console.print()
            if not confirm_action("继续FiftyOne操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            import traceback
            traceback.print_exc()
            if not confirm_action("继续?", default=True):
                break


def run_validate_operations():
    """验证操作"""
    while True:
        operation = select_validate_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'run':
                # 验证单个模型
                print_section_header("验证模型性能")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                data_path = input_path("数据集配置文件:", default="data/processed/dataset.yaml", must_exist=False)
                split = select_validation_split()
                
                # 任务类型选择
                task_choice = select_option(
                    "任务类型:",
                    choices=[
                        "自动推断（从模型名称推断）",
                        "检测 (detect)",
                        "分割 (segment)",
                        "分类 (classify)",
                        "姿势估计 (pose)",
                    ],
                    default="自动推断（从模型名称推断）"
                )
                
                if "自动推断" in task_choice:
                    task = None  # 自动推断
                elif "检测" in task_choice:
                    task = "detect"
                elif "分割" in task_choice:
                    task = "segment"
                elif "姿势" in task_choice:
                    task = "pose"
                else:  # 分类
                    task = "classify"
                
                # 验证参数（分类任务不需要 conf 和 iou）
                if task != "classify":
                    conf = input_number("置信度阈值 (训练验证用0.001，部署验证用0.25):", default=0.001, min_value=0.0, max_value=1.0)
                    iou = input_number("IoU阈值:", default=0.6, min_value=0.0, max_value=1.0)
                else:
                    # 分类任务使用默认值，但不显示
                    conf = 0.001
                    iou = 0.6
                
                batch = int(input_number("批次大小:", default=16, min_value=1))
                
                # 可选项
                save_json = confirm_action("保存JSON格式结果?", default=True)
                plots = confirm_action("生成可视化图表?", default=True)
                
                console.print()
                print_info("验证配置:")
                print_info(f"  模型: {model_path}")
                print_info(f"  数据集: {data_path}")
                print_info(f"  验证集: {split}")
                print_info(f"  任务类型: {task if task else '自动推断'}")
                
                # 只有检测和分割任务才显示 conf 和 iou
                if task != "classify":
                    print_info(f"  置信度阈值: {conf}")
                    print_info(f"  IoU阈值: {iou}")
                
                print_info(f"  批次大小: {batch}")
                
                if confirm_action("确认开始验证?"):
                    from ..commands.validate import validate_model
                    validate_model(
                        model=model_path,
                        data=data_path,
                        split=split,
                        task=task,
                        batch=batch,
                        imgsz=640,
                        conf=conf,
                        iou=iou,
                        device='auto',
                        save_json=save_json,
                        save_hybrid=False,
                        plots=plots,
                        verbose=True,
                        project=None,
                        name=None,
                    )
            
            elif operation == 'compare':
                # 比较多个模型
                print_section_header("比较多个模型")
                
                print_info("请输入要比较的模型路径，用逗号分隔")
                print_info("示例: model1.pt,model2.pt,model3.pt")
                models_str = input_text("模型路径 (逗号分隔):", default="")
                
                if not models_str:
                    print_warning("未输入模型路径")
                    continue
                
                data_path = input_path("数据集配置文件:", default="data/processed/dataset.yaml", must_exist=False)
                
                # 任务类型选择
                task_choice = select_option(
                    "任务类型:",
                    choices=[
                        "自动推断（从第一个模型名称推断）",
                        "检测 (detect)",
                        "分割 (segment)",
                        "分类 (classify)",
                        "姿势估计 (pose)",
                    ],
                    default="自动推断（从第一个模型名称推断）"
                )
                
                if "自动推断" in task_choice:
                    task = None  # 自动推断
                elif "检测" in task_choice:
                    task = "detect"
                elif "分割" in task_choice:
                    task = "segment"
                elif "姿势" in task_choice:
                    task = "pose"
                else:  # 分类
                    task = "classify"
                
                # 验证参数
                batch = int(input_number("批次大小:", default=16, min_value=1))
                
                # 分类任务不需要 conf 和 iou
                if task != "classify":
                    conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                    iou = input_number("IoU阈值:", default=0.6, min_value=0.0, max_value=1.0)
                else:
                    # 分类任务使用默认值
                    conf = 0.001
                    iou = 0.6
                
                console.print()
                print_info(f"将比较以下模型:")
                for i, model in enumerate(models_str.split(','), 1):
                    print_info(f"  {i}. {model.strip()}")
                print_info(f"任务类型: {task if task else '自动推断'}")
                print_info(f"批次大小: {batch}")
                
                # 只有检测和分割任务才显示阈值参数
                if task != "classify":
                    print_info(f"置信度阈值: {conf}")
                    print_info(f"IoU阈值: {iou}")
                
                if confirm_action("确认开始对比?"):
                    from ..commands.validate import compare_models
                    compare_models(
                        models=models_str,
                        data=data_path,
                        task=task,
                        split='val',
                        batch=batch,
                        imgsz=640,
                        conf=conf,
                        iou=iou,
                        device='auto',
                    )
            
            console.print()
            if not confirm_action("继续验证操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


@app.command()
def start():
    """启动交互式模式"""
    
    try:
        # 清屏并显示Logo
        clear_screen()
        print_logo()
        
        print_info("欢迎使用 YOLO CLI 交互式模式！")
        print_info("使用方向键选择，按回车确认\n")
        
        while True:
            try:
                choice = select_main_menu()
                
                if choice == 'exit':
                    print_success("感谢使用 YOLO CLI！再见！")
                    break
                
                elif choice == 'model':
                    run_model_operations()
                
                elif choice == 'data':
                    run_data_operations()
                
                elif choice == 'train':
                    run_train_operations()
                
                elif choice == 'quick':
                    # 一键训练
                    run_quick_train()
                
                elif choice == 'validate':
                    run_validate_operations()
                
                elif choice == 'detect':
                    run_detect_operations()
                
                elif choice == 'labelstudio':
                    # Label Studio管理
                    run_labelstudio_operations()
                
                elif choice == 'fiftyone':
                    # FiftyOne可视化
                    run_fiftyone_operations()
                
            except KeyboardInterrupt:
                console.print()
                if confirm_action("确认退出?", default=True):
                    print_success("再见！")
                    break
                continue
    
    except Exception as e:
        print_error(f"发生错误: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
