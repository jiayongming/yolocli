#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式模式"""

import typer
from pathlib import Path

from ..ui.prompts import (
    select_main_menu, select_model_operation, select_data_operation,
    select_train_operation, select_detect_operation, select_validate_operation,
    select_validation_split, select_option,
    select_yolo_version, select_task_type, select_model_size, select_device,
    select_augmentation_preset, select_export_formats,
    build_training_config, input_text, input_path, input_number,
    confirm_action,
    select_labelstudio_operation, select_labelstudio_project, input_labelstudio_config,
    select_fiftyone_operation, select_fiftyone_dataset
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
                
                # 对于检测任务，询问是否包含负样本
                include_negative = True
                if task_type == 'detect':
                    console.print()
                    print_info("💡 负样本说明：")
                    print_info("   - 无标注的图片可作为负样本训练")
                    print_info("   - 有助于减少误报，提高模型鲁棒性")
                    print_info("   - 推荐包含10-20%的负样本")
                    console.print()
                    include_negative = confirm_action("包含无标注图片作为负样本?", default=True)
                
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
                        max_workers=max_workers,
                        include_negative=include_negative
                    )
            
            elif operation == 'split':
                # 划分数据集
                print_section_header("划分数据集")
                
                task_type = select_task_type()
                
                # 配置输出目录
                output_dir = input_path("输出目录:", default="data/processed", must_exist=False)
                
                # 选择划分方式
                from ..ui.prompts import select_option
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
                                task=task_type
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
                                task=task_type
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
                            create_empty_labels=create_empty
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
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                
                from ..commands.data import verify_dataset
                verify_dataset(data_path=data_path, task=task_type)
            
            elif operation == 'stats':
                # 数据统计
                print_section_header("数据统计")
                
                task_type = select_task_type()
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                detailed = confirm_action("显示详细统计 (含正负样本统计)?", default=True)
                
                # 如果是分类任务且需要详细统计，让用户选择正类
                positive_classes_str = None
                if task_type == 'classify' and detailed:
                    from pathlib import Path
                    data_path_obj = Path(data_path)
                    
                    # 获取所有类别
                    all_classes = set()
                    for split in ['train', 'val', 'test']:
                        split_dir = data_path_obj / 'images' / split
                        if split_dir.exists():
                            classes = [d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                            all_classes.update(classes)
                    
                    if all_classes:
                        print_info(f"检测到 {len(all_classes)} 个类别: {', '.join(sorted(all_classes))}")
                        console.print()
                        print_info("💡 正负样本统计说明：")
                        print_info("   - 选择一个或多个类别作为「正类」")
                        print_info("   - 其余类别将自动归为「负类」")
                        print_info("   - 适用于异常检测、二分类等场景")
                        console.print()
                        
                        if confirm_action("是否选择正类进行正负样本统计?", default=True):
                            from ..ui.prompts import select_multiple
                            
                            console.print()
                            print_info("📋 操作说明：")
                            print_info("   1. 使用 ↑↓ 键移动")
                            print_info("   2. 使用 空格键 选择/取消选择")
                            print_info("   3. 按 回车键 确认选择")
                            console.print()
                            
                            positive_classes = select_multiple(
                                "选择正类 (空格选择，回车确认):",
                                sorted(list(all_classes))
                            )
                            
                            if positive_classes:
                                positive_classes_str = ','.join(positive_classes)
                                console.print()
                                print_success(f"✓ 已选择正类: {positive_classes_str}")
                                negative_classes = [c for c in all_classes if c not in positive_classes]
                                if negative_classes:
                                    print_info(f"  负类: {', '.join(sorted(negative_classes))}")
                            else:
                                console.print()
                                print_warning("⚠️  未选择任何正类，将跳过正负样本统计")
                
                from ..commands.data import dataset_stats
                dataset_stats(data_path=data_path, detailed=detailed, task=task_type, positive_classes=positive_classes_str)
            
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
        from ..ui.prompts import select_option
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
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                if confirm_action("确认检测?"):
                    from ..commands.detect import detect_image
                    detect_image(model=model_path, image=image_path, conf=conf, iou=0.45, 
                               output=None, save_txt=True, save_json=True, show=False, device='auto')
            
            elif operation == 'batch':
                # 批量检测
                print_section_header("批量检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                source_path = input_path("图片目录或视频:", default="test_images/", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                if confirm_action("确认批量检测?"):
                    from ..commands.detect import detect_batch
                    detect_batch(model=model_path, source=source_path, conf=conf, iou=0.45,
                               output=None, save_txt=True, save_json=True, device='auto', batch=1)
            
            elif operation == 'video':
                # 视频检测
                print_section_header("视频检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                video_path = input_path("视频路径:", default="test_video.mp4", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                show = confirm_action("实时显示?", default=False)
                
                if confirm_action("确认检测?"):
                    from ..commands.detect import detect_video
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
                
                # 对于检测任务，询问是否包含负样本
                include_negative = True
                if task_type == 'detect':
                    console.print()
                    print_info("💡 负样本说明：")
                    print_info("   - 无标注的图片可作为负样本训练")
                    print_info("   - 有助于减少误报，提高模型鲁棒性")
                    print_info("   - 推荐包含10-20%的负样本")
                    console.print()
                    include_negative = confirm_action("包含无标注图片作为负样本?", default=True)
                
                console.print()
                print_info("将执行以下操作:")
                print_info(f"  1. 导出项目 {project_id} 的标注数据")
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
                
                # 2. 下载图片
                print_section_header("步骤2: 下载图片")
                images_dir = output_path / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                print_info(f"正在下载图片到: {images_dir}")
                
                # 准备下载列表
                from ..converters.labelstudio import LabelStudioConverter
                converter = LabelStudioConverter()
                parsed_data = converter.parse_json(export_json_path, include_negative=True)
                download_list = converter.prepare_download_list(parsed_data, images_dir)
                
                print_info(f"共 {len(download_list)} 张图片需要下载")
                
                # 批量下载（使用进度条）
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
                
                # 3. 转换为YOLO格式
                print_section_header("步骤3: 转换为YOLO格式")
                
                # 调用现有的转换命令
                from ..commands.data import convert_labelstudio
                
                try:
                    convert_labelstudio(
                        input_file=str(export_json_path),
                        url=default_url,
                        token=default_token,
                        output_dir=str(output_path),
                        task=task_type,
                        format_type='json',
                        skip_existing=True,
                        max_workers=int(max_workers),
                        include_negative=include_negative
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
                            # 跳过非数据文件
                            if item.name.startswith('.') or item.name.endswith('.json') or item.name.endswith('.yaml'):
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
                    copy_to_datasets=copy_dataset
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
                
                # 检查是否有 classes.txt
                from pathlib import Path
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
                    persistent=True
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
                
                # 检查是否有 classes.txt
                from pathlib import Path
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
                    conf_threshold=conf_threshold
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
                
                # 选择数据集
                dataset_name = select_fiftyone_dataset(list(datasets))
                if dataset_name is None:
                    continue
                
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
                    ],
                    default="自动推断（从模型名称推断）"
                )
                
                if "自动推断" in task_choice:
                    task = None  # 自动推断
                elif "检测" in task_choice:
                    task = "detect"
                elif "分割" in task_choice:
                    task = "segment"
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
                    ],
                    default="自动推断（从第一个模型名称推断）"
                )
                
                if "自动推断" in task_choice:
                    task = None  # 自动推断
                elif "检测" in task_choice:
                    task = "detect"
                elif "分割" in task_choice:
                    task = "segment"
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
