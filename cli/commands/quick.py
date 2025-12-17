#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键训练命令"""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..core.config import ConfigManager
from ..core.version import YOLOVersionManager
from ..core.utils import (
    detect_device, get_device_name,
    TaskType, validate_task_type, get_model_name_with_task
)
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_step, console
)

# 导入其他命令的函数
from .data import split_dataset, generate_yaml, verify_dataset, dataset_stats, prepare_classify
from .model import download, list_models
from .train import start_training

app = typer.Typer(help="一键训练命令")


@app.command("train")
def quick_train(
    images_dir: str = typer.Option(..., "--images", "-i", help="原始图像目录"),
    labels_dir: Optional[str] = typer.Option(None, "--labels", "-l", help="原始标签目录（检测/分割必需，分类可选）"),
    classes_file: Optional[str] = typer.Option(None, "--classes", "-c", help="类别文件路径 (默认: data/raw/classes.txt)"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify)"),
    model_version: str = typer.Option("yolo11", "--version", "-v", help="YOLO版本 (yolo11/yolov8)"),
    model_size: str = typer.Option("s", "--size", "-s", help="模型大小 (n/s/m/l/x)"),
    epochs: int = typer.Option(200, "--epochs", "-e", help="训练轮数"),
    batch: int = typer.Option(16, "--batch", "-b", help="批次大小"),
    imgsz: int = typer.Option(640, "--imgsz", help="图像尺寸"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
    augmentation: str = typer.Option("balanced", "--augmentation", "-a", help="数据增强策略"),
    ratios: str = typer.Option("0.7:0.2:0.1", "--ratios", "-r", help="数据集划分比例"),
    skip_verify: bool = typer.Option(False, "--skip-verify", help="跳过数据验证"),
    skip_stats: bool = typer.Option(False, "--skip-stats", help="跳过数据统计"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="项目目录"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="实验名称"),
):
    """
    一键训练：自动完成数据处理、模型下载和训练的完整流程
    
    支持三种任务类型：检测(detect)、分割(segment)、分类(classify)
    
    示例:
    
        # 检测任务
        python yolo_cli.py quick train --task detect --images data/raw/images --labels data/raw/labels
        
        # 分割任务
        python yolo_cli.py quick train --task segment --images data/raw/images --labels data/raw/labels
        
        # 分类任务（图像按类别目录组织）
        python yolo_cli.py quick train --task classify --images data/raw/images
        
        # 分类任务（从 images + labels 转换）
        python yolo_cli.py quick train --task classify --images data/raw/images --labels data/raw/labels
    """
    
    console.print()
    print_section_header("🚀 一键训练模式")
    
    # 验证任务类型
    task = validate_task_type(task)
    task_type = TaskType.from_string(task)
    print_info(f"任务类型: {task.upper()}")
    
    # 根据任务类型调整默认参数
    if task_type == TaskType.CLASSIFY:
        if imgsz == 640:
            imgsz = 224
            print_info(f"分类任务使用图像尺寸: {imgsz}")
        if epochs == 200:
            epochs = 100
            print_info(f"分类任务使用训练轮数: {epochs}")
        if batch == 16:
            batch = 32
            print_info(f"分类任务使用批次大小: {batch}")
    
    total_steps = 8 if task == 'classify' else 7
    current_step = 0
    
    try:
        # 验证输入目录
        images_path = Path(images_dir)
        
        if not images_path.exists():
            print_error(f"图像目录不存在: {images_dir}")
            raise typer.Exit(1)
        
        # 对于检测/分割任务，labels_dir 是必需的
        if task_type != TaskType.CLASSIFY and labels_dir is None:
            print_error(f"{task} 任务需要指定 --labels 参数")
            print_info("示例: python yolo_cli.py quick train --task detect --images data/raw/images --labels data/raw/labels")
            raise typer.Exit(1)
        
        labels_path = Path(labels_dir) if labels_dir else None
        
        if labels_path and not labels_path.exists():
            print_error(f"标签目录不存在: {labels_dir}")
            raise typer.Exit(1)
        
        # 确定classes文件
        if classes_file is None:
            # 尝试在多个位置查找
            possible_classes = [
                Path("data/raw/classes.txt"),
                images_path.parent / "classes.txt",
            ]
            if labels_path:
                possible_classes.append(labels_path.parent / "classes.txt")
            
            for cls_file in possible_classes:
                if cls_file.exists():
                    classes_file = str(cls_file)
                    break
            
            # 对于分类任务，如果没有找到 classes.txt，尝试从目录结构提取
            if classes_file is None and task_type == TaskType.CLASSIFY:
                if images_path.exists():
                    subdirs = [d.name for d in images_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if subdirs:
                        # 自动创建 classes.txt
                        classes_file = str(images_path.parent / "classes.txt")
                        with open(classes_file, 'w') as f:
                            for class_name in sorted(subdirs):
                                f.write(f"{class_name}\n")
                        print_info(f"从目录结构自动生成类别文件: {classes_file}")
            
            if classes_file is None:
                print_error("未找到 classes.txt 文件")
                print_info("请使用 --classes 参数指定类别文件，或将其放在 data/raw/classes.txt")
                raise typer.Exit(1)
        
        print_info(f"类别文件: {classes_file}")
        
        # 配置路径
        config = ConfigManager()
        
        # 统一使用 data/processed 作为输出目录（所有任务类型）
        default_output_dir = config.get_path('data_processed', absolute=True)
        output_dir = default_output_dir
        
        # ========== 步骤1: 数据集划分/检查 ==========
        current_step += 1
        print_step(current_step, total_steps, "数据集划分/检查")
        console.print()
        
        # 检查数据集是否已经划分
        data_already_split = False
        if task_type == TaskType.CLASSIFY:
            # 检查分类数据集是否已经划分（检查 images/train, images/val 等目录）
            train_dir = images_path / 'images' / 'train'
            val_dir = images_path / 'images' / 'val'
            if train_dir.exists() and val_dir.exists():
                # 检查是否有类别子目录
                train_classes = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if train_classes:
                    data_already_split = True
                    output_dir = images_path  # 使用现有目录
                    print_info(f"检测到已划分的分类数据集，跳过划分步骤")
                    print_info(f"训练集类别数: {len(train_classes)}")
        else:
            # 检查检测/分割数据集是否已经划分
            train_images = images_path / 'images' / 'train'
            train_labels = images_path / 'labels' / 'train'
            val_images = images_path / 'images' / 'val'
            val_labels = images_path / 'labels' / 'val'
            if (train_images.exists() and train_labels.exists() and 
                val_images.exists() and val_labels.exists()):
                data_already_split = True
                output_dir = images_path  # 使用现有目录
                print_info(f"检测到已划分的数据集，跳过划分步骤")
        
        # 如果数据未划分，则执行划分
        if not data_already_split:
            print_info("数据集未划分，开始划分...")
            if task_type == TaskType.CLASSIFY:
                # 检查分类数据是否已经按类别组织
                classify_organized = False
                if images_path.exists():
                    subdirs = [d for d in images_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if subdirs:
                        # 检查子目录中是否有图片（判断是否为类别目录）
                        for subdir in subdirs[:3]:  # 检查前3个目录
                            images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')) + list(subdir.glob('*.jpeg'))
                            if images:
                                classify_organized = True
                                break
                
                if classify_organized:
                    # 数据已按类别组织，使用 split_dataset
                    print_info(f"检测到按类别组织的分类数据: {[d.name for d in subdirs]}")
                    split_dataset(
                        images_dir=None,
                        labels_dir=None,
                        source_dir=images_dir,
                        output_dir=str(output_dir),
                        ratios=ratios,
                        seed=42,
                        task='classify'
                    )
                else:
                    # 需要从 images + labels 组织为分类结构
                    if labels_dir:
                        print_info("检测到 images + labels 结构，转换为分类数据集...")
                        prepare_classify(
                            images_dir=images_dir,
                            labels_dir=labels_dir,
                            classes_file=classes_file,
                            output_dir=str(output_dir),
                            ratios=ratios,
                            seed=42
                        )
                    else:
                        print_error("分类任务的图像目录应按类别组织（每个类别一个子目录）")
                        print_info("或者使用 --labels 参数提供标签文件来转换数据")
                        raise typer.Exit(1)
            else:
                # 检测/分割任务：划分数据集
                if not labels_dir:
                    print_error(f"{task} 任务需要指定 --labels 参数")
                    raise typer.Exit(1)
                split_dataset(
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    source_dir=None,
                    output_dir=str(output_dir),
                    ratios=ratios,
                    seed=42,
                    task=task
                )
        
        print_success("✓ 数据集准备完成")
        console.print()
        
        # ========== 步骤2: 生成dataset.yaml ==========
        current_step += 1
        print_step(current_step, total_steps, "生成dataset.yaml配置")
        console.print()
        
        # 对于分类任务，使用划分后生成的 classes.txt
        if task_type == TaskType.CLASSIFY and not data_already_split:
            generated_classes = output_dir / 'classes.txt'
            if generated_classes.exists():
                classes_file = str(generated_classes)
                print_info(f"使用划分后生成的类别文件: {classes_file}")
        
        dataset_yaml = "data/dataset.yaml"
        generate_yaml(
            data_path=str(output_dir),
            classes_file=classes_file,
            output=dataset_yaml,
            train_dir=None,
            val_dir=None,
            test_dir=None,
            task=task
        )
        
        print_success(f"✓ 配置文件生成: {dataset_yaml}")
        console.print()
        
        # ========== 步骤3: 验证数据集 ==========
        if not skip_verify:
            current_step += 1
            print_step(current_step, total_steps, "验证数据集")
            console.print()
            
            verify_dataset(data_path=str(output_dir), task=task)
            
            print_success("✓ 数据集验证完成")
            console.print()
        else:
            print_warning("⊘ 跳过数据验证")
            console.print()
        
        # ========== 步骤4: 数据统计 ==========
        if not skip_stats:
            current_step += 1
            print_step(current_step, total_steps, "数据统计分析")
            console.print()
            
            dataset_stats(data_path=str(output_dir), detailed=True, task=task)
            
            print_success("✓ 数据统计完成")
            console.print()
        else:
            print_warning("⊘ 跳过数据统计")
            console.print()
        
        # ========== 步骤5: 检查/下载模型 ==========
        current_step += 1
        print_step(current_step, total_steps, "检查模型")
        console.print()
        
        # 标准化版本并添加任务后缀
        model_version = YOLOVersionManager.normalize_version(model_version)
        base_model_name = YOLOVersionManager.get_model_name(model_version, model_size)
        model_name = get_model_name_with_task(base_model_name, task)
        
        # 检查模型是否存在
        models_dir = config.get_path('models', absolute=True) / 'weights'
        model_path = models_dir / model_name
        
        if model_path.exists():
            print_success(f"✓ 找到模型: {model_name}")
            print_info(f"  路径: {model_path}")
        else:
            print_warning(f"⊗ 模型不存在: {model_name}")
            print_info(f"  开始下载...")
            console.print()
            
            download(
                version=model_version,
                size=[model_size],
                task=task,
                all=False,
                output_dir=str(models_dir)
            )
            
            # 验证下载是否成功
            if model_path.exists():
                print_success(f"✓ 模型下载完成: {model_name}")
                print_info(f"  路径: {model_path}")
            else:
                print_error(f"✗ 模型下载失败: {model_name}")
                raise typer.Exit(1)
        
        console.print()
        
        # ========== 步骤6: 自动检测设备 ==========
        current_step += 1
        print_step(current_step, total_steps, "配置训练环境")
        console.print()
        
        if device == 'auto':
            device = detect_device()
        
        device_name = get_device_name(device)
        print_info(f"训练设备: {device_name}")
        print_info(f"模型: {model_name}")
        print_info(f"数据集: {dataset_yaml}")
        print_info(f"训练轮数: {epochs}")
        print_info(f"批次大小: {batch}")
        print_info(f"图像尺寸: {imgsz}")
        print_info(f"数据增强: {augmentation}")
        
        console.print()
        
        # ========== 步骤7: 开始训练 ==========
        current_step += 1
        print_step(current_step, total_steps, "开始训练")
        console.print()
        
        print_info("训练即将开始，这可能需要较长时间...")
        print_info("按 Ctrl+C 可以中断训练")
        console.print()
        
        # 使用完整模型路径避免重复下载
        full_model_path = str(model_path) if model_path.exists() else model_name
        
        # 保存 project 和 name 的值用于后续显示
        actual_project = project if project else str(config.get_path('results', absolute=True) / 'training')
        
        # 生成默认名称（如果未指定）
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_stem = Path(full_model_path).stem if Path(full_model_path).exists() else model_name
            actual_name = f"{model_stem}_{timestamp}"
        else:
            actual_name = name
        
        start_training(
            model=full_model_path,
            data=dataset_yaml,
            task=task,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            augmentation=augmentation,
            patience=50,
            save_period=10,
            resume=False,
            pretrained=True,
            # 任务特定参数 - 显式传递 None
            overlap_mask=None,
            mask_ratio=None,
            dropout=None,
        )
        
        # 训练完成
        console.print()
        print_section_header("🎉 一键训练完成")
        
        print_success("所有步骤已完成！")
        console.print()
        
        # 显示训练结果位置（使用实际的保存路径）
        results_path = Path(actual_project) / actual_name
        best_model_path = results_path / 'weights' / 'best.pt'
        
        print_info("训练结果位置:")
        print_info(f"  项目目录: {results_path.absolute()}")
        if best_model_path.exists():
            print_info(f"  最佳模型: {best_model_path.absolute()}")
        else:
            print_info(f"  最佳模型: {best_model_path}")
        print_info(f"  数据集配置: {Path(dataset_yaml).absolute()}")
        
        console.print()
        print_info("下一步操作:")
        print_info("  1. 查看训练结果: ls results/training/")
        print_info(f"  2. 测试模型: python yolo_cli.py detect image <best.pt> <test.jpg>")
        print_info(f"  3. 导出模型: python yolo_cli.py model export <best.pt> --format onnx")
        
    except KeyboardInterrupt:
        console.print()
        print_warning("\n训练被用户中断")
        raise typer.Exit(130)
    except Exception as e:
        console.print()
        print_error(f"\n一键训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("resume")
def quick_resume(
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="检查点路径"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="额外训练轮数"),
):
    """
    快速恢复训练
    
    自动查找最新的检查点并恢复训练
    """
    
    print_section_header("🔄 快速恢复训练")
    
    try:
        from .train import resume_training
        
        if checkpoint is None:
            print_info("自动查找最新检查点...")
            
            # 在results目录查找最新的last.pt
            config = ConfigManager()
            results_dir = config.get_path('results', absolute=True) / 'training'
            
            if results_dir.exists():
                last_pts = list(results_dir.rglob('last.pt'))
                if last_pts:
                    checkpoint = str(max(last_pts, key=lambda p: p.stat().st_mtime))
                    print_success(f"找到检查点: {checkpoint}")
                else:
                    print_error("未找到检查点文件")
                    raise typer.Exit(1)
            else:
                print_error("训练结果目录不存在")
                raise typer.Exit(1)
        
        console.print()
        resume_training(checkpoint=checkpoint, project=None, name=None)
        
    except Exception as e:
        print_error(f"恢复训练失败: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
