#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键训练命令"""

import typer
from pathlib import Path
from typing import Optional

from ..core.config import ConfigManager
from ..core.version import YOLOVersionManager
from ..core.utils import detect_device, get_device_name
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_step, console
)

# 导入其他命令的函数
from .data import split_dataset, generate_yaml, verify_dataset, dataset_stats
from .model import download, list_models
from .train import start_training

app = typer.Typer(help="一键训练命令")


@app.command("train")
def quick_train(
    images_dir: str = typer.Option(..., "--images", "-i", help="原始图像目录"),
    labels_dir: str = typer.Option(..., "--labels", "-l", help="原始标签目录"),
    classes_file: Optional[str] = typer.Option(None, "--classes", "-c", help="类别文件路径 (默认: data/raw/classes.txt)"),
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
    
    示例:
    
        python yolo_cli.py quick train \\
            --images data/raw/images \\
            --labels data/raw/labels \\
            --version yolo11 \\
            --size s \\
            --epochs 200
    """
    
    console.print()
    print_section_header("🚀 一键训练模式")
    
    total_steps = 7
    current_step = 0
    
    try:
        # 验证输入目录
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        if not images_path.exists():
            print_error(f"图像目录不存在: {images_dir}")
            raise typer.Exit(1)
        
        if not labels_path.exists():
            print_error(f"标签目录不存在: {labels_dir}")
            raise typer.Exit(1)
        
        # 确定classes文件
        if classes_file is None:
            # 尝试在多个位置查找
            possible_classes = [
                Path("data/raw/classes.txt"),
                images_path.parent / "classes.txt",
                labels_path.parent / "classes.txt",
            ]
            
            for cls_file in possible_classes:
                if cls_file.exists():
                    classes_file = str(cls_file)
                    break
            
            if classes_file is None:
                print_error("未找到 classes.txt 文件")
                print_info("请使用 --classes 参数指定类别文件，或将其放在 data/raw/classes.txt")
                raise typer.Exit(1)
        
        print_info(f"类别文件: {classes_file}")
        
        # 配置路径
        config = ConfigManager()
        output_dir = config.get_path('data_processed', absolute=True)
        
        # ========== 步骤1: 数据集划分 ==========
        current_step += 1
        print_step(current_step, total_steps, "数据集划分")
        console.print()
        
        split_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=str(output_dir),
            ratios=ratios,
            seed=42
        )
        
        print_success("✓ 数据集划分完成")
        console.print()
        
        # ========== 步骤2: 生成dataset.yaml ==========
        current_step += 1
        print_step(current_step, total_steps, "生成dataset.yaml配置")
        console.print()
        
        dataset_yaml = "data/dataset.yaml"
        generate_yaml(
            data_path=str(output_dir),
            classes_file=classes_file,
            output=dataset_yaml,
            train_dir='images/train',
            val_dir='images/val',
            test_dir='images/test'
        )
        
        print_success(f"✓ 配置文件生成: {dataset_yaml}")
        console.print()
        
        # ========== 步骤3: 验证数据集 ==========
        if not skip_verify:
            current_step += 1
            print_step(current_step, total_steps, "验证数据集")
            console.print()
            
            verify_dataset(data_path=str(output_dir))
            
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
            
            dataset_stats(data_path=str(output_dir), detailed=True)
            
            print_success("✓ 数据统计完成")
            console.print()
        else:
            print_warning("⊘ 跳过数据统计")
            console.print()
        
        # ========== 步骤5: 检查/下载模型 ==========
        current_step += 1
        print_step(current_step, total_steps, "检查模型")
        console.print()
        
        # 标准化版本
        model_version = YOLOVersionManager.normalize_version(model_version)
        model_name = YOLOVersionManager.get_model_name(model_version, model_size)
        
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
                all=False,
                output_dir=str(models_dir)
            )
            
            print_success(f"✓ 模型下载完成: {model_name}")
        
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
        
        start_training(
            model=model_name,
            data=dataset_yaml,
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
        )
        
        # 训练完成
        console.print()
        print_section_header("🎉 一键训练完成")
        
        print_success("所有步骤已完成！")
        console.print()
        
        # 显示训练结果位置
        if project:
            results_path = Path(project)
        else:
            results_path = config.get_path('results', absolute=True) / 'training'
        
        print_info("训练结果位置:")
        print_info(f"  项目目录: {results_path}")
        print_info(f"  最佳模型: {results_path / (name or 'train') / 'weights' / 'best.pt'}")
        print_info(f"  数据集配置: {dataset_yaml}")
        
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
