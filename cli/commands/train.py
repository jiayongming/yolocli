#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练命令"""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
from ultralytics import YOLO
import yaml

from ..core.config import ConfigManager
from ..core.version import YOLOVersionManager
from ..core.utils import (
    detect_device, get_device_name, ensure_dir,
    TaskType, validate_task_type, get_model_name_with_task
)
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_training_config, print_key_value,
    console
)

app = typer.Typer(help="训练命令")

# 数据增强预设配置
AUGMENTATION_PRESETS = {
    'balanced': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'auto_augment': 'randaugment',
        'erasing': 0.2,
    },
    'conservative': {
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 0.0,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.0,
        'auto_augment': None,
        'erasing': 0.1,
    },
    'aggressive': {
        'hsv_h': 0.02,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'degrees': 0.0,
        'translate': 0.15,
        'scale': 0.6,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
    },
    'default': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
    },
}


@app.command("start")
def start_training(
    model: str = typer.Option("yolo11s.pt", "--model", "-m", help="模型名称或路径"),
    data: str = typer.Option("data/dataset.yaml", "--data", "-d", help="数据集配置文件"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify)"),
    epochs: int = typer.Option(200, "--epochs", "-e", help="训练轮数"),
    batch: int = typer.Option(16, "--batch", "-b", help="批次大小"),
    imgsz: int = typer.Option(640, "--imgsz", help="图像尺寸"),
    device: str = typer.Option("auto", "--device", help="设备 (auto/mps/cuda/cpu)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="项目目录"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="实验名称"),
    augmentation: str = typer.Option("balanced", "--augmentation", "-a", help="数据增强预设"),
    patience: int = typer.Option(50, "--patience", help="早停耐心值"),
    save_period: int = typer.Option(10, "--save-period", help="保存周期"),
    resume: bool = typer.Option(False, "--resume", "-r", help="从last.pt恢复训练"),
    pretrained: bool = typer.Option(True, "--pretrained/--from-scratch", help="使用预训练权重"),
    # 分割任务特定参数
    overlap_mask: Optional[bool] = typer.Option(None, "--overlap-mask", help="[分割] 是否允许掩码重叠"),
    mask_ratio: Optional[int] = typer.Option(None, "--mask-ratio", help="[分割] 掩码下采样比例"),
    # 分类任务特定参数
    dropout: Optional[float] = typer.Option(None, "--dropout", help="[分类] Dropout比例"),
):
    """开始训练YOLO模型"""
    
    print_section_header("开始训练")
    
    # 验证任务类型
    task = validate_task_type(task)
    task_type = TaskType.from_string(task)
    print_info(f"任务类型: {task}")
    
    # 验证数据集配置文件
    data_path = Path(data)
    if not data_path.exists():
        print_error(f"数据集配置文件不存在: {data}")
        raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    device_name = get_device_name(device)
    print_info(f"使用设备: {device_name}")
    
    # 创建项目名称
    if project is None:
        config = ConfigManager()
        project = str(config.get_path('results', absolute=True) / 'training')
    
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model).stem
        name = f"{model_name}_{timestamp}"
    
    # 确保模型名称包含正确的任务后缀
    model_path = Path(model)
    if not model_path.exists():
        # 如果是模型名称而非路径，添加任务后缀
        model = get_model_name_with_task(model, task)
        print_info(f"使用模型: {model}")
    
    # 获取数据增强配置
    if augmentation not in AUGMENTATION_PRESETS:
        print_warning(f"未知的增强预设: {augmentation}，使用 'balanced'")
        augmentation = 'balanced'
    
    aug_config = AUGMENTATION_PRESETS[augmentation].copy()
    
    # 添加任务特定配置
    task_specific_config = {}
    if task_type == TaskType.SEGMENT:
        # 分割任务特定参数
        if overlap_mask is not None:
            task_specific_config['overlap_mask'] = overlap_mask
        else:
            task_specific_config['overlap_mask'] = True
        
        if mask_ratio is not None:
            task_specific_config['mask_ratio'] = mask_ratio
        else:
            task_specific_config['mask_ratio'] = 4
    
    elif task_type == TaskType.CLASSIFY:
        # 分类任务特定参数
        if dropout is not None:
            task_specific_config['dropout'] = dropout
        
        # 分类任务通常使用较小的图像尺寸
        if imgsz == 640:
            imgsz = 224
            print_info(f"分类任务使用默认图像尺寸: {imgsz}")
    
    # 显示训练配置
    train_config = {
        '任务类型': task.upper(),
        '模型': model,
        '数据集': data,
        '训练轮数': epochs,
        '批次大小': batch,
        '图像尺寸': imgsz,
        '设备': device_name,
        '项目目录': project,
        '实验名称': name,
        '数据增强': augmentation,
        '早停耐心值': patience,
        '保存周期': save_period,
        '预训练权重': '是' if pretrained else '否',
    }
    
    # 添加任务特定配置到显示
    if task_specific_config:
        for key, value in task_specific_config.items():
            train_config[f'[{task}] {key}'] = value
    
    print_training_config(train_config)
    
    try:
        # 加载模型
        print_info("加载模型...")
        
        if resume:
            # 查找last.pt
            last_pt = Path(project) / name / 'weights' / 'last.pt'
            if last_pt.exists():
                print_info(f"从检查点恢复: {last_pt}")
                yolo_model = YOLO(str(last_pt))
            else:
                print_warning("未找到last.pt，将开始新训练")
                yolo_model = YOLO(model)
        else:
            yolo_model = YOLO(model)
        
        # 开始训练
        print_info("开始训练...")
        console.print()
        
        # 合并所有配置
        training_kwargs = {
            'data': str(data_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'project': project,
            'name': name,
            'pretrained': pretrained,
            'save': True,
            'save_period': save_period,
            'val': True,
            'plots': True,
            'patience': patience,
            'verbose': True,
            'exist_ok': True,
            'resume': resume,
            **aug_config,
            **task_specific_config,
        }
        
        results = yolo_model.train(**training_kwargs)
        
        # 训练完成
        console.print()
        print_success("训练完成！")
        
        weights_dir = Path(project) / name / 'weights'
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        
        if best_pt.exists():
            print_success(f"最佳模型: {best_pt}")
        if last_pt.exists():
            print_info(f"最后模型: {last_pt}")
        
        print_info(f"结果目录: {Path(project) / name}")
        
    except KeyboardInterrupt:
        print_warning("\n训练被用户中断")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"训练失败: {e}")
        raise typer.Exit(1)


@app.command("resume")
def resume_training(
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="检查点路径"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="项目目录"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="实验名称"),
):
    """恢复训练"""
    
    print_section_header("恢复训练")
    
    # 查找检查点
    if checkpoint is None:
        if project and name:
            checkpoint = Path(project) / name / 'weights' / 'last.pt'
        else:
            # 在默认目录中查找最新的last.pt
            config = ConfigManager()
            results_dir = config.get_path('results', absolute=True) / 'training'
            
            if results_dir.exists():
                last_pts = list(results_dir.rglob('last.pt'))
                if last_pts:
                    # 按修改时间排序，选择最新的
                    checkpoint = max(last_pts, key=lambda p: p.stat().st_mtime)
                    print_info(f"自动找到最新检查点: {checkpoint}")
    else:
        checkpoint = Path(checkpoint)
    
    if checkpoint is None or not Path(checkpoint).exists():
        print_error("未找到检查点文件")
        print_info("使用 --checkpoint 指定检查点路径")
        raise typer.Exit(1)
    
    print_info(f"检查点: {checkpoint}")
    
    try:
        # 加载模型并恢复训练
        print_info("加载检查点...")
        yolo_model = YOLO(str(checkpoint))
        
        print_info("恢复训练...")
        results = yolo_model.train(resume=True)
        
        print_success("训练完成！")
        
    except Exception as e:
        print_error(f"恢复训练失败: {e}")
        raise typer.Exit(1)


@app.command("config")
def generate_config(
    output: str = typer.Option("train_config.yaml", "--output", "-o", help="输出文件"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="配置预设 (small/medium/large)"),
):
    """生成训练配置文件"""
    
    print_section_header("生成训练配置")
    
    config = ConfigManager()
    
    # 加载预设配置
    if profile:
        try:
            config.load_profile(profile)
            print_info(f"使用预设配置: {profile}")
        except FileNotFoundError:
            print_warning(f"预设配置不存在: {profile}，使用默认配置")
    
    # 生成训练配置
    train_config = {
        'model': config.get('model.default_version', 'yolo11s.pt'),
        'data': 'data/dataset.yaml',
        'epochs': config.get('training.epochs', 200),
        'batch': config.get('training.batch', 16),
        'imgsz': config.get('training.imgsz', 640),
        'device': config.get('training.device', 'auto'),
        'patience': config.get('training.patience', 50),
        'save_period': config.get('training.save_period', 10),
        'augmentation': config.get('augmentation.default_preset', 'balanced'),
    }
    
    # 保存配置
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO 训练配置文件\n")
        yaml.dump(train_config, f, default_flow_style=False, allow_unicode=True)
    
    print_success(f"配置文件已生成: {output_path}")
    
    # 显示配置内容
    console.print("\n生成的配置:")
    for key, value in train_config.items():
        print_key_value(key, value)


@app.command("validate")
def validate_model(
    model: str = typer.Argument(..., help="模型路径"),
    data: str = typer.Option("data/dataset.yaml", "--data", "-d", help="数据集配置文件"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务类型（自动从模型推断）"),
    batch: int = typer.Option(16, "--batch", "-b", help="批次大小"),
    imgsz: int = typer.Option(640, "--imgsz", help="图像尺寸"),
    device: str = typer.Option("auto", "--device", help="设备"),
):
    """验证模型性能"""
    
    print_section_header("模型验证")
    
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 如果未指定任务类型，从模型名称推断
    if task is None:
        from ..core.utils import parse_model_name
        _, task = parse_model_name(model_path.name)
    else:
        task = validate_task_type(task)
    
    print_info(f"任务类型: {task}")
    
    data_path = Path(data)
    if not data_path.exists():
        print_error(f"数据集配置不存在: {data}")
        raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"模型: {model_path.name}")
    print_info(f"数据集: {data}")
    print_info(f"设备: {get_device_name(device)}")
    
    try:
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        print_info("开始验证...")
        results = yolo_model.val(
            data=str(data_path),
            batch=batch,
            imgsz=imgsz,
            device=device,
        )
        
        print_success("验证完成！")
        
        # 显示结果
        if hasattr(results, 'box'):
            box_metrics = results.box
            console.print("\n验证指标:")
            print_key_value("mAP50", f"{box_metrics.map50:.4f}")
            print_key_value("mAP50-95", f"{box_metrics.map:.4f}")
            print_key_value("Precision", f"{box_metrics.mp:.4f}")
            print_key_value("Recall", f"{box_metrics.mr:.4f}")
        
    except Exception as e:
        print_error(f"验证失败: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
