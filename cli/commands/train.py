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
    TaskType, validate_task_type, get_model_name_with_task, resolve_model_path
)
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_training_config, print_key_value,
    console
)

app = typer.Typer(help="训练命令")


def _is_valid_param(value, expected_type=None):
    """
    检查参数是否为有效值（不是 typer.OptionInfo 对象）
    
    Args:
        value: 参数值
        expected_type: 期望的类型（可选），可以是单个类型或类型元组
    
    Returns:
        bool: 是否为有效参数
    """
    # 检查是否为 None
    if value is None:
        return False
    
    # 检查是否为 typer.OptionInfo 对象
    if hasattr(value, '__class__') and 'OptionInfo' in value.__class__.__name__:
        return False
    
    # 如果指定了期望类型，检查类型
    if expected_type is not None:
        return isinstance(value, expected_type)
    
    return True

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

# Pose任务专用数据增强预设（避免破坏关键点结构）
POSE_AUGMENTATION_PRESETS = {
    'balanced': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,  # Pose不推荐旋转
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,  # 水平翻转需要flip_idx
        'mosaic': 0.0,  # Pose不推荐mosaic
        'mixup': 0.0,   # Pose不推荐mixup
        'auto_augment': None,
        'erasing': 0.0,
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
        'mosaic': 0.0,
        'mixup': 0.0,
        'auto_augment': None,
        'erasing': 0.0,
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
        'mosaic': 0.0,
        'mixup': 0.0,
        'auto_augment': None,
        'erasing': 0.1,
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
        'mosaic': 0.0,
        'mixup': 0.0,
        'auto_augment': None,
        'erasing': 0.0,
    },
}


@app.command("start")
def start_training(
    model: str = typer.Option("yolo11s.pt", "--model", "-m", help="模型名称或路径"),
    data: str = typer.Option("data/processed/dataset.yaml", "--data", "-d", help="数据集配置文件"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
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
    optimizer: str = typer.Option("auto", "--optimizer", "-opt", help="优化器 (auto/SGD/Adam/AdamW/NAdam/RAdam/RMSProp)"),
    freeze: Optional[int] = typer.Option(None, "--freeze", help="冻结前N层 (0=不冻结, 10=冻结前10层, None=不使用)"),
    # 分割任务特定参数
    overlap_mask: Optional[bool] = typer.Option(None, "--overlap-mask", help="[分割] 是否允许掩码重叠"),
    mask_ratio: Optional[int] = typer.Option(None, "--mask-ratio", help="[分割] 掩码下采样比例"),
    # 分类任务特定参数
    dropout: Optional[float] = typer.Option(None, "--dropout", help="[分类] Dropout比例"),
    # Pose任务特定参数
    kpt_shape: Optional[str] = typer.Option(None, "--kpt-shape", help="[Pose] 关键点形状，格式: '17,3' (关键点数,3)"),
    flip_idx: Optional[str] = typer.Option(None, "--flip-idx", help="[Pose] 水平翻转时的关键点索引映射，格式: '0,2,1,4,3,...'"),
    # 从交互模式或quick_train传递的高级配置（非CLI参数，使用 **kwargs 接收）
    **kwargs
):
    """开始训练YOLO模型"""
    
    print_section_header("开始训练")
    
    # 验证任务类型
    task = validate_task_type(task)
    task_type = TaskType.from_string(task)
    print_info(f"任务类型: {task}")
    
    # 处理数据集路径
    # 分类任务需要目录路径，检测/分割任务需要 yaml 文件
    data_path = Path(data)
    
    if task_type == TaskType.CLASSIFY:
        # 分类任务：如果传入的是 yaml 文件，需要提取数据集目录
        if data_path.is_file() and data_path.suffix in ['.yaml', '.yml']:
            # 读取 yaml 文件获取数据集路径
            with open(data_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            
            if 'path' not in yaml_content:
                print_error("dataset.yaml 中缺少 'path' 字段")
                raise typer.Exit(1)
            
            # 使用 yaml 中的 path 作为数据集根目录
            dataset_root = Path(yaml_content['path'])
            
            # 检查是否使用 images/ 子目录结构
            images_dir = dataset_root / 'images'
            if images_dir.exists() and (images_dir / 'train').exists():
                data = str(images_dir)
                print_info(f"分类任务使用数据集目录: {data}")
            else:
                data = str(dataset_root)
                print_info(f"分类任务使用数据集目录: {data}")
        elif data_path.is_dir():
            # 如果直接传入目录，直接使用
            data = str(data_path)
        else:
            print_error(f"分类任务需要数据集目录或 dataset.yaml 文件: {data}")
            raise typer.Exit(1)
    else:
        # 检测/分割任务：验证 yaml 文件
        if not data_path.exists():
            print_error(f"数据集配置文件不存在: {data}")
            raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    device_name = get_device_name(device)
    print_info(f"使用设备: {device_name}")
    
    # 创建项目名称
    # 使用辅助函数检查参数是否为有效值
    if not _is_valid_param(project, str):
        config = ConfigManager()
        project = str(config.get_path('results', absolute=True) / 'training')
    
    if not _is_valid_param(name, str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model).stem
        name = f"{model_name}_{timestamp}"
    
    # 解析模型路径（自动查找已下载的模型）
    model, found_local = resolve_model_path(model, task)
    if found_local:
        print_success(f"✓ 使用已下载的模型: {model}")
    else:
        print_info(f"使用模型: {model} (将自动下载)")
    
    # 获取数据增强配置
    # Pose任务使用专用的数据增强预设
    if task_type == TaskType.POSE:
        if augmentation not in POSE_AUGMENTATION_PRESETS:
            print_warning(f"未知的增强预设: {augmentation}，使用 'balanced'")
            augmentation = 'balanced'
        aug_config = POSE_AUGMENTATION_PRESETS[augmentation].copy()
    else:
        if augmentation not in AUGMENTATION_PRESETS:
            print_warning(f"未知的增强预设: {augmentation}，使用 'balanced'")
            augmentation = 'balanced'
        aug_config = AUGMENTATION_PRESETS[augmentation].copy()
    
    # 从 kwargs 获取自定义增强参数
    augmentation_custom = kwargs.get('augmentation_custom')
    if augmentation_custom:
        print_info("应用自定义数据增强配置...")
        aug_config.update(augmentation_custom)
        # 显示自定义的参数
        console.print()
        print_info("自定义增强参数:")
        for key, value in augmentation_custom.items():
            print_info(f"  {key}: {value}")
        console.print()
    
    # 添加任务特定配置
    task_specific_config = {}
    if task_type == TaskType.SEGMENT:
        # 分割任务特定参数
        if _is_valid_param(overlap_mask, bool):
            task_specific_config['overlap_mask'] = overlap_mask
        else:
            task_specific_config['overlap_mask'] = True
        
        if _is_valid_param(mask_ratio, int):
            task_specific_config['mask_ratio'] = mask_ratio
        else:
            task_specific_config['mask_ratio'] = 4
    
    elif task_type == TaskType.CLASSIFY:
        # 分类任务特定参数
        if _is_valid_param(dropout, (int, float)):
            task_specific_config['dropout'] = dropout
        
        # 分类任务通常使用较小的图像尺寸
        if imgsz == 640:
            imgsz = 224
            print_info(f"分类任务使用默认图像尺寸: {imgsz}")
    
    elif task_type == TaskType.POSE:
        # Pose任务特定参数
        if _is_valid_param(kpt_shape, str):
            # 解析 kpt_shape，如 "17,3"
            try:
                kpt_parts = [int(x) for x in kpt_shape.split(',')]
                task_specific_config['kpt_shape'] = kpt_parts
                print_info(f"关键点配置: {kpt_parts[0]} 个关键点")
            except ValueError:
                print_warning(f"无效的 kpt_shape 格式: {kpt_shape}，应为 '17,3' 格式")
        
        if _is_valid_param(flip_idx, str):
            # 解析 flip_idx，如 "0,2,1,4,3,..."
            try:
                flip_indices = [int(x) for x in flip_idx.split(',')]
                task_specific_config['fliplr'] = 0.5  # 启用水平翻转
                task_specific_config['flip_idx'] = flip_indices
                print_info(f"配置关键点水平翻转映射")
            except ValueError:
                print_warning(f"无效的 flip_idx 格式: {flip_idx}")
    
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
            'data': data,  # 使用处理后的 data 路径（分类任务为目录，其他为 yaml）
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
        
        # 添加优化器配置
        if optimizer != "auto":
            training_kwargs['optimizer'] = optimizer
            print_info(f"使用优化器: {optimizer}")
        
        # 添加层冻结配置
        if freeze is not None:
            training_kwargs['freeze'] = freeze
            if freeze > 0:
                print_info(f"冻结前 {freeze} 层")
            else:
                print_info("不冻结任何层")
        
        # 从 kwargs 获取并添加优化器配置
        optimizer_config = kwargs.get('optimizer_config')
        if optimizer_config:
            print_info("应用自定义优化器配置...")
            training_kwargs.update(optimizer_config)
            console.print()
            print_info("自定义优化器参数:")
            for key, value in optimizer_config.items():
                print_info(f"  {key}: {value}")
            console.print()
        
        # 从 kwargs 获取并添加损失权重配置
        loss_weights = kwargs.get('loss_weights')
        if loss_weights:
            print_info("应用自定义损失权重配置...")
            training_kwargs.update(loss_weights)
            console.print()
            print_info("自定义损失权重:")
            for key, value in loss_weights.items():
                print_info(f"  {key}: {value}")
            console.print()
        
        results = yolo_model.train(**training_kwargs)
        
        # 训练完成
        console.print()
        print_success("训练完成！")
        
        # 从YOLO结果对象获取实际保存目录
        if hasattr(results, 'save_dir'):
            save_dir = Path(results.save_dir)
        else:
            # 回退到手动构建路径
            save_dir = Path(project) / name
        
        weights_dir = save_dir / 'weights'
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        
        if best_pt.exists():
            print_success(f"最佳模型: {best_pt.absolute()}")
        if last_pt.exists():
            print_info(f"最后模型: {last_pt.absolute()}")
        
        print_info(f"结果目录: {save_dir.absolute()}")
        
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
        'data': 'data/processed/dataset.yaml',
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
    data: str = typer.Option("data/processed/dataset.yaml", "--data", "-d", help="数据集配置文件"),
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
