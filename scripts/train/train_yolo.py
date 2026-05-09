#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO模型训练脚本（集成数据增强）"""

from ultralytics import YOLO
import os
import sys
from datetime import datetime

# ============================================================================
# 数据增强预设配置（针对墙体渗水检测场景优化）
# ============================================================================

# 平衡配置（推荐，适合大多数场景）
AUGMENTATION_BALANCED = {
    'hsv_h': 0.015,      # 色调：轻微变化，保持渗水区域颜色特征
    'hsv_s': 0.7,        # 饱和度：中等增强，适应不同拍摄设备
    'hsv_v': 0.4,        # 明度：中等增强，适应不同光照环境
    'degrees': 0.0,      # 不旋转：墙体通常是垂直的
    'translate': 0.1,    # 轻微平移：增加位置变化
    'scale': 0.5,        # 缩放：适应不同拍摄距离
    'shear': 0.0,        # 不剪切：保持墙体结构
    'perspective': 0.0,  # 不透视：保持平面视角
    'flipud': 0.0,       # 不上下翻转：墙体方向固定
    'fliplr': 0.5,       # 左右翻转：50%概率
    'mosaic': 1.0,       # Mosaic增强：提高小目标检测能力
    'mixup': 0.1,        # 轻微MixUp：增加数据多样性
    'cutmix': 0.0,       # 不使用CutMix
    'copy_paste': 0.0,   # 不使用Copy-Paste
    'auto_augment': 'randaugment',  # 使用随机增强
    'erasing': 0.2,      # 轻微随机擦除：模拟遮挡场景
}

# 保守配置（最小增强，适合小数据集 <500张）
AUGMENTATION_CONSERVATIVE = {
    'hsv_h': 0.01,       # 轻微颜色变化
    'hsv_s': 0.5,
    'hsv_v': 0.3,
    'degrees': 0.0,
    'translate': 0.05,
    'scale': 0.3,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,       # 仅左右翻转
    'mosaic': 0.5,       # 降低Mosaic概率
    'mixup': 0.0,        # 不使用MixUp
    'cutmix': 0.0,
    'copy_paste': 0.0,
    'auto_augment': None,  # 不使用自动增强
    'erasing': 0.1,
}

# 激进配置（强增强，适合大数据集 >1000张）
AUGMENTATION_AGGRESSIVE = {
    'hsv_h': 0.02,       # 更强的颜色变化
    'hsv_s': 0.8,
    'hsv_v': 0.5,
    'degrees': 0.0,
    'translate': 0.15,   # 更大的平移
    'scale': 0.6,        # 更大的缩放范围
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,       # 完全启用Mosaic
    'mixup': 0.15,      # 启用MixUp
    'cutmix': 0.1,      # 启用CutMix
    'copy_paste': 0.0,
    'auto_augment': 'randaugment',
    'erasing': 0.4,      # 更强的随机擦除
}

# 默认配置（YOLO官方默认值）
AUGMENTATION_DEFAULT = {
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
    'cutmix': 0.0,
    'copy_paste': 0.0,
    'auto_augment': 'randaugment',
    'erasing': 0.4,
}

AUGMENTATION_PRESETS = {
    'balanced': AUGMENTATION_BALANCED,
    'conservative': AUGMENTATION_CONSERVATIVE,
    'aggressive': AUGMENTATION_AGGRESSIVE,
    'default': AUGMENTATION_DEFAULT,
}

def get_augmentation_config(preset='default', **kwargs):
    """
    获取数据增强配置
    
    Args:
        preset: 预设配置名称，可选：'balanced', 'conservative', 'aggressive', 'default'
        **kwargs: 自定义数据增强参数，会覆盖预设值
    
    Returns:
        dict: 数据增强参数字典
    """
    if preset not in AUGMENTATION_PRESETS:
        print(f"警告：未知的预设配置 '{preset}'，使用 'default' 配置")
        preset = 'default'
    
    config = AUGMENTATION_PRESETS[preset].copy()
    
    # 使用自定义参数覆盖预设值
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
        else:
            print(f"警告：未知的数据增强参数 '{key}'，将被忽略")
    
    return config

def print_augmentation_config(config):
    """打印数据增强配置"""
    print("\n" + "=" * 60)
    print("数据增强配置")
    print("=" * 60)
    print("颜色增强:")
    print(f"  HSV色调 (hsv_h): {config['hsv_h']}")
    print(f"  HSV饱和度 (hsv_s): {config['hsv_s']}")
    print(f"  HSV明度 (hsv_v): {config['hsv_v']}")
    print("\n几何变换:")
    print(f"  旋转角度 (degrees): {config['degrees']}")
    print(f"  平移 (translate): {config['translate']}")
    print(f"  缩放 (scale): {config['scale']}")
    print(f"  剪切 (shear): {config['shear']}")
    print(f"  透视 (perspective): {config['perspective']}")
    print("\n翻转:")
    print(f"  上下翻转 (flipud): {config['flipud']}")
    print(f"  左右翻转 (fliplr): {config['fliplr']}")
    print("\n高级增强:")
    print(f"  Mosaic: {config['mosaic']}")
    print(f"  MixUp: {config['mixup']}")
    print(f"  CutMix: {config['cutmix']}")
    print(f"  Copy-Paste: {config['copy_paste']}")
    print(f"  自动增强 (auto_augment): {config['auto_augment']}")
    print(f"  随机擦除 (erasing): {config['erasing']}")
    print("=" * 60 + "\n")

def train_model(
    model_name='yolo11s.pt',
    data_yaml='data/processed/dataset.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    device='auto',
    project='results/training',
    name=None,
    save_period=10,
    patience=50,
    augmentation_preset='default',
    # 数据增强参数（使用YOLO官方默认值）
    hsv_h=None,
    hsv_s=None,
    hsv_v=None,
    degrees=None,
    translate=None,
    scale=None,
    shear=None,
    perspective=None,
    flipud=None,
    fliplr=None,
    mosaic=None,
    mixup=None,
    cutmix=None,
    copy_paste=None,
    auto_augment=None,
    erasing=None,
):
    """训练YOLO模型"""
    
    # 自动检测设备
    if device == 'auto':
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            print("检测到Apple芯片，使用MPS加速")
        elif torch.cuda.is_available():
            device = 0
            print(f"检测到NVIDIA GPU，使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            device = -1
            print("未检测到GPU，使用CPU训练")
    
    # 创建项目名称（带时间戳）
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"workspace_{model_name.replace('.pt', '')}_{timestamp}"
    
    # 获取数据增强配置
    aug_kwargs = {}
    if hsv_h is not None: aug_kwargs['hsv_h'] = hsv_h
    if hsv_s is not None: aug_kwargs['hsv_s'] = hsv_s
    if hsv_v is not None: aug_kwargs['hsv_v'] = hsv_v
    if degrees is not None: aug_kwargs['degrees'] = degrees
    if translate is not None: aug_kwargs['translate'] = translate
    if scale is not None: aug_kwargs['scale'] = scale
    if shear is not None: aug_kwargs['shear'] = shear
    if perspective is not None: aug_kwargs['perspective'] = perspective
    if flipud is not None: aug_kwargs['flipud'] = flipud
    if fliplr is not None: aug_kwargs['fliplr'] = fliplr
    if mosaic is not None: aug_kwargs['mosaic'] = mosaic
    if mixup is not None: aug_kwargs['mixup'] = mixup
    if cutmix is not None: aug_kwargs['cutmix'] = cutmix
    if copy_paste is not None: aug_kwargs['copy_paste'] = copy_paste
    if auto_augment is not None: aug_kwargs['auto_augment'] = auto_augment
    if erasing is not None: aug_kwargs['erasing'] = erasing
    
    aug_config = get_augmentation_config(augmentation_preset, **aug_kwargs)
    
    print("=" * 60)
    print("开始训练YOLO模型")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"数据集: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"图像尺寸: {imgsz}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    print(f"项目: {project}")
    print(f"名称: {name}")
    print(f"数据增强预设: {augmentation_preset}")
    print_augmentation_config(aug_config)
    
    # 加载模型
    print(f"\n加载模型: {model_name}")
    model = YOLO(model_name)
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        save_period=save_period,
        val=True,
        plots=True,
        patience=patience,
        verbose=True,
        **aug_config,  # 应用数据增强配置
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳模型保存在: {project}/{name}/weights/best.pt")
    print(f"最后模型保存在: {project}/{name}/weights/last.pt")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练YOLO模型（集成数据增强）')
    
    # 基本训练参数
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                        help='模型名称 (yolo11n.pt, yolo11s.pt, yolo11m.pt等)')
    parser.add_argument('--data', type=str, default='data/processed/dataset.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备选择: auto(自动), 0(NVIDIA GPU), mps(Apple芯片), -1(CPU)')
    parser.add_argument('--project', type=str, default='results/training',
                        help='项目目录')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--save-period', type=int, default=10,
                        help='每N个epoch保存一次检查点')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    
    # 数据增强预设
    parser.add_argument('--augmentation', type=str, default='default',
                        choices=['balanced', 'conservative', 'aggressive', 'default'],
                        help='数据增强预设: balanced(平衡), conservative(保守), aggressive(激进), default(YOLO默认)')
    
    # 数据增强参数（YOLO官方默认值）
    parser.add_argument('--hsv-h', type=float, default=None,
                        help='HSV色调增强幅度 (默认: 0.015)')
    parser.add_argument('--hsv-s', type=float, default=None,
                        help='HSV饱和度增强幅度 (默认: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=None,
                        help='HSV明度增强幅度 (默认: 0.4)')
    parser.add_argument('--degrees', type=float, default=None,
                        help='旋转角度范围 (默认: 0.0)')
    parser.add_argument('--translate', type=float, default=None,
                        help='平移幅度 (默认: 0.1)')
    parser.add_argument('--scale', type=float, default=None,
                        help='缩放范围 (默认: 0.5)')
    parser.add_argument('--shear', type=float, default=None,
                        help='剪切变换幅度 (默认: 0.0)')
    parser.add_argument('--perspective', type=float, default=None,
                        help='透视变换幅度 (默认: 0.0)')
    parser.add_argument('--flipud', type=float, default=None,
                        help='上下翻转概率 (默认: 0.0)')
    parser.add_argument('--fliplr', type=float, default=None,
                        help='左右翻转概率 (默认: 0.5)')
    parser.add_argument('--mosaic', type=float, default=None,
                        help='Mosaic增强概率 (默认: 1.0)')
    parser.add_argument('--mixup', type=float, default=None,
                        help='MixUp增强概率 (默认: 0.0)')
    parser.add_argument('--cutmix', type=float, default=None,
                        help='CutMix增强概率 (默认: 0.0)')
    parser.add_argument('--copy-paste', type=float, default=None,
                        help='Copy-Paste增强概率 (默认: 0.0)')
    parser.add_argument('--auto-augment', type=str, default=None,
                        choices=['randaugment', 'autoaugment', 'augmix', None],
                        help='自动增强策略 (默认: randaugment)')
    parser.add_argument('--erasing', type=float, default=None,
                        help='随机擦除概率 (默认: 0.4)')
    
    args = parser.parse_args()
    
    # 处理device参数
    if args.device.isdigit() or (args.device.startswith('-') and args.device[1:].isdigit()):
        device = int(args.device)
    else:
        device = args.device
    
    # 处理auto_augment参数
    auto_augment = args.auto_augment
    if auto_augment == 'None':
        auto_augment = None
    
    train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        patience=args.patience,
        augmentation_preset=args.augmentation,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cutmix=args.cutmix,
        copy_paste=args.copy_paste,
        auto_augment=auto_augment,
        erasing=args.erasing,
    )
