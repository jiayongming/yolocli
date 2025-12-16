#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式提示"""

import questionary
from typing import List, Dict, Any, Optional
from pathlib import Path


def select_option(message: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    选择选项
    
    Args:
        message: 提示消息
        choices: 选项列表
        default: 默认选项
    
    Returns:
        str: 选中的选项
    """
    return questionary.select(
        message,
        choices=choices,
        default=default,
    ).ask()


def select_multiple(message: str, choices: List[str]) -> List[str]:
    """
    选择多个选项
    
    Args:
        message: 提示消息
        choices: 选项列表
    
    Returns:
        List[str]: 选中的选项列表
    """
    return questionary.checkbox(
        message,
        choices=choices,
    ).ask()


def confirm_action(message: str, default: bool = True) -> bool:
    """
    确认操作
    
    Args:
        message: 提示消息
        default: 默认值
    
    Returns:
        bool: 确认结果
    """
    return questionary.confirm(
        message,
        default=default,
    ).ask()


def input_text(message: str, default: str = "", validate: Optional[callable] = None) -> str:
    """
    输入文本
    
    Args:
        message: 提示消息
        default: 默认值
        validate: 验证函数
    
    Returns:
        str: 输入的文本
    """
    return questionary.text(
        message,
        default=default,
        validate=validate,
    ).ask()


def input_path(message: str, default: str = "", must_exist: bool = False) -> str:
    """
    输入路径
    
    Args:
        message: 提示消息
        default: 默认路径
        must_exist: 是否必须存在
    
    Returns:
        str: 输入的路径
    """
    def validate_path(text):
        if must_exist and not Path(text).exists():
            return "路径不存在"
        return True
    
    return questionary.path(
        message,
        default=default,
        validate=validate_path if must_exist else None,
    ).ask()


def input_number(
    message: str,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    输入数字
    
    Args:
        message: 提示消息
        default: 默认值
        min_value: 最小值
        max_value: 最大值
    
    Returns:
        float: 输入的数字
    """
    def validate_number(text):
        try:
            num = float(text)
            if min_value is not None and num < min_value:
                return f"值必须 >= {min_value}"
            if max_value is not None and num > max_value:
                return f"值必须 <= {max_value}"
            return True
        except ValueError:
            return "请输入有效的数字"
    
    default_str = str(default) if default is not None else ""
    result = questionary.text(
        message,
        default=default_str,
        validate=validate_number,
    ).ask()
    
    return float(result)


def select_yolo_version() -> str:
    """
    选择YOLO版本
    
    Returns:
        str: 选中的版本
    """
    choices = [
        "yolo11 (推荐，最新版本)",
        "yolov8 (稳定版本)",
    ]
    
    result = select_option("选择YOLO版本:", choices, default=choices[0])
    
    # 提取版本名称
    if "yolo11" in result:
        return "yolo11"
    else:
        return "yolov8"


def select_model_size() -> str:
    """
    选择模型大小
    
    Returns:
        str: 选中的大小
    """
    choices = [
        "n - Nano (最快，适合边缘设备)",
        "s - Small (快速，适合大多数应用)",
        "m - Medium (平衡，推荐)",
        "l - Large (精度优先)",
        "x - Extra Large (最高精度，最慢)",
    ]
    
    result = select_option("选择模型大小:", choices, default=choices[2])
    
    # 提取大小字母
    return result.split(' ')[0]


def select_device() -> str:
    """
    选择设备
    
    Returns:
        str: 选中的设备（如果是CUDA，返回具体GPU ID如'0'或'1,2'）
    """
    choices = [
        "auto - 自动检测 (推荐)",
        "mps - Apple Silicon",
        "cuda - NVIDIA GPU",
        "cpu - CPU",
    ]
    
    result = select_option("选择训练设备:", choices, default=choices[0])
    
    # 提取设备名称
    device = result.split(' ')[0]
    
    # 如果选择了CUDA，询问GPU ID
    if device == 'cuda':
        gpu_id = input_text(
            "输入GPU设备ID (单个: 0, 多个: 0,1,2):",
            default="0",
            validate=lambda x: True if x.replace(',', '').isdigit() else "请输入有效的GPU ID"
        )
        # 如果只有一个GPU ID，直接返回数字
        if ',' not in gpu_id:
            return gpu_id
        else:
            # 多GPU训练，返回逗号分隔的ID
            return gpu_id
    
    return device


def select_augmentation_preset() -> str:
    """
    选择数据增强预设
    
    Returns:
        str: 选中的预设
    """
    choices = [
        "balanced - 平衡配置 (推荐)",
        "conservative - 保守配置 (小数据集)",
        "aggressive - 激进配置 (大数据集)",
        "default - YOLO默认配置",
    ]
    
    result = select_option("选择数据增强策略:", choices, default=choices[0])
    
    # 提取预设名称
    return result.split(' ')[0]


def select_export_formats() -> List[str]:
    """
    选择导出格式
    
    Returns:
        List[str]: 选中的格式列表
    """
    choices = [
        "onnx - ONNX格式 (推荐，通用)",
        "torchscript - TorchScript格式",
        "tflite - TensorFlow Lite",
        "coreml - CoreML (Apple设备)",
        "engine - TensorRT Engine",
        "pb - TensorFlow SavedModel",
    ]
    
    results = select_multiple("选择导出格式 (空格选择，回车确认):", choices)
    
    # 提取格式名称
    return [r.split(' ')[0] for r in results]


def build_training_config() -> Dict[str, Any]:
    """
    交互式构建训练配置
    
    Returns:
        Dict[str, Any]: 训练配置
    """
    config = {}
    
    # YOLO版本
    config['version'] = select_yolo_version()
    
    # 模型大小
    config['model_size'] = select_model_size()
    
    # 训练参数
    config['epochs'] = int(input_number("训练轮数:", default=200, min_value=1))
    config['batch'] = int(input_number("批次大小:", default=16, min_value=1))
    config['imgsz'] = int(input_number("图像尺寸:", default=640, min_value=32))
    
    # 设备
    config['device'] = select_device()
    
    # 数据增强
    config['augmentation'] = select_augmentation_preset()
    
    # 高级选项
    if confirm_action("配置高级选项?", default=False):
        config['patience'] = int(input_number("早停耐心值:", default=50, min_value=0))
        config['save_period'] = int(input_number("保存周期 (epoch):", default=10, min_value=1))
    else:
        config['patience'] = 50
        config['save_period'] = 10
    
    return config


def select_main_menu() -> str:
    """
    显示主菜单
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "quick - 一键训练 (自动化完整流程) ⚡",
        "model - 模型管理 (下载、导出、列表)",
        "data - 数据处理 (划分、验证、统计)",
        "train - 模型训练 (训练、恢复)",
        "detect - 图像检测 (单图、批量)",
        "exit - 退出",
    ]
    
    result = select_option("请选择操作:", choices, default=choices[0])
    
    # 提取操作名称
    return result.split(' ')[0]


def select_model_operation() -> str:
    """
    选择模型操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "download - 下载预训练模型",
        "export - 导出模型",
        "list - 列出本地模型",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择模型操作:", choices)
    return result.split(' ')[0]


def select_data_operation() -> str:
    """
    选择数据操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "split - 划分数据集",
        "generate-yaml - 生成dataset.yaml",
        "verify - 验证数据集",
        "stats - 数据统计",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择数据操作:", choices)
    return result.split(' ')[0]


def select_train_operation() -> str:
    """
    选择训练操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "start - 开始训练",
        "resume - 恢复训练",
        "config - 生成配置",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择训练操作:", choices)
    return result.split(' ')[0]


def select_detect_operation() -> str:
    """
    选择检测操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "image - 单张图片检测",
        "batch - 批量检测",
        "video - 视频检测",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择检测操作:", choices)
    return result.split(' ')[0]
