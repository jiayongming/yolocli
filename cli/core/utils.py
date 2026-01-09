#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""

import os
import torch
from pathlib import Path
from typing import Union, Optional, List, Tuple
from enum import Enum


class TaskType(Enum):
    """YOLO任务类型枚举"""
    DETECT = "detect"
    SEGMENT = "segment"
    CLASSIFY = "classify"
    POSE = "pose"
    
    @classmethod
    def from_string(cls, task: str) -> 'TaskType':
        """从字符串创建TaskType
        
        Args:
            task: 任务名称字符串
            
        Returns:
            TaskType: 任务类型枚举
            
        Raises:
            ValueError: 如果任务类型无效
            TypeError: 如果task不是字符串类型
        """
        # 类型检查，防止传入OptionInfo等非字符串对象
        if not isinstance(task, str):
            raise TypeError(
                f"task参数必须是字符串类型，但收到了 {type(task).__name__} 类型。"
                f"如果在交互模式中遇到此错误，请确保正确传递了task参数。"
            )
        
        task = task.lower().strip()
        for task_type in cls:
            if task_type.value == task:
                return task_type
        raise ValueError(f"无效的任务类型: {task}. 支持的类型: {', '.join([t.value for t in cls])}")
    
    def get_model_suffix(self) -> str:
        """获取模型文件后缀
        
        Returns:
            str: 模型后缀（如 '', '-seg', '-cls', '-pose'）
        """
        if self == TaskType.DETECT:
            return ""
        elif self == TaskType.SEGMENT:
            return "-seg"
        elif self == TaskType.CLASSIFY:
            return "-cls"
        elif self == TaskType.POSE:
            return "-pose"
        return ""
    
    def __str__(self) -> str:
        return self.value


def detect_device() -> str:
    """
    自动检测最佳设备
    
    支持通过环境变量 CUDA_VISIBLE_DEVICES 指定GPU
    
    Returns:
        str: 设备名称 ('mps', '0', 'cpu')
    
    Examples:
        export CUDA_VISIBLE_DEVICES=4 python yolo_cli.py ...
        export CUDA_VISIBLE_DEVICES=0,1 python yolo_cli.py ...
    """
    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if cuda_visible_devices is not None:
        # 如果设置了CUDA_VISIBLE_DEVICES，优先使用
        # CUDA会重新映射设备ID，所以这里总是使用'0'
        if cuda_visible_devices and cuda_visible_devices.strip():
            # 非空，说明设置了GPU
            if torch.cuda.is_available():
                # 如果是多个GPU（如"0,1"），返回"0"表示使用第一个可见的GPU
                # YOLO会根据CUDA_VISIBLE_DEVICES自动使用指定的GPU
                return '0'
            else:
                # 设置了CUDA_VISIBLE_DEVICES但CUDA不可用，回退到其他设备
                pass
    
    # 常规设备检测
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return '0'
    else:
        return 'cpu'


def get_device_name(device: Union[str, int]) -> str:
    """
    获取设备的友好名称
    
    Args:
        device: 设备标识
    
    Returns:
        str: 友好的设备名称
    """
    if device == 'mps':
        return "Apple Silicon (MPS)"
    elif device == 'cpu' or device == -1:
        return "CPU"
    elif isinstance(device, int) and device >= 0:
        if torch.cuda.is_available():
            return f"NVIDIA GPU - {torch.cuda.get_device_name(device)}"
        return f"GPU {device}"
    elif isinstance(device, str) and device.isdigit():
        gpu_id = int(device)
        if torch.cuda.is_available():
            return f"NVIDIA GPU - {torch.cuda.get_device_name(gpu_id)}"
        return f"GPU {gpu_id}"
    return str(device)


def validate_paths(*paths: Union[str, Path]) -> bool:
    """
    验证路径是否存在
    
    Args:
        *paths: 要验证的路径
    
    Returns:
        bool: 所有路径是否都存在
    """
    for path in paths:
        if not Path(path).exists():
            return False
    return True


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        Path: Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节大小
    
    Returns:
        str: 格式化后的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
    
    Returns:
        int: 文件大小（字节）
    """
    return Path(file_path).stat().st_size


def get_dataset_info(data_path: Union[str, Path]) -> dict:
    """
    提取数据集信息
    
    自动检测数据集类型（检测/分割 或 分类）并统计
    
    Args:
        data_path: 数据集路径
    
    Returns:
        dict: 数据集信息
    """
    data_path = Path(data_path)
    info = {
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'test_labels': 0,
    }
    
    # 检测是否是分类数据集结构
    # 分类：images/train/class1/*.jpg, images/train/class2/*.jpg
    # 检测：images/train/*.jpg + labels/train/*.txt
    train_dir = data_path / 'images' / 'train'
    is_classify = False
    
    if train_dir.exists():
        # 检查train目录下是否有子目录（类别目录）
        subdirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        # 如果有子目录且没有labels目录，可能是分类任务
        if subdirs and not (data_path / 'labels' / 'train').exists():
            is_classify = True
    
    # 统计各个split的图像和标签数量
    for split in ['train', 'val', 'test']:
        # 尝试两种目录结构
        # 结构1: images/train/, labels/train/
        img_dir = data_path / 'images' / split
        label_dir = data_path / 'labels' / split
        
        # 结构2: train/images/, train/labels/
        if not img_dir.exists():
            img_dir = data_path / split / 'images'
            label_dir = data_path / split / 'labels'
        
        # 处理 val/valid 别名
        if not img_dir.exists() and split == 'val':
            # 尝试 valid 作为 val 的别名
            img_dir = data_path / 'images' / 'valid'
            label_dir = data_path / 'labels' / 'valid'
            
            if not img_dir.exists():
                img_dir = data_path / 'valid' / 'images'
                label_dir = data_path / 'valid' / 'labels'
        
        if not img_dir.exists():
            continue
        
        if is_classify:
            # 分类任务：递归统计类别子目录中的图像
            image_count = 0
            for class_dir in img_dir.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    image_count += len(list(class_dir.glob('*.jpg'))) + \
                                  len(list(class_dir.glob('*.png'))) + \
                                  len(list(class_dir.glob('*.jpeg')))
            info[f'{split}_images'] = image_count
            info[f'{split}_labels'] = image_count  # 分类任务：图像数=标签数
        else:
            # 检测/分割任务：扁平结构
            info[f'{split}_images'] = len(list(img_dir.glob('*.jpg'))) + \
                                      len(list(img_dir.glob('*.png'))) + \
                                      len(list(img_dir.glob('*.jpeg')))
            
            if label_dir.exists():
                info[f'{split}_labels'] = len(list(label_dir.glob('*.txt')))
    
    return info


def find_files(directory: Union[str, Path], 
               extensions: Optional[List[str]] = None,
               recursive: bool = True) -> List[Path]:
    """
    查找指定扩展名的文件
    
    Args:
        directory: 搜索目录
        extensions: 文件扩展名列表（如 ['.jpg', '.png']）
        recursive: 是否递归搜索
    
    Returns:
        List[Path]: 找到的文件列表
    """
    directory = Path(directory)
    files = []
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for ext in extensions:
        if recursive:
            files.extend(directory.rglob(f'*{ext}'))
        else:
            files.extend(directory.glob(f'*{ext}'))
    
    return sorted(files)


def safe_import(module_name: str, package_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    安全导入模块
    
    Args:
        module_name: 模块名称
        package_name: 包名称（用于错误提示）
    
    Returns:
        Tuple[bool, Optional[str]]: (是否成功, 错误信息)
    """
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        pkg = package_name or module_name
        error_msg = f"缺少依赖: {pkg}。请运行: pip install {pkg}"
        return False, error_msg


def parse_ratio_string(ratio_str: str, expected_parts: int = 3) -> List[float]:
    """
    解析比例字符串（如 "0.7:0.2:0.1"）
    
    Args:
        ratio_str: 比例字符串
        expected_parts: 期望的部分数量
    
    Returns:
        List[float]: 比例列表
    
    Raises:
        ValueError: 如果格式不正确
    """
    parts = ratio_str.split(':')
    if len(parts) != expected_parts:
        raise ValueError(f"比例格式错误，期望 {expected_parts} 个部分，得到 {len(parts)} 个")
    
    try:
        ratios = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"比例值必须是数字: {ratio_str}")
    
    total = sum(ratios)
    if abs(total - 1.0) > 0.01:
        # 自动归一化
        ratios = [r / total for r in ratios]
    
    return ratios


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        Path: 项目根目录
    """
    # 从当前文件向上查找，直到找到包含 'workspace' 的目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == 'workspace':
            return parent
        # 或者查找包含特定标志文件的目录
        if (parent / 'yolo_cli.py').exists():
            return parent
    # 如果找不到，返回当前工作目录
    return Path.cwd()


def validate_task_type(task: str) -> str:
    """
    验证任务类型
    
    Args:
        task: 任务类型字符串
        
    Returns:
        str: 验证后的任务类型（标准化为小写）
        
    Raises:
        ValueError: 如果任务类型无效
    """
    try:
        task_type = TaskType.from_string(task)
        return task_type.value
    except ValueError as e:
        raise ValueError(str(e))


def get_task_specific_config(task: str) -> dict:
    """
    获取任务特定配置
    
    Args:
        task: 任务类型
        
    Returns:
        dict: 任务特定配置字典
    """
    task_type = TaskType.from_string(task)
    
    # 基础配置
    base_config = {
        'default_conf': 0.25,
        'default_iou': 0.45,
    }
    
    # 任务特定配置
    if task_type == TaskType.DETECT:
        return {
            **base_config,
            'save_txt': True,
            'save_json': True,
        }
    elif task_type == TaskType.SEGMENT:
        return {
            **base_config,
            'save_txt': True,
            'save_json': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'retina_masks': False,
        }
    elif task_type == TaskType.CLASSIFY:
        return {
            'default_conf': 0.25,
            'top_k': 5,
            'dropout': 0.0,
        }
    elif task_type == TaskType.POSE:
        return {
            **base_config,
            'save_txt': True,
            'save_json': True,
            'kpt_shape': None,  # 从 dataset.yaml 读取
            'pose_specific': True,
        }
    
    return base_config


def get_model_name_with_task(base_model: str, task: str) -> str:
    """
    根据任务类型获取完整的模型名称
    
    Args:
        base_model: 基础模型名称（如 'yolo11s.pt' 或 'yolo11s'）
        task: 任务类型
        
    Returns:
        str: 完整的模型名称
        
    Examples:
        >>> get_model_name_with_task('yolo11s.pt', 'segment')
        'yolo11s-seg.pt'
        >>> get_model_name_with_task('yolo11s', 'classify')
        'yolo11s-cls'
    """
    task_type = TaskType.from_string(task)
    suffix = task_type.get_model_suffix()
    
    # 移除已有的任务后缀
    base = base_model.replace('-seg', '').replace('-cls', '')
    
    # 分离扩展名
    if '.' in base:
        name, ext = base.rsplit('.', 1)
        return f"{name}{suffix}.{ext}"
    else:
        return f"{base}{suffix}"


def resolve_model_path(model: str, task: Optional[str] = None) -> Tuple[str, bool]:
    """
    解析模型路径，自动查找已下载的模型
    
    此函数会按以下顺序查找模型：
    1. 如果model是绝对路径或相对路径且文件存在，直接使用
    2. 在 models/weights/ 目录中查找
    3. 返回模型名称（让YOLO自动下载）
    
    Args:
        model: 模型名称或路径（如 'yolo11s.pt', 'path/to/model.pt'）
        task: 任务类型（可选，用于确定正确的模型文件名）
        
    Returns:
        Tuple[str, bool]: (解析后的模型路径或名称, 是否找到本地文件)
        
    Examples:
        >>> resolve_model_path('yolo11s.pt', 'pose')
        ('models/weights/yolo11s-pose.pt', True)  # 如果文件存在
        >>> resolve_model_path('yolo11s.pt', 'detect')
        ('yolo11s.pt', False)  # 如果文件不存在，返回模型名
    """
    model_path = Path(model)
    
    # 1. 如果是路径且文件存在，直接使用
    if model_path.exists():
        return (str(model_path), True)
    
    # 2. 在 models/weights/ 目录中查找
    try:
        from .config import ConfigManager
        config = ConfigManager()
        weights_dir = config.get_path('models', absolute=True) / 'weights'
        
        # 如果提供了任务类型，确保模型名包含正确的后缀
        if task:
            model_with_task = get_model_name_with_task(model_path.name if model_path.suffix else model, task)
        else:
            model_with_task = model_path.name if model_path.suffix else model
        
        weights_model_path = weights_dir / model_with_task
        
        if weights_model_path.exists():
            return (str(weights_model_path), True)
    except Exception:
        pass
    
    # 3. 没有找到本地文件，返回模型名称（让YOLO自动下载）
    if task:
        model_name = get_model_name_with_task(model_path.name if model_path.suffix else model, task)
    else:
        model_name = model_path.name if model_path.suffix else model
    
    return (model_name, False)


def parse_model_name(model_name: str) -> Tuple[str, str]:
    """
    解析模型名称，提取任务类型
    
    Args:
        model_name: 模型名称（如 'yolo11s-seg.pt'）
        
    Returns:
        Tuple[str, str]: (基础名称, 任务类型)
        
    Examples:
        >>> parse_model_name('yolo11s-seg.pt')
        ('yolo11s', 'segment')
        >>> parse_model_name('yolo11s.pt')
        ('yolo11s', 'detect')
    """
    # 移除扩展名
    if '.' in model_name:
        base = model_name.rsplit('.', 1)[0]
    else:
        base = model_name
    
    # 检查任务后缀
    if base.endswith('-seg'):
        return (base[:-4], TaskType.SEGMENT.value)
    elif base.endswith('-cls'):
        return (base[:-4], TaskType.CLASSIFY.value)
    elif base.endswith('-pose'):
        return (base[:-5], TaskType.POSE.value)
    else:
        return (base, TaskType.DETECT.value)
