#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO版本管理"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re


class YOLOVersionManager:
    """YOLO版本管理器"""
    
    SUPPORTED_VERSIONS = ['yolov8', 'yolo11']
    
    # 模型大小列表
    MODEL_SIZES = ['n', 's', 'm', 'l', 'x']
    
    # 模型映射
    MODEL_MAPPING = {
        'yolov8': {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt',
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt',
        },
        'yolo11': {
            'n': 'yolo11n.pt',
            's': 'yolo11s.pt',
            'm': 'yolo11m.pt',
            'l': 'yolo11l.pt',
            'x': 'yolo11x.pt',
        },
    }
    
    # 模型参数信息
    MODEL_INFO = {
        'n': {
            'name': 'Nano',
            'description': '最小最快，适合边缘设备',
            'params': '~3M',
            'speed': '最快',
        },
        's': {
            'name': 'Small',
            'description': '小型模型，速度与精度平衡',
            'params': '~11M',
            'speed': '快',
        },
        'm': {
            'name': 'Medium',
            'description': '中型模型，推荐用于大多数任务',
            'params': '~25M',
            'speed': '中等',
        },
        'l': {
            'name': 'Large',
            'description': '大型模型，更高精度',
            'params': '~43M',
            'speed': '较慢',
        },
        'x': {
            'name': 'Extra Large',
            'description': '超大型模型，最高精度',
            'params': '~68M',
            'speed': '慢',
        },
    }
    
    @classmethod
    def get_model_name(cls, version: str, size: str) -> str:
        """
        获取模型文件名
        
        Args:
            version: 版本 ('yolov8' 或 'yolo11')
            size: 大小 ('n', 's', 'm', 'l', 'x')
        
        Returns:
            str: 模型文件名
        """
        if version not in cls.SUPPORTED_VERSIONS:
            raise ValueError(f"不支持的版本: {version}。支持的版本: {cls.SUPPORTED_VERSIONS}")
        
        if size not in cls.MODEL_SIZES:
            raise ValueError(f"不支持的模型大小: {size}。支持的大小: {cls.MODEL_SIZES}")
        
        return cls.MODEL_MAPPING[version][size]
    
    @classmethod
    def detect_model_version(cls, model_path: str) -> Optional[str]:
        """
        从模型文件名检测版本
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            Optional[str]: 版本名称，如果无法检测则返回None
        """
        model_name = Path(model_path).name
        
        # 检查是否匹配 yolov8 或 yolo11 模式
        for version in cls.SUPPORTED_VERSIONS:
            pattern = f"{version}[nslmx]\\.pt"
            if re.match(pattern, model_name.lower()):
                return version
        
        return None
    
    @classmethod
    def detect_model_size(cls, model_path: str) -> Optional[str]:
        """
        从模型文件名检测大小
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            Optional[str]: 模型大小，如果无法检测则返回None
        """
        model_name = Path(model_path).name.lower()
        
        # 查找大小字母
        for size in cls.MODEL_SIZES:
            pattern = f"(yolov8|yolo11){size}\\.pt"
            if re.match(pattern, model_name):
                return size
        
        return None
    
    @classmethod
    def parse_model_name(cls, model_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析模型文件名，提取版本和大小
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (版本, 大小)
        """
        version = cls.detect_model_version(model_path)
        size = cls.detect_model_size(model_path)
        return version, size
    
    @classmethod
    def get_all_models(cls, version: Optional[str] = None) -> List[str]:
        """
        获取所有模型列表
        
        Args:
            version: 如果指定，只返回该版本的模型
        
        Returns:
            List[str]: 模型名称列表
        """
        if version:
            if version not in cls.SUPPORTED_VERSIONS:
                raise ValueError(f"不支持的版本: {version}")
            return list(cls.MODEL_MAPPING[version].values())
        
        # 返回所有版本的所有模型
        models = []
        for ver in cls.SUPPORTED_VERSIONS:
            models.extend(cls.MODEL_MAPPING[ver].values())
        return models
    
    @classmethod
    def get_model_info(cls, size: str) -> Dict[str, str]:
        """
        获取模型信息
        
        Args:
            size: 模型大小
        
        Returns:
            Dict[str, str]: 模型信息
        """
        if size not in cls.MODEL_INFO:
            raise ValueError(f"不支持的模型大小: {size}")
        
        return cls.MODEL_INFO[size].copy()
    
    @classmethod
    def normalize_version(cls, version: str) -> str:
        """
        标准化版本名称
        
        Args:
            version: 版本字符串 (可以是 'v8', '8', 'yolov8', 'v11', '11', 'yolo11')
        
        Returns:
            str: 标准化的版本名称
        """
        version = version.lower().strip()
        
        # 映射常见的别名
        version_aliases = {
            'v8': 'yolov8',
            '8': 'yolov8',
            'v11': 'yolo11',
            '11': 'yolo11',
        }
        
        if version in version_aliases:
            version = version_aliases[version]
        
        if version not in cls.SUPPORTED_VERSIONS:
            raise ValueError(f"不支持的版本: {version}。支持的版本: {cls.SUPPORTED_VERSIONS}")
        
        return version
    
    @classmethod
    def is_valid_model_name(cls, model_name: str) -> bool:
        """
        验证模型名称是否有效
        
        Args:
            model_name: 模型名称
        
        Returns:
            bool: 是否有效
        """
        version, size = cls.parse_model_name(model_name)
        return version is not None and size is not None
    
    @classmethod
    def get_recommended_model(cls, dataset_size: int, version: str = 'yolo11') -> str:
        """
        根据数据集大小推荐模型
        
        Args:
            dataset_size: 数据集大小（图片数量）
            version: YOLO版本
        
        Returns:
            str: 推荐的模型名称
        """
        version = cls.normalize_version(version)
        
        # 根据数据集大小推荐模型
        if dataset_size < 500:
            size = 'n'  # 小数据集用小模型，防止过拟合
        elif dataset_size < 2000:
            size = 's'
        elif dataset_size < 5000:
            size = 'm'
        elif dataset_size < 10000:
            size = 'l'
        else:
            size = 'x'
        
        return cls.get_model_name(version, size)
    
    @classmethod
    def compare_models(cls, model1: str, model2: str) -> Dict[str, any]:
        """
        比较两个模型
        
        Args:
            model1: 第一个模型路径或名称
            model2: 第二个模型路径或名称
        
        Returns:
            Dict: 比较结果
        """
        v1, s1 = cls.parse_model_name(model1)
        v2, s2 = cls.parse_model_name(model2)
        
        if not all([v1, s1, v2, s2]):
            return {
                'valid': False,
                'message': '无法解析模型名称'
            }
        
        size_order = {'n': 1, 's': 2, 'm': 3, 'l': 4, 'x': 5}
        
        return {
            'valid': True,
            'model1': {
                'name': model1,
                'version': v1,
                'size': s1,
                'info': cls.get_model_info(s1),
            },
            'model2': {
                'name': model2,
                'version': v2,
                'size': s2,
                'info': cls.get_model_info(s2),
            },
            'comparison': {
                'same_version': v1 == v2,
                'model1_larger': size_order[s1] > size_order[s2],
                'model2_larger': size_order[s2] > size_order[s1],
            }
        }
