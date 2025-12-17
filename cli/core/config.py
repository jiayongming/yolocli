#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration management"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .utils import ensure_dir, get_project_root


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG_PATH = "config/default.yaml"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.project_root = get_project_root()
        
        if config_path is None:
            config_path = self.project_root / self.DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = self.project_root / config_path
        
        self.config_path = config_path
        self.config = self._load_default_config()
        
        # 如果配置文件存在，加载它
        if self.config_path.exists():
            self.load(self.config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            'model': {
                'default_version': 'yolo11',
                'default_task': 'detect',
                'weights_dir': 'models/weights',
                'pretrained': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
            },
            'training': {
                'epochs': 200,
                'batch': 16,
                'imgsz': 640,
                'patience': 50,
                'save_period': 10,
                'device': 'auto',
            },
            'augmentation': {
                'default_preset': 'balanced',
            },
            'paths': {
                'data_raw': 'data/raw',
                'data_processed': 'data/processed',
                'results': 'results',
                'models': 'models',
            },
            'detection': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'save_txt': True,
                'save_json': True,
            },
            'tasks': {
                'detect': {
                    'default_model_size': 's',
                    'default_conf': 0.25,
                    'default_iou': 0.45,
                },
                'segment': {
                    'default_model_size': 's',
                    'default_conf': 0.25,
                    'default_iou': 0.45,
                    'overlap_mask': True,
                    'mask_ratio': 4,
                    'retina_masks': False,
                },
                'classify': {
                    'default_model_size': 's',
                    'dropout': 0.0,
                    'top_k': 5,
                },
            },
        }
    
    def load(self, config_path: Union[str, Path]) -> None:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
        
        # 深度合并配置
        self._deep_merge(self.config, loaded_config)
        self.config_path = config_path
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        保存配置文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用当前配置路径
        """
        if config_path is None:
            config_path = self.config_path
        else:
            config_path = Path(config_path)
        
        # 确保目录存在
        ensure_dir(config_path.parent)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，支持嵌套 (例如: 'model.default_version')
            default: 默认值
        
        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，支持嵌套 (例如: 'model.default_version')
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 导航到倒数第二层
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """获取数据增强配置"""
        return self.config.get('augmentation', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """获取路径配置"""
        return self.config.get('paths', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """获取检测配置"""
        return self.config.get('detection', {})
    
    def get_task_config(self, task: str) -> Dict[str, Any]:
        """
        获取任务特定配置
        
        Args:
            task: 任务类型 (detect, segment, classify)
            
        Returns:
            Dict[str, Any]: 任务配置
        """
        tasks_config = self.config.get('tasks', {})
        return tasks_config.get(task, {})
    
    def get_path(self, path_key: str, absolute: bool = False) -> Path:
        """
        获取路径配置
        
        Args:
            path_key: 路径键名
            absolute: 是否返回绝对路径
        
        Returns:
            Path: 路径对象
        """
        paths = self.get_paths_config()
        path = Path(paths.get(path_key, path_key))
        
        if absolute and not path.is_absolute():
            path = self.project_root / path
        
        return path
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            bool: 配置是否有效
        """
        required_keys = [
            'model.default_version',
            'training.epochs',
            'training.batch',
            'paths.data_raw',
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                return False
        
        return True
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """
        深度合并字典（就地修改base）
        
        Args:
            base: 基础字典
            update: 更新字典
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def load_profile(self, profile_name: str) -> None:
        """
        加载预设配置文件
        
        Args:
            profile_name: 预设名称 (small, medium, large)
        """
        profile_path = self.project_root / f"config/profiles/{profile_name}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"预设配置不存在: {profile_path}")
        
        self.load(profile_path)
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path={self.config_path})"
