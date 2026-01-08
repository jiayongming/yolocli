#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label Studio 上传进度跟踪器

管理上传进度，支持断点续传功能。
"""

import json
import hashlib
import threading
import fcntl
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime


class UploadProgressTracker:
    """上传进度跟踪器"""
    
    def __init__(self, project_id: int, dataset_path: Path, url: str, progress_dir: Optional[Path] = None):
        """
        初始化进度跟踪器
        
        Args:
            project_id: Label Studio 项目ID
            dataset_path: 数据集路径
            url: Label Studio 服务器 URL
            progress_dir: 进度文件目录（默认为 .upload_progress）
        """
        self.project_id = project_id
        self.dataset_path = Path(dataset_path).resolve()
        # 规范化 URL（移除尾部斜杠，统一比较）
        self.url = url.rstrip('/')
        
        # 创建进度目录
        if progress_dir is None:
            # 默认在项目根目录下创建 .upload_progress
            progress_dir = Path.cwd() / '.upload_progress'
        
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成进度文件名：project_{id}_{dataset_hash}.json
        dataset_hash = self._hash_path(self.dataset_path)
        self.progress_file = self.progress_dir / f"project_{project_id}_{dataset_hash}.json"
        
        # 内存数据结构
        self.uploaded_files: Dict[str, int] = {}  # {filename: task_id}
        self.failed_files: List[Dict] = []  # [{filename, error, timestamp}, ...]
        self.metadata: Dict = {
            'project_id': project_id,
            'dataset_path': str(self.dataset_path),
            'url': self.url,
            'created_at': datetime.now().isoformat()
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 加载现有进度
        self._load()
    
    def _hash_path(self, path: Path) -> str:
        """生成路径的哈希值（用于文件名）"""
        return hashlib.md5(str(path).encode()).hexdigest()[:8]
    
    def _load(self):
        """从文件加载进度"""
        if not self.progress_file.exists():
            return
        
        try:
            # 检查文件大小（限制为 10MB）
            file_size = self.progress_file.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB
                print(f"警告: 进度文件过大 ({file_size / 1024 / 1024:.1f}MB)，已清除")
                self.progress_file.unlink()
                return
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据完整性
            if not isinstance(data, dict):
                raise ValueError("Invalid progress file format")
            
            # 检查是否匹配当前配置
            metadata = data.get('metadata', {})
            if metadata.get('project_id') != self.project_id:
                # 项目ID不匹配，清除旧进度
                print(f"提示: 项目ID不匹配，已清除旧进度（旧: {metadata.get('project_id')}, 新: {self.project_id})")
                # 删除旧进度文件
                if self.progress_file.exists():
                    try:
                        self.progress_file.unlink()
                    except:
                        pass
                return
            
            if metadata.get('dataset_path') != str(self.dataset_path):
                # 数据集路径不匹配，清除旧进度
                print(f"提示: 数据集路径不匹配，已清除旧进度")
                # 删除旧进度文件
                if self.progress_file.exists():
                    try:
                        self.progress_file.unlink()
                    except:
                        pass
                return
            
            # 规范化 URL 比较（移除尾部斜杠）
            old_url = (metadata.get('url') or '').rstrip('/')
            new_url = self.url.rstrip('/')
            if old_url != new_url:
                # 服务器URL不匹配，清除旧进度
                print(f"提示: 服务器URL不匹配，已清除旧进度（旧: {metadata.get('url')}, 新: {self.url})")
                # 删除旧进度文件
                if self.progress_file.exists():
                    try:
                        self.progress_file.unlink()
                    except:
                        pass
                return
            
            # 加载数据
            self.uploaded_files = data.get('uploaded_files', {})
            self.failed_files = data.get('failed_files', [])
            self.metadata = metadata
            
        except (json.JSONDecodeError, ValueError, OSError) as e:
            # 进度文件损坏，清除
            print(f"警告: 进度文件损坏，已清除: {str(e)}")
            self.uploaded_files = {}
            self.failed_files = []
            self.metadata = {}
            # 删除损坏的文件
            if self.progress_file.exists():
                try:
                    self.progress_file.unlink()
                except:
                    pass
    
    def _save(self):
        """保存进度到文件（带文件锁防止并发冲突）"""
        with self._lock:
            # 更新元数据
            self.metadata.update({
                'project_id': self.project_id,
                'dataset_path': str(self.dataset_path),
                'url': self.url,
                'last_updated': datetime.now().isoformat(),
                'total_uploaded': len(self.uploaded_files),
                'total_failed': len(self.failed_files)
            })
            
            data = {
                'metadata': self.metadata,
                'uploaded_files': self.uploaded_files,
                'failed_files': self.failed_files
            }
            
            # 原子写入（先写临时文件，再重命名）
            temp_file = self.progress_file.with_suffix('.tmp')
            lock_file = self.progress_file.with_suffix('.lock')
            
            try:
                # 使用文件锁防止并发写入
                with open(lock_file, 'w') as lock_f:
                    try:
                        # 尝试获取排他锁（非阻塞）
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        # 写入临时文件
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        # 重命名（原子操作）
                        temp_file.replace(self.progress_file)
                        
                        # 释放锁
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                        
                    except IOError:
                        # 无法获取锁，可能有其他进程正在写入
                        # 等待一小段时间后重试
                        import time
                        time.sleep(0.1)
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # 阻塞等待
                        
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        temp_file.replace(self.progress_file)
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                
                # 清理锁文件
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except:
                        pass
                        
            except Exception as e:
                # 清理临时文件
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                # 清理锁文件
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except:
                        pass
                raise e
    
    def mark_uploaded(self, filename: str, task_id: int, auto_save: bool = True):
        """
        标记文件已上传
        
        Args:
            filename: 文件名
            task_id: Label Studio 任务ID
            auto_save: 是否立即保存（默认True，批量操作时可设为False）
        """
        with self._lock:
            self.uploaded_files[filename] = task_id
            
            # 从失败列表中移除（如果存在）
            self.failed_files = [
                f for f in self.failed_files if f['filename'] != filename
            ]
        
        if auto_save:
            self._save()
    
    def mark_failed(self, filename: str, error: str, auto_save: bool = True):
        """
        标记文件上传失败
        
        Args:
            filename: 文件名
            error: 错误信息
            auto_save: 是否立即保存（默认True）
        """
        with self._lock:
            # 检查是否已在失败列表中
            existing = None
            for f in self.failed_files:
                if f['filename'] == filename:
                    existing = f
                    break
            
            if existing:
                # 更新错误信息和时间
                existing['error'] = error
                existing['timestamp'] = datetime.now().isoformat()
                existing['retry_count'] = existing.get('retry_count', 0) + 1
            else:
                # 添加新的失败记录
                self.failed_files.append({
                    'filename': filename,
                    'error': error,
                    'timestamp': datetime.now().isoformat(),
                    'retry_count': 1
                })
        
        if auto_save:
            self._save()
    
    def is_uploaded(self, filename: str) -> bool:
        """
        检查文件是否已上传
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否已上传
        """
        return filename in self.uploaded_files
    
    def get_uploaded_files(self) -> Dict[str, int]:
        """
        获取已上传文件列表
        
        Returns:
            Dict[str, int]: {filename: task_id}
        """
        with self._lock:
            return self.uploaded_files.copy()
    
    def get_failed_files(self) -> List[Dict]:
        """
        获取失败文件列表
        
        Returns:
            List[Dict]: [{filename, error, timestamp, retry_count}, ...]
        """
        with self._lock:
            return self.failed_files.copy()
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            return {
                'uploaded_count': len(self.uploaded_files),
                'failed_count': len(self.failed_files),
                'last_updated': self.metadata.get('last_updated'),
                'project_id': self.project_id,
                'dataset_path': str(self.dataset_path),
                'url': self.url,
                'progress_file': str(self.progress_file)
            }
    
    def clear(self):
        """清除所有进度"""
        with self._lock:
            self.uploaded_files = {}
            self.failed_files = []
            self.metadata = {}
        
        # 删除进度文件
        if self.progress_file.exists():
            self.progress_file.unlink()
    
    def filter_uploaded(self, filenames: List[str]) -> List[str]:
        """
        过滤掉已上传的文件
        
        Args:
            filenames: 文件名列表
            
        Returns:
            List[str]: 未上传的文件名列表
        """
        return [f for f in filenames if not self.is_uploaded(f)]
    
    def get_uploaded_set(self) -> Set[str]:
        """
        获取已上传文件集合（用于快速查询）
        
        Returns:
            Set[str]: 已上传文件名集合
        """
        with self._lock:
            return set(self.uploaded_files.keys())
    
    def batch_mark_uploaded(self, files: Dict[str, int]):
        """
        批量标记文件已上传
        
        Args:
            files: {filename: task_id}
        """
        with self._lock:
            self.uploaded_files.update(files)
            
            # 从失败列表中移除
            failed_filenames = {f['filename'] for f in self.failed_files}
            for filename in files.keys():
                if filename in failed_filenames:
                    self.failed_files = [
                        f for f in self.failed_files if f['filename'] != filename
                    ]
        
        self._save()
    
    def force_save(self):
        """
        强制保存进度（用于周期性保存或中断时保存）
        """
        self._save()
    
    def has_progress(self) -> bool:
        """
        检查是否有保存的进度（有实际上传记录）
        
        Returns:
            bool: 是否有进度
        """
        # 只有当有上传记录或失败记录时才认为有进度
        return len(self.uploaded_files) > 0 or len(self.failed_files) > 0
    
    def get_progress_info(self) -> Optional[Dict]:
        """
        获取进度摘要信息（用于显示给用户）
        
        Returns:
            Optional[Dict]: 进度信息，如果没有进度则返回 None
        """
        if not self.has_progress():
            return None
        
        with self._lock:
            # 优先使用 last_updated，否则使用 created_at
            time_str = self.metadata.get('last_updated') or self.metadata.get('created_at', 'Unknown')
            
            return {
                'dataset_name': self.dataset_path.name,
                'project_id': self.project_id,
                'uploaded_count': len(self.uploaded_files),
                'failed_count': len(self.failed_files),
                'last_updated': time_str,
                'url': self.url
            }
