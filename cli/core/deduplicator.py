#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图像去重工具模块"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from collections import defaultdict

from .utils import find_files
from ..ui.display import create_progress_bar


class ImageDeduplicator:
    """图像去重器
    
    使用文件哈希检测并删除完全相同的图片，避免数据集中的重复问题。
    """
    
    def __init__(self):
        """初始化去重器"""
        self.hash_map: Dict[str, List[Path]] = defaultdict(list)
        self.removed_files: Set[Path] = set()
        self.total_size_saved: int = 0
    
    def compute_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """计算文件的 MD5 哈希值
        
        Args:
            file_path: 图片文件路径
            chunk_size: 读取块大小（字节）
            
        Returns:
            文件的 MD5 哈希值（十六进制字符串）
        """
        md5_hash = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                # 分块读取文件，适合大文件
                while chunk := f.read(chunk_size):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception as e:
            # 如果读取失败，返回空字符串
            return ""
    
    def find_duplicates(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """扫描图片列表，查找重复的图片
        
        Args:
            image_files: 图片文件路径列表
            
        Returns:
            重复图片映射 {hash: [file1, file2, ...]}
            只包含有重复的哈希（即列表长度 > 1）
        """
        self.hash_map.clear()
        
        # 使用进度条显示扫描进度
        with create_progress_bar() as progress:
            task_id = progress.add_task("计算文件哈希", total=len(image_files))
            
            for img_file in image_files:
                file_hash = self.compute_hash(img_file)
                if file_hash:  # 只添加成功计算哈希的文件
                    self.hash_map[file_hash].append(img_file)
                progress.advance(task_id)
        
        # 只返回有重复的（列表长度 > 1）
        duplicates = {
            hash_val: files 
            for hash_val, files in self.hash_map.items() 
            if len(files) > 1
        }
        
        return duplicates
    
    def remove_duplicates(
        self, 
        duplicates_map: Dict[str, List[Path]], 
        labels_dir: Path = None,
        keep_first: bool = True
    ) -> List[Path]:
        """删除重复的图片及其对应的标签文件
        
        Args:
            duplicates_map: 重复图片映射（来自 find_duplicates）
            labels_dir: 标签文件目录（如果提供，会同时删除对应的标签）
            keep_first: 是否保留每组中的第一个文件（默认 True）
            
        Returns:
            已删除的文件路径列表
        """
        self.removed_files.clear()
        self.total_size_saved = 0
        
        for hash_val, files in duplicates_map.items():
            if keep_first:
                # 保留第一个，删除其余
                files_to_remove = files[1:]
            else:
                # 删除所有（一般不会用到）
                files_to_remove = files
            
            for file_path in files_to_remove:
                try:
                    # 记录文件大小
                    if file_path.exists():
                        self.total_size_saved += file_path.stat().st_size
                        
                        # 删除图片
                        file_path.unlink()
                        self.removed_files.add(file_path)
                        
                        # 删除对应的标签文件（如果存在）
                        if labels_dir:
                            label_file = labels_dir / f"{file_path.stem}.txt"
                            if label_file.exists():
                                label_file.unlink()
                
                except Exception as e:
                    # 删除失败时继续处理其他文件
                    print(f"警告: 无法删除 {file_path.name}: {e}")
        
        return list(self.removed_files)
    
    def generate_report(
        self, 
        duplicates_map: Dict[str, List[Path]], 
        output_path: Path,
        original_count: int = None
    ) -> None:
        """生成去重报告
        
        Args:
            duplicates_map: 重复图片映射
            output_path: 报告输出路径（.json）
            original_count: 原始图片总数（如果提供）
        """
        # 统计信息
        total_groups = len(duplicates_map)
        total_duplicates = sum(len(files) - 1 for files in duplicates_map.values())
        removed_count = len(self.removed_files)
        
        if original_count is None:
            original_count = sum(len(files) for files in duplicates_map.values())
        
        final_count = original_count - removed_count
        
        # 构建报告数据
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "original_image_count": original_count,
                "final_image_count": final_count,
                "duplicate_groups": total_groups,
                "duplicates_found": total_duplicates,
                "files_removed": removed_count,
                "space_saved_bytes": self.total_size_saved,
                "space_saved_mb": round(self.total_size_saved / (1024 * 1024), 2)
            },
            "duplicate_groups": []
        }
        
        # 添加每组重复的详细信息
        for hash_val, files in duplicates_map.items():
            group_info = {
                "hash": hash_val,
                "count": len(files),
                "kept": str(files[0]),
                "removed": [str(f) for f in files[1:]]
            }
            report_data["duplicate_groups"].append(group_info)
        
        # 保存 JSON 报告
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 同时生成可读的 TXT 报告
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("图像去重报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"生成时间: {report_data['timestamp']}\n\n")
            
            f.write("统计摘要:\n")
            f.write("-" * 70 + "\n")
            summary = report_data['summary']
            f.write(f"  原始图片数量: {summary['original_image_count']}\n")
            f.write(f"  去重后数量:   {summary['final_image_count']}\n")
            f.write(f"  重复组数:     {summary['duplicate_groups']}\n")
            f.write(f"  发现重复:     {summary['duplicates_found']} 个\n")
            f.write(f"  删除文件:     {summary['files_removed']} 个\n")
            f.write(f"  节省空间:     {summary['space_saved_mb']} MB\n")
            f.write("\n")
            
            if report_data['duplicate_groups']:
                f.write("重复文件详情:\n")
                f.write("-" * 70 + "\n")
                for i, group in enumerate(report_data['duplicate_groups'], 1):
                    f.write(f"\n第 {i} 组 (哈希: {group['hash'][:16]}...):\n")
                    f.write(f"  ✓ 保留: {group['kept']}\n")
                    for removed_file in group['removed']:
                        f.write(f"  ✗ 删除: {removed_file}\n")
            
            f.write("\n" + "=" * 70 + "\n")
    
    def get_statistics(self) -> Dict:
        """获取去重统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "removed_count": len(self.removed_files),
            "space_saved_bytes": self.total_size_saved,
            "space_saved_mb": round(self.total_size_saved / (1024 * 1024), 2)
        }
