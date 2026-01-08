#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图像去重工具模块"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from .utils import find_files
from ..ui.display import create_progress_bar, print_warning, print_info

# 尝试导入 imagehash（用于感知哈希）
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


class ImageDeduplicator:
    """图像去重器
    
    支持两种模式：
    1. 完全相同检测（MD5）：快速，只检测文件内容完全相同的图片
    2. 相似图片检测（感知哈希）：较慢，可检测视觉上相似的图片
    """
    
    def __init__(self, mode: str = 'exact', similarity_threshold: int = 8):
        """初始化去重器
        
        Args:
            mode: 去重模式
                - 'exact': 完全相同检测（MD5，默认）
                - 'similar': 相似图片检测（感知哈希，需要安装 imagehash）
            similarity_threshold: 相似度阈值（仅在 mode='similar' 时有效）
                - 0-5: 几乎相同（不同压缩质量）
                - 6-10: 很相似（轻微编辑）
                - 11-15: 相似
                - 16+: 不同
        """
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.hash_map: Dict[str, List[Path]] = defaultdict(list)
        self.removed_files: Set[Path] = set()
        self.total_size_saved: int = 0
        self.similarity_scores: Dict[Tuple[Path, Path], int] = {}  # 存储相似度分数
        
        # 检查依赖
        if mode == 'similar' and not IMAGEHASH_AVAILABLE:
            raise ImportError(
                "相似图片检测需要安装 imagehash 库。\n"
                "请运行: pip install imagehash pillow"
            )
    
    def compute_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """计算文件的哈希值（根据模式选择算法）
        
        Args:
            file_path: 图片文件路径
            chunk_size: 读取块大小（字节，仅用于 MD5）
            
        Returns:
            文件的哈希值（字符串）
        """
        if self.mode == 'exact':
            # MD5 哈希（快速，检测完全相同）
            return self._compute_md5_hash(file_path, chunk_size)
        elif self.mode == 'similar':
            # 感知哈希（较慢，检测相似图片）
            return self._compute_perceptual_hash(file_path)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def _compute_md5_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
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
    
    def _compute_perceptual_hash(self, file_path: Path) -> str:
        """计算图片的感知哈希值（pHash）
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            感知哈希值（字符串）
        """
        try:
            img = Image.open(file_path)
            # 使用 pHash（感知哈希），对图片内容变化不敏感
            phash = imagehash.phash(img)
            return str(phash)
        except Exception as e:
            # 如果读取或计算失败，返回空字符串
            return ""
    
    def find_duplicates(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """扫描图片列表，查找重复的图片
        
        Args:
            image_files: 图片文件路径列表
            
        Returns:
            重复图片映射 {hash: [file1, file2, ...]}
            - 对于 exact 模式：只包含完全相同的图片
            - 对于 similar 模式：包含相似度在阈值内的图片
        """
        self.hash_map.clear()
        self.similarity_scores.clear()
        
        if self.mode == 'exact':
            # 完全相同检测：直接比较哈希值
            return self._find_exact_duplicates(image_files)
        elif self.mode == 'similar':
            # 相似图片检测：计算汉明距离
            return self._find_similar_images(image_files)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def _find_exact_duplicates(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """查找完全相同的图片（MD5）
        
        Args:
            image_files: 图片文件路径列表
            
        Returns:
            重复图片映射 {hash: [file1, file2, ...]}
        """
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
    
    def _find_similar_images(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """查找相似的图片（感知哈希）
        
        Args:
            image_files: 图片文件路径列表
            
        Returns:
            相似图片映射 {group_id: [file1, file2, ...]}
        """
        # 计算所有图片的感知哈希
        file_hashes = {}
        
        with create_progress_bar() as progress:
            task_id = progress.add_task("计算感知哈希", total=len(image_files))
            
            for img_file in image_files:
                file_hash = self.compute_hash(img_file)
                if file_hash:
                    file_hashes[img_file] = file_hash
                progress.advance(task_id)
        
        # 比较所有图片对，找出相似的
        files = list(file_hashes.keys())
        similar_groups = []
        processed = set()
        
        print_info(f"比较 {len(files)} 张图片的相似度...")
        
        with create_progress_bar() as progress:
            task_id = progress.add_task("查找相似图片", total=len(files))
            
            for i, file1 in enumerate(files):
                if file1 in processed:
                    progress.advance(task_id)
                    continue
                
                # 创建新组
                group = [file1]
                hash1 = file_hashes[file1]
                
                # 与后续文件比较
                for file2 in files[i+1:]:
                    if file2 in processed:
                        continue
                    
                    hash2 = file_hashes[file2]
                    
                    # 计算汉明距离
                    if IMAGEHASH_AVAILABLE:
                        try:
                            h1 = imagehash.hex_to_hash(hash1)
                            h2 = imagehash.hex_to_hash(hash2)
                            distance = h1 - h2
                            
                            # 记录相似度
                            self.similarity_scores[(file1, file2)] = distance
                            
                            # 如果相似度在阈值内，添加到组
                            if distance <= self.similarity_threshold:
                                group.append(file2)
                                processed.add(file2)
                        except Exception:
                            pass
                
                # 如果组内有多个文件，添加到结果
                if len(group) > 1:
                    similar_groups.append(group)
                    processed.add(file1)
                
                progress.advance(task_id)
        
        # 转换为字典格式
        duplicates = {
            f"group_{i}": group 
            for i, group in enumerate(similar_groups)
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
            "mode": self.mode,
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
        
        # 添加相似度阈值信息（如果是相似模式）
        if self.mode == 'similar':
            report_data["similarity_threshold"] = self.similarity_threshold
        
        # 添加每组重复的详细信息
        for hash_val, files in duplicates_map.items():
            group_info = {
                "hash": hash_val,
                "count": len(files),
                "kept": str(files[0]),
                "removed": [str(f) for f in files[1:]]
            }
            
            # 添加相似度信息（如果是相似模式）
            if self.mode == 'similar':
                similarities = []
                kept_file = files[0]
                for removed_file in files[1:]:
                    key = (kept_file, removed_file)
                    if key in self.similarity_scores:
                        similarities.append({
                            "file": str(removed_file),
                            "distance": self.similarity_scores[key]
                        })
                if similarities:
                    group_info["similarities"] = similarities
            
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
            
            f.write(f"生成时间: {report_data['timestamp']}\n")
            f.write(f"去重模式: {'完全相同检测 (MD5)' if self.mode == 'exact' else f'相似图片检测 (感知哈希, 阈值={self.similarity_threshold})'}\n\n")
            
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
                    
                    # 如果是相似模式，显示相似度信息
                    if self.mode == 'similar' and 'similarities' in group:
                        for sim_info in group['similarities']:
                            distance = sim_info['distance']
                            file = sim_info['file']
                            # 添加相似度标签
                            if distance <= 5:
                                label = "几乎相同"
                            elif distance <= 10:
                                label = "很相似"
                            elif distance <= 15:
                                label = "相似"
                            else:
                                label = "略微相似"
                            f.write(f"  ✗ 删除: {file} (距离={distance}, {label})\n")
                    else:
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
