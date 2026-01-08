#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标注缩放模块 - 批量调整YOLO标注框大小"""

from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple
from datetime import datetime


class LabelScaler:
    """YOLO标注缩放器
    
    支持按比例缩放标注框（保持中心点不变），适用于：
    - 检测任务：缩放边界框
    - 分割任务：缩放边界框和多边形顶点
    - 姿态估计：只缩放边界框，关键点保持不变
    """
    
    def __init__(self):
        """初始化统计信息"""
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_annotations': 0,
            'scaled_annotations': 0,
            'skipped_annotations': 0,
            'warnings': []
        }
    
    def scale_detection_box(
        self, 
        center_x: float, 
        center_y: float, 
        width: float, 
        height: float, 
        scale: float
    ) -> Tuple[float, float, float, float]:
        """缩放检测框（保持中心点不变）
        
        Args:
            center_x: 中心点X坐标（归一化）
            center_y: 中心点Y坐标（归一化）
            width: 宽度（归一化）
            height: 高度（归一化）
            scale: 缩放比例（>0）
            
        Returns:
            缩放后的 (center_x, center_y, width, height)
        """
        # 缩放宽高
        new_width = width * scale
        new_height = height * scale
        
        # 计算缩放后框的边界
        x_min = center_x - new_width / 2
        y_min = center_y - new_height / 2
        x_max = center_x + new_width / 2
        y_max = center_y + new_height / 2
        
        # 裁剪到 [0, 1] 范围
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        x_max = min(1.0, x_max)
        y_max = min(1.0, y_max)
        
        # 重新计算中心和尺寸（放大时可能被裁剪）
        new_center_x = (x_min + x_max) / 2
        new_center_y = (y_min + y_max) / 2
        new_width = x_max - x_min
        new_height = y_max - y_min
        
        # 检查是否过小
        if new_width < 0.001 or new_height < 0.001:
            warning = f"警告：缩放后边界框过小 (w={new_width:.4f}, h={new_height:.4f})"
            if warning not in self.stats['warnings']:
                self.stats['warnings'].append(warning)
        
        return new_center_x, new_center_y, new_width, new_height
    
    def scale_segmentation_polygon(
        self, 
        points: List[float], 
        scale: float
    ) -> List[float]:
        """缩放分割多边形（相对中心点缩放）
        
        Args:
            points: 多边形顶点坐标列表 [x1, y1, x2, y2, ...]
            scale: 缩放比例
            
        Returns:
            缩放后的顶点坐标列表
        """
        if len(points) < 6:  # 至少需要3个点
            return points
        
        # 计算多边形中心
        xs = points[::2]
        ys = points[1::2]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        
        # 缩放每个点
        scaled_points = []
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            
            # 相对中心点缩放
            new_x = center_x + (x - center_x) * scale
            new_y = center_y + (y - center_y) * scale
            
            # 裁剪到 [0, 1] 范围
            new_x = max(0.0, min(1.0, new_x))
            new_y = max(0.0, min(1.0, new_y))
            
            scaled_points.extend([new_x, new_y])
        
        return scaled_points
    
    def process_label_file(
        self,
        input_file: Path,
        output_file: Path,
        scale: float,
        task: str,
        target_classes: Optional[Set[int]] = None
    ) -> int:
        """处理单个标注文件
        
        Args:
            input_file: 输入标注文件路径
            output_file: 输出标注文件路径
            scale: 缩放比例
            task: 任务类型 (detect/segment/pose)
            target_classes: 目标类别集合（None表示处理所有类别）
            
        Returns:
            处理的标注数量
        """
        if not input_file.exists():
            return 0
        
        self.stats['total_files'] += 1
        annotations_count = 0
        scaled_count = 0
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:  # 至少需要 class_id + bbox
                    f_out.write(line + '\n')
                    continue
                
                annotations_count += 1
                self.stats['total_annotations'] += 1
                
                try:
                    class_id = int(parts[0])
                    
                    # 检查是否需要处理此类别
                    if target_classes is not None and class_id not in target_classes:
                        self.stats['skipped_annotations'] += 1
                        f_out.write(line + '\n')
                        continue
                    
                    # 根据任务类型处理
                    if task == 'detect':
                        # 检测任务：只有边界框
                        if len(parts) >= 5:
                            center_x, center_y, width, height = map(float, parts[1:5])
                            new_cx, new_cy, new_w, new_h = self.scale_detection_box(
                                center_x, center_y, width, height, scale
                            )
                            new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                            f_out.write(new_line + '\n')
                            scaled_count += 1
                        else:
                            f_out.write(line + '\n')
                    
                    elif task == 'segment':
                        # 分割任务：边界框 + 多边形
                        if len(parts) >= 5:
                            # 提取并缩放边界框
                            center_x, center_y, width, height = map(float, parts[1:5])
                            new_cx, new_cy, new_w, new_h = self.scale_detection_box(
                                center_x, center_y, width, height, scale
                            )
                            
                            # 提取并缩放多边形顶点
                            polygon_coords = list(map(float, parts[5:]))
                            if len(polygon_coords) >= 6:  # 至少3个点
                                scaled_polygon = self.scale_segmentation_polygon(polygon_coords, scale)
                                polygon_str = ' '.join(f"{coord:.6f}" for coord in scaled_polygon)
                                new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f} {polygon_str}"
                            else:
                                new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                            
                            f_out.write(new_line + '\n')
                            scaled_count += 1
                        else:
                            f_out.write(line + '\n')
                    
                    elif task == 'pose':
                        # 姿态估计：只缩放边界框，关键点保持不变
                        if len(parts) >= 5:
                            center_x, center_y, width, height = map(float, parts[1:5])
                            new_cx, new_cy, new_w, new_h = self.scale_detection_box(
                                center_x, center_y, width, height, scale
                            )
                            
                            # 关键点保持不变
                            keypoints_str = ' '.join(parts[5:])
                            if keypoints_str:
                                new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f} {keypoints_str}"
                            else:
                                new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                            
                            f_out.write(new_line + '\n')
                            scaled_count += 1
                        else:
                            f_out.write(line + '\n')
                    
                    else:
                        # 未知任务类型，保持不变
                        f_out.write(line + '\n')
                
                except (ValueError, IndexError) as e:
                    # 解析错误，保持原行
                    f_out.write(line + '\n')
                    warning = f"解析错误 {input_file}: {str(e)}"
                    if warning not in self.stats['warnings']:
                        self.stats['warnings'].append(warning)
        
        if annotations_count > 0:
            self.stats['processed_files'] += 1
            self.stats['scaled_annotations'] += scaled_count
        
        return scaled_count
    
    def get_statistics(self) -> Dict:
        """获取处理统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_annotations': 0,
            'scaled_annotations': 0,
            'skipped_annotations': 0,
            'warnings': []
        }
    
    def generate_report(
        self,
        output_path: Path,
        dataset_dir: str,
        output_dir: str,
        scale: float,
        task: str,
        splits: Optional[List[str]] = None,
        classes: Optional[List[int]] = None,
        split_stats: Optional[Dict[str, Dict]] = None
    ):
        """生成调整报告
        
        Args:
            output_path: 报告文件路径
            dataset_dir: 数据集目录
            output_dir: 输出目录
            scale: 缩放比例
            task: 任务类型
            splits: 处理的子集
            classes: 处理的类别
            split_stats: 各子集的统计信息
        """
        report_file = output_path / 'adjustment_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("标注调整报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"数据集: {dataset_dir}\n")
            f.write(f"输出: {output_dir}\n")
            f.write(f"缩放比例: {scale}\n")
            f.write(f"任务类型: {task}\n")
            f.write(f"处理子集: {', '.join(splits) if splits else '全部'}\n")
            f.write(f"处理类别: {', '.join(map(str, classes)) if classes else '全部'}\n")
            f.write("\n")
            
            f.write("-" * 60 + "\n")
            f.write("统计信息\n")
            f.write("-" * 60 + "\n")
            f.write(f"总文件数: {self.stats['total_files']}\n")
            f.write(f"处理的文件数: {self.stats['processed_files']}\n")
            f.write(f"总标注数: {self.stats['total_annotations']}\n")
            f.write(f"缩放的标注数: {self.stats['scaled_annotations']}\n")
            f.write(f"跳过的标注数: {self.stats['skipped_annotations']}")
            if classes:
                f.write(" (未在目标类别中)")
            f.write("\n\n")
            
            # 子集详情
            if split_stats:
                f.write("-" * 60 + "\n")
                f.write("子集详情\n")
                f.write("-" * 60 + "\n")
                for split_name, stats in split_stats.items():
                    f.write(f"  {split_name:8s}: {stats['files']:4d} 文件, {stats['annotations']:6d} 标注\n")
                f.write("\n")
            
            # 警告
            if self.stats['warnings']:
                f.write("-" * 60 + "\n")
                f.write("警告信息\n")
                f.write("-" * 60 + "\n")
                for warning in self.stats['warnings'][:10]:  # 只显示前10条
                    f.write(f"  • {warning}\n")
                if len(self.stats['warnings']) > 10:
                    f.write(f"  ... 还有 {len(self.stats['warnings']) - 10} 条警告\n")
                f.write("\n")
            
            f.write("-" * 60 + "\n")
            f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
