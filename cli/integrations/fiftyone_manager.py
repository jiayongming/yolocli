#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FiftyOne数据集管理器"""

import yaml
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import importlib.util


class FiftyOneManager:
    """FiftyOne数据集管理器
    
    提供YOLO数据集的可视化和管理功能
    """
    
    def __init__(self):
        """初始化管理器"""
        self.fiftyone_available = self._check_fiftyone_installed()
        if self.fiftyone_available:
            import fiftyone as fo
            self.fo = fo
        else:
            self.fo = None
    
    def _check_fiftyone_installed(self) -> bool:
        """检查FiftyOne是否已安装
        
        Returns:
            bool: 是否已安装
        """
        spec = importlib.util.find_spec("fiftyone")
        return spec is not None
    
    def ensure_fiftyone(self) -> Tuple[bool, str]:
        """确保FiftyOne可用
        
        Returns:
            Tuple[bool, str]: (是否可用, 错误信息)
        """
        if not self.fiftyone_available:
            return (False, "FiftyOne未安装。请运行: pip install fiftyone")
        return (True, "")
    
    def load_yolo_dataset(
        self,
        data_yaml_path: str,
        dataset_name: Optional[str] = None,
        splits: Optional[List[str]] = None,
        persistent: bool = True,
        copy_to_datasets: bool = True,
        datasets_base_dir: Optional[str] = None
    ) -> Tuple[bool, Optional[str], str]:
        """从YOLO格式加载数据集到FiftyOne
        
        Args:
            data_yaml_path: dataset.yaml文件路径
            dataset_name: 数据集名称，如果为None则从yaml文件生成
            splits: 要加载的划分，默认['train', 'val', 'test']
            persistent: 是否持久化数据集
            copy_to_datasets: 是否先将数据集复制到datasets目录，默认True
            datasets_base_dir: datasets基础目录，默认为当前目录下的datasets
            
        Returns:
            Tuple[bool, Optional[str], str]: (是否成功, 数据集名称, 错误信息)
        """
        # 检查FiftyOne是否可用
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, None, error)
        
        try:
            # 读取dataset.yaml
            yaml_path = Path(data_yaml_path)
            if not yaml_path.exists():
                return (False, None, f"数据集配置文件不存在: {data_yaml_path}")
            
            # 提取配置信息
            # 优先使用 yaml 文件所在目录作为根目录
            dataset_root = yaml_path.parent.resolve()
            
            # 生成数据集名称（在复制前确定）
            if dataset_name is None:
                base_name = f"yolo_{dataset_root.name}"
                dataset_name = base_name
                
                # 检查名称是否已存在，如果存在则添加时间戳
                existing_datasets = self.fo.list_datasets()
                if dataset_name in existing_datasets:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dataset_name = f"{base_name}_{timestamp}"
            
            # 如果需要复制数据集
            if copy_to_datasets:
                # 确定datasets目录
                if datasets_base_dir is None:
                    datasets_base_dir = Path.cwd() / 'datasets'
                else:
                    datasets_base_dir = Path(datasets_base_dir)
                
                # 创建datasets目录
                datasets_base_dir.mkdir(parents=True, exist_ok=True)
                
                # 目标目录：datasets/{dataset_name}/
                target_dataset_dir = datasets_base_dir / dataset_name
                
                # 如果目标目录已存在，询问是否覆盖（这里直接覆盖，可以后续改进）
                if target_dataset_dir.exists():
                    # 删除旧的目录
                    shutil.rmtree(target_dataset_dir)
                
                # 复制整个数据集目录
                shutil.copytree(dataset_root, target_dataset_dir)
                
                # 更新 yaml_path 和 dataset_root 指向复制后的位置
                yaml_path = target_dataset_dir / yaml_path.name
                dataset_root = target_dataset_dir
                
                # 修改复制后的 dataset.yaml 文件，更新路径
                # 读取复制后的 yaml 文件
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    dataset_config = yaml.safe_load(f)
                
                # 保存原始的 path 值（用于路径转换）
                original_path = dataset_config.get('path', '.')
                
                # 更新 path 字段为相对路径（'.' 表示当前目录）
                # 这样数据集可以被移动而不影响使用
                dataset_config['path'] = '.'
                
                # 如果 train/val/test 路径是绝对路径，转换为相对路径
                for split_key in ['train', 'val', 'test']:
                    if split_key in dataset_config:
                        split_path_str = dataset_config[split_key]
                        split_path = Path(split_path_str)
                        
                        if split_path.is_absolute():
                            # 尝试转换为相对于新 dataset_root 的相对路径
                            try:
                                rel_path = split_path.relative_to(dataset_root)
                                dataset_config[split_key] = str(rel_path).replace('\\', '/')
                            except ValueError:
                                # 如果无法转换（路径不在 dataset_root 下）
                                # 尝试在新目录下查找对应的目录
                                dir_name = split_path.name
                                potential_path = dataset_root / dir_name
                                if potential_path.exists():
                                    dataset_config[split_key] = dir_name
                                else:
                                    # 保持使用目录名作为相对路径
                                    dataset_config[split_key] = split_path.name
                
                # 保存修改后的 yaml 文件
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            else:
                # 如果不复制，直接读取原始 yaml
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    dataset_config = yaml.safe_load(f)
            
            # 如果配置中有 path 字段，检查是否需要使用它
            if 'path' in dataset_config:
                config_path = Path(dataset_config['path'])
                
                # 如果是绝对路径，使用它
                if config_path.is_absolute():
                    if config_path.exists():
                        dataset_root = config_path
                # 如果是相对路径，但不是 '.' 或当前目录
                elif str(config_path) not in ['.', './']:
                    # 尝试从当前工作目录解析
                    test_path = Path.cwd() / config_path
                    if test_path.exists():
                        dataset_root = test_path.resolve()
                    else:
                        # 如果不存在，保持使用 yaml 所在目录
                        pass
            
            # 获取类别信息
            if 'names' in dataset_config:
                if isinstance(dataset_config['names'], dict):
                    classes = list(dataset_config['names'].values())
                elif isinstance(dataset_config['names'], list):
                    classes = dataset_config['names']
                else:
                    classes = []
            else:
                return (False, None, "数据集配置中缺少类别信息(names)")
            
            # 初始化调试信息和计数器
            sample_count = 0
            debug_info = []
            debug_info.append(f"Dataset root: {dataset_root}")
            debug_info.append(f"YAML path: {yaml_path}")
            
            # 检测是否为 Pose 任务
            is_pose = 'kpt_shape' in dataset_config
            kpt_shape = dataset_config.get('kpt_shape', None)
            if is_pose and kpt_shape:
                num_keypoints = kpt_shape[0] if isinstance(kpt_shape, list) else 17
                debug_info.append(f"Pose dataset detected with {num_keypoints} keypoints")
            else:
                num_keypoints = 0
            
            # 获取关键点标签（如果有的话）
            # 尝试从 dataset.yaml 中获取关键点名称
            keypoint_labels = dataset_config.get('keypoint_names', None)
            if not keypoint_labels and is_pose:
                # 如果没有定义，使用默认标签
                if num_keypoints == 4:
                    keypoint_labels = ['strat', 'end', 'center', 'pointer']
                elif num_keypoints == 17:
                    keypoint_labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
                else:
                    keypoint_labels = [f'kp_{i}' for i in range(num_keypoints)]
                debug_info.append(f"Using default keypoint labels: {keypoint_labels}")
            elif keypoint_labels:
                debug_info.append(f"Keypoint labels from config: {keypoint_labels}")
            
            # 确定要加载的划分
            if splits is None:
                splits = ['train', 'val', 'test']
            
            # 检查数据集是否已存在
            if dataset_name in self.fo.list_datasets():
                # 删除已存在的数据集
                self.fo.delete_dataset(dataset_name)
            
            # 创建FiftyOne数据集
            dataset = self.fo.Dataset(dataset_name, persistent=persistent)
            
            for split in splits:
                split_key = split
                if split_key not in dataset_config:
                    debug_info.append(f"Split '{split}' not in config")
                    continue
                
                # 获取图片目录路径
                split_path = dataset_config[split_key]
                debug_info.append(f"Split '{split}' path from config: {split_path}")
                
                images_dir = dataset_root / split_path
                debug_info.append(f"Constructed images_dir: {images_dir}")
                
                # 如果路径不存在，尝试解析相对路径
                if not images_dir.exists():
                    # 尝试作为绝对路径
                    images_dir = Path(split_path)
                    debug_info.append(f"Trying absolute path: {images_dir}")
                    if not images_dir.exists():
                        debug_info.append(f"Images dir not exists: {images_dir}")
                        continue
                
                # 获取标签目录（智能查找）
                # 方法1: 替换 'images' 为 'labels'
                labels_dir = Path(str(images_dir).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
                
                # 方法2: 如果方法1不存在，尝试在同级目录查找
                if not labels_dir.exists():
                    if images_dir.name == 'images' or 'images' in str(images_dir):
                        # 如果是 .../images/train，labels 在 .../labels/train
                        parent = images_dir.parent
                        relative_part = images_dir.relative_to(parent)
                        labels_dir = parent.parent / 'labels' / relative_part.name if 'images' in parent.name else parent / 'labels' / images_dir.name
                    else:
                        # 如果是 .../train，labels 可能在 .../labels 或 images 同级
                        labels_dir = images_dir.parent / 'labels' / images_dir.name
                
                # 遍历图片文件
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                    found = list(images_dir.glob(f'*{ext}'))
                    image_files.extend(found)
                
                debug_info.append(f"Found {len(image_files)} images in {images_dir}")
                debug_info.append(f"Labels dir: {labels_dir}, exists: {labels_dir.exists()}")
                
                for image_path in image_files:
                    sample_count += 1
                    # 读取对应的标签文件
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    
                    # 创建样本
                    sample = self.fo.Sample(filepath=str(image_path))
                    sample['split'] = split
                    
                    # 读取标注
                    detections = []
                    keypoints = []
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                parts = line.split()
                                if len(parts) >= 5:
                                    # YOLO格式: class_id x_center y_center width height [keypoints...]
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    
                                    # 转换为FiftyOne格式 [x, y, width, height]，左上角坐标
                                    bbox = [
                                        x_center - width / 2,
                                        y_center - height / 2,
                                        width,
                                        height
                                    ]
                                    
                                    label = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                                    
                                    # 检查是否有关键点数据
                                    if is_pose and len(parts) > 5:
                                        # Pose 格式：解析关键点
                                        # parts[5:] 包含关键点数据：x1 y1 v1 x2 y2 v2 ...
                                        kpt_data = parts[5:]
                                        expected_kpt_values = num_keypoints * 3
                                        
                                        if len(kpt_data) >= expected_kpt_values:
                                            # 解析关键点 - 为每个关键点创建独立的 Keypoint 对象
                                            for i in range(0, expected_kpt_values, 3):
                                                kpt_x = float(kpt_data[i])
                                                kpt_y = float(kpt_data[i + 1])
                                                kpt_v = float(kpt_data[i + 2])  # visibility: 0=未标注, 1=遮挡, 2=可见
                                                kpt_idx = i // 3
                                                
                                                # 只添加可见的关键点（visibility > 0）
                                                if kpt_v > 0:
                                                    # 获取关键点标签
                                                    if keypoint_labels and kpt_idx < len(keypoint_labels):
                                                        kpt_label = keypoint_labels[kpt_idx]
                                                    else:
                                                        kpt_label = f'kp_{kpt_idx}'
                                                    
                                                    # 为每个关键点创建独立的 Keypoint 对象
                                                    keypoint = self.fo.Keypoint(
                                                        label=kpt_label,  # 使用关键点的标签而不是对象类别
                                                        points=[[kpt_x, kpt_y]]  # 只包含一个点
                                                    )
                                                    keypoints.append(keypoint)
                                        
                                        # 仍然创建 Detection 对象（边界框）
                                        detection = self.fo.Detection(
                                            label=label,
                                            bounding_box=bbox
                                        )
                                        detections.append(detection)
                                    else:
                                        # 普通检测格式
                                        detection = self.fo.Detection(
                                            label=label,
                                            bounding_box=bbox
                                        )
                                        detections.append(detection)
                    
                    # 添加检测结果（边界框）
                    if detections:
                        sample['ground_truth'] = self.fo.Detections(detections=detections)
                    else:
                        # 负样本（无标注）
                        sample['ground_truth'] = self.fo.Detections(detections=[])
                    
                    # 添加关键点结果
                    if is_pose and keypoints:
                        sample['ground_truth_keypoints'] = self.fo.Keypoints(keypoints=keypoints)
                    
                    dataset.add_sample(sample)
            
            # 保存数据集
            if persistent:
                dataset.persistent = True
            
            # 检查是否成功加载样本
            if sample_count == 0:
                debug_msg = "\n".join(debug_info) if debug_info else "无调试信息"
                return (False, None, f"未能加载任何样本。\n\n调试信息:\n{debug_msg}\n\n请检查：\n1. dataset.yaml 中的路径配置\n2. 图片目录是否存在且有图片文件\n3. 路径是相对路径还是绝对路径")
            
            return (True, dataset_name, "")
            
        except Exception as e:
            return (False, None, f"加载数据集失败: {str(e)}")
    
    def launch_app(
        self,
        dataset_name: Optional[str] = None,
        port: int = 5151,
        auto_open: bool = True
    ) -> Tuple[bool, str]:
        """启动FiftyOne可视化应用
        
        Args:
            dataset_name: 数据集名称，如果为None则不加载数据集
            port: 服务端口
            auto_open: 是否自动打开浏览器
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, error)
        
        try:
            if dataset_name:
                # 加载指定数据集
                if dataset_name not in self.fo.list_datasets():
                    return (False, f"数据集 '{dataset_name}' 不存在")
                
                dataset = self.fo.load_dataset(dataset_name)
                session = self.fo.launch_app(
                    dataset,
                    port=port,
                    auto=auto_open
                )
            else:
                # 不加载数据集，只启动app
                session = self.fo.launch_app(
                    port=port,
                    auto=auto_open
                )
            
            return (True, "")
            
        except Exception as e:
            return (False, f"启动FiftyOne应用失败: {str(e)}")
    
    def list_datasets(self) -> Tuple[bool, List[str], str]:
        """列出所有FiftyOne数据集
        
        Returns:
            Tuple[bool, List[str], str]: (是否成功, 数据集列表, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, [], error)
        
        try:
            datasets = self.fo.list_datasets()
            return (True, datasets, "")
        except Exception as e:
            return (False, [], f"获取数据集列表失败: {str(e)}")
    
    def get_dataset_info(self, dataset_name: str) -> Tuple[bool, Optional[Dict], str]:
        """获取数据集详细信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Tuple[bool, Optional[Dict], str]: (是否成功, 数据集信息, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, None, error)
        
        try:
            if dataset_name not in self.fo.list_datasets():
                return (False, None, f"数据集 '{dataset_name}' 不存在")
            
            dataset = self.fo.load_dataset(dataset_name)
            
            # 收集统计信息
            info = {
                'name': dataset.name,
                'total_samples': len(dataset),
                'persistent': dataset.persistent,
                'media_type': dataset.media_type,
                'tags': dataset.tags,
            }
            
            # 按split统计
            if 'split' in dataset.get_field_schema():
                splits_info = {}
                for split in dataset.distinct('split'):
                    split_view = dataset.match(self.fo.ViewField('split') == split)
                    splits_info[split] = len(split_view)
                info['splits'] = splits_info
            
            # 类别统计
            if 'ground_truth' in dataset.get_field_schema():
                labels = dataset.distinct('ground_truth.detections.label')
                info['classes'] = labels
                info['num_classes'] = len(labels)
                
                # 统计每个类别的样本数
                class_counts = {}
                for label in labels:
                    count = len(dataset.filter_labels(
                        'ground_truth',
                        self.fo.ViewField('label') == label
                    ))
                    class_counts[label] = count
                info['class_counts'] = class_counts
            
            return (True, info, "")
            
        except Exception as e:
            return (False, None, f"获取数据集信息失败: {str(e)}")
    
    def delete_dataset(self, dataset_name: str) -> Tuple[bool, str]:
        """删除数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, error)
        
        try:
            if dataset_name not in self.fo.list_datasets():
                return (False, f"数据集 '{dataset_name}' 不存在")
            
            self.fo.delete_dataset(dataset_name)
            return (True, "")
            
        except Exception as e:
            return (False, f"删除数据集失败: {str(e)}")
    
    def delete_all_datasets(self) -> Tuple[bool, int, str]:
        """删除所有数据集
        
        Returns:
            Tuple[bool, int, str]: (是否成功, 删除数量, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, 0, error)
        
        try:
            datasets = self.fo.list_datasets()
            deleted_count = 0
            errors = []
            
            for dataset_name in datasets:
                try:
                    self.fo.delete_dataset(dataset_name)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"{dataset_name}: {str(e)}")
            
            if errors:
                error_msg = "; ".join(errors)
                return (False, deleted_count, f"部分删除失败: {error_msg}")
            
            return (True, deleted_count, "")
            
        except Exception as e:
            return (False, 0, f"删除所有数据集失败: {str(e)}")
    
    def close_app(self) -> Tuple[bool, str]:
        """关闭FiftyOne应用
        
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, error)
        
        try:
            self.fo.close_app()
            return (True, "")
        except Exception as e:
            return (False, f"关闭应用失败: {str(e)}")
    
    def _detect_pose_format(self, label_file_path: Path) -> Tuple[bool, int]:
        """检测标签文件是否为 Pose 格式
        
        Args:
            label_file_path: 标签文件路径
            
        Returns:
            Tuple[bool, int]: (是否为Pose格式, 关键点数量)
        """
        try:
            with open(label_file_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return (False, 0)
                
                parts = first_line.split()
                if len(parts) > 5:
                    # 检查是否符合 Pose 格式
                    # class_id x y w h [kp1_x kp1_y kp1_v ...] 或 class_id x y w h conf [kp1_x kp1_y kp1_v ...]
                    # 关键点数据应该是 3 的倍数
                    extra_values = len(parts) - 5
                    
                    # 情况1: class x y w h kp... (没有置信度)
                    if extra_values % 3 == 0:
                        num_keypoints = extra_values // 3
                        return (True, num_keypoints)
                    
                    # 情况2: class x y w h conf kp... (有置信度)
                    if (extra_values - 1) % 3 == 0:
                        num_keypoints = (extra_values - 1) // 3
                        return (True, num_keypoints)
                
                return (False, 0)
        except:
            return (False, 0)
    
    def add_predictions_to_dataset(
        self,
        dataset_name: str,
        predictions_dir: str,
        classes: Optional[List[str]] = None,
        field_name: str = "predictions",
        conf_threshold: float = 0.0
    ) -> Tuple[bool, Dict[str, int], str]:
        """将 YOLO 预测结果添加到 FiftyOne 数据集
        
        Args:
            dataset_name: 数据集名称
            predictions_dir: 预测结果目录（包含txt标签文件）
            classes: 类别列表（可选，不提供则自动读取）
            field_name: 预测结果字段名，默认"predictions"
            conf_threshold: 置信度阈值，过滤低置信度预测
            
        Returns:
            Tuple[bool, Dict[str, int], str]: (是否成功, 统计信息, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, {}, error)
        
        try:
            # 加载数据集
            if dataset_name not in self.fo.list_datasets():
                return (False, {}, f"数据集 '{dataset_name}' 不存在")
            
            dataset = self.fo.load_dataset(dataset_name)
            
            predictions_path = Path(predictions_dir)
            if not predictions_path.exists():
                return (False, {}, f"预测结果目录不存在: {predictions_dir}")
            
            # 如果没有提供类别列表，尝试自动读取
            if classes is None:
                # 方式1: 从预测目录读取 classes.txt
                classes_file = predictions_path / 'classes.txt'
                if classes_file.exists():
                    with open(classes_file, 'r', encoding='utf-8') as f:
                        classes = [line.strip() for line in f if line.strip()]
                
                # 方式2: 从数据集的 ground_truth 中提取类别
                if not classes and 'ground_truth' in dataset.get_field_schema():
                    classes = dataset.distinct('ground_truth.detections.label')
                    classes = sorted(classes)
                
                # 如果还是找不到，返回错误
                if not classes:
                    return (False, {}, "无法自动获取类别信息。请确保预测目录中有 classes.txt 文件，或手动指定 classes 参数")
            
            # 查找labels目录（YOLO预测结果通常保存在labels子目录）
            labels_dir = predictions_path / 'labels'
            if not labels_dir.exists():
                # 如果没有labels子目录，尝试直接使用predictions_dir
                labels_dir = predictions_path
            
            # 检测是否为 Pose 格式（检查第一个标签文件）
            is_pose = False
            num_keypoints = 0
            for label_file in labels_dir.glob('*.txt'):
                is_pose, num_keypoints = self._detect_pose_format(label_file)
                break  # 只检查第一个文件
            
            # 获取关键点标签（尝试从数据集的 ground_truth_keypoints 中提取）
            keypoint_labels = None
            if is_pose:
                # 尝试从已有的 ground_truth_keypoints 中获取标签
                if 'ground_truth_keypoints' in dataset.get_field_schema():
                    # 获取第一个有关键点的样本
                    samples_with_kpts = dataset.match(self.fo.ViewField('ground_truth_keypoints').exists())
                    if len(samples_with_kpts) > 0:
                        first_sample = samples_with_kpts.first()
                        if first_sample.ground_truth_keypoints and first_sample.ground_truth_keypoints.keypoints:
                            first_kpt = first_sample.ground_truth_keypoints.keypoints[0]
                            if hasattr(first_kpt, 'point_labels') and first_kpt.point_labels:
                                keypoint_labels = first_kpt.point_labels
                
                # 如果还是没有，使用默认标签
                if not keypoint_labels:
                    if num_keypoints == 4:
                        keypoint_labels = ['strat', 'end', 'center', 'pointer']
                    elif num_keypoints == 17:
                        keypoint_labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
                    else:
                        keypoint_labels = [f'kp_{i}' for i in range(num_keypoints)]
            
            stats = {
                'total_samples': len(dataset),
                'updated_samples': 0,
                'total_predictions': 0,
                'skipped_low_conf': 0
            }
            
            # 为每个样本添加预测结果
            for sample in dataset:
                # 获取图片文件名
                image_filename = Path(sample.filepath).stem
                label_file = labels_dir / f"{image_filename}.txt"
                
                if not label_file.exists():
                    continue
                
                # 读取预测结果
                detections = []
                keypoints = []
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:
                            # YOLO格式: class_id x_center y_center width height [confidence] [keypoints...]
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # 判断是否有置信度值
                            if is_pose:
                                # Pose 格式可能有置信度，检查数据长度
                                extra_values = len(parts) - 5
                                if (extra_values - 1) % 3 == 0:
                                    # 有置信度
                                    confidence = float(parts[5])
                                    kpt_start_idx = 6
                                else:
                                    # 没有置信度
                                    confidence = 1.0
                                    kpt_start_idx = 5
                            else:
                                # 普通检测格式
                                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                                kpt_start_idx = 6
                            
                            # 过滤低置信度预测
                            if confidence < conf_threshold:
                                stats['skipped_low_conf'] += 1
                                continue
                            
                            # 转换为FiftyOne格式 [x, y, width, height]，左上角坐标
                            bbox = [
                                x_center - width / 2,
                                y_center - height / 2,
                                width,
                                height
                            ]
                            
                            label = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                            
                            # 处理关键点数据
                            if is_pose and len(parts) > kpt_start_idx:
                                kpt_data = parts[kpt_start_idx:]
                                expected_kpt_values = num_keypoints * 3
                                
                                if len(kpt_data) >= expected_kpt_values:
                                    # 解析关键点 - 为每个关键点创建独立的 Keypoint 对象
                                    for i in range(0, expected_kpt_values, 3):
                                        kpt_x = float(kpt_data[i])
                                        kpt_y = float(kpt_data[i + 1])
                                        kpt_v = float(kpt_data[i + 2])
                                        kpt_idx = i // 3
                                        
                                        # 只添加可见的关键点
                                        if kpt_v > 0:
                                            # 获取关键点标签
                                            if keypoint_labels and kpt_idx < len(keypoint_labels):
                                                kpt_label = keypoint_labels[kpt_idx]
                                            else:
                                                kpt_label = f'kp_{kpt_idx}'
                                            
                                            # 为每个关键点创建独立的 Keypoint 对象
                                            keypoint = self.fo.Keypoint(
                                                label=kpt_label,  # 使用关键点的标签
                                                points=[[kpt_x, kpt_y]],  # 只包含一个点
                                                confidence=confidence
                                            )
                                            keypoints.append(keypoint)
                            
                            detection = self.fo.Detection(
                                label=label,
                                bounding_box=bbox,
                                confidence=confidence
                            )
                            detections.append(detection)
                            stats['total_predictions'] += 1
                
                # 添加预测结果到样本
                if detections:
                    sample[field_name] = self.fo.Detections(detections=detections)
                    if is_pose and keypoints:
                        sample[f"{field_name}_keypoints"] = self.fo.Keypoints(keypoints=keypoints)
                    sample.save()
                    stats['updated_samples'] += 1
            
            return (True, stats, "")
            
        except Exception as e:
            return (False, {}, f"添加预测结果失败: {str(e)}")
    
    def load_predictions_dataset(
        self,
        images_dir: str,
        predictions_dir: str,
        classes: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        conf_threshold: float = 0.0,
        persistent: bool = True
    ) -> Tuple[bool, Optional[str], str]:
        """从图片和预测结果创建FiftyOne数据集（纯预测，无ground truth）
        
        Args:
            images_dir: 图片目录
            predictions_dir: 预测结果目录（txt标签文件）
            classes: 类别列表（可选，不提供则自动读取）
            dataset_name: 数据集名称
            conf_threshold: 置信度阈值
            persistent: 是否持久化
            
        Returns:
            Tuple[bool, Optional[str], str]: (是否成功, 数据集名称, 错误信息)
        """
        available, error = self.ensure_fiftyone()
        if not available:
            return (False, None, error)
        
        try:
            images_path = Path(images_dir)
            predictions_path = Path(predictions_dir)
            
            if not images_path.exists():
                return (False, None, f"图片目录不存在: {images_dir}")
            
            if not predictions_path.exists():
                return (False, None, f"预测结果目录不存在: {predictions_dir}")
            
            # 如果没有提供类别列表，尝试自动读取
            if classes is None:
                # 从预测目录读取 classes.txt
                classes_file = predictions_path / 'classes.txt'
                if classes_file.exists():
                    with open(classes_file, 'r', encoding='utf-8') as f:
                        classes = [line.strip() for line in f if line.strip()]
                
                # 如果还是找不到，返回错误
                if not classes:
                    return (False, None, "无法自动获取类别信息。请确保预测目录中有 classes.txt 文件，或手动指定 classes 参数")
            
            # 生成数据集名称
            if dataset_name is None:
                dataset_name = f"predictions_{images_path.name}"
            
            # 检查数据集是否已存在
            if dataset_name in self.fo.list_datasets():
                self.fo.delete_dataset(dataset_name)
            
            # 创建数据集
            dataset = self.fo.Dataset(dataset_name, persistent=persistent)
            
            # 查找labels目录
            labels_dir = predictions_path / 'labels'
            if not labels_dir.exists():
                labels_dir = predictions_path
            
            # 遍历图片文件
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                image_files.extend(list(images_path.glob(f'*{ext}')))
                image_files.extend(list(images_path.glob(f'**/*{ext}')))
            
            sample_count = 0
            for image_path in image_files:
                # 读取对应的预测结果
                label_file = labels_dir / f"{image_path.stem}.txt"
                
                # 创建样本
                sample = self.fo.Sample(filepath=str(image_path))
                
                # 读取预测
                detections = []
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                                
                                if confidence < conf_threshold:
                                    continue
                                
                                bbox = [
                                    x_center - width / 2,
                                    y_center - height / 2,
                                    width,
                                    height
                                ]
                                
                                label = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                                
                                detection = self.fo.Detection(
                                    label=label,
                                    bounding_box=bbox,
                                    confidence=confidence
                                )
                                detections.append(detection)
                
                # 添加预测结果
                sample['predictions'] = self.fo.Detections(detections=detections)
                dataset.add_sample(sample)
                sample_count += 1
            
            if sample_count == 0:
                return (False, None, "未找到任何图片文件")
            
            return (True, dataset_name, "")
            
        except Exception as e:
            return (False, None, f"创建预测数据集失败: {str(e)}")

