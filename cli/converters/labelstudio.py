#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Label Studio data converter"""

import json
import csv
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


class LabelStudioClient:
    """Label Studio API 客户端"""
    
    def __init__(self, url: str, token: str, token_type: str = 'auto', auto_refresh: bool = True):
        """初始化客户端
        
        Args:
            url: Label Studio服务器URL（如 http://localhost:8080）
            token: API访问令牌（支持 Refresh Token 或 Access Token）
            token_type: Token类型 ('auto', 'legacy', 'bearer')
            auto_refresh: 如果是 Refresh Token，自动转换为 Access Token
        """
        self.url = url.rstrip('/')
        self.original_token = token
        self.token = token
        self.token_type = token_type
        
        # 如果启用自动刷新且 token 是 JWT 格式，检查是否为 refresh token
        if auto_refresh and token.startswith('eyJ'):
            token_payload = self._decode_jwt_payload(token)
            if token_payload and token_payload.get('token_type') == 'refresh':
                # 自动获取 access token
                access_token = self._exchange_access_token(token)
                if access_token:
                    self.token = access_token
        
        # 根据 token 类型设置认证头
        if token_type == 'bearer' or (token_type == 'auto' and self.token.startswith('eyJ')):
            # Personal Access Token (PAT) 或 JWT token
            self.headers = {'Authorization': f'Bearer {self.token}'}
        else:
            # Legacy Token (默认)
            self.headers = {'Authorization': f'Token {self.token}'}
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _decode_jwt_payload(self, token: str) -> Optional[Dict]:
        """解码 JWT token payload
        
        Args:
            token: JWT token
            
        Returns:
            Dict: payload 内容，失败返回 None
        """
        try:
            import base64
            parts = token.split('.')
            if len(parts) >= 2:
                payload = parts[1]
                # 添加 padding
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += '=' * padding
                decoded = base64.b64decode(payload)
                import json
                return json.loads(decoded)
        except Exception:
            return None
    
    def _exchange_access_token(self, refresh_token: str) -> Optional[str]:
        """使用 Refresh Token 交换 Access Token
        
        Args:
            refresh_token: Refresh Token
            
        Returns:
            str: Access Token，失败返回 None
        """
        endpoints = [
            "/api/token/refresh/",
            "/api/auth/token/refresh/",
            "/user/token/refresh/",
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.post(
                    f"{self.url}{endpoint}",
                    json={"refresh": refresh_token},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'access' in data:
                        return data['access']
            except Exception:
                continue
        
        return None
    
    def test_connection(self) -> Tuple[bool, str]:
        """测试连接和认证
        
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            # 尝试访问API根路径
            response = self.session.get(f"{self.url}/api/", timeout=10)
            if response.status_code == 200:
                return (True, "连接成功")
            elif response.status_code == 401:
                return (False, "认证失败：Token无效")
            elif response.status_code == 403:
                return (False, "认证失败：权限不足")
            else:
                return (False, f"连接失败：HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            return (False, f"无法连接到服务器：{self.url}")
        except requests.exceptions.Timeout:
            return (False, "连接超时")
        except Exception as e:
            return (False, f"未知错误：{str(e)}")
    
    def download_image(
        self, 
        image_path: str, 
        output_path: Path,
        skip_if_exists: bool = True
    ) -> Tuple[bool, str]:
        """下载图片（支持断点续传）
        
        Args:
            image_path: Label Studio中的图片路径（如 /data/upload/3/xxx.jpg）
            output_path: 本地保存路径
            skip_if_exists: 如果文件已存在是否跳过（断点续传）
            
        Returns:
            Tuple[bool, str]: (是否成功, 状态信息)
                - (True, "downloaded"): 成功下载
                - (True, "skipped"): 已存在，跳过
                - (False, error_msg): 下载失败
        """
        # 1. 断点续传：检查文件是否已存在
        if skip_if_exists and output_path.exists() and output_path.stat().st_size > 0:
            return (True, "skipped")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 2. 构建完整URL
            # Label Studio的图片URL格式：http://server/data/upload/project_id/filename
            if image_path.startswith('/'):
                image_url = f"{self.url}{image_path}"
            else:
                image_url = f"{self.url}/{image_path}"
            
            # 3. 发送HTTP GET请求（带认证header）
            response = self.session.get(image_url, timeout=30, stream=True)
            
            if response.status_code == 200:
                # 4. 保存到本地
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return (True, "downloaded")
            elif response.status_code == 404:
                return (False, "文件不存在")
            elif response.status_code in [401, 403]:
                return (False, "认证失败")
            else:
                return (False, f"HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            return (False, "下载超时")
        except requests.exceptions.ConnectionError:
            return (False, "网络连接错误")
        except Exception as e:
            return (False, f"下载失败: {str(e)}")
    
    def download_images_batch(
        self,
        image_list: List[Tuple[str, Path]],
        skip_existing: bool = True,
        max_workers: int = 4,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, int]:
        """批量下载图片（多线程）
        
        Args:
            image_list: [(label_studio_path, local_path), ...]
            skip_existing: 是否跳过已存在的文件
            max_workers: 最大并发数
            progress_callback: 进度回调函数 callback(current, total, status)
            
        Returns:
            Dict: {"downloaded": N, "skipped": M, "failed": K}
        """
        stats = {"downloaded": 0, "skipped": 0, "failed": 0}
        total = len(image_list)
        current = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_image = {
                executor.submit(
                    self.download_image, 
                    ls_path, 
                    local_path, 
                    skip_existing
                ): (ls_path, local_path)
                for ls_path, local_path in image_list
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_image):
                current += 1
                ls_path, local_path = future_to_image[future]
                
                try:
                    success, status = future.result()
                    if success:
                        if status == "downloaded":
                            stats["downloaded"] += 1
                        elif status == "skipped":
                            stats["skipped"] += 1
                    else:
                        stats["failed"] += 1
                        
                    # 调用进度回调
                    if progress_callback:
                        progress_callback(current, total, status, local_path.name)
                        
                except Exception as e:
                    stats["failed"] += 1
                    if progress_callback:
                        progress_callback(current, total, f"error: {str(e)}", local_path.name)
        
        return stats


class LabelStudioConverter:
    """Label Studio 数据转换器"""
    
    @staticmethod
    def detect_format(input_file: Path) -> str:
        """检测Label Studio导出格式
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            str: 'json' or 'csv'
        """
        suffix = input_file.suffix.lower()
        if suffix == '.json':
            return 'json'
        elif suffix == '.csv':
            return 'csv'
        else:
            # 尝试通过内容检测
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    first_char = f.read(1)
                    if first_char == '[' or first_char == '{':
                        return 'json'
                    else:
                        return 'csv'
            except Exception:
                return 'json'  # 默认
    
    @staticmethod
    def parse_json(json_file: Path, include_negative: bool = True) -> List[Dict]:
        """解析Label Studio JSON格式
        
        Args:
            json_file: JSON文件路径
            include_negative: 是否包含无标注的图片（负样本）
            
        Returns:
            List[Dict]: 解析后的数据列表
                [{
                    'image_path': str,  # Label Studio中的路径
                    'filename': str,    # 文件名
                    'annotations': [...],  # 检测任务的标注
                    'category': str,       # 分类任务的类别
                    'image_width': int,
                    'image_height': int,
                    'is_negative': bool,   # 是否为负样本（无标注）
                }, ...]
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        parsed_data = []
        
        for task in data:
            # 获取图片路径
            image_path = task['data'].get('image', '')
            if not image_path:
                continue
            
            filename = Path(image_path).name
            
            item = {
                'image_path': image_path,
                'filename': filename,
                'annotations': [],
                'category': None,
                'image_width': None,
                'image_height': None,
                'is_negative': True,  # 默认为负样本
            }
            
            # 检查是否有标注
            if task.get('annotations'):
                annotation = task['annotations'][0]
                results = annotation.get('result', [])
                
                if results:
                    item['is_negative'] = False
                    
                    # 解析标注结果
                    for result in results:
                        result_type = result.get('type', '')
                        value = result.get('value', {})
                        
                        if result_type == 'rectanglelabels':
                            # 目标检测标注
                            annotation_item = {
                                'type': 'rectangle',
                                'x': value.get('x', 0),
                                'y': value.get('y', 0),
                                'width': value.get('width', 0),
                                'height': value.get('height', 0),
                                'labels': value.get('rectanglelabels', []),
                            }
                            
                            # 获取原始图像尺寸
                            if 'original_width' in result:
                                item['image_width'] = result['original_width']
                                item['image_height'] = result['original_height']
                            
                            item['annotations'].append(annotation_item)
                            
                        elif result_type == 'choices':
                            # 分类标注
                            choices = value.get('choices', [])
                            if choices:
                                item['category'] = choices[0]
            
            # 根据 include_negative 参数决定是否添加负样本
            if include_negative or not item['is_negative']:
                parsed_data.append(item)
        
        return parsed_data
    
    @staticmethod
    def parse_csv(csv_file: Path, include_negative: bool = True) -> List[Dict]:
        """解析Label Studio CSV格式
        
        Args:
            csv_file: CSV文件路径
            include_negative: 是否包含无标注的图片（负样本）
            
        Returns:
            List[Dict]: 解析后的数据列表（格式同parse_json）
        """
        parsed_data = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                image_path = row.get('image', '')
                if not image_path:
                    continue
                
                filename = Path(image_path).name
                
                item = {
                    'image_path': image_path,
                    'filename': filename,
                    'annotations': [],
                    'category': None,
                    'image_width': None,
                    'image_height': None,
                    'is_negative': True,  # 默认为负样本
                }
                
                # 尝试解析label字段（可能是JSON字符串）
                label_str = row.get('label', '')
                if label_str:
                    try:
                        # CSV中的label字段可能是JSON数组
                        labels = json.loads(label_str)
                        if isinstance(labels, list):
                            for label in labels:
                                # 目标检测格式
                                if 'rectanglelabels' in label:
                                    annotation_item = {
                                        'type': 'rectangle',
                                        'x': label.get('x', 0),
                                        'y': label.get('y', 0),
                                        'width': label.get('width', 0),
                                        'height': label.get('height', 0),
                                        'labels': label.get('rectanglelabels', []),
                                    }
                                    
                                    # 获取原始图像尺寸
                                    if 'original_width' in label:
                                        item['image_width'] = label['original_width']
                                        item['image_height'] = label['original_height']
                                    
                                    item['annotations'].append(annotation_item)
                                    item['is_negative'] = False
                    except json.JSONDecodeError:
                        # 可能是直接的类别名
                        item['category'] = label_str
                        item['is_negative'] = False
                
                # 检查是否有直接的类别列（分类任务）
                # 根据实际CSV结构，可能有state、category等列
                for col in ['state', 'category', 'class', 'label']:
                    if col in row and row[col] and not item['category']:
                        item['category'] = row[col]
                        item['is_negative'] = False
                        break
                
                # 根据 include_negative 参数决定是否添加负样本
                if include_negative or not item['is_negative']:
                    parsed_data.append(item)
        
        return parsed_data
    
    @staticmethod
    def build_class_mapping(parsed_data: List[Dict], task_type: str) -> Dict[str, int]:
        """构建类别名称到ID的映射
        
        Args:
            parsed_data: 解析后的数据
            task_type: 任务类型 ('detect' or 'classify')
            
        Returns:
            Dict[str, int]: {'class_name': class_id}
        """
        class_names = set()
        
        if task_type == 'detect':
            # 从检测标注中收集类别
            for item in parsed_data:
                for ann in item['annotations']:
                    labels = ann.get('labels', [])
                    class_names.update(labels)
        else:  # classify
            # 从分类标注中收集类别
            for item in parsed_data:
                if item['category']:
                    class_names.add(item['category'])
        
        # 按字母顺序排序
        sorted_classes = sorted(class_names)
        
        # 创建映射
        class_mapping = {name: idx for idx, name in enumerate(sorted_classes)}
        
        return class_mapping
    
    @staticmethod
    def convert_bbox_to_yolo(
        x: float, y: float, 
        width: float, height: float
    ) -> Tuple[float, float, float, float]:
        """将Label Studio百分比坐标转换为YOLO格式
        
        Label Studio: x, y, width, height（相对百分比0-100，左上角）
        YOLO: x_center, y_center, width, height（归一化0-1，中心点）
        
        Args:
            x, y, width, height: Label Studio坐标（百分比）
            
        Returns:
            Tuple[float, float, float, float]: YOLO格式坐标
        """
        # 转换为归一化坐标（0-1）
        x_norm = x / 100.0
        y_norm = y / 100.0
        w_norm = width / 100.0
        h_norm = height / 100.0
        
        # 转换为中心点坐标
        x_center = x_norm + w_norm / 2
        y_center = y_norm + h_norm / 2
        
        # 确保在有效范围内
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))
        
        return x_center, y_center, w_norm, h_norm
    
    @staticmethod
    def prepare_download_list(
        parsed_data: List[Dict],
        output_images_dir: Path
    ) -> List[Tuple[str, Path]]:
        """准备图片下载列表
        
        Args:
            parsed_data: 解析后的标注数据
            output_images_dir: 输出图片目录
            
        Returns:
            List[Tuple[str, Path]]: [(label_studio_path, local_save_path), ...]
        """
        download_list = []
        
        for item in parsed_data:
            ls_path = item['image_path']
            filename = item['filename']
            local_path = output_images_dir / filename
            download_list.append((ls_path, local_path))
        
        return download_list
