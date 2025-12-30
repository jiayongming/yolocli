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
    
    def list_projects(self) -> Tuple[bool, List[Dict], str]:
        """获取所有项目列表
        
        Returns:
            Tuple[bool, List[Dict], str]: (是否成功, 项目列表, 错误信息)
                项目列表格式: [{"id": int, "title": str, "description": str, "task_number": int, "created_at": str}, ...]
        """
        try:
            response = self.session.get(f"{self.url}/api/projects/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Label Studio API 返回的是一个包含results的字典，或者直接是列表
                if isinstance(data, dict) and 'results' in data:
                    projects = data['results']
                elif isinstance(data, list):
                    projects = data
                else:
                    return (False, [], "返回数据格式不正确")
                
                # 提取关键信息
                project_list = []
                for proj in projects:
                    project_list.append({
                        'id': proj.get('id'),
                        'title': proj.get('title', '未命名项目'),
                        'description': proj.get('description', ''),
                        'task_number': proj.get('task_number', 0),
                        'created_at': proj.get('created_at', ''),
                    })
                
                return (True, project_list, "")
            elif response.status_code == 401:
                return (False, [], "认证失败：Token无效")
            elif response.status_code == 403:
                return (False, [], "认证失败：权限不足")
            else:
                return (False, [], f"获取项目列表失败：HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            return (False, [], f"无法连接到服务器：{self.url}")
        except requests.exceptions.Timeout:
            return (False, [], "连接超时")
        except Exception as e:
            return (False, [], f"获取项目列表失败：{str(e)}")
    
    def get_project_details(self, project_id: int) -> Tuple[bool, Optional[Dict], str]:
        """获取单个项目详情
        
        Args:
            project_id: 项目ID
            
        Returns:
            Tuple[bool, Optional[Dict], str]: (是否成功, 项目详情, 错误信息)
        """
        try:
            response = self.session.get(f"{self.url}/api/projects/{project_id}/", timeout=10)
            
            if response.status_code == 200:
                project = response.json()
                return (True, project, "")
            elif response.status_code == 404:
                return (False, None, f"项目 {project_id} 不存在")
            elif response.status_code in [401, 403]:
                return (False, None, "认证失败")
            else:
                return (False, None, f"获取项目详情失败：HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            return (False, None, f"无法连接到服务器：{self.url}")
        except requests.exceptions.Timeout:
            return (False, None, "连接超时")
        except Exception as e:
            return (False, None, f"获取项目详情失败：{str(e)}")
    
    def export_project(self, project_id: int, export_format: str = 'JSON') -> Tuple[bool, Optional[List[Dict]], str]:
        """导出项目标注数据
        
        Args:
            project_id: 项目ID
            export_format: 导出格式，默认JSON
            
        Returns:
            Tuple[bool, Optional[List[Dict]], str]: (是否成功, 标注数据, 错误信息)
        """
        try:
            # Label Studio export API endpoint
            response = self.session.get(
                f"{self.url}/api/projects/{project_id}/export",
                params={'exportType': export_format},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return (True, data, "")
            elif response.status_code == 404:
                return (False, None, f"项目 {project_id} 不存在")
            elif response.status_code in [401, 403]:
                return (False, None, "认证失败")
            else:
                return (False, None, f"导出失败：HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            return (False, None, f"无法连接到服务器：{self.url}")
        except requests.exceptions.Timeout:
            return (False, None, "导出超时（数据量可能较大）")
        except Exception as e:
            return (False, None, f"导出失败：{str(e)}")
    
    def download_project_images(
        self,
        project_id: int,
        output_dir: Path,
        skip_existing: bool = True,
        max_workers: int = 4,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Dict[str, int], str]:
        """下载项目的所有图片
        
        Args:
            project_id: 项目ID
            output_dir: 输出目录
            skip_existing: 是否跳过已存在的文件
            max_workers: 最大并发数
            progress_callback: 进度回调函数
            
        Returns:
            Tuple[bool, Dict[str, int], str]: (是否成功, 统计信息, 错误信息)
        """
        # 首先导出项目获取所有任务
        success, data, error = self.export_project(project_id)
        if not success:
            return (False, {}, f"无法导出项目数据：{error}")
        
        if not data:
            return (True, {"downloaded": 0, "skipped": 0, "failed": 0}, "项目中没有数据")
        
        # 提取所有图片路径
        image_list = []
        for task in data:
            image_path = task.get('data', {}).get('image', '')
            if image_path:
                filename = Path(image_path).name
                local_path = output_dir / filename
                image_list.append((image_path, local_path))
        
        if not image_list:
            return (True, {"downloaded": 0, "skipped": 0, "failed": 0}, "项目中没有图片")
        
        # 批量下载
        stats = self.download_images_batch(
            image_list=image_list,
            skip_existing=skip_existing,
            max_workers=max_workers,
            progress_callback=progress_callback
        )
        
        return (True, stats, "")
    
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
                            # 目标检测标注（矩形框）
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
                        
                        elif result_type == 'polygonlabels':
                            # 分割标注（多边形）- 转换为边界框用于检测任务
                            points = value.get('points', [])
                            if points:
                                # 计算多边形的边界框
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                
                                x_min = min(x_coords)
                                x_max = max(x_coords)
                                y_min = min(y_coords)
                                y_max = max(y_coords)
                                
                                annotation_item = {
                                    'type': 'polygon',  # 标记为多边形转换的
                                    'x': x_min,
                                    'y': y_min,
                                    'width': x_max - x_min,
                                    'height': y_max - y_min,
                                    'labels': value.get('polygonlabels', []),
                                    'points': points,  # 保留原始多边形点
                                }
                                
                                # 获取原始图像尺寸
                                if 'original_width' in result:
                                    item['image_width'] = result['original_width']
                                    item['image_height'] = result['original_height']
                                
                                item['annotations'].append(annotation_item)
                        
                        elif result_type == 'keypointlabels':
                            # 关键点标注 - 收集用于 Pose 格式转换
                            x = value.get('x', 0)
                            y = value.get('y', 0)
                            labels = value.get('keypointlabels', [])
                            
                            # 创建关键点数据结构
                            keypoint_item = {
                                'type': 'keypoint',
                                'x': x,
                                'y': y,
                                'labels': labels,
                                'label': labels[0] if labels else 'unknown',
                            }
                            
                            # 获取原始图像尺寸
                            if 'original_width' in result:
                                item['image_width'] = result['original_width']
                                item['image_height'] = result['original_height']
                            
                            item['annotations'].append(keypoint_item)
                            
                        elif result_type == 'choices':
                            # 分类标注
                            choices = value.get('choices', [])
                            if choices:
                                item['category'] = choices[0]
            
            # 根据 include_negative 参数决定是否添加负样本
            if include_negative or not item['is_negative']:
                parsed_data.append(item)
        
        # 后处理：将关键点标注合并为 Pose 格式
        parsed_data = LabelStudioConverter._merge_keypoints_to_pose(parsed_data)
        
        return parsed_data
    
    @staticmethod
    def _merge_keypoints_to_pose(parsed_data: List[Dict]) -> List[Dict]:
        """将多个独立的关键点标注合并为 Pose 格式
        
        Args:
            parsed_data: 解析后的数据列表
            
        Returns:
            List[Dict]: 处理后的数据列表，关键点已合并为 Pose 格式
        """
        # 首先收集所有不同的关键点标签，确定顺序
        # 使用多数投票法确定最常见的标注顺序
        from collections import Counter
        import sys
        
        all_labels = []
        label_first_occurrence = {}  # 记录每个标签第一次出现的顺序
        label_occurrence_count = {}  # 统计每个标签出现的次数，用于调试
        sample_orders = []  # 收集所有样本的关键点顺序
        
        for item in parsed_data:
            keypoints = [ann for ann in item['annotations'] if ann.get('type') == 'keypoint']
            
            # 记录有完整关键点的样本的顺序（用于多数投票）
            if len(keypoints) >= 4:
                order = tuple([kp.get('label', 'unknown') for kp in keypoints[:4]])
                sample_orders.append(order)
            
            for kp in keypoints:
                label = kp.get('label', 'unknown')
                if label not in label_first_occurrence:
                    label_first_occurrence[label] = len(all_labels)
                    all_labels.append(label)
                    label_occurrence_count[label] = 0
                label_occurrence_count[label] += 1
        
        # 使用多数投票确定最常见的顺序
        if sample_orders:
            order_counter = Counter(sample_orders)
            most_common_order, count = order_counter.most_common(1)[0]
            keypoint_order = list(most_common_order)
            
            print(f"ℹ 检测到 {len(sample_orders)} 个完整样本", file=sys.stderr)
            print(f"✓ 最常见的关键点顺序: {keypoint_order} (出现 {count}/{len(sample_orders)} 次)", file=sys.stderr)
            
            # 如果有多种顺序，显示警告
            if len(order_counter) > 1:
                print(f"⚠ 发现 {len(order_counter)} 种不同的标注顺序:", file=sys.stderr)
                for order, cnt in order_counter.most_common(3):
                    print(f"   {list(order)}: {cnt} 次", file=sys.stderr)
        else:
            # 回退到预定义顺序
            expected_order = ['strat', 'end', 'center', 'pointer']
            keypoint_order = expected_order
            print(f"⚠ 无法检测实际顺序，使用默认顺序: {expected_order}", file=sys.stderr)
        
        # 输出每个标签的统计信息（用于调试）
        if label_occurrence_count:
            import sys
            print(f"ℹ 关键点统计:", file=sys.stderr)
            for label in keypoint_order:
                count = label_occurrence_count.get(label, 0)
                print(f"  {label}: {count} 个", file=sys.stderr)
        
        processed_data = []
        inconsistent_samples = []  # 记录关键点顺序不一致的样本
        
        for item in parsed_data:
            # 检查是否有关键点标注
            keypoints = [ann for ann in item['annotations'] if ann.get('type') == 'keypoint']
            
            if not keypoints:
                # 没有关键点，保持原样
                processed_data.append(item)
                continue
            
            # 有关键点，合并为 Pose 格式
            # 创建一个字典来存储每个标签的关键点
            kp_dict = {}
            actual_labels = []  # 记录实际标注的顺序
            for kp in keypoints:
                label = kp.get('label', 'unknown')
                kp_dict[label] = (kp['x'], kp['y'])
                actual_labels.append(label)
            
            # 检查是否所有预期的关键点都存在
            missing_labels = [l for l in keypoint_order if l not in kp_dict]
            if missing_labels:
                inconsistent_samples.append({
                    'filename': item.get('filename', 'unknown'),
                    'issue': 'missing_labels',
                    'missing': missing_labels
                })
            
            # 按照预定义顺序组织关键点，如果某个关键点缺失，使用 (0, 0) 和 visibility=0
            ordered_keypoints = []
            all_x = []
            all_y = []
            
            for label in keypoint_order:
                if label in kp_dict:
                    x, y = kp_dict[label]
                    ordered_keypoints.append({
                        'x': x,
                        'y': y,
                        'visibility': 2,  # 2 = 可见
                        'label': label
                    })
                    all_x.append(x)
                    all_y.append(y)
                else:
                    # 关键点缺失
                    ordered_keypoints.append({
                        'x': 0,
                        'y': 0,
                        'visibility': 0,  # 0 = 未标注
                        'label': label
                    })
            
            # 计算包含所有关键点的边界框
            if all_x and all_y:
                min_x = min(all_x)
                max_x = max(all_x)
                min_y = min(all_y)
                max_y = max(all_y)
                
                # 添加一些边距（10%）
                margin_x = (max_x - min_x) * 0.1
                margin_y = (max_y - min_y) * 0.1
                
                bbox_x = max(0, min_x - margin_x)
                bbox_y = max(0, min_y - margin_y)
                bbox_w = min(100, max_x + margin_x) - bbox_x
                bbox_h = min(100, max_y + margin_y) - bbox_y
            else:
                # 如果没有有效的关键点，使用默认边界框
                bbox_x = 0
                bbox_y = 0
                bbox_w = 100
                bbox_h = 100
            
            # 创建 Pose 格式的标注
            pose_annotation = {
                'type': 'pose',
                'x': bbox_x,
                'y': bbox_y,
                'width': bbox_w,
                'height': bbox_h,
                'keypoints': ordered_keypoints,
                'labels': ['object'],  # 默认类别
            }
            
            # 替换原来的关键点标注
            new_item = item.copy()
            new_item['annotations'] = [pose_annotation]
            new_item['is_negative'] = False
            
            processed_data.append(new_item)
        
        # 输出关键点不一致的诊断信息
        if inconsistent_samples:
            import sys
            print(f"\n⚠ 发现 {len(inconsistent_samples)} 个样本的关键点标注不完整:", file=sys.stderr)
            for i, sample in enumerate(inconsistent_samples[:5]):  # 只显示前5个
                print(f"  {i+1}. {sample['filename']}: 缺失 {sample['missing']}", file=sys.stderr)
            if len(inconsistent_samples) > 5:
                print(f"  ... 还有 {len(inconsistent_samples) - 5} 个样本", file=sys.stderr)
            print(f"\n💡 建议: 请在 Label Studio 中检查这些样本，确保所有关键点都已标注", file=sys.stderr)
        
        return processed_data
    
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
            task_type: 任务类型 ('detect', 'pose', or 'classify')
            
        Returns:
            Dict[str, int]: {'class_name': class_id}
        """
        class_names = set()
        
        if task_type in ['detect', 'pose']:
            # 从检测/姿势标注中收集类别
            for item in parsed_data:
                for ann in item['annotations']:
                    labels = ann.get('labels', [])
                    class_names.update(labels)
        else:  # classify
            # 从分类标注中收集类别
            for item in parsed_data:
                if item['category']:
                    class_names.add(item['category'])
        
        # 如果没有找到类别，为 Pose 任务添加默认类别
        if not class_names and task_type == 'pose':
            class_names.add('object')
        
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
