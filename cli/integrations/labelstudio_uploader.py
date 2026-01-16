#!/usr/bin/env python3
"""
Label Studio上传器：将YOLO数据集上传到Label Studio

支持功能：
- 上传YOLO格式数据集到Label Studio
- 自动配置标注模板
- 支持检测和分割任务
- Token自动处理（Refresh Token转换）
- 批量上传和进度显示
"""

import json
import base64
import yaml
import requests
import xml.etree.ElementTree as ET
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import threading

from ..ui.display import print_info, print_success, print_warning, print_error


class FatalUploadError(Exception):
    """致命上传错误，需要立即停止整个上传过程"""
    def __init__(self, message: str, status_code: int = None, response_text: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class LabelStudioUploader:
    """Label Studio上传器"""
    
    def __init__(self, url: str, api_key: str, project_id: int, task_type: str = 'detect'):
        """
        初始化上传器
        
        Args:
            url: Label Studio服务器URL
            api_key: API密钥（支持Refresh Token或Access Token）
            project_id: 项目ID
            task_type: 任务类型 (detect/segment/pose/classify)
        """
        self.url = url.rstrip('/')
        self.original_token = api_key
        self.api_key = self._process_token(api_key)
        self.project_id = project_id
        self.task_type = task_type
        self.headers = self._get_auth_headers(self.api_key)
        self.classes = []
        self.keypoint_names = []  # pose任务的关键点名称
        self.keypoint_from_name = "keypoint"  # KeyPointLabels 控件的 name 属性
    
    def _decode_jwt_payload(self, token: str) -> Optional[Dict]:
        """解码JWT token payload"""
        try:
            parts = token.split('.')
            if len(parts) >= 2:
                payload = parts[1]
                # 添加padding
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += '=' * padding
                decoded = base64.b64decode(payload)
                return json.loads(decoded)
        except Exception:
            return None
    
    def _exchange_access_token(self, refresh_token: str) -> Optional[str]:
        """使用Refresh Token交换Access Token"""
        print_info("检测到Refresh Token，尝试获取Access Token...")
        
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
                        print_success("✓ 成功获取Access Token")
                        return data['access']
            except Exception:
                continue
        
        print_warning("无法从Refresh Token获取Access Token")
        return None
    
    def _process_token(self, token: str) -> str:
        """智能处理token：自动检测并转换refresh token"""
        if token.startswith('eyJ'):
            payload = self._decode_jwt_payload(token)
            if payload and payload.get('token_type') == 'refresh':
                access_token = self._exchange_access_token(token)
                if access_token:
                    return access_token
                else:
                    print_warning("⚠️  无法交换Access Token，将尝试直接使用原token")
        return token
    
    def _get_auth_headers(self, token: str) -> Dict[str, str]:
        """根据token类型设置认证头"""
        headers = {'Content-Type': 'application/json'}
        
        if token.startswith('eyJ'):
            headers['Authorization'] = f'Bearer {token}'
        else:
            headers['Authorization'] = f'Token {token}'
        
        return headers
    
    def get_project_keypoint_labels(self) -> Tuple[List[str], str]:
        """
        从 Label Studio 项目配置中获取关键点标签名称和控件名称
        
        Returns:
            Tuple[List[str], str]: (关键点标签名称列表, KeyPointLabels的name属性)
        """
        try:
            # 获取项目信息
            response = requests.get(
                f"{self.url}/api/projects/{self.project_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                return [], "keypoint"
            
            project = response.json()
            label_config = project.get('label_config', '')
            
            if not label_config:
                return [], "keypoint"
            
            # 解析 XML 配置中的 KeyPointLabels
            import re
            # 先找到 <KeyPointLabels> ... </KeyPointLabels> 块
            keypoint_block_pattern = r'<KeyPointLabels[^>]*>.*?</KeyPointLabels>'
            keypoint_blocks = re.findall(keypoint_block_pattern, label_config, re.DOTALL)
            
            if not keypoint_blocks:
                return [], "keypoint"
            
            keypoint_block = keypoint_blocks[0]
            
            # 提取 KeyPointLabels 的 name 属性
            name_pattern = r'<KeyPointLabels[^>]*name=["\']([^"\']+)["\']'
            name_match = re.search(name_pattern, keypoint_block)
            control_name = name_match.group(1) if name_match else "keypoint"
            
            # 在 KeyPointLabels 块中提取所有 <Label value="xxx"/> 标签
            label_pattern = r'<Label\s+value="([^"]+)"'
            matches = re.findall(label_pattern, keypoint_block)
            
            if matches:
                print_info(f"从 Label Studio 项目获取到关键点标签: {matches}")
                print_info(f"KeyPointLabels 控件名称: {control_name}")
                return matches, control_name
            
            return [], control_name
            
        except Exception as e:
            print_warning(f"无法从项目获取关键点标签: {str(e)}")
            return [], "keypoint"
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = requests.get(
                f"{self.url}/api/projects/{self.project_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                project_data = response.json()
                project_title = project_data.get('title', 'Unknown')
                print_success(f"✓ 连接成功")
                print_info(f"  项目: {project_title} (ID: {self.project_id})")
                return True
            elif response.status_code == 404:
                print_error(f"✗ 连接失败: 项目不存在 (ID: {self.project_id})")
                print_warning("  请确认项目ID是否正确")
                print_info("  💡 提示: 在 Label Studio 中打开项目，URL中的数字就是项目ID")
                print_info("     例如: http://10.105.3.39/projects/19/ → 项目ID是 19")
                return False
            elif response.status_code == 403:
                print_error(f"✗ 连接失败: 权限不足 (403)")
                print_warning("  请确认API密钥是否有访问该项目的权限")
                return False
            elif response.status_code == 401:
                print_error(f"✗ 连接失败: 未授权 (401)")
                print_warning("  请确认API密钥是否正确")
                return False
            else:
                print_error(f"✗ 连接失败: {response.status_code}")
                try:
                    error_detail = response.json()
                    print_warning(f"  详情: {error_detail}")
                except:
                    print_warning(f"  响应: {response.text[:200]}")
                return False
        except requests.exceptions.ConnectionError:
            print_error(f"✗ 连接异常: 无法连接到服务器")
            print_warning(f"  请确认URL是否正确: {self.url}")
            print_info("  💡 提示: 确保Label Studio服务正在运行")
            return False
        except requests.exceptions.Timeout:
            print_error(f"✗ 连接异常: 连接超时")
            print_warning(f"  服务器响应时间过长")
            return False
        except Exception as e:
            print_error(f"✗ 连接异常: {str(e)}")
            return False
    
    def get_project_task_count(self) -> Optional[int]:
        """
        获取项目任务总数
        
        Returns:
            Optional[int]: 任务数量，失败返回 None
        """
        try:
            response = requests.get(
                f"{self.url}/api/projects/{self.project_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                project_data = response.json()
                return project_data.get('task_number', 0)
            
            return None
        except Exception:
            return None
    
    def build_existing_tasks_map(self, show_progress: bool = True) -> Dict[str, int]:
        """
        构建现有任务映射表（通过 original_filename 字段）
        
        使用 Export API 获取所有任务，提取 data.original_filename 字段，
        构建 {original_filename: task_id} 映射表。
        
        原始文件名格式为 'split/filename' (例如: 'train/image.jpg')
        
        Args:
            show_progress: 是否显示进度信息
            
        Returns:
            Dict[str, int]: {'split/filename': task_id} 映射表
        """
        if show_progress:
            print_info("\n🔍 正在检查服务器上已有的任务...")
        
        try:
            # 使用 Export API 获取所有任务
            export_url = f"{self.url}/api/projects/{self.project_id}/export"
            response = requests.get(
                export_url,
                headers=self.headers,
                params={'exportType': 'JSON'},
                timeout=60
            )
            
            if response.status_code != 200:
                if show_progress:
                    print_warning(f"⚠ 无法获取任务列表: HTTP {response.status_code}")
                return {}
            
            all_tasks = response.json()
            
            if show_progress:
                print_info(f"  服务器返回 {len(all_tasks)} 个任务")
            
            # 构建映射表（兼容新旧格式）
            mapping = {}
            old_format_count = 0
            for task in all_tasks:
                task_id = task.get('id')
                data = task.get('data', {})
                original_filename = data.get('original_filename')
                split_info = data.get('split')  # 新格式会包含这个字段
                
                if task_id and original_filename:
                    # 新格式：split/filename（如 train/image.jpg）
                    mapping[original_filename] = task_id
                    
                    # 兼容旧格式：如果不包含"/"，假设是train数据
                    # （因为之前只上传了train数据）
                    if '/' not in original_filename and not split_info:
                        train_key = f"train/{original_filename}"
                        if train_key not in mapping:
                            mapping[train_key] = task_id
                            old_format_count += 1
            
            if show_progress and old_format_count > 0:
                print_info(f"  💡 检测到 {old_format_count} 个旧格式任务（已映射为 train split）")
            
            if show_progress:
                print_success(f"✓ 成功构建任务映射表（{len(mapping)} 个文件）")
            
            return mapping
            
        except requests.exceptions.Timeout:
            if show_progress:
                print_warning("⚠ 获取任务列表超时（项目可能较大）")
            return {}
        except Exception as e:
            if show_progress:
                print_warning(f"⚠ 获取任务列表失败: {str(e)}")
            return {}
    
    def check_task_exists(self, filename: str, existing_map: Dict[str, int]) -> Optional[int]:
        """
        检查文件是否已上传到 Label Studio
        
        Args:
            filename: 原始文件名
            existing_map: 现有任务映射表（由 build_existing_tasks_map 生成）
            
        Returns:
            Optional[int]: 任务ID（如果存在），否则返回 None
        """
        return existing_map.get(filename)
    
    def retry_with_backoff(self, func: Callable, max_retries: int = 3, 
                          initial_delay: float = 1.0, filename: str = "") -> Tuple[bool, Any, str]:
        """
        使用指数退避策略重试函数
        
        Args:
            func: 要重试的函数
            max_retries: 最大重试次数
            initial_delay: 初始延迟（秒）
            filename: 文件名（用于日志）
            
        Returns:
            Tuple[bool, Any, str]: (成功/失败, 返回值, 错误信息)
            
        Raises:
            FatalUploadError: 遇到致命错误时立即抛出，不重试
        """
        last_error = ""
        
        for attempt in range(max_retries + 1):  # +1 因为包括第一次尝试
            try:
                result = func()
                return (True, result, "")
            except FatalUploadError:
                # 致命错误，立即停止，不重试
                raise
            except (requests.exceptions.ConnectionError, 
                   requests.exceptions.Timeout,
                   requests.exceptions.RequestException) as e:
                last_error = str(e)
                
                if attempt < max_retries:
                    # 计算延迟时间（指数退避）
                    delay = initial_delay * (2 ** attempt)
                    
                    # 显示重试信息
                    retry_msg = f"⚠ {filename}: 网络错误，{delay:.1f}秒后重试 ({attempt + 1}/{max_retries})"
                    print_warning(retry_msg)
                    
                    time.sleep(delay)
                else:
                    # 最后一次尝试失败
                    final_error = f"网络失败（已重试 {max_retries} 次）: {last_error}"
                    return (False, None, final_error)
            except Exception as e:
                # 非网络错误，不重试
                return (False, None, f"错误: {str(e)}")
        
        return (False, None, last_error)
    
    def load_dataset_config(self, dataset_path: Path) -> Dict:
        """加载数据集配置"""
        # 尝试多个可能的配置文件位置
        possible_configs = [
            dataset_path / 'dataset.yaml',
            dataset_path / 'data.yaml',
            dataset_path / 'config.yaml'
        ]
        
        for config_path in possible_configs:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.classes = config.get('names', [])
                    if isinstance(self.classes, dict):
                        self.classes = [self.classes[i] for i in sorted(self.classes.keys())]
                    
                    # 加载pose相关配置
                    if self.task_type == 'pose':
                        # 支持两种字段名（优先使用 kpt_names）
                        self.keypoint_names = config.get('kpt_names') or config.get('keypoint_names', [])
                        if not self.keypoint_names:
                            # 如果没有 kpt_names/keypoint_names，根据 kpt_shape 生成
                            kpt_shape = config.get('kpt_shape', [])
                            if kpt_shape and len(kpt_shape) > 0:
                                kpt_count = kpt_shape[0]
                                self.keypoint_names = [f'kp_{i+1}' for i in range(kpt_count)]
                    
                    print_success(f"✓ 加载配置文件: {config_path.name}")
                    print_info(f"  类别数: {len(self.classes)}")
                    if self.task_type == 'pose' and self.keypoint_names:
                        print_info(f"  关键点数: {len(self.keypoint_names)}")
                        print_info(f"  关键点: {', '.join(self.keypoint_names[:5])}{'...' if len(self.keypoint_names) > 5 else ''}")
                    return config
        
        raise FileNotFoundError(f"未找到数据集配置文件: {dataset_path}")
    
    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """获取图片尺寸"""
        with Image.open(image_path) as img:
            return img.size
    
    def _yolo_to_labelstudio_pose(self, yolo_annotation: List[float],
                                 img_width: int, img_height: int) -> Dict:
        """
        将YOLO Pose格式转换为Label Studio KeyPoint格式
        
        格式: [class_id, x_center, y_center, width, height, kp1_x, kp1_y, kp1_v, ...]
        """
        if len(yolo_annotation) < 5:
            return None
        
        class_id = int(yolo_annotation[0])
        x_center, y_center, width, height = yolo_annotation[1:5]
        
        # 转换bbox（作为辅助信息，不显示）
        x = (x_center - width / 2) * 100
        y = (y_center - height / 2) * 100
        w = width * 100
        h = height * 100
        
        # 解析关键点
        keypoints = []
        kpt_data = yolo_annotation[5:]  # 所有关键点数据
        
        if len(kpt_data) % 3 != 0:
            print_warning(f"警告：关键点数据不完整 ({len(kpt_data)} 值)")
            return None
        
        kpt_count = len(kpt_data) // 3
        
        visible_count = 0
        for i in range(kpt_count):
            kp_x = kpt_data[i * 3]
            kp_y = kpt_data[i * 3 + 1]
            kp_v = kpt_data[i * 3 + 2]  # visibility: 0=不可见, 1=遮挡, 2=可见
            
            # 只上传可见或遮挡的关键点
            if kp_v > 0:
                visible_count += 1
                kp_label = self.keypoint_names[i] if i < len(self.keypoint_names) else f'kp_{i+1}'
                keypoints.append({
                    "type": "keypointlabels",
                    "value": {
                        "x": float(kp_x * 100),  # 转换为百分比
                        "y": float(kp_y * 100),
                        "width": 0.5,  # 关键点显示大小
                        "keypointlabels": [kp_label]  # ✓ 正确：使用 keypointlabels
                    },
                    "to_name": "image",
                    "from_name": self.keypoint_from_name,  # 使用从项目配置获取的控件名称
                    "original_width": int(img_width),
                    "original_height": int(img_height),
                    "image_rotation": 0
                })
        
        return keypoints
    
    def _yolo_to_labelstudio_bbox(self, yolo_annotation: List[float], 
                                  img_width: int, img_height: int) -> Dict:
        """
        将YOLO格式转换为Label Studio格式
        
        支持：
        - 检测格式: [class_id, x_center, y_center, width, height]
        - 分割格式: [class_id, x1, y1, x2, y2, ...]
        """
        class_id = int(yolo_annotation[0])
        coords = yolo_annotation[1:]
        
        if len(coords) == 4:
            # 检测格式
            x_center, y_center, width, height = coords
            x = float((x_center - width / 2) * 100)
            y = float((y_center - height / 2) * 100)
            w = float(width * 100)
            h = float(height * 100)
            
            return {
                "original_width": int(img_width),
                "original_height": int(img_height),
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "rectanglelabels": [self.classes[class_id]]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels"
            }
        else:
            # 分割格式（多边形）
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append([float(coords[i] * 100), float(coords[i+1] * 100)])
            
            return {
                "original_width": int(img_width),
                "original_height": int(img_height),
                "image_rotation": 0,
                "value": {
                    "points": points,
                    "polygonlabels": [self.classes[class_id]]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels"
            }
    
    def _parse_yolo_label(self, label_path: Path) -> List[List[float]]:
        """解析YOLO标注文件"""
        annotations = []
        if not label_path.exists():
            return annotations
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    values = list(map(float, line.split()))
                    annotations.append(values)
        return annotations
    
    def _get_file_upload_detail(self, file_upload_id: int) -> Optional[Dict]:
        """
        查询文件上传详情，获取重命名后的文件路径
        
        参考: https://api.labelstud.io/api-reference/api-reference/files/
        GET /api/import/file-upload/{id}
        
        Args:
            file_upload_id: 文件上传ID
            
        Returns:
            文件详情，包含重命名后的文件路径
            例如: {'id': 41563, 'file': '/data/upload/19/04484f0f-image.jpg', ...}
        """
        detail_url = f"{self.url}/api/import/file-upload/{file_upload_id}"
        
        try:
            response = requests.get(
                detail_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception:
            return None
    
    def _upload_file_to_labelstudio(self, image_path: Path) -> Dict[str, str]:
        """
        上传文件到Label Studio服务器
        
        参考 Label Studio API 文档：
        https://api.labelstud.io/api-reference/api-reference/files/
        
        尝试多个可能的上传端点：
        1. POST /api/import/file-upload (标准文件上传 API)
        2. POST /api/projects/{project_id}/import?commit_to_project=false (项目导入 API)
        """
        
        # 方案1: 标准文件上传 API (推荐)
        # 参考: https://api.labelstud.io/api-reference/api-reference/files/create
        standard_upload_url = f"{self.url}/api/import/file-upload"
        
        # 方案2: 项目导入 API (基于您的网络抓包)
        project_import_url = f"{self.url}/api/projects/{self.project_id}/import"
        
        # 依次尝试两个端点
        endpoints = [
            (standard_upload_url, {}, "标准文件上传"),
            (project_import_url, {'commit_to_project': 'false'}, "项目导入（不提交）"),
        ]
        
        last_error = None
        
        for upload_url, params, endpoint_name in endpoints:
            try:
                with open(image_path, 'rb') as f:
                    files = {
                        'file': (image_path.name, f, 'image/jpeg')
                    }
                    headers = {
                        'Authorization': self.headers['Authorization']
                    }
                    
                    response = requests.post(
                        upload_url,
                        headers=headers,
                        files=files,
                        params=params if params else None,
                        timeout=60
                    )
                    
                    if response.status_code in [200, 201]:
                        result = response.json()
                        
                        # 处理不同的响应格式
                        file_info = result
                        if isinstance(result, list) and len(result) > 0:
                            file_info = result[0]
                        
                        # 提取文件ID（优先从顶层响应）
                        file_upload_ids = result.get('file_upload_ids', []) if isinstance(result, dict) else []
                        file_id = None
                        
                        if file_upload_ids and len(file_upload_ids) > 0:
                            # 从顶层响应的 file_upload_ids 获取
                            file_id = file_upload_ids[0]
                        elif isinstance(file_info, dict):
                            # 从 file_info 获取
                            file_id = (
                                file_info.get('id') or 
                                file_info.get('file_upload_id')
                            )
                        
                        # 提取文件URL
                        file_url = None
                        if isinstance(file_info, dict):
                            file_url = (
                                file_info.get('file') or
                                file_info.get('file_upload') or
                                file_info.get('data', {}).get('image') or
                                file_info.get('url') or
                                file_info.get('path') or
                                file_info.get('location')
                            )
                        
                        # 如果有文件URL，直接使用
                        if file_url:
                            # Label Studio 任务数据中应使用相对路径，不要添加域名
                            # 正确格式：/data/upload/19/uuid-filename.jpg
                            # 错误格式：http://domain.com/data/upload/...
                            
                            # 确保路径以 / 开头（相对路径）
                            if not file_url.startswith('/') and not file_url.startswith('http'):
                                file_url = f"/{file_url}"
                            
                            # 如果路径缺少 /data 前缀，添加它
                            if file_url.startswith('/upload/'):
                                file_url = f"/data{file_url}"
                            
                            return {
                                'url': file_url,
                                'original_name': image_path.name,
                                'original_path': str(image_path),
                                'file_id': file_id,
                                'endpoint': endpoint_name
                            }
                        
                        # 如果没有URL但有file_id，查询文件详情获取真实文件名
                        elif file_id:
                            # 查询文件上传详情以获取重命名后的文件名
                            try:
                                file_detail = self._get_file_upload_detail(file_id)
                                if file_detail and 'file' in file_detail:
                                    # 从详情中获取完整的文件路径
                                    # API可能返回: /upload/19/04484f0f-filename.jpg
                                    # 需要补全为: /data/upload/19/04484f0f-filename.jpg
                                    file_url = file_detail['file']
                                    
                                    # 确保使用相对路径（不添加域名）
                                    if not file_url.startswith('/') and not file_url.startswith('http'):
                                        file_url = f"/{file_url}"
                                    
                                    # 如果路径缺少 /data 前缀，添加它
                                    # 检测：/upload/19/xxx -> /data/upload/19/xxx
                                    if file_url.startswith('/upload/'):
                                        file_url = f"/data{file_url}"
                                    
                                    return {
                                        'url': file_url,  # /data/upload/19/xxx.jpg
                                        'original_name': image_path.name,
                                        'original_path': str(image_path),
                                        'file_id': file_id,
                                        'endpoint': endpoint_name,
                                        'renamed_file': file_detail.get('file', '')
                                    }
                            except Exception as e:
                                last_error = f"{endpoint_name}: 查询文件详情失败 - {str(e)}"
                                continue
                            
                            # 如果查询失败，使用默认格式
                            file_url = f"/data/upload/{self.project_id}/{file_id}"
                            
                            return {
                                'url': file_url,
                                'original_name': image_path.name,
                                'original_path': str(image_path),
                                'file_id': file_id,
                                'endpoint': endpoint_name
                            }
                        else:
                            last_error = f"{endpoint_name}: 响应中未找到文件URL或ID。响应: {result}"
                            continue
                    else:
                        last_error = f"{endpoint_name}: 状态码 {response.status_code} - {response.text[:300]}"
                        continue
                        
            except requests.exceptions.Timeout:
                last_error = f"{endpoint_name}: 超时"
                continue
            except requests.exceptions.ConnectionError:
                last_error = f"{endpoint_name}: 连接失败"
                continue
            except Exception as e:
                last_error = f"{endpoint_name}: 异常 - {str(e)}"
                continue
        
        # 所有端点都失败
        raise Exception(
            f"文件上传失败: {image_path.name}\n"
            f"最后错误: {last_error}\n"
            f"尝试了 {len(endpoints)} 个端点\n"
            f"Label Studio 版本: 1.21.0\n"
            f"建议: \n"
            f"  1. 确认 Label Studio 配置允许文件上传\n"
            f"  2. 检查存储配置（本地存储或云存储）\n"
            f"  3. 参考: https://api.labelstud.io/api-reference/api-reference/files/"
        )
    
    def _create_task(self, image_path: Path, label_path: Path, split_name: str = None) -> Dict:
        """创建Label Studio任务（文件上传模式）
        
        Args:
            image_path: 图片路径
            label_path: 标签路径
            split_name: 数据集分割名称（如 'train', 'val', 'test'）
        """
        # 获取图片尺寸
        img_width, img_height = self._get_image_dimensions(image_path)
        
        # 解析YOLO标注
        yolo_annotations = self._parse_yolo_label(label_path)
        
        # 转换为Label Studio格式
        predictions = []
        if self.task_type == 'pose':
            # Pose任务：同时转换边界框和关键点
            for anno in yolo_annotations:
                # 1. 添加边界框标注
                bbox = self._yolo_to_labelstudio_bbox(anno[:5], img_width, img_height)  # 只取前5个值（class_id + bbox）
                if bbox:
                    predictions.append(bbox)
                
                # 2. 添加关键点标注
                keypoints = self._yolo_to_labelstudio_pose(anno, img_width, img_height)
                if keypoints:
                    # pose返回的是关键点列表，需要展开
                    predictions.extend(keypoints)
        else:
            # 检测/分割任务：转换bbox/polygon
            for anno in yolo_annotations:
                bbox = self._yolo_to_labelstudio_bbox(anno, img_width, img_height)
                predictions.append(bbox)
        
        # 上传文件到Label Studio
        file_info = self._upload_file_to_labelstudio(image_path)
        
        # 获取文件URL并确保格式正确
        file_url = file_info['url']
        
        # 确保是相对路径，不包含域名
        if file_url.startswith('http://') or file_url.startswith('https://'):
            # 如果包含域名，提取路径部分
            from urllib.parse import urlparse
            parsed = urlparse(file_url)
            file_url = parsed.path
        
        # 确保路径包含 /data 前缀
        # 正确格式：/data/upload/19/uuid-filename.jpg
        if file_url.startswith('/upload/'):
            file_url = f"/data{file_url}"
        
        # 构建 original_filename（包含 split 信息以区分同名文件）
        original_filename = file_info['original_name']
        if split_name:
            original_filename = f"{split_name}/{original_filename}"
        
        # 创建任务数据
        task = {}
        task['data'] = {
            'image': file_url,  # 使用处理后的相对路径
            'original_filename': original_filename,  # 使用 split/filename 格式
            'split': split_name,  # 额外保存 split 信息
            'image_width': img_width,
            'image_height': img_height
        }
        
        # 添加预标注结果
        if predictions:
            task['predictions'] = [{
                'result': predictions,
                'score': 1.0,
                'model_version': 'yolocli_import'
            }]
        
        # 添加元数据
        task['meta'] = {
            'original_image': str(image_path),
            'original_label': str(label_path),
            'annotation_count': len(yolo_annotations),
            'image_dimensions': f'{img_width}x{img_height}'
        }
        
        return task
    
    def upload_tasks(self, dataset_path: Path, 
                    splits: List[str] = ['train', 'val', 'test'],
                    max_images: Optional[int] = None,
                    max_workers: int = 4,
                    force: bool = False,
                    no_resume: bool = False,
                    skip_server_check: bool = False,
                    retry_times: int = 3) -> Tuple[int, int]:
        """
        批量上传任务到Label Studio（支持并发和断点续传）
        
        Args:
            dataset_path: 数据集路径
            splits: 要上传的数据集分割
            max_images: 最大上传图片数（None表示全部）
            max_workers: 最大并发数（默认4）
            force: 强制重新上传所有文件（忽略所有检查）
            no_resume: 禁用断点续传（清除进度记录）
            skip_server_check: 跳过服务器重复检测（仅使用本地缓存）
            retry_times: 重试次数（默认3）
            
        Returns:
            (成功数, 失败数)
        """
        from ..core.upload_progress import UploadProgressTracker
        
        total_uploaded = 0
        total_failed = 0
        total_skipped = 0
        
        # 初始化进度跟踪器（除非 force 模式）
        progress_tracker = None
        if not force:
            progress_tracker = UploadProgressTracker(
                project_id=self.project_id,
                dataset_path=dataset_path,
                url=self.url
            )
            
            # 如果 no_resume，清除进度
            if no_resume:
                print_info("🔄 已清除本地进度缓存")
                progress_tracker.clear()
            
            # 显示进度信息
            progress_info = progress_tracker.get_progress_info()
            if progress_info:
                print_info("\n📌 检测到之前的上传进度：")
                print_info(f"  • 数据集：{progress_info['dataset_name']}")
                print_info(f"  • 项目ID：{progress_info['project_id']}")
                print_info(f"  • 已上传：{progress_info['uploaded_count']} 个文件")
                print_info(f"  • 失败：{progress_info['failed_count']} 个文件")
                print_info(f"  • 最后更新：{progress_info['last_updated']}")
                print_info("  ✨ 将自动跳过已上传的文件")
        
        # 构建服务器现有任务映射（用于重复检测）
        existing_map = {}
        if not force and not skip_server_check:
            # 智能判断是否检查服务器
            task_count = self.get_project_task_count()
            
            if task_count is not None and task_count >= 5000:
                print_warning(f"\n⚠ 项目任务数较大（{task_count} 个），服务器检查可能较慢")
                print_info("  💡 提示：可使用 --skip-server-check 跳过服务器检查，仅使用本地缓存")
                print_info("  正在检查服务器...")
            
            existing_map = self.build_existing_tasks_map(show_progress=True)
        elif skip_server_check:
            print_info("\n⚡ 跳过服务器重复检测（仅使用本地缓存）")
        
        try:
            for split in splits:
                print_info(f"\n处理 {split} 数据集...")
                
                # 尝试不同的目录结构
                possible_dirs = [
                    (dataset_path / split / 'images', dataset_path / split / 'labels'),
                    (dataset_path / 'images' / split, dataset_path / 'labels' / split),
                ]
                
                images_dir, labels_dir = None, None
                for img_dir, lbl_dir in possible_dirs:
                    if img_dir.exists():
                        images_dir = img_dir
                        labels_dir = lbl_dir
                        break
                
                if not images_dir or not images_dir.exists():
                    print_warning(f"跳过: 未找到 {split} 图片目录")
                    continue
                
                # 获取所有图片
                image_files = sorted(
                    list(images_dir.glob('*.jpg')) + 
                    list(images_dir.glob('*.png')) +
                    list(images_dir.glob('*.jpeg'))
                )
                
                if max_images:
                    image_files = image_files[:max_images]
                
                print_info(f"找到 {len(image_files)} 张图片")
                
                # 过滤掉已上传的文件
                if progress_tracker and not force:
                    original_count = len(image_files)
                    
                    # 过滤本地缓存中的文件（使用 split/filename 作为键）
                    uploaded_set = progress_tracker.get_uploaded_set()
                    image_files = [f for f in image_files if f"{split}/{f.name}" not in uploaded_set]
                    cached_count = original_count - len(image_files)
                    
                    # 过滤服务器上已存在的文件（使用 split/filename 作为键）
                    server_exists_count = 0
                    if existing_map:
                        remaining_files = []
                        for f in image_files:
                            # 使用 split/filename 格式查找
                            split_filename = f"{split}/{f.name}"
                            if split_filename not in existing_map:
                                remaining_files.append(f)
                            else:
                                server_exists_count += 1
                        image_files = remaining_files
                    
                    # 显示统计信息
                    print_info(f"\n📊 统计信息：")
                    print_info(f"  • 总文件数：{original_count}")
                    if cached_count > 0:
                        print_success(f"  • 本地缓存已记录：{cached_count} 个（跳过）")
                    if server_exists_count > 0:
                        print_success(f"  • 服务器已存在：{server_exists_count} 个（跳过）")
                    print_info(f"  • 待上传：{len(image_files)} 个")
                    
                    total_skipped += cached_count + server_exists_count
                
                if len(image_files) == 0:
                    print_success(f"✓ {split} 数据集所有文件已上传")
                    continue
                
                print_info(f"并发数: {max_workers}")
                print_info(f"重试次数: {retry_times}")
                
                # 并发上传
                uploaded, failed = self._upload_images_concurrent(
                    image_files=image_files,
                    labels_dir=labels_dir,
                    split_name=split,
                    max_workers=max_workers,
                    progress_tracker=progress_tracker,
                    retry_times=retry_times
                )
                
                total_uploaded += uploaded
                total_failed += failed
        
        except FatalUploadError as e:
            # 致命错误：已经在 _upload_images_concurrent 中处理和保存进度了
            # 这里只需要重新抛出，让上层处理
            raise
        
        except KeyboardInterrupt:
            # 用户中断，保存进度并重新抛出
            if progress_tracker:
                print_warning("\n\n⚠️  检测到中断信号...")
                # 强制保存当前进度
                try:
                    progress_tracker.force_save()
                except:
                    pass
                stats = progress_tracker.get_stats()
                if stats['uploaded_count'] > 0:
                    print_info(f"💾 已保存进度到: {stats['progress_file']}")
            raise
        
        # 显示失败文件列表
        if progress_tracker:
            # 确保最终进度被保存
            try:
                progress_tracker.force_save()
            except:
                pass
            
            failed_files = progress_tracker.get_failed_files()
            if failed_files:
                print_warning(f"\n⚠ 失败文件列表（共 {len(failed_files)} 个）：")
                for i, failed in enumerate(failed_files[:10], 1):  # 只显示前10个
                    print_warning(f"  {i}. {failed['filename']}: {failed['error']}")
                if len(failed_files) > 10:
                    print_warning(f"  ... 还有 {len(failed_files) - 10} 个文件")
            
            # 显示进度文件位置
            stats = progress_tracker.get_stats()
            if stats['uploaded_count'] > 0:
                print_info(f"\n💾 进度已保存到: {stats['progress_file']}")
        
        return total_uploaded, total_failed
    
    def _upload_images_concurrent(self, image_files: List[Path], labels_dir: Path, 
                                  split_name: str, max_workers: int,
                                  progress_tracker=None, retry_times: int = 3) -> Tuple[int, int]:
        """
        并发上传图片（支持断点续传和重试）
        
        Args:
            image_files: 图片文件列表
            labels_dir: 标签目录
            split_name: 数据集分割名称
            max_workers: 最大并发数
            progress_tracker: 进度跟踪器（可选）
            retry_times: 重试次数（默认3）
            
        Returns:
            (成功数, 失败数)
        """
        uploaded = 0
        failed = 0
        lock = threading.Lock()
        
        def upload_single_image(idx: int, image_path: Path) -> Tuple[bool, str, Optional[int]]:
            """上传单张图片（带重试）"""
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            def _do_upload():
                """实际上传函数（用于重试）"""
                task = self._create_task(image_path, label_path, split_name=split_name)
                success = self._upload_batch([task])
                
                if not success:
                    raise Exception("上传失败")
                
                return task
            
            # 使用重试机制
            success, task, error = self.retry_with_backoff(
                func=_do_upload,
                max_retries=retry_times,
                initial_delay=1.0,
                filename=image_path.name
            )
            
            if success and task:
                anno_count = task['meta']['annotation_count']
                original_name = task['data'].get('original_filename', image_path.name)
                
                # 获取任务ID（如果有的话，从响应中提取）
                task_id = task.get('id')
                
                return True, f"[{idx}/{len(image_files)}] {original_name} ({anno_count} 个标注)", task_id
            else:
                return False, f"{image_path.name}: {error}", None
        
        # 使用线程池并发上传
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        try:
            # 提交所有任务
            futures = {
                executor.submit(upload_single_image, idx, img): (idx, img)
                for idx, img in enumerate(image_files, 1)
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                idx, image_path = futures[future]
                
                try:
                    success, message, task_id = future.result()
                    
                    if success:
                        # 更新进度跟踪器（立即保存，确保一致性）
                        # 这是唯一的真实来源
                        if progress_tracker:
                            # 使用 split/filename 格式作为唯一标识
                            file_key = f"{split_name}/{image_path.name}"
                            progress_tracker.mark_uploaded(
                                filename=file_key,
                                task_id=task_id or 0,
                                auto_save=True  # 立即保存，确保进度一致性
                            )
                        
                        # 获取当前实际上传数（从 progress_tracker）
                        with lock:
                            uploaded += 1
                            total_uploaded = len(progress_tracker.uploaded_files) if progress_tracker else uploaded
                        
                        # 显示进度（每10个或最后一个）
                        if uploaded % 10 == 0 or uploaded == len(image_files) or uploaded <= 3:
                            print_info(f"  {message}")
                        
                        # 每10个显示统计（显示本次进度 + 累计总数）
                        if uploaded % 10 == 0 or uploaded == len(image_files):
                            print_success(f"✓ 进度: 本次 {uploaded}/{len(image_files)}, 累计 {total_uploaded} 个任务 ({split_name})")
                    else:
                        with lock:
                            failed += 1
                            
                            # 标记为失败（立即保存失败信息）
                            if progress_tracker:
                                # 使用 split/filename 格式作为唯一标识
                                file_key = f"{split_name}/{image_path.name}"
                                progress_tracker.mark_failed(
                                    filename=file_key,
                                    error=message
                                )
                            
                            print_error(f"✗ {message}")
                
                except FatalUploadError as e:
                    # 致命错误：立即停止所有上传
                    print_error(f"\n\n❌ 致命错误：{str(e)}")
                    
                    if e.status_code == 401:
                        print_error("\n🔑 认证失败，可能的原因：")
                        print_error("  1. API Token 已过期")
                        print_error("  2. API Token 无效或被撤销")
                        print_error("  3. Label Studio 服务器要求重新登录")
                        print_error("\n💡 解决方法：")
                        print_error("  1. 重新登录 Label Studio 获取新的 Token")
                        print_error("  2. 更新配置文件中的 api_token")
                        print_error("  3. 使用交互式命令重新配置连接")
                    elif e.status_code == 403:
                        print_error("\n🚫 权限不足，可能的原因：")
                        print_error("  1. 当前账号没有项目的上传权限")
                        print_error("  2. 项目已被锁定或归档")
                        print_error("\n💡 解决方法：")
                        print_error("  1. 联系项目管理员授予上传权限")
                        print_error("  2. 确认项目处于活动状态")
                    
                    # 取消所有未完成的任务
                    print_warning("\n⚠️  正在停止所有上传任务...")
                    cancelled_count = 0
                    for f in futures:
                        if f != future and f.cancel():
                            cancelled_count += 1
                    
                    if cancelled_count > 0:
                        print_info(f"  已取消 {cancelled_count} 个未开始的任务")
                    
                    # 保存当前进度
                    if progress_tracker:
                        try:
                            progress_tracker.force_save()
                            stats = progress_tracker.get_stats()
                            print_info(f"\n💾 已保存进度到: {stats['progress_file']}")
                            print_info(f"  已上传: {stats['uploaded_count']} 个文件")
                            print_info(f"  失败: {stats['failed_count']} 个文件")
                        except Exception as save_error:
                            print_warning(f"⚠️  保存进度失败: {save_error}")
                    
                    # 关闭线程池（Python 3.8 兼容）
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        # Python < 3.9 不支持 cancel_futures 参数
                        executor.shutdown(wait=False)
                    
                    # 重新抛出异常
                    raise
                
                except Exception as e:
                    with lock:
                        failed += 1
                        error_msg = f"任务异常: {str(e)}"
                        
                        # 标记为失败（立即保存失败信息）
                        if progress_tracker:
                            # 使用 split/filename 格式作为唯一标识
                            file_key = f"{split_name}/{image_path.name}"
                            progress_tracker.mark_failed(
                                filename=file_key,
                                error=error_msg
                            )
                        
                        print_error(f"✗ {error_msg}")
            
            # 循环结束，不需要强制保存（因为已经实时保存了）
        
        except KeyboardInterrupt:
            # 用户中断，优雅地关闭
            print_warning("\n\n⚠️  检测到中断信号，正在停止上传...")
            
            # 取消所有未开始的任务
            cancelled_count = 0
            running_futures = []
            completed_futures = []
            
            for future in futures:
                if future.cancel():  # 只能取消未开始的任务
                    cancelled_count += 1
                elif future.done():
                    # 已完成但可能还没被主循环处理
                    completed_futures.append(future)
                else:
                    # 正在运行的任务
                    running_futures.append(future)
            
            if cancelled_count > 0:
                print_info(f"  已取消 {cancelled_count} 个未开始的任务")
            
            # 处理那些已完成但还没被主循环处理的任务
            if completed_futures:
                print_info(f"  正在保存 {len(completed_futures)} 个已完成任务的进度...")
                for future in completed_futures:
                    try:
                        idx, image_path = futures[future]
                        success, message, task_id = future.result()
                        
                        if success and progress_tracker and not progress_tracker.is_uploaded(image_path.name):
                            # 只有当文件还没被标记为上传时才保存（避免重复）
                            progress_tracker.mark_uploaded(
                                filename=image_path.name,
                                task_id=task_id or 0,
                                auto_save=True
                            )
                        elif not success and progress_tracker:
                            # 使用 split/filename 格式作为唯一标识
                            file_key = f"{split_name}/{image_path.name}"
                            progress_tracker.mark_failed(
                                filename=file_key,
                                error=message
                            )
                    except Exception as e:
                        # 处理结果时出错，忽略
                        pass
            
            # 等待正在执行的任务完成
            if running_futures:
                print_info(f"  等待 {len(running_futures)} 个正在执行的任务完成...")
                
                try:
                    done, not_done = wait(running_futures, timeout=30)
                    
                    # 处理刚刚完成的任务
                    if done:
                        for future in done:
                            try:
                                idx, image_path = futures[future]
                                success, message, task_id = future.result()
                                
                                if success and progress_tracker:
                                    progress_tracker.mark_uploaded(
                                        filename=image_path.name,
                                        task_id=task_id or 0,
                                        auto_save=True
                                    )
                                elif not success and progress_tracker:
                                    # 使用 split/filename 格式作为唯一标识
                                    file_key = f"{split_name}/{image_path.name}"
                                    progress_tracker.mark_failed(
                                        filename=file_key,
                                        error=message
                                    )
                            except Exception as e:
                                # 处理结果时出错，忽略
                                pass
                    
                    if not_done:
                        print_warning(f"  ⚠ 有 {len(not_done)} 个任务超时未完成")
                except Exception as e:
                    print_warning(f"  ⚠ 等待任务时出错: {str(e)}")
            
            # 关闭线程池
            executor.shutdown(wait=False)
            
            # 获取实际保存的进度（从 progress_tracker，这是唯一的真实来源）
            actual_uploaded = len(progress_tracker.uploaded_files) if progress_tracker else uploaded
            actual_failed = len(progress_tracker.failed_files) if progress_tracker else failed
            
            print_info(f"✓ 已保存进度: {actual_uploaded} 个成功, {actual_failed} 个失败")
            print_info("💡 下次运行时将自动从断点继续")
            
            # 重新抛出异常，让上层处理
            raise
        
        finally:
            # 确保线程池被关闭
            try:
                executor.shutdown(wait=False)
            except:
                pass
        
        # 返回实际保存的数量（从 progress_tracker，这是唯一的真实来源）
        actual_uploaded = len(progress_tracker.uploaded_files) if progress_tracker else uploaded
        actual_failed = len(progress_tracker.failed_files) if progress_tracker else failed
        return actual_uploaded, actual_failed
    
    def _upload_batch(self, tasks: List[Dict]) -> bool:
        """
        批量上传任务到Label Studio
        
        Raises:
            FatalUploadError: 遇到致命错误（如401认证失败）时抛出
        """
        url = f"{self.url}/api/projects/{self.project_id}/import"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=tasks,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                return True
            
            # 检查是否为致命错误
            elif response.status_code in [401, 403]:
                # 认证失败或权限不足，立即停止
                error_msg = f"认证失败 (HTTP {response.status_code})"
                try:
                    error_data = response.json()
                    if 'detail' in error_data:
                        error_msg += f": {error_data['detail']}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                raise FatalUploadError(
                    message=error_msg,
                    status_code=response.status_code,
                    response_text=response.text
                )
            else:
                # 其他错误，可重试
                print_error(f"错误: {response.status_code} - {response.text[:200]}")
                return False
                
        except FatalUploadError:
            # 重新抛出致命错误
            raise
        except Exception as e:
            # 网络错误等，可重试
            print_error(f"上传异常: {str(e)}")
            return False
    
    def setup_project_config(self, task_type: str = None) -> bool:
        """配置Label Studio项目（设置标注界面）"""
        if not self.classes:
            print_error("未加载类别信息，请先加载数据集配置")
            return False
        
        task = task_type or self.task_type
        
        # 根据任务类型生成不同的标注配置
        if task == 'pose':
            # Pose任务：关键点标注
            if not self.keypoint_names:
                print_error("未加载关键点信息，请确保dataset.yaml包含 kpt_names 或 keypoint_names")
                return False
            
            keypoint_labels = '\n    '.join([
                f'<Label value="{kp}" background="#{self._get_color(i)}"/>'
                for i, kp in enumerate(self.keypoint_names)
            ])
            
            # Pose任务：包含矩形框（用于框住整个对象）和关键点
            labeling_config = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="3" opacity="0.9">
    {self._generate_label_tags()}
  </RectangleLabels>
  <KeyPointLabels name="keypoint" toName="image" opacity="0.9">
    {keypoint_labels}
  </KeyPointLabels>
</View>"""
        elif task == 'segment':
            # 分割任务：只需要多边形来标注物体轮廓
            labeling_config = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
    {self._generate_label_tags()}
  </PolygonLabels>
</View>"""
        elif task in ['classify', 'classification', 'cls']:
            # 分类任务：只需要选择图片的类别
            choice_options = '\n    '.join([
                f'<Choice value="{cls}"/>'
                for cls in self.classes
            ])
            
            labeling_config = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <Choices name="choice" toName="image" choice="single">
    {choice_options}
  </Choices>
</View>"""
        else:
            # 检测任务（detect）或其他：只需要矩形框
            labeling_config = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="3" opacity="0.9">
    {self._generate_label_tags()}
  </RectangleLabels>
</View>"""
        
        # 验证配置，避免重复的标签控件
        labeling_config = self._validate_and_cleanup_config(labeling_config)
        
        url = f"{self.url}/api/projects/{self.project_id}"
        
        try:
            response = requests.patch(
                url,
                headers=self.headers,
                json={'label_config': labeling_config},
                timeout=10
            )
            
            if response.status_code == 200:
                print_success("✓ 项目配置更新成功")
                return True
            else:
                print_error(f"✗ 更新配置失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print_error(f"✗ 配置异常: {str(e)}")
            return False
    
    def _validate_and_cleanup_config(self, config: str) -> str:
        """
        验证并清理配置，移除重复的标签控件
        
        Args:
            config: Label Studio配置XML字符串
        
        Returns:
            清理后的配置字符串
        """
        try:
            # 解析XML
            root = ET.fromstring(config)
            
            # 检查并移除重复的标签控件
            seen_controls = {}  # {(tag_name, name_attr): element}
            elements_to_remove = []
            duplicates = []
            
            for elem in root.findall('.//*[@name][@toName]'):
                tag_name = elem.tag
                name_attr = elem.get('name')
                control_key = (tag_name, name_attr)
                
                if control_key in seen_controls:
                    # 发现重复的控件
                    duplicates.append(f"<{tag_name} name=\"{name_attr}\">")
                    elements_to_remove.append(elem)
                else:
                    seen_controls[control_key] = elem
            
            # 如果发现重复，显示警告并移除
            if elements_to_remove:
                print_warning(f"⚠ 检测到 {len(elements_to_remove)} 个重复的标签控件，已自动移除")
                
                for elem in elements_to_remove:
                    # 找到父元素并移除
                    for parent in root.iter():
                        if elem in list(parent):
                            parent.remove(elem)
                            break
            
            # 将清理后的XML转换回字符串
            cleaned_config = ET.tostring(root, encoding='unicode')
            return cleaned_config
            
        except ET.ParseError as e:
            print_warning(f"⚠ 无法解析配置XML: {e}，将使用原始配置")
            return config
    
    def _generate_label_tags(self) -> str:
        """生成Label标签"""
        labels = '\n    '.join([
            f'<Label value="{cls}" background="#{self._get_color(i)}"/>'
            for i, cls in enumerate(self.classes)
        ])
        return labels
    
    @staticmethod
    def _get_color(index: int) -> str:
        """为每个类别生成颜色"""
        colors = [
            "FF6B6B", "4ECDC4", "45B7D1", "FFA07A", "98D8C8",
            "F7DC6F", "BB8FCE", "85C1E2", "F8B739", "52B788",
            "E76F51", "2A9D8F", "E9C46A", "F4A261", "264653",
            "E63946", "F1FAEE", "A8DADC", "457B9D", "1D3557"
        ]
        return colors[index % len(colors)]
    
    @staticmethod
    def normalize_to_labelstudio_bbox(
        center_x: float,
        center_y: float,
        width: float,
        height: float
    ) -> Dict[str, float]:
        """
        将归一化的中心点+宽高转换为Label Studio格式（左上角+宽高，百分比）
        
        Args:
            center_x: 中心点X坐标 (0-1)
            center_y: 中心点Y坐标 (0-1)
            width: 宽度 (0-1)
            height: 高度 (0-1)
        
        Returns:
            {'x': 35.0, 'y': 40.0, 'width': 30.0, 'height': 20.0}  # 百分比
        """
        x = (center_x - width / 2) * 100
        y = (center_y - height / 2) * 100
        w = width * 100
        h = height * 100
        
        return {
            'x': float(x),
            'y': float(y),
            'width': float(w),
            'height': float(h)
        }
    
    @staticmethod
    def normalize_to_labelstudio_keypoint(x: float, y: float) -> Dict[str, float]:
        """
        将归一化的关键点坐标转换为Label Studio格式（百分比）
        
        Args:
            x: X坐标 (0-1)
            y: Y坐标 (0-1)
        
        Returns:
            {'x': 50.0, 'y': 50.0}  # 百分比
        """
        return {
            'x': float(x * 100),
            'y': float(y * 100)
        }
    
    def parse_labeling_config(self) -> Dict[str, Any]:
        """
        解析项目的XML标签配置，提取可用的标注控件
        
        Returns:
            {
                'rectanglelabels': {
                    'from_name': 'label',
                    'to_name': 'image',
                    'labels': ['circle_meter', 'gauge']
                },
                'keypointlabels': {
                    'from_name': 'keypoint',
                    'to_name': 'image',
                    'labels': ['start', 'end', 'center', 'pointer']
                }
            }
        """
        print_info("正在解析项目标签配置...")
        
        # 获取项目配置
        url = f"{self.url}/api/projects/{self.project_id}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                print_error(f"✗ 获取项目配置失败: {response.status_code}")
                return {}
            
            project = response.json()
            label_config_xml = project.get('label_config', '')
            
            if not label_config_xml:
                print_warning("项目没有标签配置")
                return {}
            
            # 解析XML
            try:
                root = ET.fromstring(label_config_xml)
            except ET.ParseError as e:
                print_error(f"✗ XML解析失败: {str(e)}")
                return {}
            
            config = {}
            
            # 查找RectangleLabels控件
            for rect_elem in root.findall('.//RectangleLabels'):
                from_name = rect_elem.get('name', 'label')
                to_name = rect_elem.get('toName', 'image')
                labels = [label.get('value') for label in rect_elem.findall('.//Label') if label.get('value')]
                
                if labels:
                    config['rectanglelabels'] = {
                        'from_name': from_name,
                        'to_name': to_name,
                        'labels': labels
                    }
                    print_info(f"  ✓ 找到矩形框标注: {len(labels)} 个类别")
            
            # 查找KeyPointLabels控件
            for kp_elem in root.findall('.//KeyPointLabels'):
                from_name = kp_elem.get('name', 'keypoint')
                to_name = kp_elem.get('toName', 'image')
                labels = [label.get('value') for label in kp_elem.findall('.//Label') if label.get('value')]
                
                if labels:
                    config['keypointlabels'] = {
                        'from_name': from_name,
                        'to_name': to_name,
                        'labels': labels
                    }
                    print_info(f"  ✓ 找到关键点标注: {len(labels)} 个关键点")
            
            if not config:
                print_warning("未找到支持的标注控件（RectangleLabels或KeyPointLabels）")
            
            return config
            
        except Exception as e:
            print_error(f"✗ 解析配置异常: {str(e)}")
            return {}
    
    def create_annotation(
        self,
        task_id: int,
        annotation_data: Dict[str, Any],
        merge_mode: str = 'add'
    ) -> Tuple[bool, Optional[str]]:
        """
        为task创建或更新annotation
        
        Args:
            task_id: 任务ID
            annotation_data: 标注数据（result格式）
            merge_mode: 合并模式 ('add', 'skip', 'overwrite_same_type')
        
        Returns:
            (是否成功, 错误信息)
        """
        # 1. 获取task及其现有annotations
        task_url = f"{self.url}/api/tasks/{task_id}/"
        
        try:
            response = requests.get(
                task_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                return False, f"获取task失败: {response.status_code}"
            
            task = response.json()
            existing_annotations = task.get('annotations', [])
            
            # 2. 根据merge_mode决定操作
            if merge_mode == 'skip' and len(existing_annotations) > 0:
                return False, "跳过（已有标注）"
            
            # 获取现有的result数组
            existing_results = []
            annotation_id = None
            if existing_annotations:
                annotation_id = existing_annotations[0]['id']
                existing_results = existing_annotations[0].get('result', [])
            
            # 3. 合并result
            new_result = annotation_data
            new_type = new_result.get('type')
            
            if merge_mode == 'add':
                # 追加：保留所有已有标注，添加新标注
                merged_results = existing_results + [new_result]
            elif merge_mode == 'overwrite_same_type':
                # 只删除同类型，保留其他类型
                filtered = [r for r in existing_results if r.get('type') != new_type]
                merged_results = filtered + [new_result]
            else:  # skip模式已在前面处理
                merged_results = existing_results + [new_result]
            
            # 4. 更新或创建annotation
            if annotation_id:
                # 更新现有annotation
                anno_url = f"{self.url}/api/annotations/{annotation_id}/"
                response = requests.patch(
                    anno_url,
                    headers=self.headers,
                    json={'result': merged_results},
                    timeout=10
                )
            else:
                # 创建新annotation
                anno_url = f"{self.url}/api/tasks/{task_id}/annotations/"
                response = requests.post(
                    anno_url,
                    headers=self.headers,
                    json={'result': [new_result]},
                    timeout=10
                )
            
            if response.status_code in [200, 201]:
                return True, None
            else:
                return False, f"API错误 ({response.status_code}): {response.text[:100]}"
                
        except Exception as e:
            return False, f"异常: {str(e)}"
    
    def batch_annotate_tasks(
        self,
        annotation_data: Dict[str, Any],
        target_type: str,
        task_filter: Dict[str, Any],
        merge_mode: str = 'add',
        dry_run: bool = False,
        max_workers: int = 4
    ) -> Dict[str, int]:
        """
        批量给tasks添加标注
        
        Args:
            annotation_data: 标注数据（result格式）
            target_type: 'annotation' 或 'prediction'
            task_filter: 筛选条件 {'mode': 'all'|'ids'|'range'|'unlabeled', ...}
            merge_mode: 合并模式 ('add', 'skip', 'overwrite_same_type')
            dry_run: 试运行模式
            max_workers: 并发数
        
        Returns:
            {'total': N, 'success': N, 'failed': N, 'skipped': N}
        """
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        
        print_info("\n开始批量打标签...")
        
        # 1. 根据filter获取tasks
        tasks = self._get_tasks_by_filter(task_filter)
        
        if not tasks:
            print_warning("未找到符合条件的tasks")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        print_info(f"找到 {len(tasks)} 个任务")
        
        if dry_run:
            print_info("【试运行模式】不会实际创建标注")
            return {'total': len(tasks), 'success': 0, 'failed': 0, 'skipped': len(tasks)}
        
        # 2. 批量处理
        stats = {'total': len(tasks), 'success': 0, 'failed': 0, 'skipped': 0}
        failed_details = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            task_progress = progress.add_task("处理进度", total=len(tasks))
            
            if target_type == 'annotation':
                # 使用annotation API
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.create_annotation,
                            task['id'],
                            annotation_data,
                            merge_mode
                        ): task['id']
                        for task in tasks
                    }
                    
                    for future in as_completed(futures):
                        task_id = futures[future]
                        success, error = future.result()
                        
                        if success:
                            stats['success'] += 1
                        elif error and "跳过" in error:
                            stats['skipped'] += 1
                        else:
                            stats['failed'] += 1
                            if error:
                                failed_details.append((task_id, error))
                        
                        progress.update(task_progress, advance=1)
            else:
                # 使用prediction API
                def create_prediction_for_task(task_id, annotation_data):
                    """为单个任务创建prediction"""
                    try:
                        # 确保annotation_data是列表格式（支持单个dict或list）
                        result_list = [annotation_data] if isinstance(annotation_data, dict) else annotation_data
                        
                        # 创建prediction数据
                        prediction_data = {
                            "task": task_id,
                            "result": result_list,
                            "score": 0.9,  # 批量打标签时使用默认分数
                            "model_version": "yolocli_batch_annotate"
                        }
                        
                        # 发送POST请求创建prediction
                        create_url = f"{self.url}/api/predictions/"
                        response = requests.post(
                            create_url,
                            headers=self.headers,
                            json=prediction_data,
                            timeout=10
                        )
                        
                        if response.status_code in [200, 201]:
                            return True, None
                        else:
                            error_msg = f"HTTP {response.status_code}"
                            try:
                                error_detail = response.json()
                                error_msg += f": {error_detail}"
                            except:
                                error_msg += f": {response.text[:200]}"
                            return False, error_msg
                    except Exception as e:
                        return False, f"创建prediction失败: {str(e)}"
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            create_prediction_for_task,
                            task['id'],
                            annotation_data
                        ): task['id']
                        for task in tasks
                    }
                    
                    for future in as_completed(futures):
                        task_id = futures[future]
                        success, error = future.result()
                        
                        if success:
                            stats['success'] += 1
                        else:
                            stats['failed'] += 1
                            if error:
                                failed_details.append((task_id, error))
                        
                        progress.update(task_progress, advance=1)
        
        # 3. 显示统计
        print()
        print_success(f"✓ 成功: {stats['success']} 个任务")
        if stats['failed'] > 0:
            print_error(f"✗ 失败: {stats['failed']} 个任务")
            if failed_details[:3]:  # 显示前3个失败详情
                print_error("\n失败详情（前3个）:")
                for task_id, error in failed_details[:3]:
                    print_error(f"  Task #{task_id}: {error}")
        if stats['skipped'] > 0:
            print_info(f"ℹ 跳过: {stats['skipped']} 个任务")
        
        return stats
    
    def _get_tasks_by_filter(self, task_filter: Dict[str, Any]) -> List[Dict]:
        """根据筛选条件获取tasks"""
        mode = task_filter.get('mode', 'all')
        
        if mode == 'ids':
            task_ids = task_filter.get('task_ids', [])
            return self._get_tasks_by_ids(task_ids)
        elif mode == 'range':
            task_range = task_filter.get('task_range')
            if task_range and len(task_range) == 2:
                start_id, end_id = task_range
                return self._get_tasks_by_range(start_id, end_id)
            else:
                print_warning(f"⚠ task_range格式错误: {task_range}")
                print_warning(f"⚠ task_filter内容: {task_filter}")
                return []
        elif mode == 'unlabeled':
            return self._get_unlabeled_tasks()
        else:  # 'all'
            return self._get_all_project_tasks()
    
    def _get_tasks_by_ids(self, task_ids: List[int]) -> List[Dict]:
        """根据ID列表获取tasks（优化版：批量获取后过滤）"""
        if not task_ids:
            return []
        
        print_info(f"正在批量获取任务（{len(task_ids)} 个ID）...")
        
        # 使用Label Studio的export API获取所有任务（更快）
        try:
            export_url = f"{self.url}/api/projects/{self.project_id}/export"
            response = requests.get(
                export_url,
                headers=self.headers,
                params={'exportType': 'JSON'},
                timeout=60
            )
            
            if response.status_code == 200:
                all_tasks = response.json()
                task_id_set = set(task_ids)
                tasks = [task for task in all_tasks if task.get('id') in task_id_set]
                
                print_success(f"✓ 找到 {len(tasks)}/{len(task_ids)} 个任务")
                return tasks
        except Exception as e:
            print_warning(f"使用export API失败，切换到逐个获取: {str(e)}")
        
        # 降级方案：逐个获取
        tasks = []
        for task_id in task_ids:
            try:
                url = f"{self.url}/api/tasks/{task_id}"
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    tasks.append(response.json())
            except Exception:
                continue
        
        print_success(f"✓ 找到 {len(tasks)}/{len(task_ids)} 个任务")
        return tasks
    
    def _get_tasks_by_range(self, start_id: int, end_id: int) -> List[Dict]:
        """根据ID范围获取tasks（优化版：批量获取后过滤）"""
        url = f"{self.url}/api/projects/{self.project_id}/tasks"
        tasks = []
        page = 1
        
        print_info(f"正在批量获取任务（ID范围: {start_id}-{end_id}）...")
        
        # 使用Label Studio的export API获取所有任务（更快）
        try:
            export_url = f"{self.url}/api/projects/{self.project_id}/export"
            print_info(f"尝试使用 export API: {export_url}")
            response = requests.get(
                export_url,
                headers=self.headers,
                params={'exportType': 'JSON'},
                timeout=60
            )
            
            print_info(f"Export API 响应状态: {response.status_code}")
            
            if response.status_code == 200:
                all_tasks = response.json()
                print_info(f"Export API 返回了 {len(all_tasks)} 个任务")
                
                # 过滤ID范围内的tasks
                for task in all_tasks:
                    task_id = task.get('id')
                    if task_id and start_id <= task_id <= end_id:
                        tasks.append(task)
                
                print_success(f"✓ 找到 {len(tasks)} 个任务（ID范围: {start_id}-{end_id}）")
                return tasks
            else:
                print_warning(f"Export API 返回非200状态: {response.status_code}")
        except Exception as e:
            print_warning(f"使用export API失败，切换到分页获取: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 降级方案：使用分页获取
        print_info(f"使用分页API获取: {url}")
        found_any = False
        
        while True:
            try:
                print_info(f"正在获取第 {page} 页...")
                response = requests.get(
                    url,
                    headers=self.headers,
                    params={'page': page, 'page_size': 100},
                    timeout=30
                )
                
                print_info(f"第 {page} 页响应状态: {response.status_code}")
                
                if response.status_code != 200:
                    print_warning(f"分页API返回非200状态: {response.status_code}")
                    break
                
                data = response.json()
                
                # 处理不同的响应格式
                if isinstance(data, dict):
                    page_tasks = data.get('tasks', data.get('results', []))
                    print_info(f"第 {page} 页返回了 {len(page_tasks)} 个任务")
                else:
                    page_tasks = data if isinstance(data, list) else []
                    print_info(f"第 {page} 页返回了 {len(page_tasks)} 个任务（列表格式）")
                
                if not page_tasks:
                    print_info(f"第 {page} 页没有任务，停止获取")
                    break
                
                found_any = True
                
                # 过滤ID范围内的tasks
                page_matched = 0
                for task in page_tasks:
                    task_id = task.get('id')
                    if task_id and start_id <= task_id <= end_id:
                        tasks.append(task)
                        page_matched += 1
                
                print_info(f"第 {page} 页匹配了 {page_matched} 个任务")
                
                # 检查是否还有更多页
                if isinstance(data, dict):
                    has_next = data.get('next')
                    print_info(f"是否有下一页: {bool(has_next)}")
                    if not has_next:
                        break
                else:
                    # 如果是列表格式，且数量不足100，说明没有更多了
                    if len(page_tasks) < 100:
                        break
                
                page += 1
                
            except Exception as e:
                print_warning(f"获取第{page}页时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        if not found_any:
            print_warning("分页获取没有返回任何任务！")
        
        print_success(f"✓ 找到 {len(tasks)} 个任务（ID范围: {start_id}-{end_id}）")
        return tasks
    
    def _get_all_project_tasks(self) -> List[Dict]:
        """获取项目所有tasks"""
        url = f"{self.url}/api/projects/{self.project_id}/tasks"
        tasks = []
        page = 1
        
        while True:
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params={'page': page, 'page_size': 100},
                    timeout=30
                )
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                if isinstance(data, list):
                    tasks.extend(data)
                    if len(data) < 100:
                        break
                else:
                    tasks.extend(data.get('tasks', []))
                    if not data.get('next'):
                        break
                
                page += 1
                
            except Exception:
                break
        
        return tasks
    
    def _get_unlabeled_tasks(self) -> List[Dict]:
        """获取未标注的tasks"""
        url = f"{self.url}/api/tasks"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params={
                    'project': self.project_id,
                    'annotations__isnull': 'true',
                    'page_size': 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else data.get('tasks', [])
            
        except Exception:
            pass
        
        return []
    
    def verify_uploaded_tasks(self, num_samples: int = 5) -> bool:
        """验证上传的任务"""
        print_info(f"\n开始验证上传的任务...")
        url = f"{self.url}/api/projects/{self.project_id}/tasks"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params={'page_size': num_samples},
                timeout=10
            )
            
            if response.status_code != 200:
                print_error(f"✗ 无法获取任务列表: {response.status_code}")
                return False
            
            tasks = response.json()
            if not tasks:
                print_warning("项目中没有任务")
                return False
            
            print_info(f"正在验证 {min(num_samples, len(tasks))} 个任务...\n")
            
            for i, task in enumerate(tasks[:num_samples], 1):
                task_id = task.get('id', 'unknown')
                data = task.get('data', {})
                predictions = task.get('predictions', [])
                meta = task.get('meta', {})
                
                original_name = data.get('original_filename', 'unknown')
                anno_count = meta.get('annotation_count', 0)
                pred_count = len(predictions[0].get('result', [])) if predictions else 0
                
                print_info(f"任务 #{task_id}: {original_name}")
                
                # 根据任务类型验证
                if self.task_type == 'pose':
                    # Pose任务：每个对象上传 1个边界框 + N个可见关键点
                    # 注意：只有visibility>0的关键点才会被上传
                    print_info(f"  原始对象数: {anno_count}, 上传的标注数: {pred_count}")
                    
                    if self.keypoint_names:
                        # 每个对象: 1个bbox + 最多N个关键点
                        min_expected = anno_count  # 至少有边界框
                        max_expected = anno_count * (1 + len(self.keypoint_names))  # bbox + 所有关键点
                        print_info(f"  说明: 每个对象包含1个边界框 + 最多{len(self.keypoint_names)}个关键点")
                        print_info(f"  期望标注数: {min_expected}~{max_expected}（不可见的关键点未上传）")
                        
                        # 验证：标注数应该在合理范围内
                        if pred_count >= min_expected and pred_count <= max_expected:
                            print_success("  ✓ 标注验证通过")
                        else:
                            print_warning(f"  ⚠️  标注数量异常（期望{min_expected}~{max_expected}，实际{pred_count}）")
                    else:
                        print_warning("  ⚠️  无法验证（缺少关键点信息）")
                else:
                    # 检测/分割任务：直接比较标注数
                    print_info(f"  原始标注数: {anno_count}, 上传的标注数: {pred_count}")
                    
                    if anno_count == pred_count:
                        print_success("  ✓ 标注数量匹配")
                    else:
                        print_warning("  ⚠️  标注数量不匹配！")
            
            print_success("\n✓ 验证完成！")
            return True
            
        except Exception as e:
            print_error(f"✗ 验证异常: {str(e)}")
            return False
    
    def get_tasks_by_ids(self, task_ids: List[int]) -> List[Dict]:
        """
        从Label Studio获取指定ID的任务
        
        Args:
            task_ids: 任务ID列表
            
        Returns:
            List[Dict]: 任务数据列表
        """
        if not task_ids:
            return []
        
        tasks = []
        for task_id in task_ids:
            try:
                # 使用单个任务API获取
                url = f"{self.url}/api/tasks/{task_id}/"
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    task = response.json()
                    tasks.append(task)
                elif response.status_code == 404:
                    print_warning(f"任务 #{task_id} 不存在")
                else:
                    print_warning(f"获取任务 #{task_id} 失败: {response.status_code}")
                    
            except Exception as e:
                print_warning(f"获取任务 #{task_id} 异常: {str(e)}")
                continue
        
        return tasks
    
    def get_tasks_by_range(self, start_id: int, end_id: int) -> List[Dict]:
        """
        获取指定ID范围的任务
        
        对于小范围（<=50个ID），逐个获取任务
        对于大范围，使用分页获取所有任务后过滤
        
        Args:
            start_id: 起始ID
            end_id: 结束ID
            
        Returns:
            List[Dict]: 任务数据列表
        """
        range_size = end_id - start_id + 1
        
        # 对于小范围，逐个获取（更精确）
        if range_size <= 50:
            task_ids = list(range(start_id, end_id + 1))
            return self.get_tasks_by_ids(task_ids)
        
        # 对于大范围，使用分页获取并过滤
        url = f"{self.url}/api/projects/{self.project_id}/tasks"
        all_tasks = []
        page = 1
        page_size = 100
        
        try:
            while True:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params={'page': page, 'page_size': page_size},
                    timeout=30
                )
                
                if response.status_code != 200:
                    print_error(f"获取任务失败: {response.status_code}")
                    break
                
                data = response.json()
                
                # 处理不同的响应格式
                if isinstance(data, dict):
                    tasks = data.get('results', data.get('tasks', []))
                elif isinstance(data, list):
                    tasks = data
                else:
                    break
                
                if not tasks:
                    break
                
                # 过滤在范围内的任务
                for task in tasks:
                    task_id = task.get('id')
                    if task_id and start_id <= task_id <= end_id:
                        all_tasks.append(task)
                
                # 检查是否还有更多页
                if isinstance(data, dict):
                    # 如果最后一个任务的ID已经超过end_id，停止
                    if tasks and tasks[-1].get('id', 0) > end_id:
                        break
                    # 检查是否还有下一页
                    if not data.get('next'):
                        break
                else:
                    break
                
                page += 1
            
            return all_tasks
            
        except Exception as e:
            print_error(f"获取任务范围异常: {str(e)}")
            return []
    
    def get_unlabeled_tasks(self) -> List[Dict]:
        """
        获取所有未标注的任务
        
        Returns:
            List[Dict]: 未标注任务列表
        """
        url = f"{self.url}/api/projects/{self.project_id}/tasks"
        all_tasks = []
        page = 1
        page_size = 100
        
        try:
            while True:
                # 尝试使用过滤参数
                response = requests.get(
                    url,
                    headers=self.headers,
                    params={
                        'page': page,
                        'page_size': page_size,
                        # 尝试不同的过滤方式
                        # 'filter': 'tasks:annotations_results=0'
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    print_error(f"获取任务失败: {response.status_code}")
                    break
                
                data = response.json()
                
                # 处理不同的响应格式
                if isinstance(data, dict):
                    tasks = data.get('results', data.get('tasks', []))
                elif isinstance(data, list):
                    tasks = data
                else:
                    break
                
                if not tasks:
                    break
                
                # 过滤未标注的任务（没有annotations或annotations为空）
                for task in tasks:
                    annotations = task.get('annotations', [])
                    # 只有当annotations为空或None时才认为是未标注
                    if not annotations:
                        all_tasks.append(task)
                
                # 检查是否还有更多页
                if isinstance(data, dict):
                    if not data.get('next'):
                        break
                else:
                    break
                
                page += 1
            
            return all_tasks
            
        except Exception as e:
            print_error(f"获取未标注任务异常: {str(e)}")
            return []
    
    def download_task_image(self, task: Dict, temp_dir: Path) -> Optional[Path]:
        """
        下载任务图片到临时目录
        
        Args:
            task: 任务数据字典
            temp_dir: 临时目录路径
            
        Returns:
            Optional[Path]: 下载后的图片路径，失败返回None
        """
        try:
            # 从任务数据中提取图片URL
            data = task.get('data', {})
            image_url = data.get('image')
            
            if not image_url:
                print_warning(f"任务 #{task.get('id')} 没有图片URL")
                return None
            
            # 处理不同类型的图片URL
            # 1. 相对路径: /data/upload/xxx.jpg
            # 2. 完整URL: http://domain.com/data/upload/xxx.jpg
            # 3. 本地文件路径: /path/to/image.jpg
            
            if image_url.startswith('http://') or image_url.startswith('https://'):
                # 完整URL，直接下载
                download_url = image_url
            elif image_url.startswith('/'):
                # 相对路径，构造完整URL
                download_url = f"{self.url}{image_url}"
            else:
                # 其他情况，尝试构造URL
                download_url = f"{self.url}/{image_url}"
            
            # 生成本地文件名
            task_id = task.get('id', 'unknown')
            # 从URL中提取文件扩展名
            if '.' in image_url:
                ext = image_url.rsplit('.', 1)[-1].split('?')[0]  # 去除查询参数
                if ext.lower() in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
                    filename = f"task_{task_id}.{ext}"
                else:
                    filename = f"task_{task_id}.jpg"
            else:
                filename = f"task_{task_id}.jpg"
            
            local_path = temp_dir / filename
            
            # 下载图片
            response = requests.get(
                download_url,
                headers={'Authorization': self.headers['Authorization']},
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                # 保存到本地
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
            else:
                print_warning(f"下载图片失败 (任务 #{task_id}): {response.status_code}")
                return None
                
        except Exception as e:
            print_warning(f"下载任务 #{task.get('id')} 图片时出错: {str(e)}")
            return None
    
    def yolo_result_to_labelstudio_format(self, result, task_type: str) -> List[Dict]:
        """
        将YOLO预测结果转换为Label Studio格式
        
        Args:
            result: YOLO Results对象
            task_type: 任务类型 (detect/segment/pose/classify)
            
        Returns:
            List[Dict]: Label Studio格式的预测结果列表
        """
        predictions = []
        img_width, img_height = result.orig_shape[1], result.orig_shape[0]
        
        # 调试信息：显示检测结果统计
        debug_info = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            debug_info.append(f"检测到 {len(result.boxes)} 个边界框")
        if hasattr(result, 'masks') and result.masks is not None:
            debug_info.append(f"检测到 {len(result.masks)} 个mask")
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            debug_info.append(f"检测到 {len(result.keypoints)} 组关键点")
        if hasattr(result, 'probs') and result.probs is not None:
            debug_info.append(f"分类结果: top1={result.probs.top1}, conf={result.probs.top1conf:.2f}")
        
        if not debug_info:
            print_warning(f"⚠ YOLO未检测到任何对象 (任务类型: {task_type})")
        
        try:
            if task_type == 'detect':
                # 检测任务：转换bbox
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    skipped_count = 0
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id < len(self.classes):
                            pred = self._yolo_to_labelstudio_bbox(
                                [class_id, *box.xywhn[0].cpu().numpy().tolist()],
                                img_width,
                                img_height
                            )
                            predictions.append(pred)
                        else:
                            skipped_count += 1
                    
                    if skipped_count > 0:
                        print_warning(f"⚠ 跳过 {skipped_count} 个对象 (class_id >= {len(self.classes)}, classes: {self.classes})")
            
            elif task_type == 'segment':
                # 分割任务：转换多边形
                if hasattr(result, 'masks') and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks
                    
                    skipped_count = 0
                    for box, mask in zip(boxes, masks):
                        class_id = int(box.cls[0])
                        if class_id < len(self.classes):
                            # 获取mask的轮廓点
                            if hasattr(mask, 'xy') and len(mask.xy) > 0:
                                # mask.xy是归一化的坐标点
                                points = mask.xy[0]  # 取第一个轮廓
                                # 构造YOLO格式: [class_id, x1, y1, x2, y2, ...]
                                yolo_anno = [class_id]
                                for point in points:
                                    yolo_anno.extend([point[0] / img_width, point[1] / img_height])
                                
                                pred = self._yolo_to_labelstudio_bbox(
                                    yolo_anno,
                                    img_width,
                                    img_height
                                )
                                predictions.append(pred)
                        else:
                            skipped_count += 1
                    
                    if skipped_count > 0:
                        print_warning(f"⚠ 跳过 {skipped_count} 个对象 (class_id >= {len(self.classes)}, classes: {self.classes})")
            
            elif task_type == 'pose':
                # 姿态估计：转换关键点和bbox
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()  # [N, num_kpts, 2]
                    keypoints_conf = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
                    boxes = result.boxes
                    
                    skipped_count = 0
                    for idx, (kp, box) in enumerate(zip(keypoints, boxes)):
                        class_id = int(box.cls[0])
                        
                        if class_id < len(self.classes):
                            # 1. 添加bbox（RectangleLabels）
                            xyxy = box.xyxy[0].cpu().numpy()
                            x_center = ((xyxy[0] + xyxy[2]) / 2) / img_width
                            y_center = ((xyxy[1] + xyxy[3]) / 2) / img_height
                            width = (xyxy[2] - xyxy[0]) / img_width
                            height = (xyxy[3] - xyxy[1]) / img_height
                            
                            # 添加bbox标注
                            bbox_anno = [class_id, x_center, y_center, width, height]
                            bbox_pred = self._yolo_to_labelstudio_bbox(
                                bbox_anno,
                                img_width,
                                img_height
                            )
                            if bbox_pred:
                                # 添加置信度
                                bbox_pred['score'] = float(box.conf[0])
                                predictions.append(bbox_pred)
                            
                            # 2. 添加关键点（KeyPointLabels）
                            # 构建YOLO格式标注
                            yolo_anno = [class_id, x_center, y_center, width, height]
                            
                            # 添加关键点
                            kpts_conf = keypoints_conf[idx] if keypoints_conf is not None else None
                            for kpt_idx, (kpt_x, kpt_y) in enumerate(kp):
                                # 归一化坐标
                                norm_x = kpt_x / img_width
                                norm_y = kpt_y / img_height
                                
                                # visibility
                                if kpts_conf is not None:
                                    conf_val = kpts_conf[kpt_idx]
                                    if conf_val < 0.3:
                                        visibility = 0
                                    elif conf_val < 0.7:
                                        visibility = 1
                                    else:
                                        visibility = 2
                                else:
                                    visibility = 2 if kpt_x > 0 and kpt_y > 0 else 0
                                
                                yolo_anno.extend([norm_x, norm_y, visibility])
                            
                            # 转换为Label Studio格式
                            keypoint_preds = self._yolo_to_labelstudio_pose(
                                yolo_anno,
                                img_width,
                                img_height
                            )
                            if keypoint_preds:
                                predictions.extend(keypoint_preds)
                        else:
                            skipped_count += 1
                    
                    # 如果有被跳过的对象，打印警告
                    if skipped_count > 0:
                        print_warning(f"⚠ 跳过 {skipped_count} 个对象 (class_id >= {len(self.classes)}, classes: {self.classes})")
            
            elif task_type == 'classify':
                # 分类任务
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top_idx = probs.top1
                    top_conf = probs.top1conf
                    
                    if top_idx < len(self.classes):
                        # 分类格式（Label Studio的Choices）
                        pred = {
                            "original_width": int(img_width),
                            "original_height": int(img_height),
                            "image_rotation": 0,
                            "value": {
                                "choices": [self.classes[int(top_idx)]]
                            },
                            "from_name": "choice",
                            "to_name": "image",
                            "type": "choices",
                            "score": float(top_conf)
                        }
                        predictions.append(pred)
        
        except Exception as e:
            print_warning(f"转换预测结果时出错: {str(e)}")
        
        # 调试信息：显示转换结果统计
        if not predictions and debug_info:
            print_warning(f"⚠ 虽然检测到对象({', '.join(debug_info)})，但转换后predictions为空")
            print_warning(f"   可能原因: class_id超出范围 (self.classes长度: {len(self.classes)})")
            if task_type == 'pose':
                print_warning(f"   或关键点配置问题 (self.keypoint_names长度: {len(self.keypoint_names)})")
        elif predictions:
            print_info(f"✓ 成功转换 {len(predictions)} 个predictions ({', '.join(debug_info)})")
        
        return predictions
    
    def upload_prediction(self, task_id: int, predictions: List[Dict], overwrite: bool = True) -> Tuple[bool, Optional[str]]:
        """
        上传预测结果到Label Studio
        
        Args:
            task_id: 任务ID
            predictions: 预测结果列表（Label Studio格式）
            overwrite: 是否覆盖已有predictions
            
        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息)
        """
        try:
            # 如果需要覆盖，先获取并删除已有的predictions
            if overwrite:
                # 获取任务的现有predictions
                get_url = f"{self.url}/api/tasks/{task_id}/predictions"
                response = requests.get(
                    get_url,
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    existing_preds = response.json()
                    # 删除所有现有predictions
                    for pred in existing_preds:
                        pred_id = pred.get('id')
                        if pred_id:
                            delete_url = f"{self.url}/api/predictions/{pred_id}"
                            requests.delete(delete_url, headers=self.headers, timeout=10)
            
            # 创建新的prediction
            create_url = f"{self.url}/api/predictions/"
            
            # 计算平均置信度
            avg_score = 0.0
            if predictions:
                scores = [p.get('score', p.get('value', {}).get('score', 0.5)) for p in predictions]
                avg_score = sum(scores) / len(scores) if scores else 0.5
            
            prediction_data = {
                "task": task_id,
                "result": predictions,
                "score": float(avg_score),
                "model_version": "yolocli_local_model"
            }
            
            # 检查predictions是否为空
            if not predictions:
                error_msg = "predictions为空"
                return False, error_msg
            
            # 调试：首次上传时显示prediction示例
            if not hasattr(self, '_first_pred_shown') and self.task_type == 'pose':
                # 对于pose任务，显示关键点prediction而不是bbox
                keypoint_pred = next((p for p in predictions if p.get('type') == 'keypointlabels'), None)
                if keypoint_pred:
                    print_info(f"\n调试：关键点prediction示例 (任务 #{task_id}):")
                    print_info(json.dumps(keypoint_pred, indent=2, ensure_ascii=False))
                self._first_pred_shown = True
            
            response = requests.post(
                create_url,
                headers=self.headers,
                json=prediction_data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                return True, None
            else:
                # 构建详细错误信息
                error_msg = f"HTTP {response.status_code}: {response.text}"
                
                print_error(f"✗ 上传prediction失败 (任务 #{task_id})")
                print_error(f"   状态码: {response.status_code}")
                print_error(f"   响应: {response.text}")
                print_info(f"   发送的predictions数量: {len(predictions)}")
                
                # 显示第一个prediction的详细信息
                if predictions:
                    print_info(f"   第一个prediction类型: {predictions[0].get('type', 'unknown')}")
                    if predictions[0].get('type') == 'keypointlabels':
                        print_info(f"   from_name: {predictions[0].get('from_name', 'N/A')}")
                        print_info(f"   labels: {predictions[0].get('value', {}).get('labels', 'N/A')}")
                        print_info(f"   完整JSON: {json.dumps(predictions[0], indent=2, ensure_ascii=False)}")
                
                return False, error_msg
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            print_error(f"✗ 上传prediction异常 (任务 #{task_id})")
            print_error(f"   异常类型: {type(e).__name__}")
            print_error(f"   异常信息: {str(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            print_error(f"   堆栈跟踪: {traceback_str}")
            
            return False, f"{error_msg}\n{traceback_str}"
    
    def predict_tasks_with_yolo(
        self,
        model_path: Path,
        task_ids: Optional[List[int]] = None,
        task_range: Optional[Tuple[int, int]] = None,
        unlabeled: bool = False,
        task_type: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = 'auto',
        max_workers: int = 4
    ) -> Tuple[int, int]:
        """
        使用YOLO模型预测Label Studio任务
        
        Args:
            model_path: YOLO模型路径
            task_ids: 任务ID列表
            task_range: 任务ID范围 (start, end)
            unlabeled: 是否预测所有未标注任务
            task_type: 任务类型（None则自动推断）
            conf: 置信度阈值
            iou: IOU阈值
            device: 设备
            max_workers: 最大并发数
            
        Returns:
            Tuple[int, int]: (成功数, 失败数)
        """
        from ultralytics import YOLO
        from ..core.utils import detect_device, get_device_name, parse_model_name
        from ..ui.display import create_progress_bar
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tempfile
        import shutil
        import threading
        
        # 获取任务列表
        print_info("\n获取任务列表...")
        if task_ids:
            tasks = self.get_tasks_by_ids(task_ids)
            filter_desc = f"任务ID: {', '.join(map(str, task_ids[:5]))}" + ('...' if len(task_ids) > 5 else '')
        elif task_range:
            tasks = self.get_tasks_by_range(task_range[0], task_range[1])
            filter_desc = f"ID范围: {task_range[0]}-{task_range[1]}"
        elif unlabeled:
            tasks = self.get_unlabeled_tasks()
            filter_desc = "所有未标注任务"
        else:
            print_error("未指定任务筛选条件")
            return (0, 0)
        
        if not tasks:
            print_warning("未找到符合条件的任务")
            return (0, 0)
        
        print_success(f"✓ 找到 {len(tasks)} 个任务 ({filter_desc})")
        
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix='yolocli_ls_predict_'))
        
        try:
            # 加载模型
            print_info("\n加载模型...")
            # 使用绝对路径以支持DDP模式
            model_path_abs = model_path.absolute() if model_path.exists() else model_path
            yolo_model = YOLO(str(model_path_abs))
            
            # 确定任务类型
            if task_type:
                actual_task = task_type
                print_info(f"使用指定任务类型: {actual_task}")
            else:
                # 自动推断
                actual_task = getattr(yolo_model, 'task', None)
                if actual_task:
                    print_info(f"从模型推断任务类型: {actual_task}")
                else:
                    _, actual_task = parse_model_name(model_path.name)
                    print_info(f"从文件名推断任务类型: {actual_task}")
            
            self.task_type = actual_task
            
            # 设置类别名称（从模型获取）
            if hasattr(yolo_model, 'names'):
                self.classes = list(yolo_model.names.values())
                print_info(f"模型类别: {len(self.classes)} 个 ({', '.join(self.classes[:3])}{'...' if len(self.classes) > 3 else ''})")
            else:
                print_warning("警告：无法从模型获取类别信息")
                self.classes = []
            
            # 设置关键点名称（如果是pose任务）
            if actual_task == 'pose':
                if hasattr(yolo_model.model, 'kpt_shape') and yolo_model.model.kpt_shape:
                    num_kpts = yolo_model.model.kpt_shape[0]
                    
                    # 优先从 Label Studio 项目配置中获取关键点名称
                    print_info("正在从 Label Studio 项目获取关键点标签...")
                    ls_keypoint_labels, ls_control_name = self.get_project_keypoint_labels()
                    
                    # 保存控件名称
                    self.keypoint_from_name = ls_control_name
                    
                    if ls_keypoint_labels and len(ls_keypoint_labels) == num_kpts:
                        # 使用 Label Studio 项目中定义的关键点标签
                        self.keypoint_names = ls_keypoint_labels
                        print_success(f"✓ 使用 Label Studio 项目关键点标签: {self.keypoint_names}")
                    else:
                        if ls_keypoint_labels:
                            print_warning(f"Label Studio 关键点数量({len(ls_keypoint_labels)})与模型不匹配({num_kpts})，使用模型配置")
                        
                        # 尝试从模型获取关键点名称（使用 kpt_names）
                        self.keypoint_names = getattr(yolo_model.model, 'kpt_names', None)
                        
                        if not self.keypoint_names:
                            # 使用默认名称
                            self.keypoint_names = [f'kp_{i+1}' for i in range(num_kpts)]
                            print_info(f"使用默认关键点名称: {self.keypoint_names}")
                    
                    print_info(f"关键点数量: {num_kpts} 个")
                else:
                    print_warning("警告：无法获取关键点信息")
                    self.keypoint_names = []
            
            # 自动检测设备
            if device == 'auto':
                device = detect_device()
            
            print_info(f"模型: {model_path.name}")
            print_info(f"任务类型: {actual_task.upper()}")
            print_info(f"设备: {get_device_name(device)}")
            print_info(f"置信度阈值: {conf}")
            if actual_task != 'classify':
                print_info(f"IOU阈值: {iou}")
            
            # 并发处理任务
            print_info(f"\n开始预测（并发数: {max_workers}）...")
            
            success_count = 0
            failed_count = 0
            failed_tasks = []  # 收集失败任务的详细信息
            lock = threading.Lock()
            
            def process_single_task(task):
                """处理单个任务"""
                task_id = task.get('id')
                
                try:
                    # 1. 下载图片
                    image_path = self.download_task_image(task, temp_dir)
                    if not image_path:
                        return False, task_id, "图片下载失败", None
                    
                    # 2. YOLO预测
                    predict_kwargs = {
                        'source': str(image_path),
                        'device': device,
                        'verbose': False,
                    }
                    
                    if actual_task != 'classify':
                        predict_kwargs['conf'] = conf
                        predict_kwargs['iou'] = iou
                    
                    results = yolo_model.predict(**predict_kwargs)
                    
                    if not results:
                        return False, task_id, "YOLO预测返回空结果", None
                    
                    # 3. 转换为Label Studio格式
                    result = results[0]
                    predictions = self.yolo_result_to_labelstudio_format(result, actual_task)
                    
                    # 如果predictions为空，标记为成功但跳过上传（没有检测到对象是正常的）
                    if not predictions:
                        return True, task_id, f"预测完成 (0 个目标，跳过上传)", None
                    
                    # 4. 上传predictions
                    success, error_msg = self.upload_prediction(task_id, predictions, overwrite=True)
                    if success:
                        obj_count = len(predictions)
                        return True, task_id, f"预测完成 ({obj_count} 个目标)", None
                    else:
                        error_detail = f"Label Studio API错误:\n{error_msg}\n\nPrediction数据:\n{json.dumps(predictions[:2], indent=2, ensure_ascii=False) if predictions else 'N/A'}"
                        return False, task_id, "上传到Label Studio失败", error_detail
                    
                except Exception as e:
                    import traceback
                    error_detail = f"{str(e)}\n{traceback.format_exc()}"
                    return False, task_id, f"处理异常: {str(e)}", error_detail
            
            # 使用线程池并发处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                with create_progress_bar() as progress:
                    task_progress = progress.add_task("预测进度", total=len(tasks))
                    
                    # 提交所有任务
                    futures = {
                        executor.submit(process_single_task, task): task
                        for task in tasks
                    }
                    
                    # 处理完成的任务
                    for future in as_completed(futures):
                        try:
                            success, task_id, message, detail = future.result()
                            
                            with lock:
                                if success:
                                    success_count += 1
                                    # 每10个显示一次进度
                                    if success_count % 10 == 0 or success_count == len(tasks):
                                        print_success(f"✓ 进度: 已成功上传 {success_count}/{len(tasks)} 个任务")
                                else:
                                    failed_count += 1
                                    failed_tasks.append({
                                        'task_id': task_id,
                                        'reason': message,
                                        'detail': detail
                                    })
                                    print_warning(f"✗ 任务 #{task_id}: {message}")
                                
                                progress.update(task_progress, advance=1)
                        
                        except Exception as e:
                            with lock:
                                failed_count += 1
                                failed_tasks.append({
                                    'task_id': 'unknown',
                                    'reason': f"Future异常: {str(e)}",
                                    'detail': None
                                })
                                progress.update(task_progress, advance=1)
                                print_warning(f"✗ 任务异常: {str(e)}")
            
            # 如果有失败任务，显示详细信息
            if failed_tasks:
                print_error("\n" + "=" * 80)
                print_error(f"失败任务详情 ({len(failed_tasks)} 个)")
                print_error("=" * 80)
                
                # 按失败原因分组
                from collections import defaultdict
                failures_by_reason = defaultdict(list)
                for failed in failed_tasks:
                    failures_by_reason[failed['reason']].append(failed['task_id'])
                
                # 显示分组统计
                for reason, task_ids in failures_by_reason.items():
                    print_error(f"\n原因: {reason}")
                    print_error(f"  影响任务数: {len(task_ids)}")
                    print_error(f"  任务ID: {', '.join(map(str, task_ids[:10]))}" + 
                               (f" ... (还有{len(task_ids)-10}个)" if len(task_ids) > 10 else ""))
                
                # 显示前3个失败任务的详细错误信息
                print_error("\n详细错误信息 (前3个):")
                for i, failed in enumerate(failed_tasks[:3], 1):
                    print_error(f"\n{i}. 任务 #{failed['task_id']}")
                    print_error(f"   原因: {failed['reason']}")
                    if failed['detail']:
                        # 显示完整的错误详情（可能很长）
                        detail_str = str(failed['detail'])
                        if len(detail_str) > 2000:
                            print_error(f"   详情: {detail_str[:2000]}...")
                            print_error(f"   (详情过长，已截断。完整长度: {len(detail_str)} 字符)")
                        else:
                            print_error(f"   详情:\n{detail_str}")
                
                print_error("=" * 80 + "\n")
            
            return (success_count, failed_count)
        
        finally:
            # 清理临时文件
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    print_info("\n临时文件已清理")
            except Exception as e:
                print_warning(f"清理临时文件失败: {str(e)}")
    
    def audit_annotations(self, show_details: bool = True, max_samples: int = 10, max_tasks: Optional[int] = None) -> Dict:
        """
        审计Label Studio项目的标注质量
        
        Args:
            show_details: 是否显示异常任务的详细信息
            max_samples: 每种异常类型显示的最大样本数
            max_tasks: 最大审计任务数（None表示审计全部，用于抽样审计）
            
        Returns:
            Dict: 审计报告
        """
        from collections import defaultdict, Counter
        
        # 使用导出API一次性获取所有任务和标注
        print_info("正在导出项目数据...")
        export_url = f"{self.url}/api/projects/{self.project_id}/export"
        
        try:
            response = requests.get(
                export_url,
                headers=self.headers,
                params={'exportType': 'JSON'},
                timeout=120  # 导出可能需要较长时间
            )
            
            if response.status_code != 200:
                print_error(f"导出失败: {response.status_code} - {response.text[:200]}")
                return {}
            
            all_tasks = response.json()
            
            if not isinstance(all_tasks, list):
                print_error(f"导出数据格式异常: {type(all_tasks)}")
                return {}
            
            print_success(f"✓ 成功导出 {len(all_tasks)} 个任务")
            
            # 如果需要抽样，截取前N个
            if max_tasks and len(all_tasks) > max_tasks:
                all_tasks = all_tasks[:max_tasks]
                print_info(f"  抽样：使用前 {max_tasks} 个任务")
        
        except Exception as e:
            print_error(f"导出异常: {str(e)}")
            return {}
        
        # 统计数据
        total_tasks = len(all_tasks)
        annotated_tasks = [t for t in all_tasks if t.get('annotations')]
        unannotated_tasks = [t for t in all_tasks if not t.get('annotations')]
        
        print_info(f"已标注: {len(annotated_tasks)} 个")
        print_info(f"未标注: {len(unannotated_tasks)} 个")
        
        # 分析标注
        print_info("\n分析标注一致性...")
        
        audit_report = {
            'summary': {
                'total_tasks': total_tasks,
                'annotated_tasks': len(annotated_tasks),
                'unannotated_tasks': len(unannotated_tasks),
                'audit_mode': '抽样' if max_tasks else '全部',
                'max_tasks': max_tasks,
            },
            'issues': {},
            'statistics': {}
        }
        
        # 检查关键点标注顺序（pose任务）
        keypoint_order_issues = self._check_keypoint_order_consistency(
            annotated_tasks, show_details, max_samples
        )
        if keypoint_order_issues:
            audit_report['issues']['keypoint_order'] = keypoint_order_issues
        
        # 检查标注完整性
        completeness_issues = self._check_annotation_completeness(
            annotated_tasks, show_details, max_samples
        )
        if completeness_issues:
            audit_report['issues']['completeness'] = completeness_issues
        
        # 检查重复标注
        duplicate_issues = self._check_duplicate_annotations(
            annotated_tasks, show_details, max_samples
        )
        if duplicate_issues:
            audit_report['issues']['duplicates'] = duplicate_issues
        
        # 显示统计摘要
        self._display_audit_summary(audit_report)
        
        return audit_report
    
    def _check_keypoint_order_consistency(self, tasks: List[Dict], show_details: bool, max_samples: int) -> Dict:
        """检查关键点标注顺序的一致性"""
        from collections import Counter
        
        keypoint_orders = []
        task_order_map = {}  # task_id -> keypoint_order
        
        for task in tasks:
            task_id = task.get('id')
            annotations = task.get('annotations', [])
            
            if not annotations:
                continue
            
            # 获取第一个annotation（通常是人工标注）
            annotation = annotations[0]
            results = annotation.get('result', [])
            
            # 提取关键点标签顺序
            keypoints = [r for r in results if r.get('type') == 'keypointlabels']
            
            if keypoints:
                # 提取关键点名称序列
                kp_names = []
                for kp in keypoints:
                    labels = kp.get('value', {}).get('keypointlabels', [])
                    if labels:
                        kp_names.extend(labels)
                
                if kp_names:
                    order_tuple = tuple(kp_names)
                    keypoint_orders.append(order_tuple)
                    task_order_map[task_id] = order_tuple
        
        if not keypoint_orders:
            return None
        
        # 统计不同的标注顺序
        order_counter = Counter(keypoint_orders)
        
        if len(order_counter) == 1:
            print_success(f"✓ 关键点标注顺序一致: {list(order_counter.keys())[0]}")
            return None
        
        # 发现不一致
        most_common_order = order_counter.most_common(1)[0]
        print_warning(f"⚠ 发现 {len(order_counter)} 种不同的关键点标注顺序")
        print_info(f"  最常见顺序: {list(most_common_order[0])} (出现 {most_common_order[1]} 次)")
        
        # 找出异常任务
        abnormal_tasks = {}
        for order, count in order_counter.items():
            if order != most_common_order[0]:
                task_ids = [tid for tid, torder in task_order_map.items() if torder == order]
                abnormal_tasks[str(list(order))] = {
                    'count': count,
                    'task_ids': task_ids[:max_samples] if show_details else []
                }
                
                if show_details:
                    print_warning(f"  异常顺序: {list(order)}")
                    print_info(f"    出现次数: {count}")
                    print_info(f"    任务ID示例: {task_ids[:max_samples]}")
        
        return {
            'total_samples': len(keypoint_orders),
            'unique_orders': len(order_counter),
            'most_common_order': {
                'order': list(most_common_order[0]),
                'count': most_common_order[1]
            },
            'abnormal_orders': abnormal_tasks
        }
    
    def _check_annotation_completeness(self, tasks: List[Dict], show_details: bool, max_samples: int) -> Dict:
        """检查标注完整性"""
        incomplete_tasks = []
        
        for task in tasks:
            task_id = task.get('id')
            annotations = task.get('annotations', [])
            
            if not annotations:
                continue
            
            annotation = annotations[0]
            results = annotation.get('result', [])
            
            # 检查是否有空标注
            if not results:
                incomplete_tasks.append(task_id)
        
        if incomplete_tasks:
            print_warning(f"⚠ 发现 {len(incomplete_tasks)} 个空标注任务")
            if show_details:
                print_info(f"  任务ID示例: {incomplete_tasks[:max_samples]}")
            
            return {
                'count': len(incomplete_tasks),
                'task_ids': incomplete_tasks[:max_samples] if show_details else []
            }
        
        return None
    
    def _check_duplicate_annotations(self, tasks: List[Dict], show_details: bool, max_samples: int) -> Dict:
        """检查重复标注"""
        duplicate_tasks = []
        
        for task in tasks:
            task_id = task.get('id')
            annotations = task.get('annotations', [])
            
            if len(annotations) > 1:
                duplicate_tasks.append({
                    'task_id': task_id,
                    'annotation_count': len(annotations)
                })
        
        if duplicate_tasks:
            print_warning(f"⚠ 发现 {len(duplicate_tasks)} 个任务有多个标注")
            if show_details:
                for item in duplicate_tasks[:max_samples]:
                    print_info(f"  任务 #{item['task_id']}: {item['annotation_count']} 个标注")
            
            return {
                'count': len(duplicate_tasks),
                'details': duplicate_tasks[:max_samples] if show_details else []
            }
        
        return None
    
    def _display_audit_summary(self, report: Dict):
        """显示审计摘要"""
        print_section_header = lambda title: print_info(f"\n{'='*60}\n{title:^60}\n{'='*60}")
        
        print_section_header("审计摘要")
        
        summary = report['summary']
        print_info(f"审计模式: {summary['audit_mode']}")
        if summary.get('max_tasks'):
            print_info(f"抽样数量: {summary['max_tasks']} 个任务")
        print_info(f"总任务数: {summary['total_tasks']}")
        print_info(f"已标注: {summary['annotated_tasks']}")
        print_info(f"未标注: {summary['unannotated_tasks']}")
        
        issues = report.get('issues', {})
        
        if not issues:
            print_success("\n✅ 未发现标注问题！")
            return
        
        print_warning(f"\n发现 {len(issues)} 类问题:")
        
        for issue_type, issue_data in issues.items():
            if issue_type == 'keypoint_order':
                print_warning(f"\n  1. 关键点顺序不一致")
                print_info(f"     - {issue_data['unique_orders']} 种不同顺序")
                print_info(f"     - {len(issue_data['abnormal_orders'])} 种异常顺序")
            
            elif issue_type == 'completeness':
                print_warning(f"\n  2. 空标注")
                print_info(f"     - {issue_data['count']} 个任务")
            
            elif issue_type == 'duplicates':
                print_warning(f"\n  3. 重复标注")
                print_info(f"     - {issue_data['count']} 个任务")

