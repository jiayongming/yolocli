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
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..ui.display import print_info, print_success, print_warning, print_error


class LabelStudioUploader:
    """Label Studio上传器"""
    
    def __init__(self, url: str, api_key: str, project_id: int, task_type: str = 'detect'):
        """
        初始化上传器
        
        Args:
            url: Label Studio服务器URL
            api_key: API密钥（支持Refresh Token或Access Token）
            project_id: 项目ID
            task_type: 任务类型 (detect/segment/pose)
        """
        self.url = url.rstrip('/')
        self.original_token = api_key
        self.api_key = self._process_token(api_key)
        self.project_id = project_id
        self.task_type = task_type
        self.headers = self._get_auth_headers(self.api_key)
        self.classes = []
        self.keypoint_names = []  # pose任务的关键点名称
    
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
                        self.keypoint_names = config.get('keypoint_names', [])
                        if not self.keypoint_names:
                            # 如果没有keypoint_names，根据kpt_shape生成
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
                    "original_width": int(img_width),
                    "original_height": int(img_height),
                    "image_rotation": 0,
                    "value": {
                        "x": float(kp_x * 100),  # 转换为百分比
                        "y": float(kp_y * 100),
                        "width": 0.5,  # 关键点显示大小
                        "keypointlabels": [kp_label]
                    },
                    "from_name": "keypoint",
                    "to_name": "image",
                    "type": "keypointlabels"
                })
        
        if visible_count == 0:
            print_warning(f"警告：{kpt_count}个关键点全部不可见 (visibility=0)")
        else:
            print_info(f"转换pose标注: {visible_count}/{kpt_count} 个可见关键点")
        
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
    
    def _create_task(self, image_path: Path, label_path: Path) -> Dict:
        """创建Label Studio任务（文件上传模式）"""
        # 获取图片尺寸
        img_width, img_height = self._get_image_dimensions(image_path)
        
        # 解析YOLO标注
        yolo_annotations = self._parse_yolo_label(label_path)
        
        # 转换为Label Studio格式
        predictions = []
        if self.task_type == 'pose':
            # Pose任务：转换关键点
            for anno in yolo_annotations:
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
        
        # 创建任务数据
        task = {}
        task['data'] = {
            'image': file_url,  # 使用处理后的相对路径
            'original_filename': file_info['original_name'],
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
                    max_workers: int = 4) -> Tuple[int, int]:
        """
        批量上传任务到Label Studio（支持并发）
        
        Args:
            dataset_path: 数据集路径
            splits: 要上传的数据集分割
            max_images: 最大上传图片数（None表示全部）
            max_workers: 最大并发数（默认4）
            
        Returns:
            (成功数, 失败数)
        """
        total_uploaded = 0
        total_failed = 0
        
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
            print_info(f"并发数: {max_workers}")
            
            # 并发上传
            uploaded, failed = self._upload_images_concurrent(
                image_files=image_files,
                labels_dir=labels_dir,
                split_name=split,
                max_workers=max_workers
            )
            
            total_uploaded += uploaded
            total_failed += failed
        
        return total_uploaded, total_failed
    
    def _upload_images_concurrent(self, image_files: List[Path], labels_dir: Path, 
                                  split_name: str, max_workers: int) -> Tuple[int, int]:
        """
        并发上传图片
        
        Args:
            image_files: 图片文件列表
            labels_dir: 标签目录
            split_name: 数据集分割名称
            max_workers: 最大并发数
            
        Returns:
            (成功数, 失败数)
        """
        uploaded = 0
        failed = 0
        lock = threading.Lock()
        
        def upload_single_image(idx: int, image_path: Path) -> Tuple[bool, str]:
            """上传单张图片"""
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            try:
                task = self._create_task(image_path, label_path)
                success = self._upload_batch([task])
                
                if success:
                    anno_count = task['meta']['annotation_count']
                    original_name = task['data'].get('original_filename', image_path.name)
                    return True, f"[{idx}/{len(image_files)}] {original_name} ({anno_count} 个标注)"
                else:
                    return False, f"上传失败: {image_path.name}"
            except Exception as e:
                return False, f"处理 {image_path.name} 时出错: {str(e)}"
        
        # 使用线程池并发上传
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(upload_single_image, idx, img): (idx, img)
                for idx, img in enumerate(image_files, 1)
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                idx, image_path = futures[future]
                
                try:
                    success, message = future.result()
                    
                    with lock:
                        if success:
                            uploaded += 1
                            # 显示进度（每10个或最后一个）
                            if uploaded % 10 == 0 or uploaded == len(image_files) or uploaded <= 3:
                                print_info(f"  {message}")
                            if uploaded % 10 == 0 or uploaded == len(image_files):
                                print_success(f"✓ 进度: 已成功上传 {uploaded}/{len(image_files)} 个任务 ({split_name})")
                        else:
                            failed += 1
                            print_error(f"✗ {message}")
                
                except Exception as e:
                    with lock:
                        failed += 1
                        print_error(f"✗ 任务异常: {str(e)}")
        
        return uploaded, failed
    
    def _upload_batch(self, tasks: List[Dict]) -> bool:
        """批量上传任务到Label Studio"""
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
            else:
                print_error(f"错误: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
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
                print_error("未加载关键点信息，请确保dataset.yaml包含keypoint_names")
                return False
            
            keypoint_labels = '\n    '.join([
                f'<Label value="{kp}" background="#{self._get_color(i)}"/>'
                for i, kp in enumerate(self.keypoint_names)
            ])
            
            labeling_config = f"""
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <KeyPointLabels name="keypoint" toName="image" opacity="0.9">
    {keypoint_labels}
  </KeyPointLabels>
</View>
"""
        else:
            # 检测/分割任务：矩形框和多边形
            labeling_config = f"""
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="3" opacity="0.9">
    {self._generate_label_tags()}
  </RectangleLabels>
  <PolygonLabels name="polygon" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
    {self._generate_label_tags()}
  </PolygonLabels>
</View>
"""
        
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
                    # Pose任务：原始标注数是对象数，上传的标注数是可见关键点数
                    # 注意：只有visibility>0的关键点才会被上传
                    print_info(f"  原始对象数: {anno_count}, 上传的可见关键点数: {pred_count}")
                    
                    if self.keypoint_names:
                        expected_total = anno_count * len(self.keypoint_names)
                        print_info(f"  说明: 每个对象有{len(self.keypoint_names)}个关键点，实际上传{pred_count}个（不可见的关键点未上传）")
                        
                        # 简单验证：关键点数应该在合理范围内
                        if pred_count > 0 and pred_count <= expected_total:
                            print_success("  ✓ 标注验证通过")
                        else:
                            print_warning(f"  ⚠️  关键点数量异常（期望≤{expected_total}，实际{pred_count}）")
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
        
        # 调试信息
        print_info(f"开始转换预测结果 (任务类型: {task_type}, 图片尺寸: {img_width}x{img_height})")
        print_info(f"模型类别数: {len(self.classes)}, 类别: {self.classes if len(self.classes) <= 5 else self.classes[:5] + ['...']}")
        
        try:
            if task_type == 'detect':
                # 检测任务：转换bbox
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id < len(self.classes):
                            pred = self._yolo_to_labelstudio_bbox(
                                [class_id, *box.xywhn[0].cpu().numpy().tolist()],
                                img_width,
                                img_height
                            )
                            predictions.append(pred)
            
            elif task_type == 'segment':
                # 分割任务：转换多边形
                if hasattr(result, 'masks') and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks
                    
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
            
            elif task_type == 'pose':
                # 姿态估计：转换关键点和bbox
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()  # [N, num_kpts, 2]
                    keypoints_conf = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
                    boxes = result.boxes
                    
                    print_info(f"检测到 {len(boxes)} 个目标")
                    
                    for idx, (kp, box) in enumerate(zip(keypoints, boxes)):
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        print_info(f"  目标 {idx+1}: class_id={class_id}, conf={conf:.2%}, classes数量={len(self.classes)}")
                        
                        if class_id < len(self.classes):
                            print_info(f"    ✓ 类别有效: {self.classes[class_id]}")
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
                                print_warning(f"警告：关键点转换结果为空 (类别: {self.classes[class_id] if class_id < len(self.classes) else class_id})")
                        else:
                            # class_id越界
                            print_warning(f"    ✗ 跳过：class_id={class_id} 超出范围 (classes数量: {len(self.classes)})")
            
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
        
        return predictions
    
    def upload_prediction(self, task_id: int, predictions: List[Dict], overwrite: bool = True) -> bool:
        """
        上传预测结果到Label Studio
        
        Args:
            task_id: 任务ID
            predictions: 预测结果列表（Label Studio格式）
            overwrite: 是否覆盖已有predictions
            
        Returns:
            bool: 是否成功
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
            
            # 调试信息：打印上传的数据
            if not predictions:
                print_warning(f"任务 #{task_id}: predictions为空列表，跳过上传")
                return False
            
            # 打印第一个prediction的结构（用于调试）
            import json
            print_info(f"任务 #{task_id}: 上传 {len(predictions)} 个predictions")
            if predictions:
                print_info(f"  第一个prediction示例: {json.dumps(predictions[0], indent=2, ensure_ascii=False)[:300]}...")
            
            response = requests.post(
                create_url,
                headers=self.headers,
                json=prediction_data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                return True
            else:
                print_warning(f"上传prediction失败 (任务 #{task_id}): {response.status_code} - {response.text[:200]}")
                return False
                
        except Exception as e:
            print_warning(f"上传prediction异常 (任务 #{task_id}): {str(e)}")
            return False
    
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
            yolo_model = YOLO(str(model_path))
            
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
                    # 尝试获取关键点名称
                    if hasattr(yolo_model.model, 'names'):
                        self.keypoint_names = getattr(yolo_model.model, 'keypoint_names', None)
                    if not hasattr(self, 'keypoint_names') or not self.keypoint_names:
                        # 使用默认名称
                        self.keypoint_names = [f'kp_{i+1}' for i in range(num_kpts)]
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
            lock = threading.Lock()
            
            def process_single_task(task):
                """处理单个任务"""
                task_id = task.get('id')
                
                try:
                    # 1. 下载图片
                    image_path = self.download_task_image(task, temp_dir)
                    if not image_path:
                        return False, f"任务 #{task_id}: 图片下载失败"
                    
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
                        return False, f"任务 #{task_id}: 预测失败"
                    
                    # 3. 转换为Label Studio格式
                    result = results[0]
                    predictions = self.yolo_result_to_labelstudio_format(result, actual_task)
                    
                    # 4. 上传predictions
                    if self.upload_prediction(task_id, predictions, overwrite=True):
                        obj_count = len(predictions)
                        return True, f"任务 #{task_id}: 预测完成 ({obj_count} 个目标)"
                    else:
                        return False, f"任务 #{task_id}: 上传失败"
                    
                except Exception as e:
                    return False, f"任务 #{task_id}: 处理异常 - {str(e)}"
            
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
                            success, message = future.result()
                            
                            with lock:
                                if success:
                                    success_count += 1
                                    # 每10个显示一次进度
                                    if success_count % 10 == 0 or success_count == len(tasks):
                                        print_success(f"✓ 进度: 已成功上传 {success_count}/{len(tasks)} 个任务")
                                else:
                                    failed_count += 1
                                    print_warning(f"✗ {message}")
                                
                                progress.update(task_progress, advance=1)
                        
                        except Exception as e:
                            with lock:
                                failed_count += 1
                                progress.update(task_progress, advance=1)
                                print_warning(f"✗ 任务异常: {str(e)}")
            
            return (success_count, failed_count)
        
        finally:
            # 清理临时文件
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    print_info("\n临时文件已清理")
            except Exception as e:
                print_warning(f"清理临时文件失败: {str(e)}")

