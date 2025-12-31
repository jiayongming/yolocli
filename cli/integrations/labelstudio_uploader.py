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
        
        for i in range(kpt_count):
            kp_x = kpt_data[i * 3]
            kp_y = kpt_data[i * 3 + 1]
            kp_v = kpt_data[i * 3 + 2]  # visibility: 0=不可见, 1=遮挡, 2=可见
            
            # 只上传可见或遮挡的关键点
            if kp_v > 0:
                kp_label = self.keypoint_names[i] if i < len(self.keypoint_names) else f'kp_{i+1}'
                keypoints.append({
                    "original_width": img_width,
                    "original_height": img_height,
                    "image_rotation": 0,
                    "value": {
                        "x": kp_x * 100,  # 转换为百分比
                        "y": kp_y * 100,
                        "width": 0.5,  # 关键点显示大小
                        "keypointlabels": [kp_label]
                    },
                    "from_name": "keypoint",
                    "to_name": "image",
                    "type": "keypointlabels"
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
            x = (x_center - width / 2) * 100
            y = (y_center - height / 2) * 100
            w = width * 100
            h = height * 100
            
            return {
                "original_width": img_width,
                "original_height": img_height,
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
                    points.append([coords[i] * 100, coords[i+1] * 100])
            
            return {
                "original_width": img_width,
                "original_height": img_height,
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

