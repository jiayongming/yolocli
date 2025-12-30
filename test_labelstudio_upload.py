#!/usr/bin/env python3
"""
Label Studio 文件上传测试脚本

用于测试和调试文件上传功能
"""

import requests
import sys
from pathlib import Path

def test_upload(url: str, api_key: str, project_id: int, image_path: str):
    """测试文件上传到 Label Studio"""
    
    print(f"🔍 测试 Label Studio 文件上传")
    print(f"   URL: {url}")
    print(f"   项目ID: {project_id}")
    print(f"   文件: {image_path}\n")
    
    # 准备头信息
    if api_key.startswith('eyJ'):
        auth_header = f'Bearer {api_key}'
    else:
        auth_header = f'Token {api_key}'
    
    # 测试的端点列表
    endpoints = [
        (f"{url}/api/projects/{project_id}/import?commit_to_project=false", "两步上传（推荐）"),
        (f"{url}/api/projects/{project_id}/import/file-upload", "项目文件上传"),
        (f"{url}/api/import/file-upload", "通用文件上传"),
        (f"{url}/api/projects/{project_id}/file-uploads", "简单文件上传"),
    ]
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ 文件不存在: {image_path}")
        return
    
    for endpoint, name in endpoints:
        print(f"📤 测试端点: {name}")
        print(f"   {endpoint}")
        
        try:
            with open(image_file, 'rb') as f:
                files = {'file': (image_file.name, f, 'image/jpeg')}
                headers = {'Authorization': auth_header}
                
                response = requests.post(
                    endpoint,
                    headers=headers,
                    files=files,
                    timeout=30
                )
                
                print(f"   状态码: {response.status_code}")
                
                if response.status_code in [200, 201]:
                    try:
                        result = response.json()
                        print(f"   ✅ 成功!")
                        print(f"   响应数据: {result}")
                        
                        # 尝试找到文件URL
                        file_url = (
                            result.get('file') or 
                            result.get('url') or 
                            result.get('path') or
                            result.get('file_url') or
                            result.get('uploaded_file') or
                            result.get('location')
                        )
                        
                        if file_url:
                            print(f"   📍 文件URL: {file_url}")
                        else:
                            print(f"   ⚠️  未找到文件URL字段")
                        
                        return True
                    except:
                        print(f"   响应内容: {response.text[:500]}")
                else:
                    print(f"   ❌ 失败")
                    print(f"   响应: {response.text[:200]}")
                    
        except requests.exceptions.Timeout:
            print(f"   ❌ 超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ 连接失败")
        except Exception as e:
            print(f"   ❌ 异常: {e}")
        
        print()
    
    print("❌ 所有端点测试失败")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("用法:")
        print(f"  python {sys.argv[0]} <URL> <API_KEY> <PROJECT_ID> <IMAGE_PATH>")
        print()
        print("示例:")
        print(f"  python {sys.argv[0]} http://10.105.3.39 YOUR_API_KEY 19 test.jpg")
        sys.exit(1)
    
    url = sys.argv[1].rstrip('/')
    api_key = sys.argv[2]
    project_id = int(sys.argv[3])
    image_path = sys.argv[4]
    
    test_upload(url, api_key, project_id, image_path)

