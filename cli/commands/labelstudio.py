"""
Label Studio集成命令

支持将YOLO数据集上传到Label Studio进行标注和审核
"""

import typer
from pathlib import Path
from typing import Optional, List

from ..core.config import ConfigManager
from ..ui.display import print_section_header, print_info, print_success, print_error, print_warning
from ..integrations.labelstudio_uploader import LabelStudioUploader


app = typer.Typer(help="Label Studio集成")


@app.command("upload")
def upload_dataset(
    dataset: str = typer.Argument(..., help="数据集路径（支持相对路径或datasets目录下的数据集名称）"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    splits: Optional[List[str]] = typer.Option(
        None, 
        "--split", 
        "-s", 
        help="要上传的数据集分割（可多次指定，默认: train val test）"
    ),
    max_images: Optional[int] = typer.Option(None, "--max", help="最大上传图片数（用于测试）"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="最大并发数（默认: 4）"),
    setup_config: bool = typer.Option(False, "--setup-config", help="配置标注模板"),
    verify: bool = typer.Option(False, "--verify", help="上传后验证结果"),
):
    """
    上传YOLO数据集到Label Studio（文件上传模式）
    
    示例:
        # 上传datasets目录下的数据集
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1
        
        # 上传指定路径的数据集
        yolo_cli labelstudio upload /path/to/dataset --url http://localhost:8080 --api-key xxx --project-id 1
        
        # 只配置标注模板（不上传数据）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --setup-config
        
        # 上传并验证
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --verify
        
        # 只上传训练集和验证集
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --split train --split val
    """
    print_section_header("上传数据集到 Label Studio")
    
    # 解析数据集路径
    dataset_path = _resolve_dataset_path(dataset)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    print_info(f"数据集路径: {dataset_path}")
    print_info(f"Label Studio: {url}")
    print_info(f"项目ID: {project_id}")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化上传器失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    print_info("\n测试连接...")
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 加载数据集配置
    try:
        uploader.load_dataset_config(dataset_path)
    except Exception as e:
        print_error(f"加载数据集配置失败: {str(e)}")
        raise typer.Exit(1)
    
    # 配置标注模板
    if setup_config:
        print_info("\n配置标注模板...")
        if uploader.setup_project_config():
            print_success("\n✅ 标注模板配置完成！")
        else:
            print_error("标注模板配置失败")
            raise typer.Exit(1)
        return
    
    # 准备上传参数
    if splits is None:
        splits = ['train', 'val', 'test']
    
    print_info(f"\n数据集分割: {', '.join(splits)}")
    print_info(f"并发数: {max_workers}")
    print_info("使用文件上传模式（Label Studio会自动管理文件）")
    
    # 上传数据集
    try:
        total_uploaded, total_failed = uploader.upload_tasks(
            dataset_path=dataset_path,
            splits=splits,
            max_images=max_images,
            max_workers=max_workers
        )
        
        print_section_header("上传完成")
        print_success(f"成功: {total_uploaded} 个任务")
        if total_failed > 0:
            print_error(f"失败: {total_failed} 个任务")
        
        # 验证上传结果
        if verify and total_uploaded > 0:
            uploader.verify_uploaded_tasks(num_samples=5)
        
    except Exception as e:
        print_error(f"上传失败: {str(e)}")
        raise typer.Exit(1)


@app.command("setup-config")
def setup_project_config(
    dataset: str = typer.Argument(..., help="数据集路径（用于读取类别信息）"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
):
    """
    仅配置Label Studio项目标注模板
    
    示例:
        yolo_cli labelstudio setup-config my_dataset --url http://localhost:8080 --api-key xxx --project-id 1
    """
    print_section_header("配置 Label Studio 标注模板")
    
    # 解析数据集路径
    dataset_path = _resolve_dataset_path(dataset)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化上传器失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 加载数据集配置
    try:
        uploader.load_dataset_config(dataset_path)
    except Exception as e:
        print_error(f"加载数据集配置失败: {str(e)}")
        raise typer.Exit(1)
    
    # 配置标注模板
    if uploader.setup_project_config():
        print_success("\n✅ 标注模板配置完成！")
    else:
        print_error("标注模板配置失败")
        raise typer.Exit(1)


@app.command("verify")
def verify_uploaded_tasks(
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    num_samples: int = typer.Option(5, "--samples", "-n", help="验证样本数量"),
):
    """
    验证已上传的任务
    
    示例:
        yolo_cli labelstudio verify --url http://localhost:8080 --api-key xxx --project-id 1 --samples 10
    """
    print_section_header("验证上传的任务")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化上传器失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 验证任务
    if uploader.verify_uploaded_tasks(num_samples=num_samples):
        print_success("\n✅ 验证完成！")
    else:
        print_error("验证失败")
        raise typer.Exit(1)


@app.command("predict")
def predict_tasks(
    model: str = typer.Argument(..., help="YOLO模型路径"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务类型 (detect/segment/pose/classify，不指定则自动推断)"),
    task_ids: Optional[List[int]] = typer.Option(None, "--task-ids", help="指定任务ID列表"),
    task_range: Optional[List[int]] = typer.Option(None, "--task-range", help="指定任务ID范围 (start end)"),
    unlabeled: bool = typer.Option(False, "--unlabeled", help="预测所有未标注的任务"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    max_workers: int = typer.Option(4, "--max-workers", "-w", help="最大并发数"),
    device: str = typer.Option("auto", "--device", "-d", help="设备 (auto/cpu/cuda)"),
):
    """
    使用本地YOLO模型预测Label Studio任务
    
    任务筛选方式（三选一）：
      - --task-ids: 指定任务ID列表，如 --task-ids 1 2 3 5 8
      - --task-range: 指定任务ID范围，如 --task-range 5900 6000
      - --unlabeled: 预测所有未标注的任务
    
    示例:
        # 预测指定任务ID列表
        yolo_cli labelstudio predict yolo11n.pt --url http://localhost:8080 --api-key xxx --project-id 1 --task-ids 1 2 3 5 8
        
        # 预测任务ID范围
        yolo_cli labelstudio predict best.pt --url http://localhost:8080 --api-key xxx --project-id 1 --task-range 5900 6000
        
        # 预测所有未标注任务
        yolo_cli labelstudio predict yolo11m.pt --url http://localhost:8080 --api-key xxx --project-id 1 --unlabeled
        
        # 显式指定任务类型
        yolo_cli labelstudio predict yolo11n-pose.pt --url http://localhost:8080 --api-key xxx --project-id 1 --task pose --unlabeled
    """
    print_section_header("Label Studio 任务预测")
    
    # 验证模型路径
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 验证任务筛选参数（三选一）
    filter_count = sum([
        task_ids is not None,
        task_range is not None,
        unlabeled
    ])
    
    if filter_count == 0:
        print_error("必须指定一种任务筛选方式: --task-ids, --task-range 或 --unlabeled")
        raise typer.Exit(1)
    
    if filter_count > 1:
        print_error("只能指定一种任务筛选方式")
        raise typer.Exit(1)
    
    # 验证 task_range 参数
    if task_range is not None:
        if len(task_range) != 2:
            print_error("--task-range 需要两个参数: 起始ID 结束ID")
            raise typer.Exit(1)
        if task_range[0] > task_range[1]:
            print_error("起始ID必须小于或等于结束ID")
            raise typer.Exit(1)
    
    print_info(f"模型: {model_path.name}")
    print_info(f"Label Studio: {url}")
    print_info(f"项目ID: {project_id}")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id, task_type=task or 'detect')
    except Exception as e:
        print_error(f"初始化失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    print_info("\n连接到 Label Studio...")
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 执行预测
    try:
        print_info("\n开始预测...")
        success, failed = uploader.predict_tasks_with_yolo(
            model_path=model_path,
            task_ids=task_ids,
            task_range=tuple(task_range) if task_range else None,
            unlabeled=unlabeled,
            task_type=task,
            conf=conf,
            iou=iou,
            device=device,
            max_workers=max_workers
        )
        
        print_section_header("预测完成")
        print_success(f"成功: {success} 个任务")
        if failed > 0:
            print_error(f"失败: {failed} 个任务")
        
    except Exception as e:
        print_error(f"预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _resolve_dataset_path(dataset: str) -> Path:
    """
    解析数据集路径
    
    支持:
    - 相对路径或绝对路径
    - datasets目录下的数据集名称
    """
    dataset_path = Path(dataset)
    
    # 如果是绝对路径或存在的相对路径，直接使用
    if dataset_path.is_absolute() or dataset_path.exists():
        return dataset_path
    
    # 尝试在datasets目录下查找
    config = ConfigManager()
    datasets_base = config.project_root / 'datasets'
    
    if datasets_base.exists():
        # 尝试作为数据集名称
        candidate = datasets_base / dataset
        if candidate.exists():
            return candidate
    
    # 返回原始路径（可能不存在，后续会检查）
    return dataset_path


if __name__ == "__main__":
    app()

