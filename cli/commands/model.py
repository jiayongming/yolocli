#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型管理命令"""

import typer
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO
import os

from ..core.config import ConfigManager
from ..core.version import YOLOVersionManager
from ..core.utils import (
    detect_device, get_device_name, format_size, get_file_size, ensure_dir,
    TaskType, validate_task_type, get_model_name_with_task, parse_model_name
)
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_model_list, print_section_header, create_progress_bar,
    print_key_value, console
)

app = typer.Typer(help="模型管理命令")


@app.command("download")
def download(
    version: str = typer.Option("yolo11", "--version", "-v", help="YOLO版本 (yolo11/yolov8)"),
    size: Optional[List[str]] = typer.Option(None, "--size", "-s", help="模型大小 (n/s/m/l/x)，可多选"),
    task: str = typer.Option("detect", "--task", "-t", help="任务类型 (detect/segment/classify/pose)"),
    all: bool = typer.Option(False, "--all", "-a", help="下载该版本所有模型"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
):
    """下载预训练模型"""
    
    print_section_header("下载预训练模型")
    
    try:
        # 验证任务类型
        task = validate_task_type(task)
        task_type = TaskType.from_string(task)
        
        # 标准化版本
        version = YOLOVersionManager.normalize_version(version)
        print_info(f"YOLO版本: {version}")
        print_info(f"任务类型: {task}")
        
        # 确定要下载的模型
        if all:
            base_models = YOLOVersionManager.get_all_models(version)
            # 为所有基础模型添加任务后缀
            models_to_download = [get_model_name_with_task(m, task) for m in base_models]
        elif size:
            base_models = [YOLOVersionManager.get_model_name(version, s) for s in size]
            models_to_download = [get_model_name_with_task(m, task) for m in base_models]
        else:
            # 默认下载 small 模型
            base_model = YOLOVersionManager.get_model_name(version, 's')
            models_to_download = [get_model_name_with_task(base_model, task)]
        
        # 确定输出目录
        if output_dir is None:
            config = ConfigManager()
            output_dir = config.get_path('models', absolute=True) / 'weights'
        else:
            output_dir = Path(output_dir)
        
        ensure_dir(output_dir)
        print_info(f"输出目录: {output_dir}")
        
        # 下载模型
        print_info(f"准备下载 {len(models_to_download)} 个模型...")
        
        with create_progress_bar() as progress:
            task = progress.add_task("下载模型", total=len(models_to_download))
            
            for model_name in models_to_download:
                progress.update(task, description=f"下载 {model_name}")
                
                try:
                    target_path = output_dir / model_name
                    
                    # 如果已存在，跳过
                    if target_path.exists():
                        print_warning(f"⚠ {model_name} 已存在，跳过")
                        progress.advance(task)
                        continue
                    
                    # 使用YOLO类会自动下载模型到缓存或当前目录
                    model = YOLO(model_name)
                    
                    # 查找模型文件的可能位置
                    possible_locations = [
                        Path(model_name),  # 当前目录
                        Path.home() / '.cache' / 'ultralytics' / model_name,  # YOLO缓存目录
                        Path.home() / '.ultralytics' / model_name,  # 旧版缓存目录
                    ]
                    
                    source_path = None
                    for location in possible_locations:
                        if location.exists():
                            source_path = location
                            break
                    
                    if source_path:
                        # 复制（而不是移动）到目标目录，保留缓存
                        import shutil
                        shutil.copy2(source_path, target_path)
                        print_success(f"✓ {model_name} 已下载到 {target_path}")
                        
                        # 如果是当前目录的临时文件，删除它
                        if source_path == Path(model_name):
                            source_path.unlink()
                    else:
                        print_warning(f"⚠ {model_name} 下载成功但未找到文件（可能在其他位置）")
                
                except Exception as e:
                    print_error(f"✗ {model_name} 下载失败: {e}")
                
                progress.advance(task)
        
        print_success(f"\n下载完成！模型保存在: {output_dir}")
        
    except Exception as e:
        print_error(f"下载失败: {e}")
        raise typer.Exit(1)


@app.command("export")
def export(
    model: str = typer.Argument(..., help="模型路径"),
    formats: Optional[List[str]] = typer.Option(
        ["onnx"], "--format", "-f",
        help="导出格式 (onnx/torchscript/tflite/coreml/engine等)"
    ),
    imgsz: int = typer.Option(640, "--imgsz", help="图像尺寸"),
    device: str = typer.Option("auto", "--device", "-d", help="设备 (auto/mps/cuda/cpu)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
):
    """导出模型为不同格式"""
    
    print_section_header("导出模型")
    
    # 解析模型路径（自动查找已下载的模型）
    from ..core.utils import resolve_model_path
    model, found_local = resolve_model_path(model)
    
    # 使用绝对路径以支持DDP模式
    model_path = Path(model).absolute() if Path(model).exists() else Path(model)
    if not model_path.exists():
        if found_local:
            print_error(f"模型文件损坏或不可访问: {model}")
        else:
            print_error(f"模型文件不存在: {model}")
            print_info(f"请先下载模型或提供完整的模型路径")
        raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    device_name = get_device_name(device)
    print_info(f"使用设备: {device_name}")
    print_info(f"模型: {model_path.name}")
    print_info(f"导出格式: {', '.join(formats)}")
    
    # 确定输出目录
    if output_dir is None:
        config = ConfigManager()
        output_dir = config.get_path('models', absolute=True) / 'exported'
    else:
        output_dir = Path(output_dir)
    
    ensure_dir(output_dir)
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 导出模型
        with create_progress_bar() as progress:
            task = progress.add_task("导出模型", total=len(formats))
            
            for fmt in formats:
                progress.update(task, description=f"导出为 {fmt.upper()}")
                
                try:
                    exported_path = yolo_model.export(
                        format=fmt,
                        imgsz=imgsz,
                        device=device,
                    )
                    
                    # 移动到输出目录
                    exported_file = Path(exported_path)
                    if exported_file.exists():
                        target_path = output_dir / exported_file.name
                        if target_path.exists():
                            target_path.unlink()
                        exported_file.rename(target_path)
                        
                        file_size = format_size(get_file_size(target_path))
                        print_success(f"✓ {fmt.upper()}: {target_path} ({file_size})")
                    else:
                        print_warning(f"⚠ {fmt.upper()}: 导出文件未找到")
                
                except Exception as e:
                    print_error(f"✗ {fmt.upper()}: 导出失败 - {e}")
                
                progress.advance(task)
        
        print_success(f"\n导出完成！文件保存在: {output_dir}")
        
    except Exception as e:
        print_error(f"导出失败: {e}")
        raise typer.Exit(1)


@app.command("list")
def list_models(
    directory: Optional[str] = typer.Option(None, "--dir", "-d", help="模型目录"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="筛选版本"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="筛选任务类型 (detect/segment/classify/pose/all)"),
):
    """列出本地模型"""
    
    print_section_header("本地模型列表")
    
    # 确定搜索目录
    if directory is None:
        config = ConfigManager()
        model_dir = config.get_path('models', absolute=True) / 'weights'
    else:
        model_dir = Path(directory)
    
    if not model_dir.exists():
        print_warning(f"目录不存在: {model_dir}")
        print_info("使用 'yolo-cli model download' 下载模型")
        return
    
    print_info(f"搜索目录: {model_dir}")
    
    # 查找模型文件
    model_files = list(model_dir.glob("*.pt"))
    
    if not model_files:
        print_warning("未找到模型文件")
        print_info("使用 'yolo-cli model download' 下载模型")
        return
    
    # 验证任务类型筛选
    if task and task.lower() != 'all':
        task = validate_task_type(task)
    
    # 整理模型信息，按任务类型分组
    models_by_task = {
        'detect': [],
        'segment': [],
        'classify': [],
    }
    
    for model_file in model_files:
        ver, size = YOLOVersionManager.parse_model_name(model_file.name)
        base_name, model_task = parse_model_name(model_file.name)
        
        # 版本筛选
        if version and ver != YOLOVersionManager.normalize_version(version):
            continue
        
        # 任务类型筛选
        if task and task.lower() != 'all' and model_task != task:
            continue
        
        model_info = {
            'name': model_file.name,
            'version': ver or 'Unknown',
            'size': size or 'Unknown',
            'task': model_task,
            'params': YOLOVersionManager.get_model_info(size)['params'] if size else 'N/A',
            'file_size': format_size(get_file_size(model_file)),
            'path': str(model_file),
        }
        models_by_task[model_task].append(model_info)
    
    # 计算总数
    total_models = sum(len(models) for models in models_by_task.values())
    
    if total_models == 0:
        print_warning(f"未找到匹配的模型")
        return
    
    # 按任务类型打印模型列表
    for task_name, models in models_by_task.items():
        if models:
            console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
            console.print(f"[bold cyan]{task_name.upper()} 模型[/bold cyan]")
            console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
            print_model_list(models)
    
    print_info(f"\n共找到 {total_models} 个模型")


@app.command("info")
def model_info(
    model: str = typer.Argument(..., help="模型路径或名称"),
):
    """显示模型详细信息"""
    
    print_section_header("模型信息")
    
    # 解析模型路径（自动查找已下载的模型）
    from ..core.utils import resolve_model_path
    model, found_local = resolve_model_path(model)
    
    model_path = Path(model)
    if not model_path.exists():
        if found_local:
            print_error(f"模型文件损坏或不可访问: {model}")
        else:
            print_error(f"模型不存在: {model}")
            print_info(f"请先下载模型或提供完整的模型路径")
        raise typer.Exit(1)
    
    # 解析模型信息
    version, size = YOLOVersionManager.parse_model_name(model_path.name)
    base_name, model_task = parse_model_name(model_path.name)
    
    print_key_value("模型名称", model_path.name)
    print_key_value("完整路径", str(model_path))
    print_key_value("文件大小", format_size(get_file_size(model_path)))
    print_key_value("任务类型", model_task.upper())
    
    if version:
        print_key_value("YOLO版本", version)
    
    if size:
        size_info = YOLOVersionManager.get_model_info(size)
        print_key_value("模型大小", f"{size.upper()} - {size_info['name']}")
        print_key_value("参数量", size_info['params'])
        print_key_value("速度", size_info['speed'])
        print_key_value("描述", size_info['description'])
    
    # 尝试加载模型获取更多信息
    try:
        print_info("\n加载模型获取详细信息...")
        yolo_model = YOLO(str(model_path))
        
        if hasattr(yolo_model, 'names') and yolo_model.names:
            print_key_value("类别数量", len(yolo_model.names))
            print_key_value("类别名称", ", ".join(yolo_model.names.values()))
    
    except Exception as e:
        print_warning(f"无法加载模型详细信息: {e}")


if __name__ == "__main__":
    app()
