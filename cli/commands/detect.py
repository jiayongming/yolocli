#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测命令"""

import typer
from pathlib import Path
from typing import Optional, List
import json
from ultralytics import YOLO

from ..core.config import ConfigManager
from ..core.utils import detect_device, get_device_name, ensure_dir, find_files
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_detection_results, print_key_value,
    create_progress_bar, console
)

app = typer.Typer(help="检测命令")


@app.command("image")
def detect_image(
    model: str = typer.Argument(..., help="模型路径"),
    image: str = typer.Argument(..., help="图片路径"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(True, "--save-txt/--no-txt", help="保存TXT结果"),
    save_json: bool = typer.Option(True, "--save-json/--no-json", help="保存JSON结果"),
    show: bool = typer.Option(False, "--show", "-s", help="显示结果"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
):
    """单张图片检测"""
    
    print_section_header("图片检测")
    
    # 验证文件存在
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    image_path = Path(image)
    if not image_path.exists():
        print_error(f"图片不存在: {image}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output is None:
        config = ConfigManager()
        output_dir = config.get_path('results', absolute=True) / 'predictions'
    else:
        output_dir = Path(output)
    
    ensure_dir(output_dir)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"模型: {model_path.name}")
    print_info(f"图片: {image_path.name}")
    print_info(f"置信度阈值: {conf}")
    print_info(f"设备: {get_device_name(device)}")
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 执行检测
        print_info("执行检测...")
        results = yolo_model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            save=True,
            save_txt=save_txt,
            save_conf=True,
            project=str(output_dir),
            name='single_image',
            device=device,
            show=show,
        )
        
        # 处理结果
        detections = []
        for result in results:
            # 检测任务类型：分类任务有 probs 属性，检测/分割任务有 boxes 属性
            if hasattr(result, 'probs') and result.probs is not None:
                # 分类任务
                probs = result.probs
                top5_indices = probs.top5
                top5_conf = probs.top5conf.tolist()
                
                for idx, conf_val in zip(top5_indices, top5_conf):
                    detection = {
                        'class': int(idx),
                        'class_name': yolo_model.names[int(idx)],
                        'confidence': float(conf_val),
                        'type': 'classification'
                    }
                    detections.append(detection)
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # 检测/分割任务
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'class': int(box.cls[0]),
                        'class_name': yolo_model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                        'type': 'detection'
                    }
                    detections.append(detection)
        
        # 保存JSON结果
        if save_json:
            output_path = output_dir / 'single_image'
            output_path.mkdir(parents=True, exist_ok=True)
            
            json_file = output_path / f"{image_path.stem}_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(detections, f, indent=2, ensure_ascii=False)
            
            print_success(f"JSON结果已保存: {json_file}")
        
        # 显示检测结果
        if detections:
            console.print()
            print_success(f"检测完成！")
            console.print()
            
            if detections[0].get('type') == 'classification':
                # 分类结果显示
                print_info("分类结果（Top-5）:")
                for i, det in enumerate(detections, 1):
                    console.print(f"  {i}. {det['class_name']}: {det['confidence']:.4f}")
            else:
                # 检测结果显示
                print_detection_results(detections, image_path.name)
                print_success(f"共检测到 {len(detections)} 个目标")
        else:
            print_warning("未检测到目标")
        
        print_info(f"结果保存在: {output_dir / 'single_image'}")
        
    except Exception as e:
        print_error(f"检测失败: {e}")
        raise typer.Exit(1)


@app.command("batch")
def detect_batch(
    model: str = typer.Argument(..., help="模型路径"),
    source: str = typer.Argument(..., help="图片目录或视频文件"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(True, "--save-txt/--no-txt", help="保存TXT结果"),
    save_json: bool = typer.Option(True, "--save-json/--no-json", help="保存JSON结果"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
    batch: int = typer.Option(1, "--batch", "-b", help="批次大小"),
):
    """批量检测"""
    
    print_section_header("批量检测")
    
    # 验证模型
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 验证源
    source_path = Path(source)
    if not source_path.exists():
        print_error(f"源路径不存在: {source}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output is None:
        config = ConfigManager()
        output_dir = config.get_path('results', absolute=True) / 'predictions'
    else:
        output_dir = Path(output)
    
    ensure_dir(output_dir)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"模型: {model_path.name}")
    print_info(f"源: {source_path}")
    print_info(f"置信度阈值: {conf}")
    print_info(f"设备: {get_device_name(device)}")
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 如果是目录，统计图片数量
        if source_path.is_dir():
            images = find_files(source_path, ['.jpg', '.jpeg', '.png', '.bmp'])
            print_info(f"找到 {len(images)} 张图片")
        else:
            print_info("检测视频文件...")
        
        # 执行检测
        print_info("开始检测...")
        results = yolo_model.predict(
            source=str(source_path),
            conf=conf,
            iou=iou,
            save=True,
            save_txt=save_txt,
            save_conf=True,
            project=str(output_dir),
            name='batch',
            device=device,
            stream=True,
            batch=batch,
        )
        
        # 处理结果
        all_detections = {}
        total_objects = 0
        
        with create_progress_bar() as progress:
            task = progress.add_task("检测进度", total=None)
            
            for i, result in enumerate(results):
                img_name = Path(result.path).name if hasattr(result, 'path') else f"image_{i}"
                
                detections = []
                
                # 检测任务类型
                if hasattr(result, 'probs') and result.probs is not None:
                    # 分类任务
                    probs = result.probs
                    top5_indices = probs.top5
                    top5_conf = probs.top5conf.tolist()
                    
                    for idx, conf_val in zip(top5_indices, top5_conf):
                        detection = {
                            'class': int(idx),
                            'class_name': yolo_model.names[int(idx)],
                            'confidence': float(conf_val),
                            'type': 'classification'
                        }
                        detections.append(detection)
                    total_objects += 1  # 分类任务计为1个结果
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # 检测/分割任务
                    boxes = result.boxes
                    for box in boxes:
                        detection = {
                            'class': int(box.cls[0]),
                            'class_name': yolo_model.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist(),
                            'type': 'detection'
                        }
                        detections.append(detection)
                    total_objects += len(detections)
                
                all_detections[img_name] = detections
                
                progress.update(task, advance=1, description=f"已处理 {i+1} 张")
        
        # 保存JSON结果
        if save_json:
            json_file = output_dir / 'batch' / 'detections.json'
            ensure_dir(json_file.parent)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_detections, f, indent=2, ensure_ascii=False)
            
            print_success(f"JSON结果已保存: {json_file}")
        
        # 显示统计
        console.print()
        print_success(f"批量检测完成！")
        print_key_value("处理图片数", len(all_detections))
        print_key_value("检测目标总数", total_objects)
        print_key_value("平均每张", f"{total_objects / len(all_detections):.2f}" if all_detections else "0")
        print_info(f"结果保存在: {output_dir / 'batch'}")
        
    except Exception as e:
        print_error(f"批量检测失败: {e}")
        raise typer.Exit(1)


@app.command("video")
def detect_video(
    model: str = typer.Argument(..., help="模型路径"),
    video: str = typer.Argument(..., help="视频文件路径"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(False, "--save-txt", help="保存TXT结果"),
    show: bool = typer.Option(False, "--show", "-s", help="实时显示"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
):
    """视频检测"""
    
    print_section_header("视频检测")
    
    # 验证模型
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 验证视频
    video_path = Path(video)
    if not video_path.exists():
        print_error(f"视频不存在: {video}")
        raise typer.Exit(1)
    
    # 确定输出目录
    if output is None:
        config = ConfigManager()
        output_dir = config.get_path('results', absolute=True) / 'predictions'
    else:
        output_dir = Path(output)
    
    ensure_dir(output_dir)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"模型: {model_path.name}")
    print_info(f"视频: {video_path.name}")
    print_info(f"置信度阈值: {conf}")
    print_info(f"设备: {get_device_name(device)}")
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 执行检测
        print_info("开始检测视频...")
        if show:
            print_info("按 'q' 退出实时预览")
        
        results = yolo_model.predict(
            source=str(video_path),
            conf=conf,
            iou=iou,
            save=True,
            save_txt=save_txt,
            project=str(output_dir),
            name='video',
            device=device,
            stream=True,
            show=show,
        )
        
        # 处理结果
        frame_count = 0
        total_objects = 0
        
        for result in results:
            frame_count += 1
            total_objects += len(result.boxes)
        
        # 显示统计
        console.print()
        print_success(f"视频检测完成！")
        print_key_value("总帧数", frame_count)
        print_key_value("检测目标总数", total_objects)
        print_key_value("平均每帧", f"{total_objects / frame_count:.2f}" if frame_count > 0 else "0")
        print_info(f"结果保存在: {output_dir / 'video'}")
        
    except KeyboardInterrupt:
        print_warning("\n检测被用户中断")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"视频检测失败: {e}")
        raise typer.Exit(1)


@app.command("webcam")
def detect_webcam(
    model: str = typer.Argument(..., help="模型路径"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
    camera: int = typer.Option(0, "--camera", help="摄像头ID"),
):
    """实时摄像头检测"""
    
    print_section_header("摄像头检测")
    
    # 验证模型
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"模型: {model_path.name}")
    print_info(f"摄像头: {camera}")
    print_info(f"置信度阈值: {conf}")
    print_info(f"设备: {get_device_name(device)}")
    print_info("按 'q' 退出")
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 执行检测
        print_info("启动摄像头...")
        results = yolo_model.predict(
            source=camera,
            conf=conf,
            iou=iou,
            device=device,
            stream=True,
            show=True,
        )
        
        # 处理结果
        for result in results:
            pass  # 实时显示由YOLO内部处理
        
    except KeyboardInterrupt:
        print_warning("\n检测被用户中断")
    except Exception as e:
        print_error(f"摄像头检测失败: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
