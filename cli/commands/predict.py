#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测/推理命令（支持检测、分割、分类）"""

import typer
from pathlib import Path
from typing import Optional, List
import json
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import shutil

from ..core.config import ConfigManager
from ..core.utils import (
    detect_device, get_device_name, ensure_dir, find_files,
    TaskType, validate_task_type, parse_model_name
)
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_detection_results, print_key_value,
    create_progress_bar, console
)

app = typer.Typer(help="预测/推理命令（检测、分割、分类）")


def generate_run_id() -> str:
    """生成唯一的运行ID（时间戳格式）
    
    Returns:
        str: 格式为 YYYYMMDD_HHMMSS 的唯一ID
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_classes_file(model, output_dir: Path):
    """保存类别列表到文件
    
    Args:
        model: YOLO模型对象
        output_dir: 输出目录
    """
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for idx in sorted(model.names.keys()):
            f.write(f"{model.names[idx]}\n")


def organize_prediction_results(yolo_output_dir: Path, organized_dir: Path):
    """整理YOLO预测结果到规范的目录结构
    
    Args:
        yolo_output_dir: YOLO原始输出目录
        organized_dir: 整理后的目标目录（包含images和labels子目录）
    """
    # 创建目标目录结构
    images_dir = organized_dir / 'images'
    labels_dir = organized_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 移动图片文件（检测结果图片）
    for img_file in yolo_output_dir.glob('*.jpg'):
        shutil.move(str(img_file), str(images_dir / img_file.name))
    for img_file in yolo_output_dir.glob('*.jpeg'):
        shutil.move(str(img_file), str(images_dir / img_file.name))
    for img_file in yolo_output_dir.glob('*.png'):
        shutil.move(str(img_file), str(images_dir / img_file.name))
    
    # 移动labels目录
    yolo_labels_dir = yolo_output_dir / 'labels'
    if yolo_labels_dir.exists():
        for label_file in yolo_labels_dir.glob('*.txt'):
            shutil.move(str(label_file), str(labels_dir / label_file.name))
        # 删除空的labels目录
        yolo_labels_dir.rmdir()
    
    # 如果YOLO输出目录为空，删除它
    if not any(yolo_output_dir.iterdir()):
        yolo_output_dir.rmdir()


@app.command("image")
def predict_image(
    model: str = typer.Argument(..., help="模型路径"),
    image: str = typer.Argument(..., help="图片路径"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务类型（自动从模型推断）"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(True, "--save-txt/--no-txt", help="保存TXT结果"),
    save_json: bool = typer.Option(True, "--save-json/--no-json", help="保存JSON结果"),
    show: bool = typer.Option(False, "--show", "-s", help="显示结果"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
    top_k: int = typer.Option(5, "--top-k", help="[分类] Top-K预测结果"),
):
    """单张图片预测（检测/分割/分类）"""
    
    print_section_header("图片预测")
    
    # 验证文件存在
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    image_path = Path(image)
    if not image_path.exists():
        print_error(f"图片不存在: {image}")
        raise typer.Exit(1)
    
    # 推断任务类型
    if task is None:
        _, task = parse_model_name(model_path.name)
    else:
        task = validate_task_type(task)
    
    task_type = TaskType.from_string(task)
    
    # 确定输出目录（使用唯一ID）
    if output is None:
        config = ConfigManager()
        run_id = generate_run_id()
        output_base = config.get_path('results', absolute=True) / 'predictions'
        output_dir = output_base / run_id
    else:
        output_dir = Path(output)
    
    ensure_dir(output_dir)
    
    # 创建临时YOLO输出目录
    yolo_temp_dir = output_dir / '_yolo_temp'
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"任务类型: {task.upper()}")
    print_info(f"模型: {model_path.name}")
    print_info(f"图片: {image_path.name}")
    if task != 'classify':
        print_info(f"置信度阈值: {conf}")
    print_info(f"设备: {get_device_name(device)}")
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 执行预测
        print_info(f"执行{task}预测...")
        
        predict_kwargs = {
            'source': str(image_path),
            'save': True,
            'project': str(yolo_temp_dir),
            'name': 'run',
            'device': device,
            'show': show,
            'exist_ok': True,
        }
        
        # 根据任务类型添加特定参数
        if task_type != TaskType.CLASSIFY:
            predict_kwargs['conf'] = conf
            predict_kwargs['iou'] = iou
            predict_kwargs['save_txt'] = save_txt
            predict_kwargs['save_conf'] = True
        
        results = yolo_model.predict(**predict_kwargs)
        
        # 处理结果（根据任务类型）
        predictions = []
        
        if task_type == TaskType.DETECT:
            # 检测任务
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    prediction = {
                        'class': int(box.cls[0]),
                        'class_name': yolo_model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                    }
                    predictions.append(prediction)
        
        elif task_type == TaskType.SEGMENT:
            # 分割任务
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks
                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        prediction = {
                            'class': int(box.cls[0]),
                            'class_name': yolo_model.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist(),
                            'mask_area': float(mask.data.sum()) if hasattr(mask, 'data') else 0,
                        }
                        predictions.append(prediction)
        
        elif task_type == TaskType.CLASSIFY:
            # 分类任务
            for result in results:
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top_indices = probs.top5  # Top5 indices
                    top_conf = probs.top5conf  # Top5 confidences
                    
                    for idx, conf_val in zip(top_indices[:top_k], top_conf[:top_k]):
                        prediction = {
                            'class': int(idx),
                            'class_name': yolo_model.names[int(idx)],
                            'confidence': float(conf_val),
                        }
                        predictions.append(prediction)
        
        elif task_type == TaskType.POSE:
            # 姿势估计任务
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()  # 关键点坐标
                    keypoints_conf = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
                    boxes = result.boxes
                    
                    for idx, (kp, box) in enumerate(zip(keypoints, boxes)):
                        prediction = {
                            'class': int(box.cls[0]),
                            'class_name': yolo_model.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist(),
                            'keypoints': kp.tolist(),
                            'keypoint_scores': keypoints_conf[idx].tolist() if keypoints_conf is not None else None,
                        }
                        predictions.append(prediction)
        
        # 整理YOLO输出结果到规范目录结构
        yolo_run_dir = yolo_temp_dir / 'run'
        if yolo_run_dir.exists():
            organize_prediction_results(yolo_run_dir, output_dir)
            # 清理临时目录
            if yolo_temp_dir.exists():
                shutil.rmtree(yolo_temp_dir)
        
        # 保存类别列表
        save_classes_file(yolo_model, output_dir)
        
        # 保存JSON结果
        if save_json:
            json_file = output_dir / f"{image_path.stem}_results.json"
            result_data = {
                'task': task,
                'image': str(image_path),
                'predictions': predictions,
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"JSON结果已保存: {json_file}")
        
        # 显示结果
        console.print()
        if task_type == TaskType.CLASSIFY:
            console.print(f"[bold cyan]分类结果 (Top-{top_k}):[/bold cyan]")
            for i, pred in enumerate(predictions, 1):
                print_info(f"  {i}. {pred['class_name']}: {pred['confidence']:.4f}")
        else:
            print_detection_results(predictions, image_path.name)
        
        if predictions:
            result_word = "类别" if task == 'classify' else "目标"
            print_success(f"{task}预测完成！共{len(predictions)} 个{result_word}")
        else:
            print_warning(f"未{task}到结果")
        
        print_info(f"结果保存在: {output_dir}")
        print_info(f"  - 图片: {output_dir / 'images'}")
        if save_txt:
            print_info(f"  - 标签: {output_dir / 'labels'}")
        
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
    
    # 确定输出目录（使用唯一ID）
    if output is None:
        config = ConfigManager()
        run_id = generate_run_id()
        output_base = config.get_path('results', absolute=True) / 'predictions'
        output_dir = output_base / run_id
    else:
        output_dir = Path(output)
    
    ensure_dir(output_dir)
    
    # 创建临时YOLO输出目录
    yolo_temp_dir = output_dir / '_yolo_temp'
    
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
            project=str(yolo_temp_dir),
            name='run',
            device=device,
            stream=True,
            batch=batch,
            exist_ok=True,
        )
        
        # 处理结果
        all_detections = {}
        total_objects = 0
        
        with create_progress_bar() as progress:
            task = progress.add_task("检测进度", total=None)
            
            for i, result in enumerate(results):
                img_name = Path(result.path).name if hasattr(result, 'path') else f"image_{i}"
                
                detections = []
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'class': int(box.cls[0]),
                        'class_name': yolo_model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                    }
                    detections.append(detection)
                
                all_detections[img_name] = detections
                total_objects += len(detections)
                
                progress.update(task, advance=1, description=f"已处理 {i+1} 张")
        
        # 整理YOLO输出结果到规范目录结构
        yolo_run_dir = yolo_temp_dir / 'run'
        if yolo_run_dir.exists():
            organize_prediction_results(yolo_run_dir, output_dir)
            # 清理临时目录
            if yolo_temp_dir.exists():
                shutil.rmtree(yolo_temp_dir)
        
        # 保存类别列表
        save_classes_file(yolo_model, output_dir)
        
        # 保存JSON结果
        if save_json:
            json_file = output_dir / 'detections.json'
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_detections, f, indent=2, ensure_ascii=False)
            
            print_success(f"JSON结果已保存: {json_file}")
        
        # 显示统计
        console.print()
        print_success(f"批量检测完成！")
        print_key_value("处理图片数", len(all_detections))
        print_key_value("检测目标总数", total_objects)
        print_key_value("平均每张", f"{total_objects / len(all_detections):.2f}" if all_detections else "0")
        print_info(f"结果保存在: {output_dir}")
        print_info(f"  - 图片: {output_dir / 'images'}")
        if save_txt:
            print_info(f"  - 标签: {output_dir / 'labels'}")
        
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
    
    # 确定输出目录（使用唯一ID）
    if output is None:
        config = ConfigManager()
        run_id = generate_run_id()
        output_base = config.get_path('results', absolute=True) / 'predictions'
        output_dir = output_base / run_id
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
