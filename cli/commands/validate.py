#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型验证命令"""

import typer
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
from ultralytics import YOLO
import yaml
import numpy as np

from ..core.utils import (
    detect_device, get_device_name, ensure_dir,
    TaskType, validate_task_type, parse_model_name
)
from ..ui.display import (
    print_success, print_error, print_info, print_warning,
    print_section_header, print_key_value, console
)
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="模型验证命令")


def safe_float(value):
    """
    安全地将numpy数组或标量转换为float
    
    Args:
        value: 任意类型的值（numpy数组、标量、None等）
    
    Returns:
        float: 转换后的浮点数
    """
    if value is None:
        return 0.0
    # 处理numpy数组
    if isinstance(value, np.ndarray):
        return float(value.mean()) if len(value) > 0 else 0.0
    # 处理numpy标量类型（numpy.float64, numpy.int32等）
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except (TypeError, ValueError, AttributeError):
            pass
    # 处理普通Python类型
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@app.command("run")
def validate_model(
    model: str = typer.Argument(..., help="模型路径"),
    data: str = typer.Option("data/dataset.yaml", "--data", "-d", help="数据集配置文件"),
    split: str = typer.Option("val", "--split", help="验证数据集 (val/test/train)"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务类型（自动从模型推断）"),
    batch: int = typer.Option(16, "--batch", "-b", help="批次大小"),
    imgsz: int = typer.Option(640, "--imgsz", help="图像尺寸"),
    conf: float = typer.Option(0.001, "--conf", help="置信度阈值"),
    iou: float = typer.Option(0.6, "--iou", help="IoU阈值"),
    device: str = typer.Option("auto", "--device", help="设备 (auto/mps/cuda/cpu)"),
    save_json: bool = typer.Option(True, "--save-json/--no-save-json", help="保存JSON格式结果"),
    save_hybrid: bool = typer.Option(False, "--save-hybrid/--no-save-hybrid", help="保存混合标签"),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="生成可视化图表"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="详细输出"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="结果保存目录"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="验证实验名称"),
):
    """
    验证模型在数据集上的性能
    
    支持检测(detect)、分割(segment)和分类(classify)三种任务类型。
    会计算并显示 mAP、精确率、召回率等关键指标。
    
    示例:
    
    \b
      # 基本验证
      yolo-cli validate run models/best.pt
      
    \b
      # 指定数据集和置信度阈值
      yolo-cli validate run models/best.pt --data data/dataset.yaml --conf 0.25
      
    \b
      # 在测试集上验证
      yolo-cli validate run models/best.pt --split test
      
    \b
      # 保存详细结果
      yolo-cli validate run models/best.pt --save-json --plots --project results/validation
    """
    
    print_section_header("模型验证")
    
    # 验证模型文件
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 推断或验证任务类型
    if task is None:
        _, task = parse_model_name(model_path.name)
        if task is None:
            print_warning("无法从模型名称推断任务类型，使用默认值: detect")
            task = "detect"
    else:
        task = validate_task_type(task)
    
    task_type = TaskType.from_string(task)
    print_info(f"任务类型: {task.upper()}")
    
    # 处理数据集路径
    data_path = Path(data)
    
    if task_type == TaskType.CLASSIFY:
        # 分类任务需要目录路径
        if data_path.is_file() and data_path.suffix in ['.yaml', '.yml']:
            with open(data_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            
            if 'path' not in yaml_content:
                print_error("dataset.yaml 中缺少 'path' 字段")
                raise typer.Exit(1)
            
            dataset_root = Path(yaml_content['path'])
            images_dir = dataset_root / 'images'
            if images_dir.exists():
                data = str(images_dir)
            else:
                data = str(dataset_root)
            print_info(f"分类任务使用数据集目录: {data}")
        elif data_path.is_dir():
            data = str(data_path)
        else:
            print_error(f"分类任务需要数据集目录或 dataset.yaml 文件")
            raise typer.Exit(1)
    else:
        # 检测/分割任务需要 yaml 文件
        if not data_path.exists():
            print_error(f"数据集配置文件不存在: {data}")
            raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    device_name = get_device_name(device)
    
    # 设置结果保存路径
    if project is None:
        from ..core.config import ConfigManager
        config = ConfigManager()
        project = str(config.get_path('results', absolute=True) / 'validation')
    
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_path.stem
        name = f"{model_name}_{timestamp}"
    
    # 显示验证配置
    console.print()
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("参数", style="cyan")
    config_table.add_column("值", style="green")
    
    config_table.add_row("模型", model_path.name)
    config_table.add_row("数据集", data)
    config_table.add_row("验证集", split)
    config_table.add_row("任务类型", task.upper())
    config_table.add_row("批次大小", str(batch))
    config_table.add_row("图像尺寸", str(imgsz))
    config_table.add_row("置信度阈值", str(conf))
    config_table.add_row("IoU阈值", str(iou))
    config_table.add_row("设备", device_name)
    config_table.add_row("保存目录", f"{project}/{name}")
    
    console.print(config_table)
    console.print()
    
    try:
        # 加载模型
        print_info("加载模型...")
        yolo_model = YOLO(str(model_path))
        
        # 开始验证
        print_info("开始验证...")
        console.print()
        
        validation_kwargs = {
            'data': data,
            'split': split,
            'batch': batch,
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou,
            'device': device,
            'save_json': save_json,
            'save_hybrid': save_hybrid,
            'plots': plots,
            'verbose': verbose,
            'project': project,
            'name': name,
        }
        
        results = yolo_model.val(**validation_kwargs)
        
        console.print()
        print_success("验证完成！")
        console.print()
        
        # 显示结果
        _display_validation_results(results, task_type, model_path.name)
        
        # 保存结果摘要
        if save_json:
            summary = _generate_results_summary(results, task_type, model_path, data, split, conf, iou)
            summary_path = Path(project) / name / "validation_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print_info(f"验证摘要已保存: {summary_path}")
        
        # 显示结果文件位置
        result_dir = Path(project) / name
        if result_dir.exists():
            print_info(f"结果目录: {result_dir.absolute()}")
            
            if plots:
                print_info("📊 可视化图表已生成")
        
    except KeyboardInterrupt:
        print_warning("\n验证被用户中断")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"验证失败: {e}")
        import traceback
        if verbose:
            console.print(traceback.format_exc())
        raise typer.Exit(1)


def _display_validation_results(results, task_type: TaskType, model_name: str):
    """显示验证结果"""
    
    # 根据任务类型显示不同的指标
    if task_type == TaskType.DETECT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            
            # 创建指标表格
            metrics_table = Table(title="🎯 检测指标", show_header=True, header_style="bold cyan")
            metrics_table.add_column("指标", style="cyan", width=20)
            metrics_table.add_column("值", style="green", justify="right", width=15)
            
            metrics_table.add_row("mAP@0.5", f"{safe_float(box_metrics.map50):.4f}")
            metrics_table.add_row("mAP@0.5:0.95", f"{safe_float(box_metrics.map):.4f}")
            metrics_table.add_row("精确率 (Precision)", f"{safe_float(box_metrics.mp):.4f}")
            metrics_table.add_row("召回率 (Recall)", f"{safe_float(box_metrics.mr):.4f}")
            
            # F1分数（如果存在）
            if hasattr(box_metrics, 'f1') and box_metrics.f1 is not None:
                metrics_table.add_row("F1分数", f"{safe_float(box_metrics.f1):.4f}")
            
            console.print(metrics_table)
            
            # 显示每个类别的结果
            if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > 0:
                console.print()
                
                class_table = Table(title="📋 各类别指标 (AP@0.5)", show_header=True, header_style="bold magenta")
                class_table.add_column("类别", style="cyan")
                class_table.add_column("AP@0.5", style="green", justify="right")
                
                # 获取类别名称
                if hasattr(results, 'names'):
                    class_names = [results.names[i] for i in range(len(box_metrics.ap50))]
                else:
                    class_names = [f"class_{i}" for i in range(len(box_metrics.ap50))]
                
                for class_name, ap50 in zip(class_names, box_metrics.ap50):
                    # 使用safe_float确保正确转换
                    class_table.add_row(class_name, f"{safe_float(ap50):.4f}")
                
                console.print(class_table)
    
    elif task_type == TaskType.SEGMENT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            mask_metrics = results.seg if hasattr(results, 'seg') else None
            
            # 创建指标表格
            metrics_table = Table(title="🎯 分割指标", show_header=True, header_style="bold cyan")
            metrics_table.add_column("指标类型", style="cyan", width=15)
            metrics_table.add_column("指标", style="yellow", width=20)
            metrics_table.add_column("值", style="green", justify="right", width=15)
            
            # 边界框指标
            metrics_table.add_row("边界框", "mAP@0.5", f"{safe_float(box_metrics.map50):.4f}")
            metrics_table.add_row("", "mAP@0.5:0.95", f"{safe_float(box_metrics.map):.4f}")
            metrics_table.add_row("", "Precision", f"{safe_float(box_metrics.mp):.4f}")
            metrics_table.add_row("", "Recall", f"{safe_float(box_metrics.mr):.4f}")
            
            # 掩码指标
            if mask_metrics:
                metrics_table.add_row("", "", "")
                metrics_table.add_row("掩码", "mAP@0.5", f"{safe_float(mask_metrics.map50):.4f}")
                metrics_table.add_row("", "mAP@0.5:0.95", f"{safe_float(mask_metrics.map):.4f}")
                metrics_table.add_row("", "Precision", f"{safe_float(mask_metrics.mp):.4f}")
                metrics_table.add_row("", "Recall", f"{safe_float(mask_metrics.mr):.4f}")
            
            console.print(metrics_table)
    
    elif task_type == TaskType.CLASSIFY:
        if hasattr(results, 'top1') and hasattr(results, 'top5'):
            # 分类指标
            metrics_table = Table(title="🎯 分类指标", show_header=True, header_style="bold cyan")
            metrics_table.add_column("指标", style="cyan", width=20)
            metrics_table.add_column("值", style="green", justify="right", width=15)
            
            metrics_table.add_row("Top-1 准确率", f"{safe_float(results.top1):.4f}")
            metrics_table.add_row("Top-5 准确率", f"{safe_float(results.top5):.4f}")
            
            console.print(metrics_table)


def _generate_results_summary(results, task_type: TaskType, model_path: Path, 
                              data: str, split: str, conf: float, iou: float) -> dict:
    """生成验证结果摘要"""
    
    summary = {
        'model': str(model_path),
        'model_name': model_path.name,
        'task': task_type.value,
        'dataset': data,
        'split': split,
        'conf_threshold': conf,
        'iou_threshold': iou,
        'timestamp': datetime.now().isoformat(),
    }
    
    # 根据任务类型添加指标
    if task_type == TaskType.DETECT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            summary['metrics'] = {
                'mAP50': safe_float(box_metrics.map50),
                'mAP50_95': safe_float(box_metrics.map),
                'precision': safe_float(box_metrics.mp),
                'recall': safe_float(box_metrics.mr),
            }
            
            if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > 0:
                if hasattr(results, 'names'):
                    class_names = [results.names[i] for i in range(len(box_metrics.ap50))]
                else:
                    class_names = [f"class_{i}" for i in range(len(box_metrics.ap50))]
                
                # 确保AP值被正确转换为Python float
                summary['per_class'] = {
                    name: safe_float(ap) 
                    for name, ap in zip(class_names, box_metrics.ap50)
                }
    
    elif task_type == TaskType.SEGMENT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            mask_metrics = results.seg if hasattr(results, 'seg') else None
            
            summary['metrics'] = {
                'box': {
                    'mAP50': safe_float(box_metrics.map50),
                    'mAP50_95': safe_float(box_metrics.map),
                    'precision': safe_float(box_metrics.mp),
                    'recall': safe_float(box_metrics.mr),
                }
            }
            
            if mask_metrics:
                summary['metrics']['mask'] = {
                    'mAP50': safe_float(mask_metrics.map50),
                    'mAP50_95': safe_float(mask_metrics.map),
                    'precision': safe_float(mask_metrics.mp),
                    'recall': safe_float(mask_metrics.mr),
                }
    
    elif task_type == TaskType.CLASSIFY:
        if hasattr(results, 'top1') and hasattr(results, 'top5'):
            summary['metrics'] = {
                'top1_accuracy': safe_float(results.top1),
                'top5_accuracy': safe_float(results.top5),
            }
    
    return summary


@app.command("compare")
def compare_models(
    models: str = typer.Argument(..., help="模型路径，用逗号分隔"),
    data: str = typer.Option("data/dataset.yaml", "--data", "-d", help="数据集配置文件"),
    split: str = typer.Option("val", "--split", help="验证数据集"),
    conf: float = typer.Option(0.001, "--conf", help="置信度阈值"),
    iou: float = typer.Option(0.6, "--iou", help="IoU阈值"),
    device: str = typer.Option("auto", "--device", help="设备"),
):
    """
    比较多个模型的性能
    
    示例:
    
    \b
      yolo-cli validate compare model1.pt,model2.pt,model3.pt --data data/dataset.yaml
    """
    
    print_section_header("模型性能比较")
    
    # 解析模型路径
    model_paths = [Path(m.strip()) for m in models.split(',')]
    
    # 验证模型文件
    for model_path in model_paths:
        if not model_path.exists():
            print_error(f"模型不存在: {model_path}")
            raise typer.Exit(1)
    
    # 验证数据集
    data_path = Path(data)
    if not data_path.exists():
        print_error(f"数据集配置文件不存在: {data}")
        raise typer.Exit(1)
    
    # 自动检测设备
    if device == 'auto':
        device = detect_device()
    
    print_info(f"将比较 {len(model_paths)} 个模型")
    print_info(f"数据集: {data}")
    print_info(f"设备: {get_device_name(device)}")
    console.print()
    
    # 存储所有结果
    all_results = []
    
    # 验证每个模型
    for i, model_path in enumerate(model_paths, 1):
        print_info(f"[{i}/{len(model_paths)}] 验证模型: {model_path.name}")
        
        try:
            yolo_model = YOLO(str(model_path))
            results = yolo_model.val(
                data=str(data_path),
                split=split,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
                plots=False,
            )
            
            all_results.append({
                'name': model_path.name,
                'path': str(model_path),
                'results': results,
            })
            
            print_success(f"✓ {model_path.name} 验证完成")
            
        except Exception as e:
            print_error(f"✗ {model_path.name} 验证失败: {e}")
            continue
    
    if not all_results:
        print_error("没有模型验证成功")
        raise typer.Exit(1)
    
    # 显示比较表格
    console.print()
    print_section_header("性能比较结果")
    
    comparison_table = Table(show_header=True, header_style="bold cyan")
    comparison_table.add_column("模型", style="cyan")
    comparison_table.add_column("mAP@0.5", style="green", justify="right")
    comparison_table.add_column("mAP@0.5:0.95", style="green", justify="right")
    comparison_table.add_column("Precision", style="yellow", justify="right")
    comparison_table.add_column("Recall", style="yellow", justify="right")
    
    best_map50 = -1
    best_model = None
    
    for result in all_results:
        if hasattr(result['results'], 'box') and result['results'].box:
            box_metrics = result['results'].box
            map50 = safe_float(box_metrics.map50)
            
            if map50 > best_map50:
                best_map50 = map50
                best_model = result['name']
            
            comparison_table.add_row(
                result['name'],
                f"{safe_float(box_metrics.map50):.4f}",
                f"{safe_float(box_metrics.map):.4f}",
                f"{safe_float(box_metrics.mp):.4f}",
                f"{safe_float(box_metrics.mr):.4f}",
            )
    
    console.print(comparison_table)
    console.print()
    
    if best_model:
        print_success(f"🏆 最佳模型: {best_model} (mAP@0.5: {best_map50:.4f})")


if __name__ == "__main__":
    app()

