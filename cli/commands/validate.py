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
    
    # 只有检测和分割任务才显示 conf 和 iou 阈值
    if task_type in [TaskType.DETECT, TaskType.SEGMENT]:
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
        
        # 从模型本身检测真实的任务类型（优先于文件名推断）
        if hasattr(yolo_model, 'task'):
            model_task = yolo_model.task
            # 映射 YOLO 内部任务名称到我们的任务类型
            task_mapping = {
                'classify': 'classify',
                'detect': 'detect',
                'segment': 'segment',
                'pose': 'detect',  # 姿态估计也归类为检测
                'obb': 'detect',   # 旋转框检测也归类为检测
            }
            if model_task in task_mapping:
                detected_task = task_mapping[model_task]
                if task != detected_task:
                    print_warning(f"从模型检测到任务类型: {detected_task.upper()} (文件名推断: {task.upper()})")
                    task = detected_task
                    task_type = TaskType.from_string(task)
                    print_info(f"使用模型实际任务类型: {task.upper()}")
        
        # 开始验证
        print_info("开始验证...")
        console.print()
        
        validation_kwargs = {
            'data': data,
            'split': split,
            'batch': batch,
            'imgsz': imgsz,
            'device': device,
            'save_json': save_json,
            'save_hybrid': save_hybrid,
            'plots': plots,
            'verbose': verbose,
            'project': project,
            'name': name,
        }
        
        # 只有检测和分割任务需要 conf 和 iou 参数
        # 分类任务不使用这些参数，避免影响结果
        if task_type in [TaskType.DETECT, TaskType.SEGMENT]:
            validation_kwargs['conf'] = conf
            validation_kwargs['iou'] = iou
        
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
            
            # 计算F1分数（如果不存在）
            precision = safe_float(box_metrics.mp)
            recall = safe_float(box_metrics.mr)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 创建综合指标表格
            metrics_table = Table(title="🎯 检测指标 - 综合评估", show_header=True, header_style="bold cyan")
            metrics_table.add_column("指标", style="cyan", width=25)
            metrics_table.add_column("值", style="green", justify="right", width=15)
            metrics_table.add_column("说明", style="dim", width=40)
            
            # mAP指标
            metrics_table.add_row(
                "mAP@0.5", 
                f"{safe_float(box_metrics.map50):.4f}",
                "IoU=0.5时的平均精度"
            )
            metrics_table.add_row(
                "mAP@0.5:0.95", 
                f"{safe_float(box_metrics.map):.4f}",
                "IoU从0.5到0.95的平均精度"
            )
            
            # 核心指标
            metrics_table.add_row("", "", "")  # 分隔行
            metrics_table.add_row(
                "精确率 (Precision)", 
                f"{precision:.4f}",
                "预测为正的样本中真正为正的比例"
            )
            metrics_table.add_row(
                "召回率 (Recall)", 
                f"{recall:.4f}",
                "所有正样本中被正确预测的比例"
            )
            metrics_table.add_row(
                "F1 分数", 
                f"{f1_score:.4f}",
                "精确率和召回率的调和平均数"
            )
            
            # 计算并显示准确率
            # 准确率始终显示，确保统计需求得到满足
            accuracy = None
            accuracy_method = ""
            
            # 方法1: 尝试从混淆矩阵计算（最准确）
            try:
                if hasattr(box_metrics, 'confusion_matrix') and box_metrics.confusion_matrix is not None:
                    cm = box_metrics.confusion_matrix
                    if hasattr(cm, 'matrix') and cm.matrix is not None and cm.matrix.size > 0:
                        matrix = cm.matrix
                        total = safe_float(matrix.sum())
                        correct = safe_float(np.trace(matrix))
                        if total > 0:
                            accuracy = correct / total
                            accuracy_method = "基于混淆矩阵"
            except:
                pass
            
            # 方法2: 如果没有混淆矩阵，从precision和recall计算检测准确率
            # 检测准确率定义为: TP / (TP + FP + FN)
            # 从已知的precision和recall可以推导：
            # 设TP为真正例数
            # Precision = TP/(TP+FP) => FP = TP/Precision - TP = TP(1/Precision - 1)
            # Recall = TP/(TP+FN) => FN = TP/Recall - TP = TP(1/Recall - 1)
            # Accuracy = TP/(TP+FP+FN) = TP/(TP + TP(1/P-1) + TP(1/R-1))
            #          = TP/(TP(1 + 1/P - 1 + 1/R - 1)) = 1/(1/P + 1/R - 1)
            #          = 1/((R+P-PR)/(PR)) = PR/(P+R-PR)
            if accuracy is None and precision > 0 and recall > 0:
                accuracy = (precision * recall) / (precision + recall - precision * recall)
                accuracy_method = "基于P&R推导"
            
            # 方法3: 如果仍无法计算，使用mAP作为综合性能指标
            if accuracy is None:
                accuracy = safe_float(box_metrics.map50)
                accuracy_method = "使用mAP@0.5"
            
            # 显示准确率
            if accuracy is not None:
                metrics_table.add_row(
                    "准确率 (Accuracy)",
                    f"{accuracy:.4f}",
                    f"检测准确性指标 [{accuracy_method}]"
                )
            
            console.print(metrics_table)
            
            # 显示详细统计信息
            console.print()
            stats_table = Table(title="📊 详细统计", show_header=True, header_style="bold yellow")
            stats_table.add_column("统计项", style="cyan", width=25)
            stats_table.add_column("值", style="green", justify="right", width=15)
            
            # 获取统计信息
            if hasattr(results, 'speed'):
                speed = results.speed
                if isinstance(speed, dict):
                    total_time = sum(speed.values())
                    stats_table.add_row("总推理时间 (ms)", f"{total_time:.2f}")
                    if 'preprocess' in speed:
                        stats_table.add_row("  - 预处理", f"{speed['preprocess']:.2f}")
                    if 'inference' in speed:
                        stats_table.add_row("  - 推理", f"{speed['inference']:.2f}")
                    if 'postprocess' in speed:
                        stats_table.add_row("  - 后处理", f"{speed['postprocess']:.2f}")
            
            if hasattr(results, 'seen'):
                stats_table.add_row("验证图像数", str(results.seen))
            
            console.print(stats_table)
            
            # 显示每个类别的详细结果
            if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > 0:
                console.print()
                
                class_table = Table(
                    title="📋 各类别详细指标", 
                    show_header=True, 
                    header_style="bold magenta"
                )
                class_table.add_column("类别", style="cyan", width=20)
                class_table.add_column("AP@0.5", style="green", justify="right", width=12)
                class_table.add_column("AP@0.5:0.95", style="green", justify="right", width=14)
                class_table.add_column("Precision", style="yellow", justify="right", width=12)
                class_table.add_column("Recall", style="yellow", justify="right", width=12)
                class_table.add_column("F1", style="blue", justify="right", width=10)
                
                # 获取类别名称
                if hasattr(results, 'names'):
                    class_names = [results.names[i] for i in range(len(box_metrics.ap50))]
                else:
                    class_names = [f"class_{i}" for i in range(len(box_metrics.ap50))]
                
                # 获取每个类别的指标
                for idx, class_name in enumerate(class_names):
                    ap50_val = safe_float(box_metrics.ap50[idx]) if idx < len(box_metrics.ap50) else 0.0
                    ap_val = safe_float(box_metrics.ap[idx]) if hasattr(box_metrics, 'ap') and idx < len(box_metrics.ap) else 0.0
                    
                    # 获取每个类别的精确率和召回率（如果有）
                    if hasattr(box_metrics, 'p') and idx < len(box_metrics.p):
                        class_precision = safe_float(box_metrics.p[idx])
                    else:
                        class_precision = precision
                    
                    if hasattr(box_metrics, 'r') and idx < len(box_metrics.r):
                        class_recall = safe_float(box_metrics.r[idx])
                    else:
                        class_recall = recall
                    
                    # 计算类别F1
                    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0.0
                    
                    class_table.add_row(
                        class_name,
                        f"{ap50_val:.4f}",
                        f"{ap_val:.4f}",
                        f"{class_precision:.4f}",
                        f"{class_recall:.4f}",
                        f"{class_f1:.4f}"
                    )
                
                console.print(class_table)
    
    elif task_type == TaskType.SEGMENT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            mask_metrics = results.seg if hasattr(results, 'seg') else None
            
            # 计算边界框的F1分数
            box_precision = safe_float(box_metrics.mp)
            box_recall = safe_float(box_metrics.mr)
            box_f1 = 2 * (box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) > 0 else 0.0
            
            # 创建指标表格
            metrics_table = Table(title="🎯 分割指标 - 综合评估", show_header=True, header_style="bold cyan")
            metrics_table.add_column("指标类型", style="cyan", width=15)
            metrics_table.add_column("指标", style="yellow", width=20)
            metrics_table.add_column("值", style="green", justify="right", width=15)
            
            # 边界框指标
            metrics_table.add_row("边界框", "mAP@0.5", f"{safe_float(box_metrics.map50):.4f}")
            metrics_table.add_row("", "mAP@0.5:0.95", f"{safe_float(box_metrics.map):.4f}")
            metrics_table.add_row("", "Precision", f"{box_precision:.4f}")
            metrics_table.add_row("", "Recall", f"{box_recall:.4f}")
            metrics_table.add_row("", "F1 Score", f"{box_f1:.4f}")
            
            # 掩码指标
            if mask_metrics:
                mask_precision = safe_float(mask_metrics.mp)
                mask_recall = safe_float(mask_metrics.mr)
                mask_f1 = 2 * (mask_precision * mask_recall) / (mask_precision + mask_recall) if (mask_precision + mask_recall) > 0 else 0.0
                
                metrics_table.add_row("", "", "")
                metrics_table.add_row("掩码", "mAP@0.5", f"{safe_float(mask_metrics.map50):.4f}")
                metrics_table.add_row("", "mAP@0.5:0.95", f"{safe_float(mask_metrics.map):.4f}")
                metrics_table.add_row("", "Precision", f"{mask_precision:.4f}")
                metrics_table.add_row("", "Recall", f"{mask_recall:.4f}")
                metrics_table.add_row("", "F1 Score", f"{mask_f1:.4f}")
            
            console.print(metrics_table)
            
            # 显示详细统计信息
            console.print()
            stats_table = Table(title="📊 详细统计", show_header=True, header_style="bold yellow")
            stats_table.add_column("统计项", style="cyan", width=25)
            stats_table.add_column("值", style="green", justify="right", width=15)
            
            if hasattr(results, 'speed'):
                speed = results.speed
                if isinstance(speed, dict):
                    total_time = sum(speed.values())
                    stats_table.add_row("总推理时间 (ms)", f"{total_time:.2f}")
                    if 'preprocess' in speed:
                        stats_table.add_row("  - 预处理", f"{speed['preprocess']:.2f}")
                    if 'inference' in speed:
                        stats_table.add_row("  - 推理", f"{speed['inference']:.2f}")
                    if 'postprocess' in speed:
                        stats_table.add_row("  - 后处理", f"{speed['postprocess']:.2f}")
            
            if hasattr(results, 'seen'):
                stats_table.add_row("验证图像数", str(results.seen))
            
            console.print(stats_table)
            
            # 显示每个类别的详细结果
            if mask_metrics and hasattr(mask_metrics, 'ap50') and len(mask_metrics.ap50) > 0:
                console.print()
                
                class_table = Table(
                    title="📋 各类别分割指标", 
                    show_header=True, 
                    header_style="bold magenta"
                )
                class_table.add_column("类别", style="cyan", width=20)
                class_table.add_column("Mask AP@0.5", style="green", justify="right", width=14)
                class_table.add_column("Mask AP@0.5:0.95", style="green", justify="right", width=16)
                class_table.add_column("Precision", style="yellow", justify="right", width=12)
                class_table.add_column("Recall", style="yellow", justify="right", width=12)
                class_table.add_column("F1", style="blue", justify="right", width=10)
                
                # 获取类别名称
                if hasattr(results, 'names'):
                    class_names = [results.names[i] for i in range(len(mask_metrics.ap50))]
                else:
                    class_names = [f"class_{i}" for i in range(len(mask_metrics.ap50))]
                
                for idx, class_name in enumerate(class_names):
                    ap50_val = safe_float(mask_metrics.ap50[idx]) if idx < len(mask_metrics.ap50) else 0.0
                    ap_val = safe_float(mask_metrics.ap[idx]) if hasattr(mask_metrics, 'ap') and idx < len(mask_metrics.ap) else 0.0
                    
                    if hasattr(mask_metrics, 'p') and idx < len(mask_metrics.p):
                        class_precision = safe_float(mask_metrics.p[idx])
                    else:
                        class_precision = mask_precision
                    
                    if hasattr(mask_metrics, 'r') and idx < len(mask_metrics.r):
                        class_recall = safe_float(mask_metrics.r[idx])
                    else:
                        class_recall = mask_recall
                    
                    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0.0
                    
                    class_table.add_row(
                        class_name,
                        f"{ap50_val:.4f}",
                        f"{ap_val:.4f}",
                        f"{class_precision:.4f}",
                        f"{class_recall:.4f}",
                        f"{class_f1:.4f}"
                    )
                
                console.print(class_table)
    
    elif task_type == TaskType.CLASSIFY:
        if hasattr(results, 'top1') and hasattr(results, 'top5'):
            # 分类指标
            metrics_table = Table(title="🎯 分类指标 - 综合评估", show_header=True, header_style="bold cyan")
            metrics_table.add_column("指标", style="cyan", width=25)
            metrics_table.add_column("值", style="green", justify="right", width=15)
            metrics_table.add_column("说明", style="dim", width=40)
            
            top1_acc = safe_float(results.top1)
            top5_acc = safe_float(results.top5)
            
            metrics_table.add_row(
                "Top-1 准确率", 
                f"{top1_acc:.4f}",
                "预测概率最高的类别正确的比例"
            )
            metrics_table.add_row(
                "Top-5 准确率", 
                f"{top5_acc:.4f}",
                "正确类别在前5个预测中的比例"
            )
            
            # 如果有混淆矩阵，显示更多指标
            if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
                cm = results.confusion_matrix
                if hasattr(cm, 'matrix') and cm.matrix is not None:
                    matrix = cm.matrix
                    # 计算宏平均精确率、召回率、F1
                    n_classes = matrix.shape[0] if len(matrix.shape) > 1 else 1
                    
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    for i in range(n_classes):
                        tp = safe_float(matrix[i, i]) if i < matrix.shape[0] and i < matrix.shape[1] else 0
                        fp = safe_float(matrix[:, i].sum() - matrix[i, i]) if i < matrix.shape[1] else 0
                        fn = safe_float(matrix[i, :].sum() - matrix[i, i]) if i < matrix.shape[0] else 0
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)
                    
                    if precisions:
                        metrics_table.add_row("", "", "")
                        metrics_table.add_row(
                            "宏平均精确率", 
                            f"{np.mean(precisions):.4f}",
                            "各类别精确率的平均值"
                        )
                        metrics_table.add_row(
                            "宏平均召回率", 
                            f"{np.mean(recalls):.4f}",
                            "各类别召回率的平均值"
                        )
                        metrics_table.add_row(
                            "宏平均F1分数", 
                            f"{np.mean(f1_scores):.4f}",
                            "各类别F1分数的平均值"
                        )
            
            console.print(metrics_table)
            
            # 显示详细统计信息
            console.print()
            stats_table = Table(title="📊 详细统计", show_header=True, header_style="bold yellow")
            stats_table.add_column("统计项", style="cyan", width=25)
            stats_table.add_column("值", style="green", justify="right", width=15)
            
            if hasattr(results, 'speed'):
                speed = results.speed
                if isinstance(speed, dict):
                    total_time = sum(speed.values())
                    stats_table.add_row("总推理时间 (ms)", f"{total_time:.2f}")
                    if 'preprocess' in speed:
                        stats_table.add_row("  - 预处理", f"{speed['preprocess']:.2f}")
                    if 'inference' in speed:
                        stats_table.add_row("  - 推理", f"{speed['inference']:.2f}")
                    if 'postprocess' in speed:
                        stats_table.add_row("  - 后处理", f"{speed['postprocess']:.2f}")
            
            if hasattr(results, 'seen'):
                stats_table.add_row("验证图像数", str(results.seen))
            
            console.print(stats_table)
            
            # 如果有每个类别的详细信息，显示出来
            if hasattr(results, 'names') and hasattr(results, 'confusion_matrix'):
                cm = results.confusion_matrix
                if hasattr(cm, 'matrix') and cm.matrix is not None:
                    console.print()
                    
                    class_table = Table(
                        title="📋 各类别详细指标", 
                        show_header=True, 
                        header_style="bold magenta"
                    )
                    class_table.add_column("类别", style="cyan", width=20)
                    class_table.add_column("Accuracy", style="green", justify="right", width=12)
                    class_table.add_column("Precision", style="yellow", justify="right", width=12)
                    class_table.add_column("Recall", style="yellow", justify="right", width=12)
                    class_table.add_column("F1 Score", style="blue", justify="right", width=12)
                    class_table.add_column("Support", style="dim", justify="right", width=10)
                    
                    matrix = cm.matrix
                    n_classes = matrix.shape[0] if len(matrix.shape) > 1 else 1
                    class_names = [results.names[i] if hasattr(results, 'names') else f"Class {i}" for i in range(n_classes)]
                    
                    for i, class_name in enumerate(class_names):
                        if i >= matrix.shape[0] or i >= matrix.shape[1]:
                            continue
                            
                        tp = safe_float(matrix[i, i])
                        fp = safe_float(matrix[:, i].sum() - matrix[i, i])
                        fn = safe_float(matrix[i, :].sum() - matrix[i, i])
                        tn = safe_float(matrix.sum() - tp - fp - fn)
                        
                        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        support = int(tp + fn)
                        
                        class_table.add_row(
                            class_name,
                            f"{accuracy:.4f}",
                            f"{precision:.4f}",
                            f"{recall:.4f}",
                            f"{f1:.4f}",
                            str(support)
                        )
                    
                    console.print(class_table)


def _generate_results_summary(results, task_type: TaskType, model_path: Path, 
                              data: str, split: str, conf: float, iou: float) -> dict:
    """生成验证结果摘要"""
    
    summary = {
        'model': str(model_path),
        'model_name': model_path.name,
        'task': task_type.value,
        'dataset': data,
        'split': split,
        'timestamp': datetime.now().isoformat(),
    }
    
    # 只有检测和分割任务才记录 conf 和 iou 阈值
    if task_type in [TaskType.DETECT, TaskType.SEGMENT]:
        summary['conf_threshold'] = conf
        summary['iou_threshold'] = iou
    
    # 根据任务类型添加指标
    if task_type == TaskType.DETECT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            
            # 计算F1分数
            precision = safe_float(box_metrics.mp)
            recall = safe_float(box_metrics.mr)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 计算准确率（使用多种方法）
            accuracy = None
            accuracy_method = ""
            
            # 方法1: 从混淆矩阵计算
            try:
                if hasattr(box_metrics, 'confusion_matrix') and box_metrics.confusion_matrix is not None:
                    cm = box_metrics.confusion_matrix
                    if hasattr(cm, 'matrix') and cm.matrix is not None and cm.matrix.size > 0:
                        matrix = cm.matrix
                        total = safe_float(matrix.sum())
                        correct = safe_float(np.trace(matrix))
                        if total > 0:
                            accuracy = correct / total
                            accuracy_method = "confusion_matrix"
            except:
                pass
            
            # 方法2: 从precision和recall推导
            if accuracy is None and precision > 0 and recall > 0:
                accuracy = (precision * recall) / (precision + recall - precision * recall)
                accuracy_method = "derived_from_pr"
            
            # 方法3: 使用mAP@0.5作为后备
            if accuracy is None:
                accuracy = safe_float(box_metrics.map50)
                accuracy_method = "map50"
            
            summary['metrics'] = {
                'mAP50': safe_float(box_metrics.map50),
                'mAP50_95': safe_float(box_metrics.map),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'accuracy_method': accuracy_method,
            }
            
            # 添加性能统计
            if hasattr(results, 'speed'):
                summary['performance'] = {
                    'speed_ms': results.speed if isinstance(results.speed, dict) else {},
                }
            
            if hasattr(results, 'seen'):
                summary['statistics'] = {
                    'images_validated': int(results.seen),
                }
            
            # 每个类别的详细指标
            if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > 0:
                if hasattr(results, 'names'):
                    class_names = [results.names[i] for i in range(len(box_metrics.ap50))]
                else:
                    class_names = [f"class_{i}" for i in range(len(box_metrics.ap50))]
                
                per_class = {}
                for idx, name in enumerate(class_names):
                    class_metrics = {
                        'ap50': safe_float(box_metrics.ap50[idx]) if idx < len(box_metrics.ap50) else 0.0,
                        'ap50_95': safe_float(box_metrics.ap[idx]) if hasattr(box_metrics, 'ap') and idx < len(box_metrics.ap) else 0.0,
                    }
                    
                    # 添加每个类别的精确率和召回率
                    if hasattr(box_metrics, 'p') and idx < len(box_metrics.p):
                        class_precision = safe_float(box_metrics.p[idx])
                        class_metrics['precision'] = class_precision
                    
                    if hasattr(box_metrics, 'r') and idx < len(box_metrics.r):
                        class_recall = safe_float(box_metrics.r[idx])
                        class_metrics['recall'] = class_recall
                    
                    # 计算类别F1
                    if 'precision' in class_metrics and 'recall' in class_metrics:
                        p = class_metrics['precision']
                        r = class_metrics['recall']
                        class_metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                    
                    per_class[name] = class_metrics
                
                summary['per_class'] = per_class
    
    elif task_type == TaskType.SEGMENT:
        if hasattr(results, 'box') and results.box:
            box_metrics = results.box
            mask_metrics = results.seg if hasattr(results, 'seg') else None
            
            # 边界框F1
            box_precision = safe_float(box_metrics.mp)
            box_recall = safe_float(box_metrics.mr)
            box_f1 = 2 * (box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) > 0 else 0.0
            
            summary['metrics'] = {
                'box': {
                    'mAP50': safe_float(box_metrics.map50),
                    'mAP50_95': safe_float(box_metrics.map),
                    'precision': box_precision,
                    'recall': box_recall,
                    'f1_score': box_f1,
                }
            }
            
            if mask_metrics:
                # 掩码F1
                mask_precision = safe_float(mask_metrics.mp)
                mask_recall = safe_float(mask_metrics.mr)
                mask_f1 = 2 * (mask_precision * mask_recall) / (mask_precision + mask_recall) if (mask_precision + mask_recall) > 0 else 0.0
                
                summary['metrics']['mask'] = {
                    'mAP50': safe_float(mask_metrics.map50),
                    'mAP50_95': safe_float(mask_metrics.map),
                    'precision': mask_precision,
                    'recall': mask_recall,
                    'f1_score': mask_f1,
                }
                
                # 每个类别的掩码指标
                if hasattr(mask_metrics, 'ap50') and len(mask_metrics.ap50) > 0:
                    if hasattr(results, 'names'):
                        class_names = [results.names[i] for i in range(len(mask_metrics.ap50))]
                    else:
                        class_names = [f"class_{i}" for i in range(len(mask_metrics.ap50))]
                    
                    per_class = {}
                    for idx, name in enumerate(class_names):
                        class_metrics = {
                            'mask_ap50': safe_float(mask_metrics.ap50[idx]) if idx < len(mask_metrics.ap50) else 0.0,
                            'mask_ap50_95': safe_float(mask_metrics.ap[idx]) if hasattr(mask_metrics, 'ap') and idx < len(mask_metrics.ap) else 0.0,
                        }
                        
                        if hasattr(mask_metrics, 'p') and idx < len(mask_metrics.p):
                            class_precision = safe_float(mask_metrics.p[idx])
                            class_metrics['precision'] = class_precision
                        
                        if hasattr(mask_metrics, 'r') and idx < len(mask_metrics.r):
                            class_recall = safe_float(mask_metrics.r[idx])
                            class_metrics['recall'] = class_recall
                        
                        if 'precision' in class_metrics and 'recall' in class_metrics:
                            p = class_metrics['precision']
                            r = class_metrics['recall']
                            class_metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                        
                        per_class[name] = class_metrics
                    
                    summary['per_class'] = per_class
            
            # 添加性能统计
            if hasattr(results, 'speed'):
                summary['performance'] = {
                    'speed_ms': results.speed if isinstance(results.speed, dict) else {},
                }
            
            if hasattr(results, 'seen'):
                summary['statistics'] = {
                    'images_validated': int(results.seen),
                }
    
    elif task_type == TaskType.CLASSIFY:
        if hasattr(results, 'top1') and hasattr(results, 'top5'):
            summary['metrics'] = {
                'top1_accuracy': safe_float(results.top1),
                'top5_accuracy': safe_float(results.top5),
            }
            
            # 如果有混淆矩阵，添加宏平均指标
            if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
                cm = results.confusion_matrix
                if hasattr(cm, 'matrix') and cm.matrix is not None:
                    matrix = cm.matrix
                    n_classes = matrix.shape[0] if len(matrix.shape) > 1 else 1
                    
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    per_class = {}
                    class_names = [results.names[i] if hasattr(results, 'names') else f"class_{i}" for i in range(n_classes)]
                    
                    for i, class_name in enumerate(class_names):
                        if i >= matrix.shape[0] or i >= matrix.shape[1]:
                            continue
                            
                        tp = safe_float(matrix[i, i])
                        fp = safe_float(matrix[:, i].sum() - matrix[i, i])
                        fn = safe_float(matrix[i, :].sum() - matrix[i, i])
                        tn = safe_float(matrix.sum() - tp - fp - fn)
                        
                        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        support = int(tp + fn)
                        
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)
                        
                        per_class[class_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'support': support,
                        }
                    
                    if precisions:
                        summary['metrics']['macro_avg_precision'] = float(np.mean(precisions))
                        summary['metrics']['macro_avg_recall'] = float(np.mean(recalls))
                        summary['metrics']['macro_avg_f1'] = float(np.mean(f1_scores))
                    
                    summary['per_class'] = per_class
            
            # 添加性能统计
            if hasattr(results, 'speed'):
                summary['performance'] = {
                    'speed_ms': results.speed if isinstance(results.speed, dict) else {},
                }
            
            if hasattr(results, 'seen'):
                summary['statistics'] = {
                    'images_validated': int(results.seen),
                }
    
    return summary


@app.command("compare")
def compare_models(
    models: str = typer.Argument(..., help="模型路径，用逗号分隔"),
    data: str = typer.Option("data/dataset.yaml", "--data", "-d", help="数据集配置文件"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务类型（自动从第一个模型推断）"),
    split: str = typer.Option("val", "--split", help="验证数据集"),
    batch: int = typer.Option(16, "--batch", "-b", help="批次大小"),
    imgsz: int = typer.Option(640, "--imgsz", help="图像尺寸"),
    conf: float = typer.Option(0.001, "--conf", help="置信度阈值"),
    iou: float = typer.Option(0.6, "--iou", help="IoU阈值"),
    device: str = typer.Option("auto", "--device", help="设备"),
):
    """
    比较多个模型的性能
    
    支持检测(detect)、分割(segment)和分类(classify)三种任务类型。
    任务类型可自动从第一个模型名称推断，或手动指定。
    
    示例:
    
    \b
      # 自动推断任务类型
      yolo-cli validate compare model1.pt,model2.pt,model3.pt --data data/dataset.yaml
      
    \b
      # 手动指定任务类型
      yolo-cli validate compare model1-cls.pt,model2-cls.pt --task classify --data data/images
    """
    
    print_section_header("模型性能比较")
    
    # 解析模型路径
    model_paths = [Path(m.strip()) for m in models.split(',')]
    
    # 验证模型文件
    for model_path in model_paths:
        if not model_path.exists():
            print_error(f"模型不存在: {model_path}")
            raise typer.Exit(1)
    
    # 推断或验证任务类型（从第一个模型推断）
    if task is None:
        _, task = parse_model_name(model_paths[0].name)
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
            data_path = Path(data)
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
            
            # 根据任务类型设置验证参数
            val_kwargs = {
                'data': data if task_type == TaskType.CLASSIFY else str(data_path),
                'split': split,
                'batch': batch,
                'imgsz': imgsz,
                'device': device,
                'verbose': False,
                'plots': False,
            }
            
            # 只有检测和分割任务需要 conf 和 iou 参数
            if task_type in [TaskType.DETECT, TaskType.SEGMENT]:
                val_kwargs['conf'] = conf
                val_kwargs['iou'] = iou
            
            results = yolo_model.val(**val_kwargs)
            
            # 尝试获取相对路径
            try:
                relative_path = str(model_path.relative_to(Path.cwd()))
            except (ValueError, AttributeError):
                relative_path = str(model_path)
            
            all_results.append({
                'name': model_path.name,
                'path': str(model_path),
                'relative_path': relative_path,
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
    print_section_header(f"性能比较结果 - {task.upper()}")
    
    # 检查是否有重名模型
    model_names = [r['name'] for r in all_results]
    has_duplicate_names = len(model_names) != len(set(model_names))
    
    comparison_table = Table(show_header=True, header_style="bold cyan")
    
    # 如果有重名，显示完整路径；否则只显示文件名
    if has_duplicate_names:
        comparison_table.add_column("模型路径", style="cyan", width=50)
    else:
        comparison_table.add_column("模型", style="cyan", width=30)
    
    # 根据任务类型设置不同的列标题
    if task_type == TaskType.CLASSIFY:
        comparison_table.add_column("Top-1", style="green", justify="right", width=10)
        comparison_table.add_column("Top-5", style="green", justify="right", width=10)
    else:
        comparison_table.add_column("mAP@0.5", style="green", justify="right", width=10)
        comparison_table.add_column("mAP@0.5:0.95", style="green", justify="right", width=12)
    
    comparison_table.add_column("Precision", style="yellow", justify="right", width=10)
    comparison_table.add_column("Recall", style="yellow", justify="right", width=10)
    comparison_table.add_column("F1", style="blue", justify="right", width=10)
    comparison_table.add_column("Accuracy", style="magenta", justify="right", width=10)
    comparison_table.add_column("速度(ms)", style="dim", justify="right", width=12)
    
    best_map50 = -1
    best_model_display = None
    best_model_path = None
    best_f1 = -1
    best_f1_model_display = None
    best_f1_model_path = None
    best_accuracy = -1
    best_accuracy_model_display = None
    best_accuracy_model_path = None
    
    for result in all_results:
        # 确定显示名称
        display_name = result['path'] if has_duplicate_names else result['name']
        
        # 计算推理速度
        total_speed = 0.0
        if hasattr(result['results'], 'speed'):
            speed = result['results'].speed
            if isinstance(speed, dict):
                total_speed = sum(speed.values())
        
        # 处理检测任务
        if hasattr(result['results'], 'box') and result['results'].box:
            box_metrics = result['results'].box
            map50 = safe_float(box_metrics.map50)
            precision = safe_float(box_metrics.mp)
            recall = safe_float(box_metrics.mr)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 计算准确率（与单一模型验证一致的三级计算方法）
            accuracy = None
            
            # 方法1: 从混淆矩阵计算
            try:
                if hasattr(box_metrics, 'confusion_matrix') and box_metrics.confusion_matrix is not None:
                    cm = box_metrics.confusion_matrix
                    if hasattr(cm, 'matrix') and cm.matrix is not None and cm.matrix.size > 0:
                        matrix = cm.matrix
                        total = safe_float(matrix.sum())
                        correct = safe_float(np.trace(matrix))
                        if total > 0:
                            accuracy = correct / total
            except:
                pass
            
            # 方法2: 从precision和recall推导
            if accuracy is None and precision > 0 and recall > 0:
                accuracy = (precision * recall) / (precision + recall - precision * recall)
            
            # 方法3: 使用mAP@0.5作为后备
            if accuracy is None:
                accuracy = map50
            
            # 更新最佳模型
            if map50 > best_map50:
                best_map50 = map50
                best_model_display = display_name
                best_model_path = result['path']
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_f1_model_display = display_name
                best_f1_model_path = result['path']
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_model_display = display_name
                best_accuracy_model_path = result['path']
            
            comparison_table.add_row(
                display_name,
                f"{map50:.4f}",
                f"{safe_float(box_metrics.map):.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1_score:.4f}",
                f"{accuracy:.4f}",
                f"{total_speed:.1f}" if total_speed > 0 else "N/A",
            )
        
        # 处理分割任务
        elif hasattr(result['results'], 'masks') and result['results'].masks:
            mask_metrics = result['results'].masks
            map50 = safe_float(mask_metrics.map50)
            precision = safe_float(mask_metrics.mp)
            recall = safe_float(mask_metrics.mr)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 对分割任务，准确率使用与检测相同的计算方法
            accuracy = None
            if precision > 0 and recall > 0:
                accuracy = (precision * recall) / (precision + recall - precision * recall)
            if accuracy is None:
                accuracy = map50
            
            # 更新最佳模型
            if map50 > best_map50:
                best_map50 = map50
                best_model_display = display_name
                best_model_path = result['path']
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_f1_model_display = display_name
                best_f1_model_path = result['path']
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_model_display = display_name
                best_accuracy_model_path = result['path']
            
            comparison_table.add_row(
                display_name,
                f"{map50:.4f}",
                f"{safe_float(mask_metrics.map):.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1_score:.4f}",
                f"{accuracy:.4f}",
                f"{total_speed:.1f}" if total_speed > 0 else "N/A",
            )
        
        # 处理分类任务
        elif hasattr(result['results'], 'top1') and hasattr(result['results'], 'top5'):
            top1 = safe_float(result['results'].top1)
            top5 = safe_float(result['results'].top5)
            
            # 分类任务使用top1作为准确率
            accuracy = top1
            
            # 分类任务的F1需要从混淆矩阵计算
            f1_score = 0.0
            precision = 0.0
            recall = 0.0
            
            try:
                if hasattr(result['results'], 'confusion_matrix') and result['results'].confusion_matrix is not None:
                    cm = result['results'].confusion_matrix
                    if hasattr(cm, 'matrix') and cm.matrix is not None and cm.matrix.size > 0:
                        matrix = cm.matrix
                        # 计算宏平均（分类任务没有背景类，使用全部类别）
                        num_classes = matrix.shape[0]
                        precisions = []
                        recalls = []
                        f1_scores = []
                        
                        for i in range(num_classes):
                            tp = safe_float(matrix[i, i])
                            fp = safe_float(matrix[:, i].sum() - matrix[i, i])
                            fn = safe_float(matrix[i, :].sum() - matrix[i, i])
                            
                            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                            
                            precisions.append(p)
                            recalls.append(r)
                            f1_scores.append(f1)
                        
                        # 宏平均：先计算每个类别的指标，再取平均
                        precision = np.mean(precisions) if precisions else 0.0
                        recall = np.mean(recalls) if recalls else 0.0
                        f1_score = np.mean(f1_scores) if f1_scores else 0.0
            except:
                pass
            
            # 更新最佳模型
            if top1 > best_map50:  # 分类任务用top1替代mAP
                best_map50 = top1
                best_model_display = display_name
                best_model_path = result['path']
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_f1_model_display = display_name
                best_f1_model_path = result['path']
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_model_display = display_name
                best_accuracy_model_path = result['path']
            
            comparison_table.add_row(
                display_name,
                f"{top1:.4f}",  # Top-1作为主要指标
                f"{top5:.4f}",  # Top-5
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1_score:.4f}",
                f"{accuracy:.4f}",
                f"{total_speed:.1f}" if total_speed > 0 else "N/A",
            )
    
    console.print(comparison_table)
    console.print()
    
    # 显示最佳模型（显示路径以便区分，根据任务类型调整指标名称）
    if best_model_path:
        if task_type == TaskType.CLASSIFY:
            print_success(f"🏆 最高Top-1准确率: {best_model_path} ({best_map50:.4f})")
        else:
            print_success(f"🏆 最高mAP@0.5: {best_model_path} ({best_map50:.4f})")
    if best_f1_model_path:
        print_success(f"🏆 最高F1分数: {best_f1_model_path} ({best_f1:.4f})")
    if best_accuracy_model_path:
        if task_type == TaskType.CLASSIFY:
            print_success(f"🏆 最高Top-1准确率: {best_accuracy_model_path} ({best_accuracy:.4f})")
        else:
            print_success(f"🏆 最高准确率: {best_accuracy_model_path} ({best_accuracy:.4f})")


if __name__ == "__main__":
    app()

