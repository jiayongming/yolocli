#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型评估脚本"""

from ultralytics import YOLO
import json
from pathlib import Path

def evaluate_model(
    model_path,
    data_yaml='data/processed/dataset.yaml',
    conf_threshold=0.25,
    iou_threshold=0.45,
    split='val',
    save_dir='results/validation'
):
    """评估模型性能"""
    
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"数据集: {data_yaml}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    print(f"评估集: {split}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 评估
    metrics = model.val(
        data=data_yaml,
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,  # 生成评估图表
        save_json=True,  # 保存JSON格式结果
        save_hybrid=True,  # 保存混合标签
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果 - 综合指标")
    print("=" * 60)
    
    # 核心指标
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1_score:.4f}")
    
    # 准确率计算
    accuracy = None
    accuracy_method = ""
    
    # 方法1: 从混淆矩阵计算
    if hasattr(metrics.box, 'confusion_matrix') and metrics.box.confusion_matrix is not None:
        cm = metrics.box.confusion_matrix
        if hasattr(cm, 'matrix') and cm.matrix is not None:
            import numpy as np
            matrix = cm.matrix
            total = float(matrix.sum())
            correct = float(np.trace(matrix))
            if total > 0:
                accuracy = correct / total
                accuracy_method = "混淆矩阵"
    
    # 方法2: 从precision和recall推导
    if accuracy is None and precision > 0 and recall > 0:
        accuracy = (precision * recall) / (precision + recall - precision * recall)
        accuracy_method = "P&R推导"
    
    # 方法3: 使用mAP@0.5
    if accuracy is None:
        accuracy = float(metrics.box.map50)
        accuracy_method = "mAP@0.5"
    
    if accuracy is not None:
        print(f"准确率 (Accuracy): {accuracy:.4f} [{accuracy_method}]")
    
    # 每个类别的详细结果
    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        print("\n" + "=" * 60)
        print("各类别详细指标")
        print("=" * 60)
        
        # 自动获取类别名称
        if hasattr(metrics, 'names'):
            class_names = [metrics.names[i] for i in range(len(metrics.box.ap50))]
        else:
            class_names = ['waterpoll', 'active_leak']  # 备用类别名称
        
        print(f"{'类别':<15} {'AP@0.5':<10} {'AP@0.5:0.95':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(class_names):
            if i >= len(metrics.box.ap50):
                break
                
            ap50_val = float(metrics.box.ap50[i])
            ap_val = float(metrics.box.ap[i]) if hasattr(metrics.box, 'ap') and i < len(metrics.box.ap) else 0.0
            
            # 获取每个类别的精确率和召回率（如果有）
            if hasattr(metrics.box, 'p') and i < len(metrics.box.p):
                class_precision = float(metrics.box.p[i])
            else:
                class_precision = precision
            
            if hasattr(metrics.box, 'r') and i < len(metrics.box.r):
                class_recall = float(metrics.box.r[i])
            else:
                class_recall = recall
            
            # 计算类别F1
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0.0
            
            print(f"{class_name:<15} {ap50_val:<10.4f} {ap_val:<12.4f} {class_precision:<10.4f} {class_recall:<10.4f} {class_f1:<10.4f}")
    
    # 保存评估结果（增强版）
    results_dict = {
        'model_path': str(model_path),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'metrics': {
            'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy if accuracy is not None else 0.0,
            'accuracy_method': accuracy_method if accuracy_method else 'unknown',
        }
    }
    
    # 添加每个类别的详细指标
    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        if hasattr(metrics, 'names'):
            class_names = [metrics.names[i] for i in range(len(metrics.box.ap50))]
        else:
            class_names = ['waterpoll', 'active_leak']
        
        per_class = {}
        for i, class_name in enumerate(class_names):
            if i >= len(metrics.box.ap50):
                break
                
            class_metrics = {
                'ap50': float(metrics.box.ap50[i]),
                'ap50_95': float(metrics.box.ap[i]) if hasattr(metrics.box, 'ap') and i < len(metrics.box.ap) else 0.0,
            }
            
            if hasattr(metrics.box, 'p') and i < len(metrics.box.p):
                class_precision = float(metrics.box.p[i])
                class_metrics['precision'] = class_precision
            
            if hasattr(metrics.box, 'r') and i < len(metrics.box.r):
                class_recall = float(metrics.box.r[i])
                class_metrics['recall'] = class_recall
            
            if 'precision' in class_metrics and 'recall' in class_metrics:
                p = class_metrics['precision']
                r = class_metrics['recall']
                class_metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            
            per_class[class_name] = class_metrics
        
        results_dict['per_class'] = per_class
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results_file = save_path / f"evaluation_results_{Path(model_path).stem}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {results_file}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估YOLO模型')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--data', type=str, default='data/processed/dataset.yaml',
                        help='数据集配置文件')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU阈值')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='评估数据集')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        split=args.split,
    )
