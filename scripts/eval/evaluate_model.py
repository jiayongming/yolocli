#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型评估脚本"""

from ultralytics import YOLO
import json
from pathlib import Path

def evaluate_model(
    model_path,
    data_yaml='data/dataset.yaml',
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
    print("评估结果")
    print("=" * 60)
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"精确率 (Precision): {metrics.box.mp:.4f}")
    print(f"召回率 (Recall): {metrics.box.mr:.4f}")
    
    # 每个类别的结果
    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        print("\n各类别mAP@0.5:")
        class_names = ['waterpoll', 'active_leak']  # 与dataset.yaml中的类别名称保持一致
        for i, (class_name, ap50) in enumerate(zip(class_names, metrics.box.ap50)):
            print(f"  {class_name}: {ap50:.4f}")
    
    # 保存评估结果
    results_dict = {
        'model_path': str(model_path),
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
    }
    
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
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
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
