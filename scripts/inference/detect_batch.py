#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量图片检测"""

from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

def detect_batch(
    model_path,
    images_dir,
    output_dir='results/predictions',
    conf_threshold=0.25,
):
    """批量检测图片"""
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有图片
    images_path = Path(images_dir)
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 批量检测
    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='batch_detection',
    )
    
    # 统计结果
    total_detections = 0
    summary = []
    
    for result, img_file in zip(results, image_files):
        detections = []
        for box in result.boxes:
            detection = {
                'class': int(box.cls[0]),
                'class_name': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
            }
            detections.append(detection)
            total_detections += 1
        
        summary.append({
            'image': img_file.name,
            'detections_count': len(detections),
            'detections': detections,
        })
    
    # 保存汇总结果
    summary_file = Path(output_dir) / 'batch_detection' / 'summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_images': len(image_files),
            'total_detections': total_detections,
            'results': summary,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n批量检测完成！")
    print(f"总图片数: {len(image_files)}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张: {total_detections/len(image_files):.2f} 个目标")
    print(f"汇总结果已保存到: {summary_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量图片检测')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--images', type=str, required=True,
                        help='图片目录')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--output', type=str, default='results/predictions',
                        help='输出目录')
    
    args = parser.parse_args()
    
    detect_batch(
        model_path=args.model,
        images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
    )
