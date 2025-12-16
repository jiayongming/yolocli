#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单张图片检测"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import json

def detect_image(
    model_path,
    image_path,
    output_dir='results/predictions',
    conf_threshold=0.25,
    save_txt=True,
    save_json=True,
):
    """检测单张图片"""
    
    # 加载模型
    model = YOLO(model_path)
    
    # 执行检测
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        save_txt=save_txt,
        save_conf=True,
        project=output_dir,
        name='single_image',
    )
    
    # 处理结果
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detection = {
                'class': int(box.cls[0]),
                'class_name': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist(),
            }
            detections.append(detection)
    
    # 保存JSON结果
    if save_json:
        output_path = Path(output_dir) / 'single_image'
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_file = output_path / f"{Path(image_path).stem}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已保存到: {json_file}")
    
    print(f"检测到 {len(detections)} 个目标")
    return detections

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='单张图片检测')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--image', type=str, required=True,
                        help='图片路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--output', type=str, default='results/predictions',
                        help='输出目录')
    
    args = parser.parse_args()
    
    detect_image(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        conf_threshold=args.conf,
    )
