#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出模型为不同格式"""

from ultralytics import YOLO
from pathlib import Path

def export_model(
    model_path,
    formats=['onnx', 'torchscript'],
    imgsz=640,
    device='auto',  # 自动检测：'auto', 0 (NVIDIA GPU), 'mps' (Apple芯片), -1 (CPU)
    output_dir='models/exported'
):
    """导出模型"""
    
    # 自动检测设备
    if device == 'auto':
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'  # Apple芯片
            print("检测到Apple芯片，使用MPS加速")
        elif torch.cuda.is_available():
            device = 0  # NVIDIA GPU
            print(f"检测到NVIDIA GPU，使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            device = -1  # CPU
            print("未检测到GPU，使用CPU导出")
    
    print("=" * 60)
    print("模型导出")
    print("=" * 60)
    print(f"设备: {device}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(model_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 导出为不同格式
    for fmt in formats:
        print(f"\n导出为 {fmt.upper()} 格式...")
        try:
            exported_path = model.export(
                format=fmt,
                imgsz=imgsz,
                device=device,
            )
            print(f"✓ 导出成功: {exported_path}")
            
            # 移动到输出目录
            exported_file = Path(exported_path)
            if exported_file.exists():
                new_path = output_path / exported_file.name
                exported_file.rename(new_path)
                print(f"  已移动到: {new_path}")
        except Exception as e:
            print(f"✗ 导出失败: {e}")
    
    print("\n" + "=" * 60)
    print("模型导出完成！")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='导出YOLO模型')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--formats', type=str, nargs='+',
                        default=['onnx', 'torchscript'],
                        help='导出格式')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备选择: auto(自动), 0(NVIDIA GPU), mps(Apple芯片), -1(CPU)')
    parser.add_argument('--output', type=str, default='models/exported',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 处理device参数：如果是数字字符串，转换为int
    if args.device.isdigit() or (args.device.startswith('-') and args.device[1:].isdigit()):
        device = int(args.device)
    else:
        device = args.device
    
    export_model(
        model_path=args.model,
        formats=args.formats,
        imgsz=args.imgsz,
        device=device,
        output_dir=args.output,
    )
