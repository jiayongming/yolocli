#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速修复 dataset.yaml，添加缺失的 keypoint_names"""

import yaml
from pathlib import Path

yaml_file = Path("data/processed/dataset.yaml")

if yaml_file.exists():
    # 读取现有配置
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查是否有 kpt_shape
    if 'kpt_shape' in config:
        kpt_count = config['kpt_shape'][0] if isinstance(config['kpt_shape'], list) else config['kpt_shape']
        
        # 如果没有 keypoint_names，添加它
        if 'keypoint_names' not in config:
            print(f"检测到 {kpt_count} 个关键点，正在添加 keypoint_names...")
            
            if kpt_count == 4:
                config['keypoint_names'] = ['strat', 'end', 'center', 'pointer']
            elif kpt_count == 17:
                config['keypoint_names'] = [
                    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                ]
            else:
                config['keypoint_names'] = [f'kp_{i}' for i in range(kpt_count)]
            
            # 保存修改后的配置
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write("# YOLO 数据集配置文件\n")
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"✓ 已添加 keypoint_names: {config['keypoint_names']}")
            print(f"✓ dataset.yaml 已更新")
        else:
            print("✓ keypoint_names 已存在，无需修改")
    else:
        print("这不是 Pose 数据集配置（没有 kpt_shape）")
else:
    print(f"文件不存在: {yaml_file}")

