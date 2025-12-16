#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式模式"""

import typer
from pathlib import Path

from ..ui.prompts import (
    select_main_menu, select_model_operation, select_data_operation,
    select_train_operation, select_detect_operation,
    select_yolo_version, select_model_size, select_device,
    select_augmentation_preset, select_export_formats,
    build_training_config, input_text, input_path, input_number,
    confirm_action
)
from ..ui.display import (
    print_logo, clear_screen, print_section_header,
    print_success, print_error, print_info, print_warning,
    console
)
from ..core.config import ConfigManager
from ..core.version import YOLOVersionManager

# 导入命令函数
from . import model, data, train, detect

app = typer.Typer(help="交互式模式")


def run_model_operations():
    """模型管理操作"""
    while True:
        operation = select_model_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'download':
                # 下载模型
                print_section_header("下载预训练模型")
                
                version = select_yolo_version()
                size_str = select_model_size()
                
                sizes = [size_str] if size_str != 'all' else None
                download_all = (size_str == 'all')
                
                if confirm_action(f"确认下载 {version} {size_str if not download_all else '所有'} 模型?"):
                    from ..commands.model import download
                    download(version=version, size=sizes, all=download_all, output_dir=None)
            
            elif operation == 'export':
                # 导出模型
                print_section_header("导出模型")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                formats = select_export_formats()
                
                if formats and confirm_action(f"确认导出为 {', '.join(formats)} 格式?"):
                    from ..commands.model import export
                    export(model=model_path, formats=formats, imgsz=640, device='auto', output_dir=None)
            
            elif operation == 'list':
                # 列出模型
                from ..commands.model import list_models
                list_models(directory=None, version=None)
            
            console.print()
            if not confirm_action("继续模型操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def run_data_operations():
    """数据处理操作"""
    while True:
        operation = select_data_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'split':
                # 划分数据集
                print_section_header("划分数据集")
                
                images_dir = input_path("图像目录:", default="data/raw/images", must_exist=False)
                labels_dir = input_path("标签目录:", default="data/raw/labels", must_exist=False)
                
                ratios = input_text("划分比例 (train:val:test):", default="0.7:0.2:0.1")
                
                if confirm_action("确认划分数据集?"):
                    from ..commands.data import split_dataset
                    split_dataset(images_dir=images_dir, labels_dir=labels_dir, output_dir=None, ratios=ratios, seed=42)
            
            elif operation == 'generate-yaml':
                # 生成dataset.yaml
                print_section_header("生成 dataset.yaml")
                
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                output = input_text("输出文件:", default="data/dataset.yaml")
                
                if confirm_action("确认生成配置文件?"):
                    from ..commands.data import generate_yaml
                    generate_yaml(data_path=data_path, classes_file=None, output=output, 
                                 train_dir='images/train', val_dir='images/val', test_dir='images/test')
            
            elif operation == 'verify':
                # 验证数据集
                print_section_header("验证数据集")
                
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                
                from ..commands.data import verify_dataset
                verify_dataset(data_path=data_path)
            
            elif operation == 'stats':
                # 数据统计
                print_section_header("数据统计")
                
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                detailed = confirm_action("显示详细统计?", default=True)
                
                from ..commands.data import dataset_stats
                dataset_stats(data_path=data_path, detailed=detailed)
            
            console.print()
            if not confirm_action("继续数据操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def run_train_operations():
    """训练操作"""
    while True:
        operation = select_train_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'start':
                # 开始训练
                print_section_header("配置训练")
                
                config = build_training_config()
                
                # 数据集路径
                data_path = input_path("数据集配置文件:", default="data/dataset.yaml", must_exist=True)
                
                # 模型名称
                model_name = YOLOVersionManager.get_model_name(config['version'], config['model_size'])
                
                print_info(f"训练配置:")
                print_info(f"  模型: {model_name}")
                print_info(f"  数据集: {data_path}")
                print_info(f"  训练轮数: {config['epochs']}")
                print_info(f"  批次大小: {config['batch']}")
                print_info(f"  图像尺寸: {config['imgsz']}")
                print_info(f"  设备: {config['device']}")
                print_info(f"  数据增强: {config['augmentation']}")
                
                if confirm_action("确认开始训练?"):
                    from ..commands.train import start_training
                    start_training(
                        model=model_name,
                        data=data_path,
                        epochs=config['epochs'],
                        batch=config['batch'],
                        imgsz=config['imgsz'],
                        device=config['device'],
                        project=None,
                        name=None,
                        augmentation=config['augmentation'],
                        patience=config['patience'],
                        save_period=config['save_period'],
                        resume=False,
                        pretrained=True,
                    )
            
            elif operation == 'resume':
                # 恢复训练
                print_section_header("恢复训练")
                
                checkpoint = input_path("检查点路径 (留空自动查找):", default="results/training/last.pt", must_exist=False)
                
                if confirm_action("确认恢复训练?"):
                    from ..commands.train import resume_training
                    resume_training(checkpoint=checkpoint if checkpoint else None, project=None, name=None)
            
            elif operation == 'config':
                # 生成配置
                print_section_header("生成训练配置")
                
                output = input_text("输出文件:", default="train_config.yaml")
                
                from ..commands.train import generate_config
                generate_config(output=output, profile=None)
            
            console.print()
            if not confirm_action("继续训练操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


def run_quick_train():
    """一键训练操作"""
    print_section_header("⚡ 一键训练")
    
    try:
        print_info("一键训练会自动完成以下步骤:")
        print_info("  1. 数据集划分")
        print_info("  2. 生成配置文件")
        print_info("  3. 验证数据集")
        print_info("  4. 数据统计")
        print_info("  5. 检查/下载模型")
        print_info("  6. 开始训练")
        console.print()
        
        # 获取参数
        images_dir = input_path("图像目录:", default="data/raw/images", must_exist=False)
        labels_dir = input_path("标签目录:", default="data/raw/labels", must_exist=False)
        
        # 训练配置
        config = build_training_config()
        
        # 高级选项
        skip_verify = not confirm_action("验证数据集?", default=True)
        skip_stats = not confirm_action("统计数据分布?", default=True)
        
        console.print()
        print_info("配置摘要:")
        print_info(f"  图像目录: {images_dir}")
        print_info(f"  标签目录: {labels_dir}")
        print_info(f"  YOLO版本: {config['version']}")
        print_info(f"  模型大小: {config['model_size']}")
        print_info(f"  训练轮数: {config['epochs']}")
        print_info(f"  批次大小: {config['batch']}")
        print_info(f"  设备: {config['device']}")
        
        console.print()
        if not confirm_action("确认开始一键训练?", default=True):
            print_warning("已取消")
            return
        
        # 执行一键训练
        from ..commands.quick import quick_train
        
        quick_train(
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes_file=None,
            model_version=config['version'],
            model_size=config['model_size'],
            epochs=config['epochs'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            device=config['device'],
            augmentation=config['augmentation'],
            ratios="0.7:0.2:0.1",
            skip_verify=skip_verify,
            skip_stats=skip_stats,
            project=None,
            name=None,
        )
        
    except Exception as e:
        print_error(f"一键训练失败: {e}")


def run_detect_operations():
    """检测操作"""
    while True:
        operation = select_detect_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'image':
                # 单张图片检测
                print_section_header("单张图片检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                image_path = input_path("图片路径:", default="test.jpg", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                if confirm_action("确认检测?"):
                    from ..commands.detect import detect_image
                    detect_image(model=model_path, image=image_path, conf=conf, iou=0.45, 
                               output=None, save_txt=True, save_json=True, show=False, device='auto')
            
            elif operation == 'batch':
                # 批量检测
                print_section_header("批量检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                source_path = input_path("图片目录或视频:", default="test_images/", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                if confirm_action("确认批量检测?"):
                    from ..commands.detect import detect_batch
                    detect_batch(model=model_path, source=source_path, conf=conf, iou=0.45,
                               output=None, save_txt=True, save_json=True, device='auto', batch=1)
            
            elif operation == 'video':
                # 视频检测
                print_section_header("视频检测")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                video_path = input_path("视频路径:", default="test_video.mp4", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                show = confirm_action("实时显示?", default=False)
                
                if confirm_action("确认检测?"):
                    from ..commands.detect import detect_video
                    detect_video(model=model_path, video=video_path, conf=conf, iou=0.45,
                               output=None, save_txt=False, show=show, device='auto')
            
            console.print()
            if not confirm_action("继续检测操作?", default=False):
                break
                
        except Exception as e:
            print_error(f"操作失败: {e}")
            if not confirm_action("继续?", default=True):
                break


@app.command()
def start():
    """启动交互式模式"""
    
    try:
        # 清屏并显示Logo
        clear_screen()
        print_logo()
        
        print_info("欢迎使用 YOLO CLI 交互式模式！")
        print_info("使用方向键选择，按回车确认\n")
        
        while True:
            try:
                choice = select_main_menu()
                
                if choice == 'exit':
                    print_success("感谢使用 YOLO CLI！再见！")
                    break
                
                elif choice == 'model':
                    run_model_operations()
                
                elif choice == 'data':
                    run_data_operations()
                
                elif choice == 'train':
                    run_train_operations()
                
                elif choice == 'quick':
                    # 一键训练
                    run_quick_train()
                
                elif choice == 'detect':
                    run_detect_operations()
                
            except KeyboardInterrupt:
                console.print()
                if confirm_action("确认退出?", default=True):
                    print_success("再见！")
                    break
                continue
    
    except Exception as e:
        print_error(f"发生错误: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
