#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式模式"""

import typer
from pathlib import Path

from ..ui.prompts import (
    select_main_menu, select_model_operation, select_data_operation,
    select_train_operation, select_detect_operation, select_validate_operation,
    select_validation_split,
    select_yolo_version, select_task_type, select_model_size, select_device,
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
from . import model, data, train, detect, validate

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
                task_type = select_task_type()
                size_str = select_model_size()
                
                sizes = [size_str] if size_str != 'all' else None
                download_all = (size_str == 'all')
                
                if confirm_action(f"确认下载 {version} {task_type} {size_str if not download_all else '所有'} 模型?"):
                    from ..commands.model import download
                    download(version=version, size=sizes, task=task_type, all=download_all, output_dir=None)
            
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
            if operation == 'convert-labelstudio':
                # 转换Label Studio数据
                print_section_header("转换Label Studio数据")
                
                input_file = input_path("Label Studio导出文件 (JSON/CSV):", default="labelstudioexport/project.json", must_exist=False)
                url = input_text("Label Studio URL:", default="http://localhost:8080")
                token = input_text("访问令牌 (支持Refresh Token):")
                task_type = select_task_type()
                max_workers = input_number("并发下载线程数:", default=4, min_value=1, max_value=16)
                
                if confirm_action(f"确认转换 {task_type} 数据?"):
                    from ..commands.data import convert_labelstudio
                    convert_labelstudio(
                        input_file=input_file,
                        url=url,
                        token=token,
                        output_dir=None,
                        task=task_type,
                        format_type='auto',
                        skip_existing=True,
                        max_workers=max_workers
                    )
            
            elif operation == 'split':
                # 划分数据集
                print_section_header("划分数据集")
                
                task_type = select_task_type()
                
                if task_type == 'classify':
                    # 分类任务
                    source_dir = input_path("源目录 (已按类别组织):", default="data/raw/images", must_exist=False)
                    ratios = input_text("划分比例 (train:val:test):", default="0.7:0.2:0.1")
                    
                    if confirm_action("确认划分分类数据集?"):
                        from ..commands.data import split_dataset
                        split_dataset(
                            images_dir=None,
                            labels_dir=None,
                            source_dir=source_dir,
                            output_dir=None,
                            ratios=ratios,
                            seed=42,
                            task=task_type
                        )
                else:
                    # 检测/分割任务
                    images_dir = input_path("图像目录:", default="data/raw/images", must_exist=False)
                    labels_dir = input_path("标签目录:", default="data/raw/labels", must_exist=False)
                    ratios = input_text("划分比例 (train:val:test):", default="0.7:0.2:0.1")
                    
                    if confirm_action("确认划分数据集?"):
                        from ..commands.data import split_dataset
                        split_dataset(
                            images_dir=images_dir,
                            labels_dir=labels_dir,
                            source_dir=None,
                            output_dir=None,
                            ratios=ratios,
                            seed=42,
                            task=task_type
                        )
            
            elif operation == 'generate-yaml':
                # 生成dataset.yaml
                print_section_header("生成 dataset.yaml")
                
                task_type = select_task_type()
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                output = input_text("输出文件:", default="data/dataset.yaml")
                
                if confirm_action("确认生成配置文件?"):
                    from ..commands.data import generate_yaml
                    generate_yaml(
                        data_path=data_path,
                        classes_file=None,
                        output=output,
                        train_dir=None,
                        val_dir=None,
                        test_dir=None,
                        task=task_type
                    )
            
            elif operation == 'verify':
                # 验证数据集
                print_section_header("验证数据集")
                
                task_type = select_task_type()
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                
                from ..commands.data import verify_dataset
                verify_dataset(data_path=data_path, task=task_type)
            
            elif operation == 'stats':
                # 数据统计
                print_section_header("数据统计")
                
                task_type = select_task_type()
                data_path = input_path("数据集路径:", default="data/processed", must_exist=False)
                detailed = confirm_action("显示详细统计 (含正负样本统计)?", default=True)
                
                # 如果是分类任务且需要详细统计，让用户选择正类
                positive_classes_str = None
                if task_type == 'classify' and detailed:
                    from pathlib import Path
                    data_path_obj = Path(data_path)
                    
                    # 获取所有类别
                    all_classes = set()
                    for split in ['train', 'val', 'test']:
                        split_dir = data_path_obj / 'images' / split
                        if split_dir.exists():
                            classes = [d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                            all_classes.update(classes)
                    
                    if all_classes:
                        print_info(f"检测到 {len(all_classes)} 个类别: {', '.join(sorted(all_classes))}")
                        console.print()
                        print_info("💡 正负样本统计说明：")
                        print_info("   - 选择一个或多个类别作为「正类」")
                        print_info("   - 其余类别将自动归为「负类」")
                        print_info("   - 适用于异常检测、二分类等场景")
                        console.print()
                        
                        if confirm_action("是否选择正类进行正负样本统计?", default=True):
                            from ..ui.prompts import select_multiple
                            
                            console.print()
                            print_info("📋 操作说明：")
                            print_info("   1. 使用 ↑↓ 键移动")
                            print_info("   2. 使用 空格键 选择/取消选择")
                            print_info("   3. 按 回车键 确认选择")
                            console.print()
                            
                            positive_classes = select_multiple(
                                "选择正类 (空格选择，回车确认):",
                                sorted(list(all_classes))
                            )
                            
                            if positive_classes:
                                positive_classes_str = ','.join(positive_classes)
                                console.print()
                                print_success(f"✓ 已选择正类: {positive_classes_str}")
                                negative_classes = [c for c in all_classes if c not in positive_classes]
                                if negative_classes:
                                    print_info(f"  负类: {', '.join(sorted(negative_classes))}")
                            else:
                                console.print()
                                print_warning("⚠️  未选择任何正类，将跳过正负样本统计")
                
                from ..commands.data import dataset_stats
                dataset_stats(data_path=data_path, detailed=detailed, task=task_type, positive_classes=positive_classes_str)
            
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
        
        # 选择任务类型
        task_type = select_task_type()
        
        # 根据任务类型获取不同的数据路径
        # 一键训练从原始数据开始，自动完成划分
        if task_type == 'classify':
            images_dir = input_path("图像目录 (按类别组织):", default="data/raw/images", must_exist=False)
            labels_dir = images_dir  # 分类任务图像和标签目录相同
        else:
            images_dir = input_path("图像目录:", default="data/raw/images", must_exist=False)
            labels_dir = input_path("标签目录:", default="data/raw/labels", must_exist=False)
        
        # 训练配置
        config = build_training_config()
        
        # 高级选项
        skip_verify = not confirm_action("验证数据集?", default=True)
        skip_stats = not confirm_action("统计数据分布?", default=True)
        
        console.print()
        print_info("配置摘要:")
        print_info(f"  任务类型: {task_type}")
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
            task=task_type,
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


def run_validate_operations():
    """验证操作"""
    while True:
        operation = select_validate_operation()
        
        if operation == 'back':
            break
        
        try:
            if operation == 'run':
                # 验证单个模型
                print_section_header("验证模型性能")
                
                model_path = input_path("模型路径:", default="results/training/best.pt", must_exist=False)
                data_path = input_path("数据集配置文件:", default="data/dataset.yaml", must_exist=False)
                split = select_validation_split()
                
                # 验证参数
                conf = input_number("置信度阈值 (训练验证用0.001，部署验证用0.25):", default=0.001, min_value=0.0, max_value=1.0)
                iou = input_number("IoU阈值:", default=0.6, min_value=0.0, max_value=1.0)
                batch = int(input_number("批次大小:", default=16, min_value=1))
                
                # 可选项
                save_json = confirm_action("保存JSON格式结果?", default=True)
                plots = confirm_action("生成可视化图表?", default=True)
                
                console.print()
                print_info("验证配置:")
                print_info(f"  模型: {model_path}")
                print_info(f"  数据集: {data_path}")
                print_info(f"  验证集: {split}")
                print_info(f"  置信度阈值: {conf}")
                print_info(f"  IoU阈值: {iou}")
                print_info(f"  批次大小: {batch}")
                
                if confirm_action("确认开始验证?"):
                    from ..commands.validate import validate_model
                    validate_model(
                        model=model_path,
                        data=data_path,
                        split=split,
                        task=None,  # 自动推断
                        batch=batch,
                        imgsz=640,
                        conf=conf,
                        iou=iou,
                        device='auto',
                        save_json=save_json,
                        save_hybrid=False,
                        plots=plots,
                        verbose=True,
                        project=None,
                        name=None,
                    )
            
            elif operation == 'compare':
                # 比较多个模型
                print_section_header("比较多个模型")
                
                print_info("请输入要比较的模型路径，用逗号分隔")
                print_info("示例: model1.pt,model2.pt,model3.pt")
                models_str = input_text("模型路径 (逗号分隔):", default="")
                
                if not models_str:
                    print_warning("未输入模型路径")
                    continue
                
                data_path = input_path("数据集配置文件:", default="data/dataset.yaml", must_exist=False)
                conf = input_number("置信度阈值:", default=0.25, min_value=0.0, max_value=1.0)
                
                console.print()
                print_info(f"将比较以下模型:")
                for i, model in enumerate(models_str.split(','), 1):
                    print_info(f"  {i}. {model.strip()}")
                
                if confirm_action("确认开始对比?"):
                    from ..commands.validate import compare_models
                    compare_models(
                        models=models_str,
                        data=data_path,
                        split='val',
                        conf=conf,
                        iou=0.6,
                        device='auto',
                    )
            
            console.print()
            if not confirm_action("继续验证操作?", default=False):
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
                
                elif choice == 'validate':
                    run_validate_operations()
                
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
