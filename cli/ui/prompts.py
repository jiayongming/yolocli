#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式提示"""

import os
import warnings

# 禁用终端 CPR (Cursor Position Request) 警告
# 这解决了在某些终端环境（如 Docker、CI/CD、SSH）下的兼容性问题
os.environ.setdefault('PROMPT_TOOLKIT_NO_CPR', '1')
warnings.filterwarnings('ignore', message='.*cursor position requests.*')

import questionary
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def select_option(message: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    选择选项
    
    Args:
        message: 提示消息
        choices: 选项列表
        default: 默认选项
    
    Returns:
        str: 选中的选项
    """
    return questionary.select(
        message,
        choices=choices,
        default=default,
    ).ask()


def select_multiple(message: str, choices: List[str]) -> List[str]:
    """
    选择多个选项
    
    Args:
        message: 提示消息
        choices: 选项列表
    
    Returns:
        List[str]: 选中的选项列表
    """
    return questionary.checkbox(
        message,
        choices=choices,
    ).ask()


def confirm_action(message: str, default: bool = True) -> bool:
    """
    确认操作
    
    Args:
        message: 提示消息
        default: 默认值
    
    Returns:
        bool: 确认结果
    """
    return questionary.confirm(
        message,
        default=default,
    ).ask()


def input_text(message: str, default: str = "", validate: Optional[callable] = None) -> str:
    """
    输入文本
    
    Args:
        message: 提示消息
        default: 默认值
        validate: 验证函数
    
    Returns:
        str: 输入的文本
    """
    return questionary.text(
        message,
        default=default,
        validate=validate,
    ).ask()


def input_path(message: str, default: str = "", must_exist: bool = False) -> str:
    """
    输入路径
    
    Args:
        message: 提示消息
        default: 默认路径
        must_exist: 是否必须存在
    
    Returns:
        str: 输入的路径
    """
    def validate_path(text):
        if must_exist and not Path(text).exists():
            return "路径不存在"
        return True
    
    return questionary.path(
        message,
        default=default,
        validate=validate_path if must_exist else None,
    ).ask()


def input_number(
    message: str,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    输入数字
    
    Args:
        message: 提示消息
        default: 默认值
        min_value: 最小值
        max_value: 最大值
    
    Returns:
        float: 输入的数字
    """
    def validate_number(text):
        try:
            num = float(text)
            if min_value is not None and num < min_value:
                return f"值必须 >= {min_value}"
            if max_value is not None and num > max_value:
                return f"值必须 <= {max_value}"
            return True
        except ValueError:
            return "请输入有效的数字"
    
    default_str = str(default) if default is not None else ""
    result = questionary.text(
        message,
        default=default_str,
        validate=validate_number,
    ).ask()
    
    return float(result)


def select_yolo_version() -> str:
    """
    选择YOLO版本
    
    Returns:
        str: 选中的版本
    """
    choices = [
        "yolo11 (推荐，最新版本)",
        "yolov8 (稳定版本)",
    ]
    
    result = select_option("选择YOLO版本:", choices, default=choices[0])
    
    # 提取版本名称
    if "yolo11" in result:
        return "yolo11"
    else:
        return "yolov8"


def select_task_type() -> str:
    """
    选择任务类型
    
    Returns:
        str: 选中的任务类型
    """
    choices = [
        "detect - 目标检测",
        "segment - 实例分割",
        "classify - 图像分类",
        "pose - 姿势估计",
    ]
    
    result = select_option("选择任务类型:", choices, default=choices[0])
    
    # 提取任务类型
    return result.split(' ')[0]


def select_model_size() -> str:
    """
    选择模型大小
    
    Returns:
        str: 选中的大小
    """
    choices = [
        "n - Nano (最快，适合边缘设备)",
        "s - Small (快速，适合大多数应用)",
        "m - Medium (平衡，推荐)",
        "l - Large (精度优先)",
        "x - Extra Large (最高精度，最慢)",
    ]
    
    result = select_option("选择模型大小:", choices, default=choices[2])
    
    # 提取大小字母
    return result.split(' ')[0]


def select_device() -> str:
    """
    选择设备
    
    Returns:
        str: 选中的设备（如果是CUDA，返回具体GPU ID如'0'或'1,2'）
    """
    choices = [
        "auto - 自动检测 (推荐)",
        "mps - Apple Silicon",
        "cuda - NVIDIA GPU",
        "cpu - CPU",
    ]
    
    result = select_option("选择训练设备:", choices, default=choices[0])
    
    # 提取设备名称
    device = result.split(' ')[0]
    
    # 如果选择了CUDA，询问GPU ID
    if device == 'cuda':
        gpu_id = input_text(
            "输入GPU设备ID (单个: 0, 多个: 0,1,2):",
            default="0",
            validate=lambda x: True if x.replace(',', '').isdigit() else "请输入有效的GPU ID"
        )
        # 如果只有一个GPU ID，直接返回数字
        if ',' not in gpu_id:
            return gpu_id
        else:
            # 多GPU训练，返回逗号分隔的ID
            return gpu_id
    
    return device


def select_augmentation_preset() -> str:
    """
    选择数据增强预设
    
    Returns:
        str: 选中的预设
    """
    choices = [
        "balanced - 平衡配置 (推荐)",
        "conservative - 保守配置 (小数据集)",
        "aggressive - 激进配置 (大数据集)",
        "default - YOLO默认配置",
    ]
    
    result = select_option("选择数据增强策略:", choices, default=choices[0])
    
    # 提取预设名称
    return result.split(' ')[0]


def select_optimizer() -> str:
    """
    选择优化器
    
    Returns:
        str: 选中的优化器
    """
    choices = [
        "auto - 自动选择 (推荐)",
        "SGD - 随机梯度下降 (YOLO默认)",
        "Adam - Adam优化器",
        "AdamW - Adam with weight decay",
        "NAdam - Nesterov-accelerated Adam",
        "RAdam - Rectified Adam",
        "RMSProp - RMSProp优化器",
    ]
    
    result = select_option("选择优化器:", choices, default=choices[0])
    return result.split(' ')[0]


def select_export_formats() -> List[str]:
    """
    选择导出格式
    
    Returns:
        List[str]: 选中的格式列表
    """
    choices = [
        "onnx - ONNX格式 (推荐，通用)",
        "torchscript - TorchScript格式",
        "tflite - TensorFlow Lite",
        "coreml - CoreML (Apple设备)",
        "engine - TensorRT Engine",
        "pb - TensorFlow SavedModel",
    ]
    
    results = select_multiple("选择导出格式 (空格选择，回车确认):", choices)
    
    # 提取格式名称
    return [r.split(' ')[0] for r in results]


def build_training_config() -> Dict[str, Any]:
    """
    交互式构建训练配置
    
    Returns:
        Dict[str, Any]: 训练配置
    """
    config = {}
    
    # YOLO版本
    config['version'] = select_yolo_version()
    
    # 任务类型
    config['task'] = select_task_type()
    
    # 模型大小
    config['model_size'] = select_model_size()
    
    # 训练参数
    config['epochs'] = int(input_number("训练轮数:", default=200, min_value=1))
    config['batch'] = int(input_number("批次大小:", default=16, min_value=1))
    config['imgsz'] = int(input_number("图像尺寸:", default=640, min_value=32))
    
    # 设备
    config['device'] = select_device()
    
    # 数据增强
    config['augmentation'] = select_augmentation_preset()
    
    # 高级选项
    if confirm_action("配置高级选项?", default=False):
        from ..ui.display import console, print_info, print_section_header
        
        console.print()
        print_section_header("⚙️ 高级选项配置")
        
        # 1. 训练控制参数
        console.print()
        print_info("📊 训练控制参数：")
        config['patience'] = int(input_number(
            "早停耐心值 (连续多少轮无改善后停止):", 
            default=50, 
            min_value=0
        ))
        config['save_period'] = int(input_number(
            "保存周期 (每隔多少轮保存一次模型):", 
            default=10, 
            min_value=1
        ))
        
        # 2. 数据增强配置
        console.print()
        print_info("🎨 数据增强配置：")
        print_info("   当前使用预设: " + config['augmentation'])
        print_info("   您可以选择微调各项增强参数，或使用预设默认值")
        console.print()
        
        if confirm_action("自定义数据增强参数?", default=False):
            console.print()
            print_section_header("🎨 数据增强详细配置")
            
            # 获取当前预设的默认值
            from ..commands.train import AUGMENTATION_PRESETS
            preset = AUGMENTATION_PRESETS.get(config['augmentation'], AUGMENTATION_PRESETS['balanced'])
            
            config['augmentation_custom'] = {}
            
            # 颜色空间增强
            console.print()
            print_info("📐 颜色空间增强 (HSV调整)：")
            config['augmentation_custom']['hsv_h'] = float(input_number(
                "  HSV-Hue增益 (色调偏移，0.0-1.0，推荐0.01-0.02):",
                default=preset['hsv_h'],
                min_value=0.0,
                max_value=1.0
            ))
            config['augmentation_custom']['hsv_s'] = float(input_number(
                "  HSV-Saturation增益 (饱和度，0.0-1.0，推荐0.5-0.8):",
                default=preset['hsv_s'],
                min_value=0.0,
                max_value=1.0
            ))
            config['augmentation_custom']['hsv_v'] = float(input_number(
                "  HSV-Value增益 (明度，0.0-1.0，推荐0.3-0.5):",
                default=preset['hsv_v'],
                min_value=0.0,
                max_value=1.0
            ))
            
            # 几何变换
            console.print()
            print_info("🔄 几何变换：")
            config['augmentation_custom']['degrees'] = float(input_number(
                "  旋转角度 (度数，0.0表示不旋转):",
                default=preset['degrees'],
                min_value=0.0,
                max_value=180.0
            ))
            config['augmentation_custom']['translate'] = float(input_number(
                "  平移比例 (图像宽高比例，0.0-0.5，推荐0.05-0.15):",
                default=preset['translate'],
                min_value=0.0,
                max_value=0.5
            ))
            config['augmentation_custom']['scale'] = float(input_number(
                "  缩放比例 (0.0-1.0，推荐0.3-0.6):",
                default=preset['scale'],
                min_value=0.0,
                max_value=1.0
            ))
            config['augmentation_custom']['shear'] = float(input_number(
                "  剪切角度 (度数，0.0表示不剪切):",
                default=preset['shear'],
                min_value=0.0,
                max_value=45.0
            ))
            config['augmentation_custom']['perspective'] = float(input_number(
                "  透视变换 (0.0-0.001，推荐0.0或极小值):",
                default=preset['perspective'],
                min_value=0.0,
                max_value=0.001
            ))
            
            # 翻转
            console.print()
            print_info("🔃 翻转增强：")
            config['augmentation_custom']['flipud'] = float(input_number(
                "  上下翻转概率 (0.0-1.0，0.0=不翻转):",
                default=preset['flipud'],
                min_value=0.0,
                max_value=1.0
            ))
            config['augmentation_custom']['fliplr'] = float(input_number(
                "  左右翻转概率 (0.0-1.0，推荐0.5):",
                default=preset['fliplr'],
                min_value=0.0,
                max_value=1.0
            ))
            
            # 高级增强
            console.print()
            print_info("🚀 高级增强技术：")
            config['augmentation_custom']['mosaic'] = float(input_number(
                "  Mosaic增强概率 (0.0-1.0，4图拼接，推荐1.0):",
                default=preset['mosaic'],
                min_value=0.0,
                max_value=1.0
            ))
            config['augmentation_custom']['mixup'] = float(input_number(
                "  MixUp增强概率 (0.0-1.0，图像混合，推荐0.0-0.15):",
                default=preset['mixup'],
                min_value=0.0,
                max_value=1.0
            ))
            config['augmentation_custom']['erasing'] = float(input_number(
                "  随机擦除概率 (0.0-1.0，推荐0.1-0.4):",
                default=preset['erasing'],
                min_value=0.0,
                max_value=1.0
            ))
            
            # AutoAugment
            console.print()
            print_info("🤖 自动增强策略：")
            if confirm_action("  启用AutoAugment (RandAugment策略)?", default=(preset.get('auto_augment') is not None)):
                config['augmentation_custom']['auto_augment'] = 'randaugment'
            else:
                config['augmentation_custom']['auto_augment'] = None
            
            console.print()
            print_info("✓ 数据增强自定义配置完成")
        else:
            config['augmentation_custom'] = None
        
        # 3. 优化器配置
        console.print()
        print_info("⚡ 优化器配置：")
        
        # 优化器类型选择
        config['optimizer_type'] = select_optimizer()
        print_info(f"   选择的优化器: {config['optimizer_type']}")
        
        if confirm_action("自定义优化器参数?", default=False):
            config['optimizer'] = {}
            
            # 学习率
            config['optimizer']['lr0'] = float(input_number(
                "  初始学习率 (推荐: 检测0.01, 分类0.001):",
                default=0.01,
                min_value=0.0001,
                max_value=1.0
            ))
            config['optimizer']['lrf'] = float(input_number(
                "  最终学习率比例 (lr_final = lr0 * lrf，推荐0.01):",
                default=0.01,
                min_value=0.0001,
                max_value=1.0
            ))
            
            # 动量和权重衰减
            config['optimizer']['momentum'] = float(input_number(
                "  SGD动量 (推荐0.937):",
                default=0.937,
                min_value=0.0,
                max_value=0.999
            ))
            config['optimizer']['weight_decay'] = float(input_number(
                "  权重衰减 (L2正则化，推荐0.0005):",
                default=0.0005,
                min_value=0.0,
                max_value=0.01
            ))
            
            # Warmup
            config['optimizer']['warmup_epochs'] = float(input_number(
                "  Warmup轮数 (推荐3.0):",
                default=3.0,
                min_value=0.0,
                max_value=10.0
            ))
            config['optimizer']['warmup_momentum'] = float(input_number(
                "  Warmup初始动量 (推荐0.8):",
                default=0.8,
                min_value=0.0,
                max_value=0.999
            ))
            config['optimizer']['warmup_bias_lr'] = float(input_number(
                "  Warmup偏置学习率 (推荐0.1):",
                default=0.1,
                min_value=0.0,
                max_value=1.0
            ))
        else:
            config['optimizer'] = None
        
        # 4. 层冻结配置
        console.print()
        print_info("❄️ 层冻结配置（用于迁移学习）：")
        if confirm_action("冻结模型层?", default=False):
            freeze_layers = int(input_number(
                "  冻结前N层 (0=不冻结, 10=冻结前10层, 推荐: 检测10, 分割12):",
                default=10,
                min_value=0,
                max_value=100
            ))
            if freeze_layers > 0:
                config['freeze'] = freeze_layers
                print_info(f"   将冻结前 {freeze_layers} 层")
            else:
                config['freeze'] = None
                print_info("   不冻结任何层")
        else:
            config['freeze'] = None
        
        # 5. 损失函数权重
        console.print()
        print_info("⚖️ 损失函数权重：")
        if confirm_action("自定义损失函数权重?", default=False):
            config['loss_weights'] = {}
            
            if config['task'] == 'detect' or config['task'] == 'segment':
                config['loss_weights']['box'] = float(input_number(
                    "  边界框损失权重 (推荐7.5):",
                    default=7.5,
                    min_value=0.1,
                    max_value=20.0
                ))
                config['loss_weights']['cls'] = float(input_number(
                    "  分类损失权重 (推荐0.5):",
                    default=0.5,
                    min_value=0.1,
                    max_value=10.0
                ))
                config['loss_weights']['dfl'] = float(input_number(
                    "  DFL损失权重 (推荐1.5):",
                    default=1.5,
                    min_value=0.1,
                    max_value=10.0
                ))
            
            if config['task'] == 'classify':
                config['loss_weights']['label_smoothing'] = float(input_number(
                    "  标签平滑 (0.0-0.1，推荐0.0):",
                    default=0.0,
                    min_value=0.0,
                    max_value=0.1
                ))
        else:
            config['loss_weights'] = None
        
        # 6. Pose任务特定配置
        if config['task'] == 'pose':
            console.print()
            print_info("🤸 Pose任务特定配置：")
            
            # 关键点配置
            preset_choice = select_option(
                "选择关键点预设:",
                ["COCO (17个关键点 - 人体)", "自定义"],
                default="COCO (17个关键点 - 人体)"
            )
            
            if "COCO" in preset_choice:
                config['kpt_shape'] = [17, 3]
                config['flip_idx'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
                print_info("  使用COCO 17关键点配置")
            else:
                kpt_count = int(input_number(
                    "  关键点数量:",
                    default=17,
                    min_value=1
                ))
                config['kpt_shape'] = [kpt_count, 3]
                config['flip_idx'] = None
                print_info(f"  配置 {kpt_count} 个关键点")
        
        console.print()
        print_info("✓ 高级选项配置完成")
        
    else:
        config['patience'] = 50
        config['save_period'] = 10
        config['augmentation_custom'] = None
        config['optimizer_type'] = 'auto'
        config['optimizer'] = None
        config['freeze'] = None
        config['loss_weights'] = None
    
    return config


def select_main_menu() -> str:
    """
    显示主菜单
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "quick - 一键训练 (自动化完整流程) ⚡",
        "model - 模型管理 (下载、导出、列表)",
        "data - 数据处理 (划分、验证、统计)",
        "train - 模型训练 (训练、恢复)",
        "validate - 模型验证 (性能评估、模型对比)",
        "detect - 图像检测 (单图、批量)",
        "labelstudio - Label Studio管理 (获取项目、下载数据) 🆕",
        "fiftyone - FiftyOne可视化 (数据集查看、管理) 🆕",
        "exit - 退出",
    ]
    
    result = select_option("请选择操作:", choices, default=choices[0])
    
    # 提取操作名称
    return result.split(' ')[0]


def select_model_operation() -> str:
    """
    选择模型操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "download - 下载预训练模型",
        "export - 导出模型",
        "list - 列出本地模型",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择模型操作:", choices)
    return result.split(' ')[0]


def select_data_operation() -> str:
    """
    选择数据操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "convert-labelstudio - 转换Label Studio数据",
        "split - 划分数据集",
        "generate-yaml - 生成dataset.yaml",
        "verify - 验证数据集",
        "stats - 数据统计",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择数据操作:", choices)
    return result.split(' ')[0]


def select_train_operation() -> str:
    """
    选择训练操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "start - 开始训练",
        "resume - 恢复训练",
        "config - 生成配置",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择训练操作:", choices)
    return result.split(' ')[0]


def select_detect_operation() -> str:
    """
    选择检测操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "image - 单张图片检测",
        "batch - 批量检测",
        "video - 视频检测",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择检测操作:", choices)
    return result.split(' ')[0]


def select_validate_operation() -> str:
    """
    选择验证操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "run - 验证单个模型",
        "compare - 比较多个模型",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择验证操作:", choices)
    return result.split(' ')[0]


def select_validation_split() -> str:
    """
    选择验证数据集
    
    Returns:
        str: 选中的数据集
    """
    choices = [
        "val - 验证集 (用于模型选择)",
        "test - 测试集 (用于最终评估)",
        "train - 训练集 (检查过拟合)",
    ]
    
    result = select_option("选择验证数据集:", choices, default=choices[0])
    return result.split(' ')[0]


def select_labelstudio_operation() -> str:
    """
    选择Label Studio操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "upload - 上传数据集到Label Studio",
        "batch-annotate - 批量打标签",
        "predict - 使用本地模型预测任务",
        "audit - 审计标注质量",
        "list - 列出所有项目",
        "fetch - 获取项目数据",
        "config - 配置Label Studio连接",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择Label Studio操作:", choices)
    return result.split(' ')[0]


def select_labelstudio_project(projects: List[Dict]) -> Optional[Dict]:
    """
    从项目列表中选择项目
    
    Args:
        projects: 项目列表
    
    Returns:
        Optional[Dict]: 选中的项目，或None表示返回
    """
    if not projects:
        return None
    
    # 构建选项列表
    choices = []
    for proj in projects:
        title = proj.get('title', '未命名项目')
        task_num = proj.get('task_number', 0)
        proj_id = proj.get('id', 0)
        choices.append(f"{proj_id} - {title} ({task_num} 任务)")
    
    choices.append("返回")
    
    result = select_option("选择项目:", choices)
    
    if result == "返回":
        return None
    
    # 提取项目ID
    proj_id = int(result.split(' - ')[0])
    
    # 找到对应项目
    for proj in projects:
        if proj.get('id') == proj_id:
            return proj
    
    return None


def input_labelstudio_config() -> Dict[str, str]:
    """
    输入Label Studio配置
    
    Returns:
        Dict[str, str]: 配置字典 {'url': str, 'token': str}
    """
    url = input_text("Label Studio URL:", default="http://localhost:8080")
    token = input_text("访问令牌 (支持Refresh Token):")
    
    return {'url': url, 'token': token}


def select_fiftyone_operation() -> str:
    """
    选择FiftyOne操作
    
    Returns:
        str: 选中的操作
    """
    choices = [
        "load - 加载数据集（Ground Truth）",
        "load_predictions - 加载预测结果",
        "add_predictions - 添加预测到现有数据集",
        "launch - 启动可视化",
        "list - 列出所有数据集",
        "info - 查看数据集信息",
        "delete - 删除数据集",
        "back - 返回主菜单",
    ]
    
    result = select_option("选择FiftyOne操作:", choices)
    return result.split(' ')[0]


def select_fiftyone_dataset(datasets: List[str], allow_delete_all: bool = False) -> Optional[str]:
    """
    从数据集列表中选择
    
    Args:
        datasets: 数据集名称列表
        allow_delete_all: 是否允许"删除全部"选项（仅在删除场景下使用）
    
    Returns:
        Optional[str]: 选中的数据集名称，"__DELETE_ALL__"表示删除全部，None表示返回
    """
    if not datasets:
        return None
    
    choices = list(datasets)
    if allow_delete_all and len(datasets) > 1:
        choices.insert(0, "🗑️  删除全部数据集")
    choices.append("返回")
    
    result = select_option("选择数据集:", choices)
    
    if result == "返回":
        return None
    elif result == "🗑️  删除全部数据集":
        return "__DELETE_ALL__"
    
    return result


def select_task_filter_mode() -> str:
    """
    选择Label Studio任务筛选方式
    
    Returns:
        str: 筛选方式 ('ids', 'range', 'unlabeled')
    """
    choices = [
        "按任务ID列表筛选（输入多个ID，逗号分隔）",
        "按任务ID范围筛选（输入起始ID和结束ID）",
        "仅处理未标注的任务",
    ]
    
    result = select_option("选择任务筛选方式:", choices)
    
    if "ID列表" in result:
        return 'ids'
    elif "ID范围" in result:
        return 'range'
    else:
        return 'unlabeled'


def input_task_ids() -> List[int]:
    """
    输入任务ID列表
    
    Returns:
        List[int]: 任务ID列表
    """
    while True:
        ids_str = input_text(
            "输入任务ID（逗号分隔）:",
            default="1,2,3"
        )
        
        try:
            # 解析逗号分隔的ID
            ids = [int(id_str.strip()) for id_str in ids_str.split(',') if id_str.strip()]
            if ids:
                return ids
            else:
                console.print("[red]✗ 请至少输入一个任务ID[/red]")
        except ValueError:
            console.print("[red]✗ 请输入有效的数字ID（用逗号分隔）[/red]")


def input_task_range() -> Tuple[int, int]:
    """
    输入任务ID范围
    
    Returns:
        Tuple[int, int]: (起始ID, 结束ID)
    """
    while True:
        start_id = input_number(
            "起始任务ID:",
            min_value=1
        )
        
        end_id = input_number(
            "结束任务ID:",
            min_value=start_id
        )
        
        if start_id <= end_id:
            return (int(start_id), int(end_id))
        else:
            console.print("[red]✗ 起始ID必须小于或等于结束ID[/red]")


def select_task_type_for_predict() -> Optional[str]:
    """
    选择预测任务类型
    
    Returns:
        Optional[str]: 任务类型 (None表示自动推断, 或 'detect'/'segment'/'pose'/'classify')
    """
    choices = [
        "自动推断（从模型推断）",
        "检测 (detect)",
        "分割 (segment)",
        "姿态估计 (pose)",
        "分类 (classify)",
    ]
    
    result = select_option("任务类型:", choices)
    
    if "自动推断" in result:
        return None
    elif "检测" in result:
        return 'detect'
    elif "分割" in result:
        return 'segment'
    elif "姿态" in result:
        return 'pose'
    else:
        return 'classify'


def select_annotation_type() -> str:
    """
    选择标注类型（矩形框或关键点）
    
    Returns:
        str: 'rectangle' 或 'keypoint'
    """
    choices = [
        "矩形框标注 (RectangleLabels)",
        "关键点标注 (KeyPointLabels)",
    ]
    
    result = select_option("选择要添加的标注类型:", choices)
    
    return 'rectangle' if "矩形框" in result else 'keypoint'


def select_target_type() -> str:
    """
    选择目标类型（annotation or prediction）
    
    Returns:
        str: 'annotation' 或 'prediction'
    """
    choices = [
        "Annotation - 正式标注（用于训练，可直接导出）",
        "Prediction - 预测结果（需要人工审核后才能作为标注）",
    ]
    
    result = select_option("选择标注目标类型:", choices)
    
    return 'annotation' if "Annotation" in result else 'prediction'


def select_merge_mode(has_existing_annotations: bool = False) -> str:
    """
    选择标注合并模式
    
    Args:
        has_existing_annotations: 是否检测到已有标注的tasks
    
    Returns:
        str: 'add', 'skip', 或 'overwrite_same_type'
    """
    from .display import print_warning
    
    if has_existing_annotations:
        print_warning("\n⚠️  检测到部分tasks已有标注")
    
    choices = [
        "追加模式 - 在已有标注基础上添加新标注（推荐） ✅",
        "跳过模式 - 跳过所有已有标注的tasks",
        "覆盖同类型 - 替换同类型标注，保留其他类型",
    ]
    
    result = select_option("选择标注合并模式:", choices)
    
    if "追加" in result:
        return 'add'
    elif "跳过" in result:
        return 'skip'
    else:
        return 'overwrite_same_type'


def input_rectangle_annotation(available_labels: List[str]) -> Dict[str, Any]:
    """
    交互式输入矩形框标注
    
    Args:
        available_labels: 可用的标签列表
    
    Returns:
        {
            'label': 'circle_meter',
            'center_x': 0.5,
            'center_y': 0.5,
            'width': 0.3,
            'height': 0.2
        }
    """
    from .display import print_info
    
    print_info("\n📐 请输入矩形框标注信息（坐标为归一化值，范围0-1）")
    print_info("提示：0.5表示图片中心，0表示左/上边缘，1表示右/下边缘\n")
    
    label = select_option("选择类别:", available_labels)
    
    center_x = input_number(
        "中心点X坐标（0-1）:",
        min_value=0.0,
        max_value=1.0,
        default=0.5
    )
    
    center_y = input_number(
        "中心点Y坐标（0-1）:",
        min_value=0.0,
        max_value=1.0,
        default=0.5
    )
    
    width = input_number(
        "矩形框宽度（0-1）:",
        min_value=0.0,
        max_value=1.0,
        default=0.3
    )
    
    height = input_number(
        "矩形框高度（0-1）:",
        min_value=0.0,
        max_value=1.0,
        default=0.2
    )
    
    return {
        'label': label,
        'center_x': center_x,
        'center_y': center_y,
        'width': width,
        'height': height
    }


def input_keypoint_annotations(keypoint_labels: List[str]) -> List[Dict[str, Any]]:
    """
    交互式输入关键点标注
    
    Args:
        keypoint_labels: 关键点标签列表
    
    Returns:
        [
            {'label': 'start', 'x': 0.3, 'y': 0.4, 'visible': True},
            {'label': 'end', 'x': 0.7, 'y': 0.4, 'visible': True},
            ...
        ]
    """
    from .display import print_info
    
    print_info(f"\n🎯 请为 {len(keypoint_labels)} 个关键点输入坐标（归一化值，范围0-1）\n")
    
    keypoints = []
    for i, kp_label in enumerate(keypoint_labels, 1):
        print_info(f"关键点 {i}/{len(keypoint_labels)}: {kp_label}")
        
        x = input_number(
            f"  X坐标（0-1）:",
            min_value=0.0,
            max_value=1.0,
            default=0.5
        )
        
        y = input_number(
            f"  Y坐标（0-1）:",
            min_value=0.0,
            max_value=1.0,
            default=0.5
        )
        
        visible = confirm(
            f"  该关键点是否可见？",
            default=True
        )
        
        keypoints.append({
            'label': kp_label,
            'x': x,
            'y': y,
            'visible': visible
        })
        
        print()  # 空行分隔
    
    return keypoints
