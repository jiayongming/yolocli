#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""美化输出和显示"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from rich.tree import Tree
from rich import box
from typing import List, Dict, Optional, Any
import sys


# 全局console对象
console = Console()


def print_logo():
    """打印YOLO CLI Logo"""
    logo = """
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║   ██╗   ██╗ ██████╗ ██╗      ██████╗     ║
    ║   ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ║
    ║    ╚████╔╝ ██║   ██║██║     ██║   ██║    ║
    ║     ╚██╔╝  ██║   ██║██║     ██║   ██║    ║
    ║      ██║   ╚██████╔╝███████╗╚██████╔╝    ║
    ║      ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝     ║
    ║                                           ║
    ║          CLI Framework v1.0.0             ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
    """
    console.print(logo, style="bold cyan")


def print_success(message: str):
    """打印成功消息"""
    console.print(f"✓ {message}", style="bold green")


def print_error(message: str):
    """打印错误消息"""
    console.print(f"✗ {message}", style="bold red")


def print_warning(message: str):
    """打印警告消息"""
    console.print(f"⚠ {message}", style="bold yellow")


def print_info(message: str):
    """打印信息消息"""
    console.print(f"ℹ {message}", style="bold blue")


def print_step(step: int, total: int, message: str):
    """打印步骤消息"""
    console.print(f"[{step}/{total}] {message}", style="bold cyan")


def create_progress_bar() -> Progress:
    """创建进度条"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


def print_section_header(title: str):
    """打印章节标题"""
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()


def print_panel(content: str, title: str = "", style: str = "cyan"):
    """打印面板"""
    panel = Panel(content, title=title, border_style=style, box=box.ROUNDED)
    console.print(panel)


def create_table(
    title: str,
    columns: List[str],
    rows: List[List[Any]],
    show_header: bool = True,
    show_lines: bool = False,
) -> Table:
    """
    创建表格
    
    Args:
        title: 表格标题
        columns: 列名列表
        rows: 行数据列表
        show_header: 是否显示表头
        show_lines: 是否显示分隔线
    
    Returns:
        Table: 表格对象
    """
    table = Table(
        title=title,
        show_header=show_header,
        show_lines=show_lines,
        box=box.ROUNDED,
        title_style="bold cyan",
    )
    
    # 添加列
    for col in columns:
        table.add_column(col, style="cyan", no_wrap=False)
    
    # 添加行
    for row in rows:
        table.add_row(*[str(item) for item in row])
    
    return table


def print_table(
    title: str,
    columns: List[str],
    rows: List[List[Any]],
    show_header: bool = True,
    show_lines: bool = False,
):
    """打印表格"""
    table = create_table(title, columns, rows, show_header, show_lines)
    console.print(table)


def print_model_list(models: List[Dict[str, Any]]):
    """
    打印模型列表
    
    Args:
        models: 模型信息列表
    """
    if not models:
        print_warning("没有找到模型")
        return
    
    columns = ["模型名称", "版本", "大小", "参数量", "文件大小", "路径"]
    rows = []
    
    for model in models:
        rows.append([
            model.get('name', 'N/A'),
            model.get('version', 'N/A'),
            model.get('size', 'N/A'),
            model.get('params', 'N/A'),
            model.get('file_size', 'N/A'),
            model.get('path', 'N/A'),
        ])
    
    print_table("可用模型列表", columns, rows, show_lines=True)


def print_dataset_info(info: Dict[str, Any]):
    """
    打印数据集信息
    
    Args:
        info: 数据集信息字典
    """
    columns = ["数据集", "图像数量", "标签数量"]
    rows = [
        ["训练集", info.get('train_images', 0), info.get('train_labels', 0)],
        ["验证集", info.get('val_images', 0), info.get('val_labels', 0)],
        ["测试集", info.get('test_images', 0), info.get('test_labels', 0)],
    ]
    
    total_images = sum([info.get('train_images', 0), info.get('val_images', 0), info.get('test_images', 0)])
    total_labels = sum([info.get('train_labels', 0), info.get('val_labels', 0), info.get('test_labels', 0)])
    
    rows.append(["总计", total_images, total_labels])
    
    print_table("数据集统计", columns, rows, show_lines=True)


def print_training_config(config: Dict[str, Any]):
    """
    打印训练配置
    
    Args:
        config: 训练配置字典
    """
    content = []
    
    for key, value in config.items():
        content.append(f"{key}: [bold cyan]{value}[/bold cyan]")
    
    print_panel("\n".join(content), title="训练配置", style="green")


def print_detection_results(results: List[Dict[str, Any]], image_name: str):
    """
    打印检测结果
    
    Args:
        results: 检测结果列表
        image_name: 图像名称
    """
    if not results:
        print_warning(f"{image_name}: 未检测到目标")
        return
    
    console.print(f"\n[bold cyan]{image_name}[/bold cyan] - 检测到 {len(results)} 个目标:")
    
    columns = ["类别", "置信度", "边界框 (x1, y1, x2, y2)"]
    rows = []
    
    for result in results:
        bbox = result.get('bbox', [])
        bbox_str = f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})" if len(bbox) >= 4 else "N/A"
        
        rows.append([
            result.get('class_name', 'N/A'),
            f"{result.get('confidence', 0):.2%}",
            bbox_str,
        ])
    
    print_table("", columns, rows)


def print_segmentation_results(results: List[Dict[str, Any]], image_name: str):
    """
    打印分割结果
    
    Args:
        results: 分割结果列表
        image_name: 图像名称
    """
    if not results:
        print_warning(f"{image_name}: 未分割到目标")
        return
    
    console.print(f"\n[bold cyan]{image_name}[/bold cyan] - 分割到 {len(results)} 个目标:")
    
    columns = ["类别", "置信度", "边界框", "掩码面积"]
    rows = []
    
    for result in results:
        bbox = result.get('bbox', [])
        bbox_str = f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})" if len(bbox) >= 4 else "N/A"
        mask_area = result.get('mask_area', 0)
        
        rows.append([
            result.get('class_name', 'N/A'),
            f"{result.get('confidence', 0):.2%}",
            bbox_str,
            f"{mask_area:.0f} px" if mask_area > 0 else "N/A",
        ])
    
    print_table("", columns, rows)


def print_classification_results(results: List[Dict[str, Any]], image_name: str, top_k: int = 5):
    """
    打印分类结果
    
    Args:
        results: 分类结果列表（Top-K）
        image_name: 图像名称
        top_k: 显示Top-K结果
    """
    if not results:
        print_warning(f"{image_name}: 无分类结果")
        return
    
    console.print(f"\n[bold cyan]{image_name}[/bold cyan] - Top-{min(top_k, len(results))} 分类结果:")
    
    columns = ["排名", "类别", "置信度", "概率条"]
    rows = []
    
    for i, result in enumerate(results[:top_k], 1):
        confidence = result.get('confidence', 0)
        bar_length = int(confidence * 20)  # 20个字符的进度条
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        rows.append([
            f"#{i}",
            result.get('class_name', 'N/A'),
            f"{confidence:.2%}",
            bar,
        ])
    
    print_table("", columns, rows)


def create_tree(root_label: str) -> Tree:
    """
    创建树形结构
    
    Args:
        root_label: 根节点标签
    
    Returns:
        Tree: 树对象
    """
    return Tree(root_label, style="bold cyan")


def print_tree(tree: Tree):
    """打印树形结构"""
    console.print(tree)


def print_command_help(command: str, description: str, examples: List[str]):
    """
    打印命令帮助
    
    Args:
        command: 命令名称
        description: 命令描述
        examples: 使用示例列表
    """
    print_section_header(f"命令: {command}")
    
    console.print(f"[bold]描述:[/bold] {description}\n")
    
    if examples:
        console.print("[bold]使用示例:[/bold]")
        for i, example in enumerate(examples, 1):
            console.print(f"  {i}. [cyan]{example}[/cyan]")
    
    console.print()


def confirm(message: str, default: bool = True) -> bool:
    """
    确认对话框
    
    Args:
        message: 提示消息
        default: 默认值
    
    Returns:
        bool: 用户选择
    """
    suffix = " [Y/n]" if default else " [y/N]"
    console.print(f"[yellow]{message}{suffix}[/yellow]", end=" ")
    
    try:
        response = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print()
        return False
    
    if not response:
        return default
    
    return response in ['y', 'yes', '是']


def print_separator():
    """打印分隔符"""
    console.rule(style="dim")


def clear_screen():
    """清空屏幕"""
    console.clear()


def print_status(message: str, status: str = "info"):
    """
    打印状态消息
    
    Args:
        message: 消息内容
        status: 状态类型 (info/success/warning/error)
    """
    styles = {
        'info': ('ℹ', 'blue'),
        'success': ('✓', 'green'),
        'warning': ('⚠', 'yellow'),
        'error': ('✗', 'red'),
    }
    
    symbol, color = styles.get(status, ('•', 'white'))
    console.print(f"{symbol} {message}", style=f"bold {color}")


def print_key_value(key: str, value: Any, key_style: str = "cyan", value_style: str = "white"):
    """
    打印键值对
    
    Args:
        key: 键
        value: 值
        key_style: 键的样式
        value_style: 值的样式
    """
    console.print(f"[{key_style}]{key}:[/{key_style}] [{value_style}]{value}[/{value_style}]")


def print_dict(data: Dict[str, Any], indent: int = 0):
    """
    递归打印字典
    
    Args:
        data: 字典数据
        indent: 缩进级别
    """
    prefix = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            console.print(f"{prefix}[cyan]{key}:[/cyan]")
            print_dict(value, indent + 1)
        else:
            console.print(f"{prefix}[cyan]{key}:[/cyan] [white]{value}[/white]")


def print_banner(text: str, style: str = "bold cyan"):
    """
    打印横幅
    
    Args:
        text: 横幅文本
        style: 样式
    """
    console.print()
    console.rule(f"[{style}]{text}[/{style}]", style=style)
    console.print()
