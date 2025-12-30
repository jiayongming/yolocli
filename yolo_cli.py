#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO CLI - YOLO推理快捷操作框架

一个功能完整的命令行工具，用于YOLO模型的训练、推理和管理。

作者: YOLO CLI Team
版本: 1.0.0
"""

import os
import warnings

# 禁用终端 CPR (Cursor Position Request) 警告
# 这解决了在某些终端环境（如 Docker、CI/CD、SSH）下的兼容性问题
os.environ.setdefault('PROMPT_TOOLKIT_NO_CPR', '1')
warnings.filterwarnings('ignore', message='.*cursor position requests.*')

import typer
from typing import Optional
from rich.console import Console

from cli.commands import model, data, train, detect, interactive, quick, predict, validate, labelstudio
from cli.ui.display import print_logo, print_info
from cli import __version__

# 创建主应用
app = typer.Typer(
    name="yolo-cli",
    help="YOLO推理CLI快捷操作框架 - 统一管理YOLO模型训练、推理和部署",
    add_completion=True,
    rich_markup_mode="rich",
)

# 注册子命令
app.add_typer(quick.app, name="quick", help="一键训练 (自动化完整流程) ⚡")
app.add_typer(model.app, name="model", help="模型管理 (下载、导出、列表)")
app.add_typer(data.app, name="data", help="数据处理 (划分、验证、统计)")
app.add_typer(train.app, name="train", help="模型训练 (启动、恢复、配置)")
app.add_typer(validate.app, name="validate", help="模型验证 (性能评估、模型比较)")
app.add_typer(predict.app, name="predict", help="模型预测 (检测、分割、分类)")
app.add_typer(labelstudio.app, name="labelstudio", help="Label Studio集成 (上传数据集)")
# 向后兼容：保留detect作为predict的别名
app.add_typer(detect.app, name="detect", help="目标检测 (图片、视频、批量) [别名]")

console = Console()


@app.command("interactive-mode")
def interactive_mode():
    """启动交互式模式 🎮"""
    interactive.start()


@app.command()
def version():
    """显示版本信息"""
    console.print(f"[bold cyan]YOLO CLI[/bold cyan] version [bold green]{__version__}[/bold green]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="显示版本信息",
    ),
):
    """
    YOLO CLI - YOLO推理快捷操作框架
    
    支持 YOLOv8 和 YOLO11，提供完整的模型训练、推理和管理功能。
    支持三种任务：目标检测(detect)、实例分割(segment)、图像分类(classify)。
    
    使用 --help 查看各个命令的详细帮助信息。
    
    快速开始:
    
      - 交互式模式:  yolo-cli interactive-mode
      - 下载模型:    yolo-cli model download --version yolo11 --size s --task segment
      - 训练模型:    yolo-cli train start --model yolo11s.pt --data data/processed/dataset.yaml --task detect
      - 预测图片:    yolo-cli predict image <model> <image>
    """
    if version_flag:
        console.print(f"[bold cyan]YOLO CLI[/bold cyan] version [bold green]{__version__}[/bold green]")
        raise typer.Exit()
    
    # 如果没有子命令，显示帮助信息
    if ctx.invoked_subcommand is None:
        print_logo()
        console.print()
        console.print("[bold cyan]YOLO CLI - YOLO推理快捷操作框架[/bold cyan]")
        console.print()
        console.print("支持 YOLOv8 和 YOLO11，提供完整的模型训练、推理和管理功能。")
        console.print()
        console.print("[bold]可用命令:[/bold]")
        console.print("  [cyan]quick[/cyan]              一键训练 (自动化完整流程) ⚡")
        console.print("  [cyan]model[/cyan]              模型管理 (下载、导出、列表)")
        console.print("  [cyan]data[/cyan]               数据处理 (划分、验证、统计)")
        console.print("  [cyan]train[/cyan]              模型训练 (启动、恢复、配置)")
        console.print("  [cyan]validate[/cyan]           模型验证 (性能评估、模型比较)")
        console.print("  [cyan]predict[/cyan]            模型预测 (检测、分割、分类)")
        console.print("  [cyan]labelstudio[/cyan]        Label Studio集成 (上传数据集)")
        console.print("  [cyan]detect[/cyan]             目标检测 (图片、视频、批量) [别名]")
        console.print("  [cyan]interactive-mode[/cyan]   交互式模式 🎮")
        console.print()
        console.print("[bold]快速开始:[/bold]")
        console.print("  [dim]# 一键训练（推荐）⚡[/dim]")
        console.print("  python yolo_cli.py quick train --images data/raw/images --labels data/raw/labels")
        console.print()
        console.print("  [dim]# 交互式模式[/dim]")
        console.print("  python yolo_cli.py interactive-mode")
        console.print()
        console.print("  [dim]# 训练模型[/dim]")
        console.print("  python yolo_cli.py train start --model yolo11s.pt --data data/processed/dataset.yaml")
        console.print()
        console.print("  [dim]# 验证模型[/dim]")
        console.print("  python yolo_cli.py validate run results/training/best.pt --data data/processed/dataset.yaml")
        console.print()
        console.print("  [dim]# 检测图片[/dim]")
        console.print("  python yolo_cli.py detect image results/training/best.pt test.jpg")
        console.print()
        console.print("使用 [cyan]--help[/cyan] 查看各个命令的详细帮助信息")
        console.print("示例: [dim]python yolo_cli.py model --help[/dim]")
        console.print()


def cli():
    """CLI入口点"""
    app()


if __name__ == "__main__":
    cli()
