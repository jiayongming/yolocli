#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测命令（向后兼容，实际调用 predict 命令）"""

import typer
from typing import Optional

# 导入 predict 命令中的函数
from .predict import predict_image as _predict_image
from .predict import detect_batch as _predict_batch
from .predict import detect_video as _predict_video

app = typer.Typer(help="检测命令（向后兼容）")


@app.command("image")
def detect_image(
    model: str = typer.Argument(..., help="模型路径"),
    image: str = typer.Argument(..., help="图片路径"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(True, "--save-txt/--no-txt", help="保存TXT结果"),
    save_json: bool = typer.Option(True, "--save-json/--no-json", help="保存JSON结果"),
    show: bool = typer.Option(False, "--show", "-s", help="显示结果"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
):
    """单张图片检测（向后兼容，调用 predict image）"""
    # 直接调用 predict.py 中的函数
    _predict_image(
        model=model,
        image=image,
        task='detect',
        conf=conf,
        iou=iou,
        output=output,
        save_txt=save_txt,
        save_json=save_json,
        show=show,
        device=device,
        top_k=5  # 默认值
    )


@app.command("batch")
def detect_batch(
    model: str = typer.Argument(..., help="模型路径"),
    source: str = typer.Argument(..., help="图片目录或视频文件"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(True, "--save-txt/--no-txt", help="保存TXT结果"),
    save_json: bool = typer.Option(True, "--save-json/--no-json", help="保存JSON结果"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
    batch: int = typer.Option(1, "--batch", "-b", help="批次大小"),
):
    """批量检测（向后兼容，调用 predict batch）"""
    # 直接调用 predict.py 中的函数
    _predict_batch(
        model=model,
        source=source,
        conf=conf,
        iou=iou,
        output=output,
        save_txt=save_txt,
        save_json=save_json,
        device=device,
        batch=batch
    )


@app.command("video")
def detect_video(
    model: str = typer.Argument(..., help="模型路径"),
    video: str = typer.Argument(..., help="视频文件路径"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录"),
    save_txt: bool = typer.Option(False, "--save-txt", help="保存TXT结果"),
    show: bool = typer.Option(False, "--show", "-s", help="实时显示"),
    device: str = typer.Option("auto", "--device", "-d", help="设备"),
):
    """视频检测（向后兼容，调用 predict video）"""
    # 直接调用 predict.py 中的函数
    _predict_video(
        model=model,
        video=video,
        conf=conf,
        iou=iou,
        output=output,
        save_txt=save_txt,
        show=show,
        device=device
    )
