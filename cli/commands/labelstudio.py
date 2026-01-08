"""
Label Studio集成命令

支持将YOLO数据集上传到Label Studio进行标注和审核
"""

import typer
from pathlib import Path
from typing import Optional, List

from ..core.config import ConfigManager
from ..ui.display import print_section_header, print_info, print_success, print_error, print_warning
from ..integrations.labelstudio_uploader import LabelStudioUploader


app = typer.Typer(help="Label Studio集成")


@app.command("upload")
def upload_dataset(
    dataset: str = typer.Argument(..., help="数据集路径（支持相对路径或datasets目录下的数据集名称）"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    splits: Optional[List[str]] = typer.Option(
        None, 
        "--split", 
        "-s", 
        help="要上传的数据集分割（可多次指定，默认: train val test）"
    ),
    max_images: Optional[int] = typer.Option(None, "--max", help="最大上传图片数（用于测试）"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="最大并发数（默认: 4）"),
    setup_config: bool = typer.Option(False, "--setup-config", help="配置标注模板"),
    verify: bool = typer.Option(False, "--verify", help="上传后验证结果"),
    force: bool = typer.Option(False, "--force", help="强制重新上传所有文件（忽略所有检查，会创建重复任务）"),
    no_resume: bool = typer.Option(False, "--no-resume", help="禁用断点续传（清除本地进度记录重新开始）"),
    skip_server_check: bool = typer.Option(False, "--skip-server-check", help="跳过服务器重复检测（仅使用本地缓存，适合大项目）"),
    verify_duplicates: bool = typer.Option(False, "--verify-duplicates", help="强制检查服务器重复（即使项目很大）"),
    retry_times: int = typer.Option(3, "--retry-times", help="网络失败重试次数（默认: 3）"),
):
    """
    上传YOLO数据集到Label Studio（文件上传模式，支持断点续传）
    
    示例:
        # 正常上传（自动断点续传 + 智能重复检测）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1
        
        # 网络中断后，重新运行相同命令，自动从断点继续
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1
        
        # 大项目快速上传（跳过服务器重复检查，仅用本地缓存）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --skip-server-check
        
        # 强制重新上传所有文件（会创建重复任务，慎用！）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --force
        
        # 清除本地缓存重新开始（但仍会检测服务器避免重复）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --no-resume
        
        # 自定义重试次数（网络极不稳定时）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --retry-times 5
        
        # 只配置标注模板（不上传数据）
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --setup-config
        
        # 只上传训练集和验证集
        yolo_cli labelstudio upload my_dataset --url http://localhost:8080 --api-key xxx --project-id 1 --split train --split val
    """
    print_section_header("上传数据集到 Label Studio")
    
    # 解析数据集路径
    dataset_path = _resolve_dataset_path(dataset)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    print_info(f"数据集路径: {dataset_path}")
    print_info(f"Label Studio: {url}")
    print_info(f"项目ID: {project_id}")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化上传器失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    print_info("\n测试连接...")
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 加载数据集配置
    try:
        uploader.load_dataset_config(dataset_path)
    except Exception as e:
        print_error(f"加载数据集配置失败: {str(e)}")
        raise typer.Exit(1)
    
    # 配置标注模板
    if setup_config:
        print_info("\n配置标注模板...")
        if uploader.setup_project_config():
            print_success("\n✅ 标注模板配置完成！")
        else:
            print_error("标注模板配置失败")
            raise typer.Exit(1)
        return
    
    # 准备上传参数
    if splits is None:
        splits = ['train', 'val', 'test']
    
    # 处理参数冲突
    if verify_duplicates and skip_server_check:
        print_warning("⚠ --verify-duplicates 和 --skip-server-check 冲突，将启用服务器检查")
        skip_server_check = False
    
    print_info(f"\n数据集分割: {', '.join(splits)}")
    print_info(f"并发数: {max_workers}")
    print_info(f"重试次数: {retry_times}")
    
    if force:
        print_warning("⚠ 强制模式：将重新上传所有文件（可能创建重复任务）")
    elif no_resume:
        print_info("ℹ️  已禁用断点续传，将清除本地进度重新开始")
    elif skip_server_check:
        print_info("⚡ 跳过服务器检查模式：仅使用本地缓存")
    else:
        print_info("✨ 断点续传模式：自动跳过已上传文件")
    
    print_info("使用文件上传模式（Label Studio会自动管理文件）")
    
    # 上传数据集
    try:
        total_uploaded, total_failed = uploader.upload_tasks(
            dataset_path=dataset_path,
            splits=splits,
            max_images=max_images,
            max_workers=max_workers,
            force=force,
            no_resume=no_resume,
            skip_server_check=skip_server_check,
            retry_times=retry_times
        )
        
        print_section_header("上传完成")
        print_success(f"成功: {total_uploaded} 个任务")
        if total_failed > 0:
            print_error(f"失败: {total_failed} 个任务")
        
        # 验证上传结果
        if verify and total_uploaded > 0:
            uploader.verify_uploaded_tasks(num_samples=5)
        
    except KeyboardInterrupt:
        # 用户中断
        print_warning("\n\n⚠️  上传已被用户中断")
        print_info("💾 进度已保存，下次运行时将自动继续")
        raise typer.Exit(130)  # 130 is the standard exit code for Ctrl+C
        
    except Exception as e:
        print_error(f"上传失败: {str(e)}")
        raise typer.Exit(1)


@app.command("setup-config")
def setup_project_config(
    dataset: str = typer.Argument(..., help="数据集路径（用于读取类别信息）"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
):
    """
    仅配置Label Studio项目标注模板
    
    示例:
        yolo_cli labelstudio setup-config my_dataset --url http://localhost:8080 --api-key xxx --project-id 1
    """
    print_section_header("配置 Label Studio 标注模板")
    
    # 解析数据集路径
    dataset_path = _resolve_dataset_path(dataset)
    if not dataset_path.exists():
        print_error(f"数据集路径不存在: {dataset_path}")
        raise typer.Exit(1)
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化上传器失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 加载数据集配置
    try:
        uploader.load_dataset_config(dataset_path)
    except Exception as e:
        print_error(f"加载数据集配置失败: {str(e)}")
        raise typer.Exit(1)
    
    # 配置标注模板
    if uploader.setup_project_config():
        print_success("\n✅ 标注模板配置完成！")
    else:
        print_error("标注模板配置失败")
        raise typer.Exit(1)


@app.command("verify")
def verify_uploaded_tasks(
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    num_samples: int = typer.Option(5, "--samples", "-n", help="验证样本数量"),
):
    """
    验证已上传的任务
    
    示例:
        yolo_cli labelstudio verify --url http://localhost:8080 --api-key xxx --project-id 1 --samples 10
    """
    print_section_header("验证上传的任务")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化上传器失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 验证任务
    if uploader.verify_uploaded_tasks(num_samples=num_samples):
        print_success("\n✅ 验证完成！")
    else:
        print_error("验证失败")
        raise typer.Exit(1)


@app.command("audit")
def audit_annotations(
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="导出报告文件名（将保存到results/audit/）"),
    show_details: bool = typer.Option(True, "--details/--no-details", help="显示异常任务的详细信息"),
    max_samples: int = typer.Option(10, "--max-samples", help="每种异常类型显示的最大样本数"),
    max_tasks: Optional[int] = typer.Option(None, "--max-tasks", help="最大审计任务数（默认全部，用于抽样审计）"),
):
    """
    审计Label Studio项目的标注质量
    
    检查项：
      - 关键点标注顺序一致性（pose任务）
      - 缺失标注
      - 重复标注
      - 标注格式异常
    
    示例:
        # 审计所有任务（默认）
        yolo_cli labelstudio audit --url http://localhost:8080 --api-key xxx --project-id 1
        
        # 抽样审计（只审计前500个任务）
        yolo_cli labelstudio audit --url http://localhost:8080 --api-key xxx --project-id 1 --max-tasks 500
        
        # 导出报告（将保存到results/audit/目录）
        yolo_cli labelstudio audit --url http://localhost:8080 --api-key xxx --project-id 1 --output audit_report.json
        
        # 只显示统计，不显示详细信息
        yolo_cli labelstudio audit --url http://localhost:8080 --api-key xxx --project-id 1 --no-details
    """
    print_section_header("Label Studio 标注审计")
    
    print_info(f"Label Studio: {url}")
    print_info(f"项目ID: {project_id}")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    print_info("\n连接到 Label Studio...")
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 执行审计
    try:
        print_info("\n开始审计...")
        audit_report = uploader.audit_annotations(
            show_details=show_details,
            max_samples=max_samples,
            max_tasks=max_tasks
        )
        
        # 导出报告
        if output and audit_report:
            try:
                import json
                from ..core.config import ConfigManager
                
                config = ConfigManager()
                # 创建 results/audit 目录
                audit_dir = config.project_root / 'results' / 'audit'
                audit_dir.mkdir(parents=True, exist_ok=True)
                
                # 如果用户输入的是绝对路径，直接使用
                # 否则，保存到 results/audit 目录
                output_path = Path(output)
                if not output_path.is_absolute():
                    # 只取文件名，放到 results/audit 目录
                    output_path = audit_dir / output_path.name
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(audit_report, f, indent=2, ensure_ascii=False)
                
                # 验证文件是否存在
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    # 显示相对路径
                    try:
                        rel_path = output_path.relative_to(config.project_root)
                        print_success(f"\n✅ 报告已导出到: {rel_path}")
                    except ValueError:
                        # 如果无法计算相对路径，显示绝对路径
                        print_success(f"\n✅ 报告已导出到: {output_path}")
                    print_info(f"   文件大小: {file_size:,} 字节")
                else:
                    print_error(f"\n✗ 报告文件未创建: {output_path}")
            except Exception as e:
                print_error(f"\n✗ 导出报告失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print_section_header("审计完成")
        
    except Exception as e:
        print_error(f"审计失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("predict")
def predict_tasks(
    model: str = typer.Argument(..., help="YOLO模型路径"),
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务类型 (detect/segment/pose/classify，不指定则自动推断)"),
    task_ids: Optional[List[int]] = typer.Option(None, "--task-ids", help="指定任务ID列表"),
    task_range: Optional[List[int]] = typer.Option(None, "--task-range", help="指定任务ID范围 (start end)"),
    unlabeled: bool = typer.Option(False, "--unlabeled", help="预测所有未标注的任务"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="置信度阈值"),
    iou: float = typer.Option(0.45, "--iou", help="IOU阈值"),
    max_workers: int = typer.Option(4, "--max-workers", "-w", help="最大并发数"),
    device: str = typer.Option("auto", "--device", "-d", help="设备 (auto/cpu/cuda)"),
):
    """
    使用本地YOLO模型预测Label Studio任务
    
    任务筛选方式（三选一）：
      - --task-ids: 指定任务ID列表，如 --task-ids 1 2 3 5 8
      - --task-range: 指定任务ID范围，如 --task-range 5900 6000
      - --unlabeled: 预测所有未标注的任务
    
    示例:
        # 预测指定任务ID列表
        yolo_cli labelstudio predict yolo11n.pt --url http://localhost:8080 --api-key xxx --project-id 1 --task-ids 1 2 3 5 8
        
        # 预测任务ID范围
        yolo_cli labelstudio predict best.pt --url http://localhost:8080 --api-key xxx --project-id 1 --task-range 5900 6000
        
        # 预测所有未标注任务
        yolo_cli labelstudio predict yolo11m.pt --url http://localhost:8080 --api-key xxx --project-id 1 --unlabeled
        
        # 显式指定任务类型
        yolo_cli labelstudio predict yolo11n-pose.pt --url http://localhost:8080 --api-key xxx --project-id 1 --task pose --unlabeled
    """
    print_section_header("Label Studio 任务预测")
    
    # 验证模型路径
    model_path = Path(model)
    if not model_path.exists():
        print_error(f"模型不存在: {model}")
        raise typer.Exit(1)
    
    # 验证任务筛选参数（三选一）
    filter_count = sum([
        task_ids is not None,
        task_range is not None,
        unlabeled
    ])
    
    if filter_count == 0:
        print_error("必须指定一种任务筛选方式: --task-ids, --task-range 或 --unlabeled")
        raise typer.Exit(1)
    
    if filter_count > 1:
        print_error("只能指定一种任务筛选方式")
        raise typer.Exit(1)
    
    # 验证 task_range 参数
    if task_range is not None:
        if len(task_range) != 2:
            print_error("--task-range 需要两个参数: 起始ID 结束ID")
            raise typer.Exit(1)
        if task_range[0] > task_range[1]:
            print_error("起始ID必须小于或等于结束ID")
            raise typer.Exit(1)
    
    print_info(f"模型: {model_path.name}")
    print_info(f"Label Studio: {url}")
    print_info(f"项目ID: {project_id}")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id, task_type=task or 'detect')
    except Exception as e:
        print_error(f"初始化失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    print_info("\n连接到 Label Studio...")
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 执行预测
    try:
        print_info("\n开始预测...")
        success, failed = uploader.predict_tasks_with_yolo(
            model_path=model_path,
            task_ids=task_ids,
            task_range=tuple(task_range) if task_range else None,
            unlabeled=unlabeled,
            task_type=task,
            conf=conf,
            iou=iou,
            device=device,
            max_workers=max_workers
        )
        
        print_section_header("预测完成")
        print_success(f"成功: {success} 个任务")
        if failed > 0:
            print_error(f"失败: {failed} 个任务")
        
    except Exception as e:
        print_error(f"预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("batch-annotate")
def batch_annotate_command(
    url: str = typer.Option(..., "--url", "-u", help="Label Studio服务器URL"),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API密钥"),
    project_id: int = typer.Option(..., "--project-id", "-p", help="Label Studio项目ID"),
    annotation_type: str = typer.Option(..., "--type", "-t", help="标注类型: rectangle, keypoint"),
    target: str = typer.Option("annotation", "--target", help="目标类型: annotation（正式标注）, prediction（预测结果）"),
    merge_mode: str = typer.Option("add", "--merge-mode", "-m", help="合并模式: add（追加）, skip（跳过已标注）, overwrite_same_type（覆盖同类型）"),
    task_ids: Optional[List[int]] = typer.Option(None, "--task-ids", help="指定task ID列表"),
    task_range: Optional[List[int]] = typer.Option(None, "--task-range", help="task ID范围 (start end)"),
    unlabeled: bool = typer.Option(False, "--unlabeled", help="仅处理未标注的tasks"),
    dry_run: bool = typer.Option(False, "--dry-run", help="试运行，不实际创建"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="并发数"),
):
    """
    批量给Label Studio tasks添加标注
    
    示例:
        # 给已有关键点的tasks添加矩形框（不影响关键点）
        yolo-cli labelstudio batch-annotate \\
          --url http://10.105.3.39/ \\
          --api-key xxx \\
          --project-id 9 \\
          --type rectangle \\
          --target annotation \\
          --merge-mode add \\
          --task-range 100 200
        
        # 只标注空白tasks
        yolo-cli labelstudio batch-annotate \\
          --url http://10.105.3.39/ \\
          --project-id 9 \\
          --type rectangle \\
          --merge-mode skip \\
          --unlabeled
        
        # 试运行预览
        yolo-cli labelstudio batch-annotate \\
          --url http://10.105.3.39/ \\
          --project-id 9 \\
          --type rectangle \\
          --dry-run
    """
    print_section_header("批量打标签到 Label Studio")
    
    print_info(f"Label Studio: {url}")
    print_info(f"项目ID: {project_id}")
    print_info(f"标注类型: {annotation_type}")
    print_info(f"目标类型: {target}")
    print_info(f"合并模式: {merge_mode}")
    
    # 初始化上传器
    try:
        uploader = LabelStudioUploader(url, api_key, project_id)
    except Exception as e:
        print_error(f"初始化失败: {str(e)}")
        raise typer.Exit(1)
    
    # 测试连接
    print_info("\n测试连接...")
    if not uploader.test_connection():
        print_error("连接失败，请检查URL和API密钥")
        raise typer.Exit(1)
    
    # 解析项目标签配置
    print_info("\n解析项目标签配置...")
    labeling_config = uploader.parse_labeling_config()
    
    if not labeling_config:
        print_error("无法解析项目标签配置")
        raise typer.Exit(1)
    
    # 验证标注类型是否可用
    if annotation_type == 'rectangle' and 'rectanglelabels' not in labeling_config:
        print_error("项目中没有配置RectangleLabels标注控件")
        raise typer.Exit(1)
    
    if annotation_type == 'keypoint' and 'keypointlabels' not in labeling_config:
        print_error("项目中没有配置KeyPointLabels标注控件")
        raise typer.Exit(1)
    
    # 交互式输入标注内容
    from ..ui.prompts import input_rectangle_annotation, input_keypoint_annotations
    
    if annotation_type == 'rectangle':
        config = labeling_config['rectanglelabels']
        annotation_input = input_rectangle_annotation(config['labels'])
        
        # 转换为Label Studio格式
        bbox = uploader.normalize_to_labelstudio_bbox(
            annotation_input['center_x'],
            annotation_input['center_y'],
            annotation_input['width'],
            annotation_input['height']
        )
        
        annotation_data = {
            'original_width': 100,  # 将由实际图片尺寸动态调整
            'original_height': 100,
            'image_rotation': 0,
            'value': {
                **bbox,
                'rotation': 0,
                'rectanglelabels': [annotation_input['label']]
            },
            'from_name': config['from_name'],
            'to_name': config['to_name'],
            'type': 'rectanglelabels'
        }
    else:  # keypoint
        config = labeling_config['keypointlabels']
        keypoints_input = input_keypoint_annotations(config['labels'])
        
        # 暂时不支持批量关键点标注（因为每个图片的关键点位置不同）
        print_warning("批量关键点标注暂未实现（每个图片的关键点位置通常不同）")
        print_info("建议使用交互模式或模型预测方式")
        raise typer.Exit(0)
    
    # 构建task filter
    task_filter = {'mode': 'all'}
    
    if task_ids:
        task_filter = {'mode': 'ids', 'task_ids': task_ids}
    elif task_range:
        if len(task_range) != 2:
            print_error(f"task范围需要两个参数: --task-range <起始ID> <结束ID>")
            raise typer.Exit(1)
        start, end = task_range[0], task_range[1]
        if start > end:
            print_error(f"起始ID ({start}) 不能大于结束ID ({end})")
            raise typer.Exit(1)
        task_filter = {'mode': 'range', 'task_range': (start, end)}
    elif unlabeled:
        task_filter = {'mode': 'unlabeled'}
    
    # 执行批量标注
    try:
        stats = uploader.batch_annotate_tasks(
            annotation_data=annotation_data,
            target_type=target,
            task_filter=task_filter,
            merge_mode=merge_mode,
            dry_run=dry_run,
            max_workers=max_workers
        )
        
        print_section_header("批量标注完成")
        
        if stats['total'] > 0:
            print_success(f"\n✅ 成功: {stats['success']} 个任务")
            if stats['failed'] > 0:
                print_error(f"❌ 失败: {stats['failed']} 个任务")
            if stats['skipped'] > 0:
                print_info(f"ℹ️  跳过: {stats['skipped']} 个任务")
        
    except Exception as e:
        print_error(f"批量标注失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _resolve_dataset_path(dataset: str) -> Path:
    """
    解析数据集路径
    
    支持:
    - 相对路径或绝对路径
    - datasets目录下的数据集名称
    """
    dataset_path = Path(dataset)
    
    # 如果是绝对路径或存在的相对路径，直接使用
    if dataset_path.is_absolute() or dataset_path.exists():
        return dataset_path
    
    # 尝试在datasets目录下查找
    config = ConfigManager()
    datasets_base = config.project_root / 'datasets'
    
    if datasets_base.exists():
        # 尝试作为数据集名称
        candidate = datasets_base / dataset
        if candidate.exists():
            return candidate
    
    # 返回原始路径（可能不存在，后续会检查）
    return dataset_path


if __name__ == "__main__":
    app()

