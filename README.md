# YOLO CLI - YOLO推理快捷操作框架

一个功能完整、易于使用的命令行工具，用于YOLO模型的训练、推理和管理。支持YOLOv8和YOLO11，提供交互式操作界面和丰富的命令行选项。

**✨ 支持三种任务类型：**
- 🎯 **目标检测 (Detection)**: 检测图像中的目标并标注边界框
- 🎨 **实例分割 (Segmentation)**: 检测目标并生成精确的像素级掩码
- 📊 **图像分类 (Classification)**: 对整个图像进行分类

## ✨ 特性

- ⚡ **一键训练**: 自动完成数据处理、模型下载和训练的完整流程
- 🎯 **完整工作流**: 涵盖模型下载、数据处理、训练、验证、推理、导出全流程
- 📊 **模型验证增强**: 全面的性能评估，包含准确率、精确率、召回率、F1值等完整指标 🆕
- 📈 **灵活数据划分**: 支持按比例或按样本数划分数据集，适合不同实验需求 🆕
- 🔄 **Label Studio 集成**: 从 Label Studio 直接转换标注数据，支持断点续传
- 🎨 **FiftyOne 可视化**: 数据集管理、预测结果展示、Ground Truth vs Predictions 对比分析 🆕
- 🎮 **交互式模式**: 友好的交互式界面，引导式操作，支持所有新功能 🆕
- 🔄 **多版本支持**: 同时支持YOLOv8和YOLO11，自动版本管理
- 🎯 **多任务支持**: 支持检测、分割、分类三种任务类型
- 🎨 **美化输出**: 使用Rich库提供彩色输出、进度条、表格等
- ⚙️ **灵活配置**: YAML配置文件，支持多种预设（小/中/大数据集）
- 🚀 **智能设备**: 自动检测最佳设备（MPS/CUDA/CPU），支持环境变量控制
- 📊 **数据增强**: 内置多种数据增强策略（保守/平衡/激进）
- 🔧 **多GPU支持**: 支持单卡、多卡训练，适合共享服务器环境
- 📦 **易于扩展**: 模块化设计，易于添加新功能

## 🆕 最新更新 (v1.3.0 - 2025-12-22)

### ⚙️ 交互式训练高级配置 🎨
- ✅ **详细数据增强配置**：13+个独立参数，每个都有注解和默认值
  - 颜色空间增强（HSV-H/S/V）
  - 几何变换（旋转、平移、缩放、剪切、透视）
  - 翻转增强（上下/左右翻转）
  - 高级增强（Mosaic、MixUp、随机擦除、AutoAugment）
- ✅ **优化器参数配置**：学习率、动量、权重衰减、Warmup等
- ✅ **损失函数权重**：边界框、分类、DFL损失权重自定义
- ✅ **智能默认值**：基于预设提供合理默认值，便于微调
- ✅ **交互式引导**：每个参数都有详细说明和推荐范围

### 🆚 模型对比增强 🎯
- ✅ **任务类型参数**：新增 `--task` 参数，支持手动指定或自动推断 🆕
- ✅ **智能路径显示**：自动检测同名模型，显示完整路径以区分
- ✅ **完整指标对比**：新增 Accuracy、F1 分数、推理速度
- ✅ **多任务支持**：检测、分割、分类任务全面支持
- ✅ **动态列标题**：根据任务类型显示相应指标（mAP/Top-1）
- ✅ **最佳模型标注**：显示最高mAP、F1、Accuracy的模型路径

### 📊 验证指标全面增强
- ✅ **任务类型智能识别**：支持 `--task` 参数，自动从模型名推断
- ✅ **参数完全一致性**：单独验证和模型对比使用完全相同的参数（batch、imgsz、conf、iou）🆕
- ✅ **参数智能过滤**：分类任务不使用 conf/iou 参数，确保结果准确性 🆕
- ✅ 新增 **F1 分数**、**准确率 (Accuracy)** 指标
- ✅ 每个类别的完整指标（Precision, Recall, F1, AP）
- ✅ 推理速度详细统计
- ✅ 增强的JSON输出，便于统计分析
- ✅ 修复分类任务宏平均 F1 计算方法

### 📈 数据集划分新功能
- ✅ 新增 **按样本数划分** 功能 (`--counts 100:30:10`)
- ✅ 保留原有 **按比例划分** 功能 (`--ratios 0.7:0.2:0.1`)
- ✅ 智能处理样本数不足情况
- ✅ **交互式一键训练**：数据集划分前移到第一步
- ✅ 适合快速实验和数据增量训练

---

## 📋 目录

- [安装](#-安装)
- [任务类型说明](#-任务类型说明)
- [快速开始](#-快速开始)
  - [目标检测快速开始](#目标检测快速开始)
  - [实例分割快速开始](#实例分割快速开始)
  - [图像分类快速开始](#图像分类快速开始)
- [Label Studio 数据转换](#-label-studio-数据转换)
- [FiftyOne 可视化与预测分析](#-fiftyone-可视化与预测分析) 🆕
- [数据准备](#-数据准备)
- [完整工作流程](#-完整工作流程)
- [模型验证](#-模型验证) 🆕
- [GPU配置指南](#-gpu配置指南)
- [命令参考](#-命令参考)
- [使用示例](#-使用示例)
- [配置文件](#-配置文件)
- [数据增强策略](#-数据增强策略)
- [优化器选择](#-优化器选择) 🆕
- [层冻结](#️-层冻结freeze-layers) 🆕
- [设备支持](#-设备支持)
- [常见问题](#-常见问题)
- [项目结构](#-项目结构)

## 🚀 安装

### 1. 克隆仓库（如果适用）

```bash
cd workspace
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python yolo_cli.py --version
```

## 🎯 任务类型说明

YOLO CLI支持三种计算机视觉任务，每种任务有不同的应用场景和数据格式要求：

### 1. 目标检测 (Detection)
- **功能**: 检测图像中的目标并用矩形框标注位置
- **输出**: 边界框坐标 + 类别 + 置信度
- **应用**: 人脸检测、车辆检测、物体计数等
- **数据格式**: YOLO格式标签（每行：`class_id x_center y_center width height`）
- **模型后缀**: `yolo11s.pt` (无后缀)

### 2. 实例分割 (Segmentation)
- **功能**: 检测目标并生成精确的像素级掩码轮廓
- **输出**: 边界框 + 多边形掩码 + 类别 + 置信度
- **应用**: 医学图像分割、图像编辑、自动驾驶等
- **数据格式**: YOLO分割格式（每行：`class_id x1 y1 x2 y2 ... xn yn`多边形坐标点）
- **模型后缀**: `yolo11s-seg.pt`

### 3. 图像分类 (Classification)
- **功能**: 对整张图像进行分类判断
- **输出**: Top-K类别 + 对应概率
- **应用**: 图像分类、质量检测、场景识别等
- **数据格式**: 目录结构组织（每个类别一个文件夹）
- **模型后缀**: `yolo11s-cls.pt`

## ⚡ 快速开始

### 目标检测快速开始

如果你已经准备好了检测数据，使用一键训练命令可以自动完成所有步骤：

```bash
python yolo_cli.py quick train \
  --task detect \
  --images data/raw/images \
  --labels data/raw/labels
```

### 实例分割快速开始

分割任务需要多边形标注数据：

```bash
python yolo_cli.py quick train \
  --task segment \
  --images data/raw/images \
  --labels data/raw/labels
```

### 图像分类快速开始

分类任务支持两种数据组织方式：

**方式1：按类别目录组织（推荐）**

图像已按类别组织在子目录中：
```
data/raw/images/
  ├── class1/
  │   ├── img1.jpg
  │   └── img2.jpg
  └── class2/
      ├── img3.jpg
      └── img4.jpg
```

```bash
python yolo_cli.py quick train \
  --task classify \
  --images data/raw/images
```

**方式2：从标签文件转换**

如果你的数据是 images + labels 分开的格式，可以指定 labels 参数：

```bash
python yolo_cli.py quick train \
  --task classify \
  --images data/raw/images \
  --labels data/raw/labels
```

这个命令会自动：
1. ✅ 划分数据集（train/val/test）
2. ✅ 生成dataset.yaml配置
3. ✅ 验证数据集
4. ✅ 统计数据分布
5. ✅ 检查并下载模型（如果需要）
6. ✅ 开始训练

**自定义参数：**
```bash
# 检测/分割任务
python yolo_cli.py quick train \
  --task detect \
  --images data/raw/images \
  --labels data/raw/labels \
  --version yolo11 \
  --size s \
  --epochs 200 \
  --batch 16 \
  --device 0

# 分类任务（使用默认参数）
python yolo_cli.py quick train \
  --task classify \
  --images data/raw/images \
  --version yolo11 \
  --size s \
  --epochs 100 \
  --batch 32 \
  --device 0

# 分类任务（自定义训练轮数）
python yolo_cli.py quick train \
  --task classify \
  --images data/raw/images \
  --epochs 200 \
  --batch 32 \
  --device 0
```

**数据集划分方式** 🆕：
```bash
# 方式1: 按比例划分（默认，使用全部数据）
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --ratios 0.7:0.2:0.1

# 方式2: 按样本数划分（从数据集中抽取固定数量）
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --counts 100:30:10
```

**参数说明：**
- 如果不指定 `--epochs`、`--batch`、`--imgsz`，系统会根据任务类型使用合适的默认值
- 显式指定的参数值会完全按照你的设置执行（不会被自动调整）
- `--ratios` 和 `--counts` 不能同时使用，如果都不指定，默认使用 `--ratios 0.7:0.2:0.1`


### 交互式模式（推荐新手）

如果你想要更多控制，使用交互式模式，它会引导你完成所有操作：

```bash
python yolo_cli.py interactive-mode
```

**交互式模式特色** 🆕：
- ✅ 图形化菜单选择，无需记忆命令
- ✅ 智能参数提示和默认值
- ✅ **任务类型智能选择**：支持自动推断或手动指定（detect/segment/classify）🆕
- ✅ **数据集划分支持两种方式**：按比例或按样本数
- ✅ 每步操作前确认，避免误操作
- ✅ 实时显示操作进度和结果

### 命令行模式

#### 1. 下载预训练模型

```bash
# 下载YOLO11 small模型
python yolo_cli.py model download --version yolo11 --size s

# 下载所有YOLO11模型
python yolo_cli.py model download --version yolo11 --all
```

#### 2. 准备数据集

**检测/分割任务：**
```bash
# 方式1: 按比例划分（传统方式）
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --ratios 0.7:0.2:0.1

# 方式2: 按样本数划分 🆕
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --counts 100:30:10

# 生成dataset.yaml
python yolo_cli.py data generate-yaml \
  --path data/processed \
  --output data/dataset.yaml

# 验证数据集
python yolo_cli.py data verify --path data/processed
```

**分类任务：**
```bash
# 方式1: 按比例划分
python yolo_cli.py data split \
  --source data/raw/images \
  --task classify \
  --ratios 0.7:0.2:0.1

# 方式2: 按样本数划分 🆕
python yolo_cli.py data split \
  --source data/raw/images \
  --task classify \
  --counts 200:50:20

# 生成dataset.yaml
python yolo_cli.py data generate-yaml \
  --path data/processed \
  --task classify \
  --output data/dataset.yaml
```

#### 3. 训练模型

```bash
# 检测任务训练
python yolo_cli.py train start \
  --model yolo11s.pt \
  --task detect \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16

# 分割任务训练
python yolo_cli.py train start \
  --model yolo11s-seg.pt \
  --task segment \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16

# 分类任务训练
python yolo_cli.py train start \
  --model yolo11s-cls.pt \
  --task classify \
  --data data/dataset.yaml \
  --epochs 100 \
  --batch 32 \
  --imgsz 224

# 使用特定的数据增强策略
python yolo_cli.py train start \
  --model yolo11s.pt \
  --task detect \
  --data data/dataset.yaml \
  --augmentation aggressive

# 使用特定的优化器
python yolo_cli.py train start \
  --model yolo11s.pt \
  --task detect \
  --data data/dataset.yaml \
  --optimizer AdamW
```

## 🔄 Label Studio 数据转换

从 Label Studio 标注平台直接转换数据为 YOLO 训练格式！

### 快速开始

```bash
# 从 Label Studio 导出并转换目标检测数据
python yolo_cli.py data convert-labelstudio \
  --input labelstudioexport/project-3.json \
  --url http://localhost:8080 \
  --token your_api_token_here \
  --task detect

# 转换分类数据
python yolo_cli.py data convert-labelstudio \
  --input labelstudioexport/project-6.json \
  --url http://localhost:8080 \
  --token your_api_token_here \
  --task classify
```

### 核心特性

- ✅ **智能 Token 处理**: 支持 Refresh Token，自动转换为 Access Token
- ✅ **批量下载图片**: 从 Label Studio API 自动下载所有标注图片
- ✅ **负样本支持**: 自动包含无标注图片作为负样本（检测任务）🆕
- ✅ **断点续传**: 已下载文件自动跳过，支持中断后继续
- ✅ **多线程并发**: 可配置并发数（`--max-workers`，默认 4）
- ✅ **格式自动检测**: 支持 JSON 和 CSV 导出格式
- ✅ **标准 YOLO 格式**: 输出到 `data/raw/` 目录

### 输出目录结构

**检测任务**:
```
data/raw/
├── images/          # 所有下载的图片
├── labels/          # YOLO 格式标签
└── classes.txt      # 类别列表
```

**分类任务**:
```
data/raw/
├── images/          # 按类别组织的图片
│   ├── class1/
│   ├── class2/
│   └── ...
└── classes.txt      # 类别列表
```

### 负样本支持 🆕

对于检测任务，系统**自动包含无标注的图片作为负样本**：

**为什么需要负样本？**
- ✅ 减少误报（False Positives）- 模型学习识别"背景"
- ✅ 提高鲁棒性 - 适应真实场景中的无目标图像
- ✅ 标准做法 - COCO、Pascal VOC 等数据集都包含负样本
- ✅ 推荐比例 - 10-20% 负样本

**如何工作？**
```bash
# 默认包含负样本（推荐）
python yolo_cli.py data convert-labelstudio \
  --input export.json \
  --url http://localhost:8080 \
  --token YOUR_TOKEN \
  --task detect \
  --include-negative  # 默认开启

# 如果不想包含负样本
python yolo_cli.py data convert-labelstudio \
  --input export.json \
  --url http://localhost:8080 \
  --token YOUR_TOKEN \
  --task detect \
  --no-negative  # 显式关闭
```

**输出示例**：
```
✓ 解析完成：找到 150 个任务
  正样本（有标注）: 120
  负样本（无标注）: 30
  负样本比例: 20.0%

下载统计:
  ✓ 已下载: 145
  ⊙ 已跳过: 5
  总计: 150

✓ 生成了 120 个标签文件（正样本）
✓ 创建了 30 个空标签文件（负样本）
  负样本有助于减少误报，提高模型鲁棒性
```

**交互式模式**：
在交互模式中，系统会询问是否包含负样本，并提供说明。

### 完整工作流

**检测任务**:
```bash
# 1. 转换数据
python yolo_cli.py data convert-labelstudio \
  -i labelstudioexport/project.json \
  -u http://localhost:8080 \
  -t YOUR_TOKEN \
  --task detect

# 2. 划分数据集
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed

# 3. 生成配置
python yolo_cli.py data generate-yaml \
  --path data/processed \
  --classes data/raw/classes.txt

# 4. 训练
python yolo_cli.py train --data data/dataset.yaml
```

**分类任务**:
```bash
# 1. 转换数据
python yolo_cli.py data convert-labelstudio \
  -i labelstudioexport/project.json \
  -u http://localhost:8080 \
  -t YOUR_TOKEN \
  --task classify

# 2. 划分数据集
python yolo_cli.py data split \
  --source data/raw/images \
  --task classify

# 3. 训练
python yolo_cli.py train \
  --task classify \
  --data data/processed
```

### 获取 Token

在 Label Studio 界面：**Account & Settings** → **Personal Access Token**

💡 **提示**: 命令支持 Refresh Token 和 Access Token，会自动识别和转换

---

## 🎨 FiftyOne 可视化与预测分析

使用 FiftyOne 可视化数据集、模型预测结果，并进行深度分析！

### 核心功能

- ✅ **数据集可视化**: 查看标注数据（Ground Truth）
- ✅ **预测结果展示**: 可视化模型的检测/分割/分类结果
- ✅ **性能对比**: Ground Truth vs Predictions 对比分析
- ✅ **错误诊断**: 快速找出误检、漏检等问题
- ✅ **数据集管理**: 统一管理所有数据集到 `datasets/` 目录

### 安装 FiftyOne

```bash
pip install fiftyone
```

### 快速开始

#### 1. 加载数据集到 FiftyOne

```bash
python yolo_cli.py interactive
# 选择 "FiftyOne 可视化"
# 选择 "load - 加载数据集（Ground Truth）"
# 输入 dataset.yaml 路径: data/processed/dataset.yaml
# 数据集会自动复制到 datasets/ 目录并修改路径配置
```

#### 2. 启动可视化

```bash
python yolo_cli.py interactive
# 选择 "FiftyOne 可视化"
# 选择 "launch - 启动可视化"
# 选择要查看的数据集
# 浏览器会自动打开 FiftyOne App
```

### 预测结果可视化

#### 场景 1: 只查看预测结果（无标注）

适用于新图片推理，没有 ground truth 标注的情况。

```bash
# 1. 对图片进行推理（确保使用 --save-txt）
python yolo_cli.py predict batch \
    models/weights/best.pt \
    new_images/ \
    --conf 0.25 \
    --save-txt \
    --output results/new_predictions

# 2. 加载预测结果到 FiftyOne
python yolo_cli.py interactive
# 选择 "FiftyOne 可视化"
# 选择 "load_predictions - 加载预测结果"
# 图片目录: new_images/
# 预测目录: results/new_predictions/batch_predict
# 类别列表: person,car,dog,cat  (根据你的模型)
```

#### 场景 2: 对比 Ground Truth 和 Predictions（推荐）

适用于在验证集/测试集上评估模型性能。

```bash
# 1. 加载数据集（包含 ground truth）
python yolo_cli.py interactive
# → FiftyOne 可视化 → load
# 输入: data/processed/dataset.yaml

# 2. 在验证集上运行推理
python yolo_cli.py predict batch \
    models/weights/best.pt \
    data/processed/images/val \
    --conf 0.25 \
    --save-txt \
    --output results/val_predictions

# 3. 添加预测结果到数据集
python yolo_cli.py interactive
# → FiftyOne 可视化 → add_predictions
# 选择数据集: yolo_processed
# 预测目录: results/val_predictions/batch_predict/labels
# 类别列表: person,car,dog  (与数据集类别一致)
# 字段名: predictions

# 4. 启动可视化对比
# → FiftyOne 可视化 → launch
# 在 App 中可以看到：
#   - ground_truth (绿色框) - 真实标注
#   - predictions (蓝色框) - 模型预测
```

### FiftyOne App 功能

启动 FiftyOne App 后，你可以：

**基础功能：**
- 📊 查看数据集统计信息（样本数、类别分布）
- 🔍 按类别、置信度、Split（train/val/test）筛选
- 📷 并排查看图片和标注
- 🎯 点击样本查看详细信息

**预测分析功能：**
- 🆚 对比 Ground Truth 和 Predictions
- 📈 查看混淆矩阵
- 🎚️ 调整置信度阈值实时过滤
- ❌ 找出误检（False Positives）
- ⚠️ 找出漏检（False Negatives）
- 📉 按置信度排序查看低质量预测

### 对比多个模型

可以添加多个预测字段来对比不同模型的性能：

```bash
# 模型 A 的预测
python yolo_cli.py predict batch modelA.pt data/val/images --save-txt
python yolo_cli.py interactive
# → add_predictions → 字段名: predictions_modelA

# 模型 B 的预测
python yolo_cli.py predict batch modelB.pt data/val/images --save-txt
python yolo_cli.py interactive
# → add_predictions → 字段名: predictions_modelB

# 在 FiftyOne 中对比三个字段：
# - ground_truth (真实标注)
# - predictions_modelA (模型A)
# - predictions_modelB (模型B)
```

### 数据集管理

FiftyOne 会自动将数据集复制到 `datasets/` 目录进行统一管理：

```
datasets/
├── ls_project_8/          # Label Studio 项目 8
│   ├── dataset.yaml       # 自动修改为相对路径
│   ├── images/
│   └── labels/
├── my_custom_dataset/     # 手动加载的数据集
│   ├── dataset.yaml
│   ├── images/
│   └── labels/
└── predictions_20251225/  # 预测结果数据集
    ├── images/
    └── labels/
```

**好处：**
- ✅ 统一管理所有数据集
- ✅ `dataset.yaml` 路径自动修正为相对路径
- ✅ 数据集可移动和分享
- ✅ 已添加到 `.gitignore`，不占用版本控制空间

### YOLO 预测结果格式

预测命令必须使用 `--save-txt` 参数保存标签文件：

```
results/predictions/detect/batch_predict/
├── labels/               # ← FiftyOne 读取此目录
│   ├── image1.txt       # 每行: class_id x y w h confidence
│   ├── image2.txt
│   └── image3.txt
└── predictions/          # 可视化结果（可选）
    ├── image1.jpg
    └── image2.jpg
```

**txt 文件格式（YOLO 格式）：**
```
# class_id x_center y_center width height confidence
0 0.512 0.345 0.234 0.456 0.95
1 0.678 0.234 0.123 0.234 0.87
```

### 完整工作流示例

```bash
# === 步骤 1: 训练模型 ===
python yolo_cli.py quick-train data/dataset.yaml --model yolo11n.pt --epochs 50

# === 步骤 2: 加载数据集到 FiftyOne ===
python yolo_cli.py interactive
# → FiftyOne 可视化 → load
# 路径: data/processed/dataset.yaml

# === 步骤 3: 在验证集上推理 ===
python yolo_cli.py predict batch \
    runs/detect/train/weights/best.pt \
    data/processed/images/val \
    --conf 0.25 \
    --save-txt

# === 步骤 4: 添加预测结果 ===
python yolo_cli.py interactive
# → FiftyOne 可视化 → add_predictions
# 选择数据集、输入预测目录和类别

# === 步骤 5: 分析和改进 ===
# → launch → 在 App 中分析误检和漏检
# → 导出问题样本
# → 改进数据标注或增强
# → 重新训练
```

### FiftyOne 操作菜单

```
FiftyOne 可视化:
├── load - 加载数据集（Ground Truth）
├── load_predictions - 加载预测结果
├── add_predictions - 添加预测到现有数据集
├── launch - 启动可视化
├── list - 列出所有数据集
├── info - 查看数据集信息
└── delete - 删除数据集
```

### 使用技巧

**技巧 1: 快速找出模型弱点**
```
1. 在 FiftyOne 中按类别分组
2. 按置信度排序
3. 找出低置信度但正确的预测（模型不确定）
4. 找出高置信度但错误的预测（过度自信）
```

**技巧 2: 置信度阈值调优**
```
在 FiftyOne 中实时调整置信度滑块：
- 低阈值 (0.1): 更多预测，可能有误检
- 中等阈值 (0.25): 平衡
- 高阈值 (0.5): 只保留高置信度预测
找到最佳平衡点
```

**技巧 3: 导出问题样本**
```
1. 在 FiftyOne 中筛选出误检/漏检的样本
2. 导出这些样本
3. 重新标注或增强数据
4. 添加到训练集重新训练
```

---

#### 4. 推理预测

```bash
# 单张图片预测（自动识别任务类型）
python yolo_cli.py predict image \
  results/training/best.pt \
  test.jpg

# 分割任务预测
python yolo_cli.py predict image \
  results/training/best.pt \
  test.jpg \
  --task segment

# 分类任务预测（Top-5）
python yolo_cli.py predict image \
  results/training/best.pt \
  test.jpg \
  --task classify \
  --top-k 5

# 批量预测
python yolo_cli.py predict batch \
  results/training/best.pt \
  test_images/

# 视频预测
python yolo_cli.py predict video \
  results/training/best.pt \
  video.mp4 \
  --show

# 向后兼容：detect命令仍然可用
python yolo_cli.py detect image \
  results/training/best.pt \
  test.jpg
```

#### 5. 导出模型

```bash
# 导出为ONNX格式
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx

# 导出为多种格式
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx torchscript tflite
```

## 📂 数据准备

在开始训练前，需要根据任务类型按照以下格式准备数据集。

### 目标检测数据格式

#### 数据集目录结构

```
data/raw/
├── images/          # 所有图片文件
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── labels/          # YOLO格式标签文件
│   ├── img001.txt   # 与图片同名
│   ├── img002.txt
│   └── ...
└── classes.txt      # 类别列表（每行一个类别名）
```

### classes.txt 格式

每行一个类别名称，例如：

```
person
car
dog
cat
bicycle
```

#### 检测标签文件格式

每个标签文件（.txt）包含该图片中所有目标的标注信息，每行一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 类别索引（从0开始，对应classes.txt中的行号）
- `x_center, y_center`: 目标中心点坐标（归一化到0-1）
- `width, height`: 目标宽高（归一化到0-1）

**示例（img001.txt）：**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

### 实例分割数据格式

#### 分割标签文件格式

每个标签文件（.txt）包含多边形坐标点，每行一个实例：

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```

- `class_id`: 类别索引（从0开始）
- `x1 y1 ... xn yn`: 多边形的N个顶点坐标（归一化到0-1）
- 至少需要3个点（6个坐标值）

**示例（img001.txt）：**
```
0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4
1 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7
```

### 图像分类数据格式

#### 分类目录结构

分类任务使用目录结构组织，每个类别一个文件夹（统一使用 `images/` 目录）：

```
data/processed/
└── images/
    ├── train/
    │   ├── class1/
    │   │   ├── img001.jpg
    │   │   └── img002.jpg
    │   ├── class2/
    │   │   └── img003.jpg
    │   └── class3/
    │       └── img004.jpg
    └── val/
        ├── class1/
        └── class2/
```

或者使用标签文件（每个文件包含单个类别ID）：

**img001.txt**:
```
0
```

然后使用命令自动组织：

```bash
python yolo_cli.py data prepare-classify \
  --images data/raw/images \
  --labels data/raw/labels \
  --classes data/raw/classes.txt
```

### 快速准备命令

```bash
# 创建数据目录
mkdir -p data/raw/images data/raw/labels

# 将你的图片复制到images目录
cp -r /path/to/your/images/* data/raw/images/

# 将你的标签复制到labels目录
cp -r /path/to/your/labels/* data/raw/labels/

# 创建classes.txt
cat > data/raw/classes.txt << EOF
class1
class2
class3
EOF
```

## 📖 完整工作流程

本节详细介绍从零开始准备数据集、训练模型到部署的完整流程。

### 步骤1：准备数据集

#### 1.1 创建数据目录结构

首先，创建必要的目录结构：

```bash
cd workspace

# 创建数据目录
mkdir -p data/raw/images
mkdir -p data/raw/labels
mkdir -p data/processed
mkdir -p models/weights
mkdir -p results
```

#### 1.2 准备原始数据

将你的数据集放置到 `data/raw/` 目录下（参考上面的[数据准备](#-数据准备)部分）：

```
data/raw/
├── images/          # 存放所有原始图片
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── labels/          # 存放对应的YOLO格式标签
│   ├── image001.txt
│   ├── image002.txt
│   └── ...
└── classes.txt      # 类别定义文件
```

**classes.txt 格式示例：**
```
waterpoll
active_leak
fireequipment
```

**YOLO标签格式 (.txt)：**

每个图片对应一个同名的txt文件，每行格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```

其中所有坐标值都是归一化的（0-1之间），例如：
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

#### 1.3 验证数据格式

确保图片和标签文件名匹配（除了扩展名）：

```bash
# 查看图片数量
ls data/raw/images/*.jpg | wc -l

# 查看标签数量
ls data/raw/labels/*.txt | wc -l

# 两者数量应该相等（除了classes.txt）
```

### 步骤2：数据集处理

#### 2.1 划分数据集

将原始数据划分为训练集、验证集和测试集。**支持两种划分方式** 🆕：

**方式1: 按比例划分（传统方式）**

```bash
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --ratios 0.7:0.2:0.1 \
  --seed 42
```

**方式2: 按样本数划分（新功能）** 🆕

适用于从大数据集中抽取固定数量的样本：

```bash
# 从300个样本中抽取：训练集100个，验证集30个，测试集10个
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --counts 100:30:10 \
  --seed 42
```

**参数说明：**
- `--images`: 原始图片目录
- `--labels`: 原始标签目录
- `--output`: 输出目录（默认：data/processed）
- `--ratios`: 划分比例，格式为 train:val:test（如：0.7:0.2:0.1）
- `--counts`: 划分样本数，格式为 train:val:test（如：100:30:10）🆕
- `--seed`: 随机种子，保证可重现性

**使用场景建议：**

| 场景 | 使用方式 | 示例 |
|------|---------|------|
| 使用全部数据 | `--ratios` | `--ratios 0.7:0.2:0.1` |
| 快速原型验证 | `--counts` | `--counts 50:15:5` |
| 数据增量实验 | `--counts` | 第1轮: `--counts 100:30:10`<br>第2轮: `--counts 200:60:20` |
| 从大数据集抽样 | `--counts` | 从3000个中抽取: `--counts 100:30:10` |

**注意事项：**
- `--ratios` 和 `--counts` 二选一，不能同时使用
- 如果两者都不指定，默认使用 `--ratios 0.7:0.2:0.1`
- 使用 `--counts` 时，如果请求的样本数大于可用样本数，会自动使用所有样本并保持比例
- 使用 `--counts` 时，如果请求的样本数小于可用样本数，会随机抽取指定数量的样本
- 使用 `--seed` 参数确保每次划分结果一致，便于复现实验

执行后，会生成以下目录结构：

```
data/processed/
├── images/
│   ├── train/       # 训练集图片
│   ├── val/         # 验证集图片
│   └── test/        # 测试集图片
├── labels/
│   ├── train/       # 训练集标签
│   ├── val/         # 验证集标签
│   └── test/        # 测试集标签
└── split_statistics.txt  # 划分统计信息
```

**负样本支持** 🆕

如果图片目录中有缺失标签文件的图片，可以将它们作为负样本包含：

```bash
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --create-empty-labels  # 为缺失标签的图片创建空标签
```

**特性：**
- ✅ 自动为缺失标签的图片创建空的 `.txt` 文件
- ✅ 这些图片将作为负样本（背景）参与训练
- ✅ 统计输出会区分正样本和负样本
- ✅ 有助于减少误报，提高模型鲁棒性

**输出示例：**
```
找到 150 个有效样本
  正样本（有标注）: 120
  负样本（无标注）: 30 - 已创建空标签文件
  
数据集划分结果:
┏━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓
┃ 数据集 ┃ 样本数 ┃ 正样本 ┃ 负样本 ┃ 比例   ┃
┡━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩
│ 训练集 │  105  │   84  │  21  │ 70.0% │
│ 验证集 │   30  │   24  │   6  │ 20.0% │
│ 测试集 │   15  │   12  │   3  │ 10.0% │
│ 总计   │  150  │  120  │  30  │ 100.0%│
└──────┴──────┴──────┴──────┴──────┘
```

**使用场景：**
- 有一些未标注的背景图片
- 从 Label Studio 导出时未使用 `--include-negative`
- 希望手动添加负样本以提高模型质量

#### 2.2 生成数据集配置文件

创建YOLO训练所需的dataset.yaml配置文件：

```bash
python yolo_cli.py data generate-yaml \
  --path data/processed \
  --classes data/raw/classes.txt \
  --output data/dataset.yaml
```

生成的 `data/dataset.yaml` 内容示例：

```yaml
path: data/processed
train: images/train
val: images/val
test: images/test
names:
  0: waterpoll
  1: active_leak
nc: 2
```

#### 2.3 验证数据集

在训练前，验证数据集的完整性：

```bash
# 基本验证
python yolo_cli.py data verify --path data/processed

# 详细统计（包含类别分布和正负样本统计）
python yolo_cli.py data stats --path data/processed --detailed --task detect
```

验证内容包括：
- 检查图片和标签文件是否一一对应
- 验证标签文件格式是否正确
- 统计各类别的分布情况
- 检测潜在的数据问题

#### 正负样本统计

在详细统计模式下，系统会自动统计正负样本分布，帮助你评估数据集质量：

**定义**：
- **正样本** ✅: 包含标注对象的图像（标签文件存在且非空）
- **负样本** ❌: 不包含标注对象的图像（标签文件不存在或为空）

**输出示例**：

```
正负样本分布
┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ 数据集  ┃ 正样本  ┃ 负样本  ┃ 总样本  ┃ 正样本比例 ┃
┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ TRAIN   │ 450     │ 50      │ 500     │ 90.0%      │
│ VAL     │ 120     │ 10      │ 130     │ 92.3%      │
│ TEST    │ 60      │ 5       │ 65      │ 92.3%      │
│ 总计    │ 630     │ 65      │ 695     │ 90.6%      │
└─────────┴─────────┴─────────┴─────────┴────────────┘

平均标注对象数:
  TRAIN: 2.45 个对象/图像 (总计 1103 个对象)
  VAL: 2.38 个对象/图像 (总计 285 个对象)
  TEST: 2.52 个对象/图像 (总计 151 个对象)
```

**质量评估标准**：

| 正样本比例 | 评估结果 | 建议操作 |
|-----------|---------|---------|
| > 95% | 🟢 优秀 | 数据集干净，可以直接训练 |
| 80-95% | 🟡 正常 | 可以训练，注意监控效果 |
| 60-80% | 🟠 警告 | 建议检查和清理负样本 |
| < 60% | 🔴 问题 | 必须清理数据后再训练 |

**平均对象数评估**：

| 平均对象数 | 场景 | 建议 |
|-----------|------|------|
| < 1 | 目标稀疏 | 可能需要调整anchor大小 |
| 1-5 | 正常范围 | 使用标准训练参数 |
| 5-10 | 目标较密 | 可以适当增加batch size |
| > 10 | 目标密集 | 建议更大batch size和更多训练轮数 |

**应用场景**：

1. **训练前数据检查** - 识别标签缺失或标注不完整
2. **数据平衡评估** - 了解正负样本比例，避免样本不平衡
3. **数据清理效果评估** - 对比清理前后的统计结果
4. **标注密度分析** - 评估每张图像的平均标注对象数

**分类任务的正负样本统计**：

对于分类任务，你可以指定哪些类别为"正类"，其余为"负类"，这在异常检测等场景中特别有用。

**命令行模式**：
```bash
# 指定正类（适用于异常检测等场景）
python yolo_cli.py data stats \
  --path data/processed \
  --detailed \
  --task classify \
  --positive-classes "normal,good"
```

**交互式模式**：
1. 选择 `数据处理` → `数据统计`
2. 选择任务类型：`classify`
3. 输入数据集路径
4. 确认显示详细统计
5. 系统会自动检测所有类别，并让你从列表中选择正类
6. **重要操作说明**：
   - 使用 **↑↓ 方向键** 移动光标
   - 使用 **空格键** 选择/取消选择正类（可多选）
   - 按 **回车键** 确认选择
   - ⚠️ 注意：必须至少选择一个正类，否则无法进行正负样本统计

**交互式选择示例**：

```
检测到 3 个类别: closed, open, unlock

💡 正负样本统计说明：
   - 选择一个或多个类别作为「正类」
   - 其余类别将自动归为「负类」
   - 适用于异常检测、二分类等场景

? 是否选择正类进行正负样本统计? Yes

📋 操作说明：
   1. 使用 ↑↓ 键移动
   2. 使用 空格键 选择/取消选择
   3. 按 回车键 确认选择

? 选择正类 (空格选择，回车确认):
  ◯ closed
  ◉ open      ← 空格键已选中（显示为 ◉）
  ◯ unlock

✓ 已选择正类: open
  负类: closed, unlock
```

**输出示例**（分类任务）：
```
正负样本分布
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ 数据集  ┃ 正类样本  ┃ 负类样本  ┃ 总样本  ┃ 正类比例   ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ TRAIN   │ 300       │ 200       │ 500     │ 60.0%      │
│ VAL     │ 80        │ 50        │ 130     │ 61.5%      │
│ TEST    │ 40        │ 25        │ 65      │ 61.5%      │
│ 总计    │ 420       │ 275       │ 695     │ 60.4%      │
└─────────┴───────────┴───────────┴─────────┴────────────┘

各类别详细分布:

正类:
  good: train=150, val=40, test=20 (总计 210)
  normal: train=150, val=40, test=20 (总计 210)

负类:
  defect: train=120, val=30, test=15 (总计 165)
  anomaly: train=80, val=20, test=10 (总计 110)
```

**注意事项**：
- 检测/分割任务：自动统计有标注对象（正）和无标注对象（负）的图像
- 分类任务：需要手动指定正类，其余类别自动归为负类
- 空标签文件和无标签文件都视为负样本（仅检测/分割任务）
- 格式错误的标注会在 `verify` 命令中报告

### 步骤3：下载预训练模型

根据你的数据集大小选择合适的模型：

```bash
# 小数据集（<500张）：使用nano模型
python yolo_cli.py model download --version yolo11 --size n

# 中等数据集（500-2000张）：使用small模型
python yolo_cli.py model download --version yolo11 --size s

# 大数据集（>2000张）：使用medium模型
python yolo_cli.py model download --version yolo11 --size m

# 下载多个模型
python yolo_cli.py model download --version yolo11 --size n s m

# 查看已下载的模型
python yolo_cli.py model list
```

### 步骤4：训练模型

#### 4.1 基础训练

使用默认参数开始训练：

```bash
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640
```

#### 4.2 根据数据集大小选择配置

**小数据集（<500张）：**
```bash
python yolo_cli.py train start \
  --model yolo11n.pt \
  --data data/dataset.yaml \
  --epochs 150 \
  --batch 8 \
  --augmentation conservative \
  --patience 30
```

**中等数据集（500-2000张）：**
```bash
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16 \
  --augmentation balanced \
  --patience 50
```

**大数据集（>2000张）：**
```bash
python yolo_cli.py train start \
  --model yolo11m.pt \
  --data data/dataset.yaml \
  --epochs 300 \
  --batch 32 \
  --augmentation aggressive \
  --patience 100
```

#### 4.3 指定设备

```bash
# 自动选择最佳设备（推荐）
python yolo_cli.py train start --device auto ...

# 使用Apple Silicon (M1/M2/M3)
python yolo_cli.py train start --device mps ...

# 使用NVIDIA GPU - 单卡
python yolo_cli.py train start --device 0 ...

# 使用NVIDIA GPU - 多卡
python yolo_cli.py train start --device 0,1,2 ...

# 使用CPU
python yolo_cli.py train start --device cpu ...
```

**GPU设备选择说明：**
- `--device 0`：使用第一块GPU（GPU ID为0）
- `--device 1`：使用第二块GPU（GPU ID为1）
- `--device 0,1,2`：使用多块GPU（ID为0、1、2）进行分布式训练
- `--device cuda`：使用默认GPU（通常是GPU 0）

**通过环境变量指定GPU：**

你也可以通过 `CUDA_VISIBLE_DEVICES` 环境变量在启动时指定要使用的GPU，无需在命令中添加 `--device` 参数：

```bash
# 使用单个GPU (ID: 4)
export CUDA_VISIBLE_DEVICES=4
python yolo_cli.py train start --model yolo11s.pt --data data/dataset.yaml

# 或者一行命令
CUDA_VISIBLE_DEVICES=4 python yolo_cli.py train start --model yolo11s.pt --data data/dataset.yaml

# 使用多个GPU (ID: 0,1,2)
CUDA_VISIBLE_DEVICES=0,1,2 python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels

# 在交互式模式中
export CUDA_VISIBLE_DEVICES=3
python yolo_cli.py interactive-mode
# 系统会自动检测并使用GPU 3
```

**优势：**
- ✅ 在启动时就确定GPU，避免程序运行后占用错误的GPU
- ✅ 适合多用户共享服务器环境
- ✅ 与其他深度学习框架的使用习惯一致

#### 4.4 自定义实验名称和输出

```bash
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --project results/my_project \
  --name exp_001 \
  --epochs 200
```

训练结果将保存在：
```
results/my_project/exp_001/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最后一个epoch的模型
├── results.png      # 训练曲线
├── confusion_matrix.png
├── val_batch0_pred.jpg
└── ...
```

#### 4.5 恢复中断的训练

如果训练被中断，可以从检查点恢复：

```bash
# 自动查找最新的检查点
python yolo_cli.py train resume

# 指定检查点路径
python yolo_cli.py train resume \
  --checkpoint results/my_project/exp_001/weights/last.pt
```

### 步骤5：验证模型性能

训练完成后，使用验证命令全面评估模型性能：

#### 5.1 基本验证

**命令行模式：**
```bash
python yolo_cli.py validate run \
  results/training/best.pt \
  --data data/dataset.yaml
```

**交互式模式：** 🆕
```bash
python yolo_cli.py interactive-mode
# 选择 "验证操作" → "验证单个模型"
# 系统会引导你完成：
#   1. 输入模型路径
#   2. 输入数据集路径
#   3. 选择验证集 (val/test/train)
#   4. 选择任务类型（自动推断/手动指定）← 新增
#   5. 配置其他参数
```

#### 5.2 自定义验证参数

```bash
python yolo_cli.py validate run \
  results/training/best.pt \
  --data data/dataset.yaml \
  --split test \
  --conf 0.25 \
  --iou 0.6 \
  --save-json \
  --plots
```

**参数说明：**
- `--split`: 验证数据集 (val/test/train)
- `--conf`: 置信度阈值（默认：0.001）
- `--iou`: IoU阈值（默认：0.6）
- `--save-json`: 保存JSON格式结果
- `--plots`: 生成可视化图表

#### 5.3 验证指标说明 🆕 增强版

**✨ 新增全面的评估指标**，满足统计分析需求！

**核心指标速查表**

| 指标 | 公式 | 含义 | 何时关注 |
|------|------|------|----------|
| **Precision (精确率)** | TP / (TP + FP) | 预测为正的样本中真正为正的比例 | 想减少误报时 |
| **Recall (召回率)** | TP / (TP + FN) | 所有正样本中被正确预测的比例 | 想减少漏报时 |
| **F1 Score** 🆕 | 2 × (P × R) / (P + R) | 精确率和召回率的调和平均 | 平衡精确率和召回率时 |
| **Accuracy (准确率)** 🆕 | PR/(P+R-PR) | 检测准确性综合指标 | 评估整体性能 |
| **mAP@0.5** | - | IoU=0.5时的平均精度 | 快速评估定位能力 |
| **mAP@0.5:0.95** | - | COCO标准，更严格 | 严格评估定位精度 |

> **TP** = True Positive（真阳性）, **FP** = False Positive（假阳性/误报）  
> **TN** = True Negative（真阴性）, **FN** = False Negative（假阴性/漏报）

**各任务类型支持的指标**

对于**检测任务**：
- ✅ mAP@0.5, mAP@0.5:0.95
- ✅ 精确率、召回率、F1分数
- ✅ **准确率** - 始终显示，自动使用最佳计算方法：
  - 优先使用混淆矩阵（如果可用）
  - 否则从Precision和Recall推导：`Accuracy = (P×R)/(P+R-P×R)`
  - 最后使用mAP@0.5作为后备
- ✅ 推理速度统计（预处理/推理/后处理）
- ✅ 每类别详细指标（AP@0.5, AP@0.5:0.95, Precision, Recall, F1）

对于**分割任务**：
- ✅ 边界框指标：完整指标集
- ✅ 掩码指标：Mask mAP, Precision, Recall, F1
- ✅ 每类别分割指标：完整掩码级别指标

对于**分类任务**：
- ✅ Top-1准确率、Top-5准确率
- ✅ 宏平均精确率、召回率、F1分数
- ✅ 每类别详细指标（Accuracy, Precision, Recall, F1, Support）

**📊 JSON输出格式**

验证结果保存在 `validation_summary.json`，包含完整的统计数据：

```json
{
  "metrics": {
    "mAP50": 0.8542,
    "mAP50_95": 0.6234,
    "precision": 0.8123,
    "recall": 0.7856,
    "f1_score": 0.7987,      // 新增
    "accuracy": 0.8234        // 新增（如果可用）
  },
  "per_class": {              // 各类别详细指标
    "person": {
      "ap50": 0.9012,
      "precision": 0.8567,    // 新增
      "recall": 0.8234,       // 新增
      "f1_score": 0.8398      // 新增
    }
  },
  "performance": {            // 性能统计
    "speed_ms": {
      "preprocess": 2.5,
      "inference": 12.3,
      "postprocess": 3.8
    }
  }
}
```

**💡 使用技巧**

```bash
# 查看所有核心指标
cat results/validation/*/validation_summary.json | jq '.metrics'

# 查看F1分数
cat results/validation/*/validation_summary.json | jq '.metrics.f1_score'

# 查看各类别F1
cat results/validation/*/validation_summary.json | jq '.per_class | to_entries[] | {class: .key, f1: .value.f1_score}'

# 导出为CSV（导入Excel）
cat results/validation/*/validation_summary.json | jq -r '
  .per_class | to_entries[] | 
  [.key, .value.ap50, .value.precision, .value.recall, .value.f1_score] | 
  @csv
' > metrics.csv
```

**🎯 应用场景指南**

| 场景 | 优先指标 | 阈值建议 |
|------|----------|----------|
| 安全监控（减少漏报） | Recall | --conf 0.1 |
| 质量检测（减少误报） | Precision | --conf 0.5 |
| 平衡应用 | F1 Score | --conf 0.25 |
| 类别不平衡 | 每类别指标 + Macro Avg | - |

#### 5.4 比较多个模型

比较不同模型的性能，找出最佳模型：

**命令行模式：**
```bash
python yolo_cli.py validate compare \
  model1.pt,model2.pt,model3.pt \
  --data data/dataset.yaml \
  --task detect \
  --conf 0.25
```

**交互式模式：** 🆕
```bash
python yolo_cli.py interactive-mode
# 选择 "验证操作" → "比较多个模型"
# 系统会引导你完成：
#   1. 输入模型路径（逗号分隔）
#   2. 输入数据集路径
#   3. 选择任务类型（自动推断/手动指定）← 新增
#   4. 配置其他参数
```

会生成性能对比表格，自动标识最佳模型。

#### 5.5 结果文件

验证完成后，结果保存在 `results/validation/` 目录：
```
results/validation/
├── model_name_timestamp/
│   ├── validation_summary.json    # 验证结果摘要
│   ├── confusion_matrix.png       # 混淆矩阵
│   ├── F1_curve.png              # F1曲线
│   ├── PR_curve.png              # Precision-Recall曲线
│   └── results.json              # 详细结果
```

### 步骤6：使用模型进行推理

#### 6.1 单张图片检测

```bash
python yolo_cli.py detect image \
  results/training/best.pt \
  path/to/test_image.jpg \
  --conf 0.25
```

结果将保存在：
```
results/predictions/single_image/
├── test_image.jpg          # 标注后的图片
├── test_image_results.json # JSON格式的检测结果
└── labels/
    └── test_image.txt      # YOLO格式的标签
```

#### 6.2 批量检测

检测整个文件夹的图片：

```bash
python yolo_cli.py detect batch \
  results/training/best.pt \
  path/to/test_images/ \
  --conf 0.25 \
  --batch 4
```

#### 6.3 视频检测

```bash
# 检测视频并保存结果
python yolo_cli.py detect video \
  results/training/best.pt \
  path/to/video.mp4 \
  --conf 0.25

# 实时显示检测结果
python yolo_cli.py detect video \
  results/training/best.pt \
  path/to/video.mp4 \
  --conf 0.25 \
  --show
```

#### 6.4 摄像头实时检测

```bash
python yolo_cli.py detect webcam \
  results/training/best.pt \
  --conf 0.25 \
  --camera 0
```

### 步骤7：模型导出

将训练好的模型导出为部署格式：

#### 7.1 导出为ONNX（推荐）

```bash
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx \
  --imgsz 640
```

#### 7.2 导出为多种格式

```bash
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx torchscript tflite \
  --imgsz 640 \
  --output models/exported
```

支持的格式：
- `onnx`: ONNX格式（推荐，通用性好）
- `torchscript`: TorchScript格式
- `tflite`: TensorFlow Lite（移动端）
- `coreml`: CoreML（Apple设备）
- `engine`: TensorRT Engine（NVIDIA GPU）
- `pb`: TensorFlow SavedModel

### 步骤8：模型管理

#### 8.1 查看本地模型

```bash
# 列出所有模型
python yolo_cli.py model list

# 筛选特定版本
python yolo_cli.py model list --version yolo11

# 指定目录
python yolo_cli.py model list --dir models/weights
```

#### 8.2 查看模型详细信息

```bash
python yolo_cli.py model info results/training/best.pt
```

### 完整示例脚本

#### 方式1：一键训练（推荐）⚡

```bash
#!/bin/bash

# 1. 准备数据（将图片和标签放到data/raw/目录）
mkdir -p data/raw/images data/raw/labels
# cp -r /path/to/your/images/* data/raw/images/
# cp -r /path/to/your/labels/* data/raw/labels/
# cp /path/to/classes.txt data/raw/

# 2. 一键训练（自动完成所有步骤）
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --version yolo11 \
  --size s \
  --epochs 200 \
  --batch 16 \
  --device auto

# 3. 验证模型
python yolo_cli.py validate run \
  results/training/best.pt \
  --data data/dataset.yaml \
  --plots

# 4. 测试推理
python yolo_cli.py detect image \
  results/training/best.pt \
  test_image.jpg

# 5. 导出模型
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx

echo "完成！"
```

#### 方式2：逐步执行（完整控制）

```bash
#!/bin/bash

# 1. 创建目录结构
mkdir -p data/raw/images data/raw/labels

# 2. 将你的数据集复制到raw目录
# cp -r /path/to/your/images/* data/raw/images/
# cp -r /path/to/your/labels/* data/raw/labels/
# cp /path/to/classes.txt data/raw/

# 3. 划分数据集
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --ratios 0.7:0.2:0.1

# 4. 生成配置文件
python yolo_cli.py data generate-yaml \
  --path data/processed \
  --classes data/raw/classes.txt \
  --output data/dataset.yaml

# 5. 验证数据集
python yolo_cli.py data verify --path data/processed
python yolo_cli.py data stats --path data/processed --detailed --task detect

# 6. 下载预训练模型
python yolo_cli.py model download --version yolo11 --size s

# 7. 训练模型
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16 \
  --augmentation balanced \
  --device auto

# 8. 验证模型
python yolo_cli.py validate run \
  results/training/best.pt \
  --data data/dataset.yaml \
  --conf 0.25 \
  --save-json \
  --plots

# 9. 测试推理
python yolo_cli.py detect image \
  results/training/best.pt \
  data/processed/images/test/sample.jpg

# 10. 导出模型
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx

echo "完成！"
```

### 使用交互式模式

如果你不熟悉命令行，可以使用交互式模式，它会一步步引导你：

```bash
python yolo_cli.py interactive
```

交互式模式提供：
- 📋 主菜单：选择操作类型（模型/数据/训练/检测/验证）
- 🔍 参数提示：智能推荐和验证输入
- ✅ 确认步骤：每步操作前确认
- 📊 实时反馈：显示操作结果和进度
- 📈 数据统计：支持详细统计，包含正负样本分布

**在交互式模式中一键训练** 🆕：

1. 启动交互式模式：`python yolo_cli.py interactive-mode`
2. 选择 `一键训练 (自动化完整流程)`
3. **步骤1 - 数据集划分配置** 🆕：
   - **选择划分方式**：
     - **按比例划分**：适合常规训练，使用全部数据
     - **按样本数划分**：适合快速实验，从数据集中抽取固定数量的样本
   - **输入划分参数**：
     - 按比例：输入 `0.7:0.2:0.1`（train:val:test）
     - 按样本数：输入 `100:30:10`（train:val:test）
4. **步骤2 - 训练参数配置**：
   - YOLO版本、任务类型、模型大小
   - 训练轮数、批次大小、图像尺寸
   - 设备选择、数据增强预设
   - **高级选项配置** 🆕（可选）：
     - **训练控制参数**：早停耐心值、保存周期
     - **数据增强详细配置** 🎨（可选）：
       - 颜色空间增强（HSV-H/S/V）
       - 几何变换（旋转、平移、缩放、剪切、透视）
       - 翻转增强（上下翻转、左右翻转）
       - 高级增强（Mosaic、MixUp、随机擦除）
       - 自动增强策略（AutoAugment/RandAugment）
     - **优化器配置** ⚡（可选）：
       - 学习率设置（初始/最终）
       - 动量和权重衰减
       - Warmup参数
     - **损失函数权重** ⚖️（可选）：
       - 边界框损失、分类损失、DFL损失（检测/分割）
       - 标签平滑（分类）
5. **步骤3 - 数据路径配置**：
   - 输入图像目录和标签目录
6. **步骤4 - 数据验证选项**：
   - 选择是否验证数据集和统计数据分布
7. **确认并执行**：
   - 显示完整配置摘要
   - 确认后自动完成：数据集划分 → 配置生成 → 数据验证 → 模型下载 → 开始训练

**在交互式模式中划分数据集** 🆕：

1. 启动交互式模式：`python yolo_cli.py interactive-mode`
2. 选择 `数据处理` → `划分数据集`
3. 选择任务类型（detect/segment/classify）
4. **选择划分方式**：
   - **按比例划分**：适合常规训练，使用全部数据
   - **按样本数划分**：适合快速实验，从数据集中抽取固定数量的样本 🆕
5. 根据选择的方式输入参数：
   - 按比例：输入 `0.7:0.2:0.1`（train:val:test）
   - 按样本数：输入 `100:30:10`（train:val:test）
6. 对于检测/分割任务，可选择是否为缺失标签创建空文件（负样本）
7. 确认后自动完成数据集划分

**在交互式模式中使用数据统计**：

1. 启动交互式模式：`python yolo_cli.py interactive-mode`
2. 选择 `数据处理` → `数据统计`
3. 选择任务类型（detect/segment/classify）
4. 输入数据集路径（默认：data/processed）
5. 确认显示详细统计（包含正负样本统计）
6. 系统会自动显示完整的统计信息，包括正负样本分布、平均对象数和类别分布

## 📊 模型验证

模型验证功能提供全面的性能评估工具，帮助你深入了解模型表现。

### 核心功能

#### ✨ 单模型验证

验证单个模型的性能，获取详细指标和可视化报告：

```bash
# 基本验证（自动检测任务类型）🆕
python yolo_cli.py validate run results/training/best.pt

# 在测试集上验证（使用实际部署阈值）
python yolo_cli.py validate run best.pt --split test --conf 0.25

# 显式指定任务类型（推荐用于分类模型）
python yolo_cli.py validate run best.pt --task classify --data data/processed

# 完整配置验证（保存所有结果）
python yolo_cli.py validate run best.pt \
  --split test \
  --conf 0.25 \
  --iou 0.6 \
  --save-json \
  --plots \
  --project results/final_validation
```

**🎯 智能任务类型检测** 🆕

系统会自动从模型本身检测任务类型，无需手动指定：
- ✅ 加载模型后，从模型对象获取真实任务类型
- ✅ 优先级：模型实际类型 > 文件名推断 > 手动指定
- ✅ 自动适配不同任务的评估指标和参数
- 💡 建议：首次验证新模型时，可以显式指定 `--task` 以确保准确

#### 🆚 模型对比

同时验证多个模型，快速找出最佳模型：

```bash
# 比较不同大小的模型（自动推断任务类型）
python yolo_cli.py validate compare \
  yolo11n.pt,yolo11s.pt,yolo11m.pt \
  --data data/dataset.yaml

# 比较分类模型（手动指定任务类型）
python yolo_cli.py validate compare \
  model1-cls.pt,model2-cls.pt \
  --task classify \
  --data data/images

# 比较分割模型
python yolo_cli.py validate compare \
  model1-seg.pt,model2-seg.pt \
  --task segment \
  --data data/dataset.yaml

# 比较不同训练配置的checkpoint
python yolo_cli.py validate compare \
  exp1/best.pt,exp2/best.pt,exp3/best.pt \
  --conf 0.25

# 比较不同路径下的同名模型 🆕
python yolo_cli.py validate compare \
  results/exp1/best.pt,results/exp2/best.pt,results/exp3/best.pt \
  --data data/dataset.yaml
```

**任务类型智能识别** 🤖

- ✅ **自动推断**：从第一个模型文件名推断（如 `model-cls.pt` → classify）
- ✅ **手动指定**：使用 `--task` 参数明确指定任务类型
- ✅ **优先级**：手动指定 > 文件名推断 > 默认detect
- 💡 **建议**：对于非标准命名的模型，建议使用 `--task` 参数

**🆕 增强功能（v1.2.0）**：

1. **智能路径显示** 🎯
   - ✅ 自动检测同名模型
   - ✅ 有重名时显示完整路径，无重名时只显示文件名
   - ✅ 最佳模型显示完整路径，便于定位

2. **完整指标对比** 📊
   - ✅ mAP@0.5, mAP@0.5:0.95
   - ✅ Precision（精确率）
   - ✅ Recall（召回率）
   - ✅ F1 分数（新增）
   - ✅ Accuracy（准确率，新增）
   - ✅ 推理速度（毫秒，新增）

3. **多任务支持** 🔄
   - ✅ 检测任务：完整的检测指标
   - ✅ 分割任务：掩码指标对比
   - ✅ 分类任务：Top-1/Top-5准确率

**示例输出**：

```
━━━━━━━━━━━━━━━━━━━━━━━━ 性能比较结果 ━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ 模型路径                   ┃ mAP@0.5  ┃ mAP@0.5:0.95 ┃ Precision┃ Recall ┃ F1     ┃ Accuracy ┃ 速度(ms)   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ results/exp1/best.pt       │ 0.8542   │ 0.6234       │ 0.8123   │ 0.7856 │ 0.7987 │ 0.7645   │ 45.3       │
│ results/exp2/best.pt       │ 0.8721   │ 0.6512       │ 0.8345   │ 0.8012 │ 0.8174 │ 0.7892   │ 47.1       │
│ results/exp3/best.pt       │ 0.8456   │ 0.6123       │ 0.8012   │ 0.7923 │ 0.7967 │ 0.7543   │ 43.8       │
└────────────────────────────┴──────────┴──────────────┴──────────┴────────┴────────┴──────────┴────────────┘

🏆 最高mAP@0.5: results/exp2/best.pt (0.8721)
🏆 最高F1分数: results/exp2/best.pt (0.8174)
🏆 最高准确率: results/exp2/best.pt (0.7892)
```

### 性能指标说明

#### 检测任务（Detection）

```
🎯 检测指标
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 指标               ┃ 值            ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ mAP@0.5            │ 0.8542        │  ← IoU=0.5时的平均精度
│ mAP@0.5:0.95       │ 0.6234        │  ← COCO标准指标（更严格）
│ 精确率 (Precision) │ 0.8123        │  ← 预测正确率
│ 召回率 (Recall)    │ 0.7856        │  ← 目标检出率
└────────────────────┴───────────────┘

📋 各类别指标 (AP@0.5)
┏━━━━━━━━━━┳━━━━━━━━━┓
┃ 类别     ┃ AP@0.5  ┃
┡━━━━━━━━━━╇━━━━━━━━━┩
│ person   │ 0.9123  │
│ car      │ 0.8456  │
└──────────┴─────────┘
```

**指标解释：**
- **mAP@0.5**: 检测框与真实框IoU≥0.5时的平均精度，常用基准
- **mAP@0.5:0.95**: IoU从0.5到0.95的平均精度，更全面的评估指标
- **Precision（精确率）**: 预测为正样本中实际为正的比例，高精确率=低误报
- **Recall（召回率）**: 实际正样本中被检测到的比例，高召回率=低漏报

#### 分割任务（Segmentation）

分割任务会显示边界框和掩码两套指标：

```
🎯 分割指标
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 指标类型  ┃ 指标               ┃ 值            ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 边界框    │ mAP@0.5            │ 0.8542        │
│           │ mAP@0.5:0.95       │ 0.6234        │
│           │                    │               │
│ 掩码      │ mAP@0.5            │ 0.8321        │
│           │ mAP@0.5:0.95       │ 0.6012        │
└───────────┴────────────────────┴───────────────┘
```

#### 分类任务（Classification）

```
🎯 分类指标
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 指标          ┃ 值            ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Top-1 准确率  │ 0.9234        │  ← 最高概率类别正确的比例
│ Top-5 准确率  │ 0.9876        │  ← 前5个概率中包含正确答案
└───────────────┴───────────────┘
```

**分类模型验证示例** 🆕：
```bash
# 自动检测任务类型（推荐）
python yolo_cli.py validate run best.pt --data data/processed

# 或显式指定（更保险）
python yolo_cli.py validate run best.pt --task classify --data data/processed

# 完整验证
python yolo_cli.py validate run best.pt \
  --task classify \
  --data data/processed \
  --split test \
  --save-json \
  --plots
```

**注意**：
- 分类模型的 `validation_summary.json` 中不会包含 `conf_threshold` 和 `iou_threshold`（这些仅用于检测/分割任务）
- 系统会自动从模型加载后检测真实任务类型，即使文件名是 `best.pt` 也能正确识别

### 主要参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--split` | 验证数据集 | val | val（模型选择）<br>test（最终评估） |
| `--conf` | 置信度阈值 | 0.001 | 训练验证：0.001<br>部署验证：0.25-0.5 |
| `--iou` | IoU阈值 | 0.6 | 0.5-0.7 |
| `--batch` | 批次大小 | 16 | 根据GPU内存调整 |
| `--plots` | 生成图表 | True | 重要验证时启用 |
| `--save-json` | 保存结果 | True | 建议启用 |

### 验证结果文件

验证完成后，结果保存在 `results/validation/` 目录：

```
results/validation/
└── model_name_20251218_143022/
    ├── validation_summary.json    # 验证结果摘要
    ├── confusion_matrix.png       # 混淆矩阵
    ├── F1_curve.png              # F1分数曲线
    ├── PR_curve.png              # Precision-Recall曲线
    ├── P_curve.png               # Precision曲线
    ├── R_curve.png               # Recall曲线
    └── results.json              # 详细结果数据
```

### 典型使用场景

#### 1. 训练后选择最佳模型

```bash
# 训练完成后立即验证
python yolo_cli.py train start --model yolo11s.pt --data data/dataset.yaml
python yolo_cli.py validate run results/training/best.pt --plots
```

#### 2. 对比不同模型选择最优

```bash
# 比较不同大小的模型
python yolo_cli.py validate compare \
  yolo11n.pt,yolo11s.pt,yolo11m.pt \
  --data data/dataset.yaml
```

#### 3. 部署前最终验证

```bash
# 在测试集上使用实际部署阈值验证
python yolo_cli.py validate run best.pt \
  --split test \
  --conf 0.25 \
  --save-json \
  --plots \
  --project results/final_validation \
  --name production_model_v1
```

#### 4. 性能监控

```bash
# 定期验证模型性能
python yolo_cli.py validate run production_model.pt \
  --data latest_dataset.yaml \
  --save-json
```

### 最佳实践

1. **多阶段验证**
   - 训练中：使用默认阈值（0.001）快速评估
   - 模型选择：在验证集上比较多个模型
   - 最终确认：在测试集上用实际部署阈值验证

2. **保存重要结果**
   - 使用 `--project` 和 `--name` 组织验证结果
   - 启用 `--save-json` 保存详细数据
   - 启用 `--plots` 生成可视化报告

3. **合理设置阈值**
   - 训练验证：使用低阈值（0.001）评估模型潜力
   - 部署验证：使用实际阈值（0.25-0.5）评估实际效果

### 查看帮助

```bash
# 主命令帮助
python yolo_cli.py validate --help

# 子命令帮助
python yolo_cli.py validate run --help
python yolo_cli.py validate compare --help
```

### 验证示例

运行验证并查看详细指标：

```bash
# 基本验证
python yolo_cli.py validate run models/best.pt

# 完整验证（保存JSON和图表）
python yolo_cli.py validate run models/best.pt \
  --split test \
  --conf 0.25 \
  --save-json \
  --plots
```

## 🎮 GPU配置指南

### 三种GPU指定方式

#### 方式1: 自动检测（推荐新手）

```bash
python yolo_cli.py train start --device auto ...
```

系统会自动检测并使用最佳设备：
1. Apple Silicon (MPS) - 如果可用
2. NVIDIA GPU (CUDA) - 如果可用  
3. CPU - 作为后备

#### 方式2: 命令参数指定

```bash
# 单GPU
python yolo_cli.py train start --device 0 ...       # 使用GPU 0
python yolo_cli.py train start --device 1 ...       # 使用GPU 1

# 多GPU
python yolo_cli.py train start --device 0,1,2 ...   # 使用GPU 0,1,2

# 其他设备
python yolo_cli.py train start --device mps ...     # Apple Silicon
python yolo_cli.py train start --device cpu ...     # CPU
```

#### 方式3: 环境变量指定（推荐共享服务器）

通过 `CUDA_VISIBLE_DEVICES` 环境变量指定GPU，这是多用户环境的**最佳实践**。

**单GPU：**
```bash
# 方式A: 导出环境变量
export CUDA_VISIBLE_DEVICES=4
python yolo_cli.py train start --model yolo11s.pt --data data/dataset.yaml
python yolo_cli.py detect image best.pt test.jpg
# 所有命令都会使用GPU 4

# 方式B: 一行命令（临时指定）
CUDA_VISIBLE_DEVICES=4 python yolo_cli.py train start ...
```

**多GPU：**
```bash
# 使用GPU 0, 1, 2
export CUDA_VISIBLE_DEVICES=0,1,2
python yolo_cli.py train start ...

# 交互式模式
export CUDA_VISIBLE_DEVICES=3
python yolo_cli.py interactive-mode
```

**优势：**
- ✅ 提前锁定GPU，避免占用错误的显卡
- ✅ 适合多用户共享服务器
- ✅ 与PyTorch、TensorFlow等框架一致
- ✅ 一次设置，所有命令生效

### 多用户环境示例

假设服务器有8块GPU，多个用户同时训练：

```bash
# 用户A - 使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1
cd ~/project_a
python yolo_cli.py train start ...

# 用户B - 使用GPU 2和3
export CUDA_VISIBLE_DEVICES=2,3
cd ~/project_b
python yolo_cli.py train start ...

# 用户C - 使用GPU 4
export CUDA_VISIBLE_DEVICES=4
cd ~/project_c
python yolo_cli.py quick train --images data/raw/images --labels data/raw/labels
```

### 检查GPU使用

```bash
# 查看GPU状态
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi
```

## 📚 命令参考

### 主命令

```bash
python yolo_cli.py [COMMAND] [OPTIONS]
```

### 可用命令列表

- **quick** - 一键训练（自动化完整流程）⚡
- **model** - 模型管理（下载、导出、列表）
- **data** - 数据处理（划分、验证、统计）
- **train** - 模型训练（启动、恢复、配置）
- **validate** - 模型验证（性能评估、模型比较）✨
- **predict** - 模型预测（检测、分割、分类）
- **detect** - 目标检测（图片、视频、批量）[别名]
- **interactive-mode** - 交互式模式 🎮

### 命令详细说明

#### `quick` - 一键训练 ⚡

```bash
# 一键训练
python yolo_cli.py quick train [OPTIONS]
  --images TEXT          原始图像目录 (必需)
  --labels TEXT          原始标签目录 (必需)
  --classes TEXT         类别文件路径
  --task TEXT            任务类型 (detect/segment/classify, 默认: detect)
  --version TEXT         YOLO版本 (默认: yolo11)
  --size TEXT            模型大小 (默认: s)
  --epochs INTEGER       训练轮数 (检测/分割默认: 200, 分类默认: 100)
  --batch INTEGER        批次大小 (检测/分割默认: 16, 分类默认: 32)
  --imgsz INTEGER        图像尺寸 (检测/分割默认: 640, 分类默认: 224)
  --device TEXT          设备 (默认: auto)
  --augmentation TEXT    数据增强策略 (默认: balanced)
  --ratios TEXT          数据集划分比例 (如: 0.7:0.2:0.1, 默认: 0.7:0.2:0.1) 🆕
  --counts TEXT          数据集划分样本数 (如: 100:30:10) 🆕
  --skip-verify          跳过数据验证
  --skip-stats           跳过数据统计
  --project TEXT         项目目录
  --name TEXT            实验名称
```

**注意事项：**
- 📊 **数据集划分方式**：`--ratios` 和 `--counts` 不能同时使用
  - 使用 `--ratios`：按比例划分，使用全部数据
  - 使用 `--counts`：按样本数划分，随机抽取指定数量
  - 如果都不指定，默认使用 `--ratios 0.7:0.2:0.1`
- 显式指定的参数值会覆盖默认值
- 例如：`--epochs 200` 在分类任务中也会使用 200 轮（不会被改为 100）

**使用示例：**

```bash
# 使用比例划分（默认）
python yolo_cli.py quick train --images data/raw/images --labels data/raw/labels

# 明确指定比例
python yolo_cli.py quick train --images data/raw/images --labels data/raw/labels --ratios 0.8:0.15:0.05

# 按样本数划分（快速实验）🆕
python yolo_cli.py quick train --images data/raw/images --labels data/raw/labels --counts 100:30:10

# 完整参数示例
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --task detect \
  --version yolo11 \
  --size s \
  --counts 100:30:10 \
  --epochs 100 \
  --batch 16 \
  --device 0
```

```bash
# 快速恢复训练
python yolo_cli.py quick resume [OPTIONS]
  --checkpoint TEXT      检查点路径
  --epochs INTEGER       额外训练轮数
```

#### `model` - 模型管理

```bash
# 下载预训练模型
python yolo_cli.py model download [OPTIONS]
  --version TEXT     YOLO版本 (yolo11/yolov8)
  --size TEXT        模型大小 (n/s/m/l/x)
  --all              下载所有模型
  --output TEXT      输出目录

# 导出模型
python yolo_cli.py model export MODEL [OPTIONS]
  --format TEXT      导出格式 (onnx/torchscript/tflite/coreml等)
  --imgsz INTEGER    图像尺寸
  --device TEXT      设备选择
  --output TEXT      输出目录

# 列出本地模型
python yolo_cli.py model list [OPTIONS]
  --dir TEXT         模型目录
  --version TEXT     筛选版本

# 显示模型信息
python yolo_cli.py model info MODEL
```

#### `data` - 数据处理

```bash
# 划分数据集
python yolo_cli.py data split [OPTIONS]
  --images TEXT                 图像目录 (必需，检测/分割任务)
  --labels TEXT                 标签目录 (必需，检测/分割任务)
  --source TEXT                 源目录 (必需，分类任务)
  --output TEXT                 输出目录
  --ratios TEXT                 划分比例 (如: 0.7:0.2:0.1)
  --counts TEXT                 划分样本数 (如: 100:30:10) 🆕
  --seed INTEGER                随机种子
  --task TEXT                   任务类型 (detect/segment/classify)
  --create-empty-labels         为缺失标签的图片创建空标签（负样本）
  --no-empty-labels             不创建空标签（默认）

注意:
  - --ratios 和 --counts 二选一，不能同时使用
  - 默认使用 --ratios 0.7:0.2:0.1

# 生成dataset.yaml
python yolo_cli.py data generate-yaml [OPTIONS]
  --path TEXT        数据集路径
  --classes TEXT     类别文件
  --output TEXT      输出文件
  --train TEXT       训练集目录
  --val TEXT         验证集目录
  --test TEXT        测试集目录

# 验证数据集
python yolo_cli.py data verify [OPTIONS]
  --path TEXT        数据集路径

# 数据统计
python yolo_cli.py data stats [OPTIONS]
  --path TEXT              数据集路径
  --detailed               显示详细统计（包括正负样本统计）
  --task TEXT              任务类型 (detect/segment/classify)
  --positive-classes TEXT  正类列表（逗号分隔，仅用于分类任务）

示例:
  # 检测任务统计
  python yolo_cli.py data stats --path data/processed --detailed --task detect
  
  # 分类任务统计（指定正类）
  python yolo_cli.py data stats --path data/processed --detailed --task classify --positive-classes "normal,good"
```

#### `train` - 模型训练

```bash
# 开始训练
python yolo_cli.py train start [OPTIONS]
  --model TEXT           模型名称或路径
  --data TEXT            数据集配置文件
  --epochs INTEGER       训练轮数
  --batch INTEGER        批次大小
  --imgsz INTEGER        图像尺寸
  --device TEXT          设备 (auto/mps/cuda/cpu)
  --optimizer TEXT       优化器 (auto/SGD/Adam/AdamW/NAdam/RAdam/RMSProp)
  --freeze INTEGER       冻结前N层 (0=不冻结, 10=冻结前10层)
  --project TEXT         项目目录
  --name TEXT            实验名称
  --augmentation TEXT    数据增强预设
  --patience INTEGER     早停耐心值
  --save-period INTEGER  保存周期
  --resume               从last.pt恢复
  --pretrained           使用预训练权重

# 恢复训练
python yolo_cli.py train resume [OPTIONS]
  --checkpoint TEXT      检查点路径
  --project TEXT         项目目录
  --name TEXT            实验名称

# 生成训练配置
python yolo_cli.py train config [OPTIONS]
  --output TEXT          输出文件
  --profile TEXT         配置预设 (small/medium/large)

# 验证模型（已移至 validate 命令）
python yolo_cli.py train validate MODEL [OPTIONS]
  --data TEXT            数据集配置文件
  --batch INTEGER        批次大小
  --imgsz INTEGER        图像尺寸
  --device TEXT          设备
```

#### `validate` - 模型验证

```bash
# 运行验证
python yolo_cli.py validate run MODEL [OPTIONS]
  --data TEXT            数据集配置文件 (默认: data/dataset.yaml)
  --split TEXT           验证数据集 (val/test/train, 默认: val)
  --task TEXT            任务类型 (detect/segment/classify, 自动推断)
  --batch INTEGER        批次大小 (默认: 16)
  --imgsz INTEGER        图像尺寸 (默认: 640)
  --conf FLOAT           置信度阈值 (默认: 0.001)
  --iou FLOAT            IoU阈值 (默认: 0.6)
  --device TEXT          设备 (默认: auto)
  --save-json            保存JSON格式结果
  --save-hybrid          保存混合标签
  --plots                生成可视化图表
  --verbose              详细输出
  --project TEXT         结果保存目录
  --name TEXT            验证实验名称

# 比较多个模型
python yolo_cli.py validate compare MODELS [OPTIONS]
  --data TEXT            数据集配置文件 (默认: data/dataset.yaml)
  --task TEXT            任务类型 (detect/segment/classify, 自动从第一个模型推断)
  --split TEXT           验证数据集 (默认: val)
  --batch INTEGER        批次大小 (默认: 16)
  --imgsz INTEGER        图像尺寸 (默认: 640)
  --conf FLOAT           置信度阈值 (默认: 0.001, 仅检测/分割)
  --iou FLOAT            IoU阈值 (默认: 0.6, 仅检测/分割)
  --device TEXT          设备 (默认: auto)

示例:
  # 基本验证（自动推断任务类型）
  python yolo_cli.py validate run best.pt
  
  # 验证分类模型（手动指定任务类型）
  python yolo_cli.py validate run model-cls.pt --task classify --data data/images
  
  # 在测试集上验证并保存详细结果
  python yolo_cli.py validate run best.pt --split test --conf 0.25 --save-json --plots
  
  # 比较多个检测模型
  python yolo_cli.py validate compare model1.pt,model2.pt,model3.pt --data data/dataset.yaml
  
  # 比较多个分类模型（手动指定任务类型）
  python yolo_cli.py validate compare model1.pt,model2.pt --task classify --data data/images
```

#### `detect` - 目标检测

```bash
# 单张图片检测
python yolo_cli.py detect image MODEL IMAGE [OPTIONS]
  --conf FLOAT           置信度阈值
  --iou FLOAT            IOU阈值
  --output TEXT          输出目录
  --save-txt/--no-txt    保存TXT结果
  --save-json/--no-json  保存JSON结果
  --show                 显示结果
  --device TEXT          设备

# 批量检测
python yolo_cli.py detect batch MODEL SOURCE [OPTIONS]
  --conf FLOAT           置信度阈值
  --iou FLOAT            IOU阈值
  --output TEXT          输出目录
  --save-txt/--no-txt    保存TXT结果
  --save-json/--no-json  保存JSON结果
  --device TEXT          设备
  --batch INTEGER        批次大小

# 视频检测
python yolo_cli.py detect video MODEL VIDEO [OPTIONS]
  --conf FLOAT           置信度阈值
  --iou FLOAT            IOU阈值
  --output TEXT          输出目录
  --save-txt             保存TXT结果
  --show                 实时显示
  --device TEXT          设备

# 摄像头检测
python yolo_cli.py detect webcam MODEL [OPTIONS]
  --conf FLOAT           置信度阈值
  --iou FLOAT            IOU阈值
  --device TEXT          设备
  --camera INTEGER       摄像头ID
```

#### `interactive` - 交互式模式

```bash
python yolo_cli.py interactive-mode
```

**功能亮点**：
- 📋 图形化菜单，无需记忆命令
- 🎯 智能参数提示和默认值
- 📊 **数据集划分支持两种方式** 🆕：
  - 按比例划分：使用全部数据（如：0.7:0.2:0.1）
  - 按样本数划分：抽取固定数量（如：100:30:10）
- ✅ 每步操作前确认
- 🔄 支持所有命令行功能

**数据集划分交互流程**：
1. 选择任务类型（detect/segment/classify）
2. **选择划分方式**（按比例/按样本数）🆕
3. 输入相应参数
4. 对于检测任务，可选择是否包含负样本
5. 确认并执行

## 💡 使用示例

### 示例1：一键训练（最简单）⚡

```bash
# 1. 准备数据
mkdir -p data/raw/images data/raw/labels
# 将图片复制到 data/raw/images/
# 将标签复制到 data/raw/labels/
# 创建 data/raw/classes.txt

# 2. 一键训练
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels

# 3. 验证模型（可选但推荐）
python yolo_cli.py validate run results/training/best.pt

# 完成！模型保存在 results/training/*/weights/best.pt
```

### 示例1-2：从大数据集中抽取样本训练 🆕

**命令行方式**：

```bash
# 场景：有300个样本，但只想用其中140个来训练
# 目标：训练集100个，验证集30个，测试集10个

# 1. 按样本数划分数据集
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --counts 100:30:10

# 2. 生成配置和训练（后续步骤相同）
python yolo_cli.py data generate-yaml --path data/processed
python yolo_cli.py train start --model yolo11s.pt --data data/dataset.yaml
```

**交互式方式** 🎮：

```bash
# 启动交互式模式
python yolo_cli.py interactive-mode

# 在交互界面中（一键训练流程）：
# 1. 选择：一键训练 (自动化完整流程)
# 
# 【步骤1: 数据集划分配置】- 优先配置 🆕
# 2. 选择划分方式：按样本数划分 (推荐用于快速实验)
# 3. 输入样本数：100:30:10
# 
# 【步骤2: 训练参数配置】
# 4. YOLO版本: yolo11
# 5. 任务类型: detect
# 6. 模型大小: s
# 7-11. 其他训练参数...
# 
# 【步骤3: 数据路径配置】
# 12. 图像目录: data/raw/images
# 13. 标签目录: data/raw/labels
# 
# 【步骤4: 高级选项】
# 14. 验证数据集? Yes
# 15. 统计数据分布? Yes
# 
# 【配置摘要与确认】
# 16. 显示完整配置摘要
# 17. 确认开始一键训练
#
# 输出示例：
# ℹ 划分方式: 按样本数
# ℹ 目标样本数: 训练=100, 验证=30, 测试=10
# ℹ 找到 300 个有效样本
# ℹ 将从 300 个样本中随机抽取 140 个
# 
# 数据集划分结果
# ┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
# ┃ 数据集 ┃ 样本数 ┃ 比例   ┃
# ┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
# │ 训练集 │ 100    │ 71.4%  │
# │ 验证集 │ 30     │ 21.4%  │
# │ 测试集 │ 10     │ 7.1%   │
# │ 总计   │ 140    │ 100.0% │
# └────────┴────────┴────────┘
# ✓ 数据集准备完成
# ✓ 配置文件生成完成
# ✓ 数据验证完成
# ✓ 开始训练...
```

### 示例2：完整的训练流程（逐步控制）

```bash
# 1. 下载模型
python yolo_cli.py model download --version yolo11 --size s

# 2. 准备数据
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels

# 3. 生成配置
python yolo_cli.py data generate-yaml \
  --path data/processed

# 4. 验证数据
python yolo_cli.py data verify --path data/processed

# 5. 训练模型
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16 \
  --augmentation balanced

# 6. 验证模型
python yolo_cli.py validate run \
  results/training/best.pt \
  --data data/dataset.yaml \
  --conf 0.25 \
  --plots

# 7. 测试模型
python yolo_cli.py detect image \
  results/training/best.pt \
  test.jpg
```

### 示例2：使用交互式模式（推荐新手）🎮

交互式模式提供图形化菜单，特别适合新手和需要精确控制的场景：

```bash
# 启动交互式模式
python yolo_cli.py interactive-mode

# 典型操作流程（一键训练）：
# 1. 主菜单 → 选择"一键训练 (自动化完整流程)"
# 
# 【步骤1: 数据集划分配置】🆕 优先配置
# 2. 划分方式 → 选择"按样本数划分 (推荐用于快速实验)"
# 3. 输入样本数 → 100:30:10 (训练:验证:测试)
# 
# 【步骤2: 训练参数配置】
# 4. YOLO版本 → yolo11
# 5. 任务类型 → detect
# 6. 模型大小 → s
# 7. 训练轮数 → 100
# 8. 批次大小 → 16
# 9. 图像尺寸 → 640
# 10. 设备 → 0 (或 auto)
# 11. 数据增强预设 → balanced
# 12. 配置高级选项? → Yes 🆕
#     ┌─ ⚙️ 高级选项配置 ─┐
#     │ 📊 训练控制参数:
#     │   - 早停耐心值: 50
#     │   - 保存周期: 10
#     │ 
#     │ 🎨 数据增强配置:
#     │   - 自定义数据增强参数? → Yes
#     │     ├─ HSV-Hue增益: 0.015
#     │     ├─ HSV-Saturation增益: 0.7
#     │     ├─ HSV-Value增益: 0.4
#     │     ├─ 平移比例: 0.1
#     │     ├─ 缩放比例: 0.5
#     │     ├─ 左右翻转概率: 0.5
#     │     ├─ Mosaic增强: 1.0
#     │     ├─ MixUp增强: 0.1
#     │     ├─ 随机擦除: 0.2
#     │     └─ AutoAugment: Yes
#     │ 
#     │ ⚡ 优化器配置:
#     │   - 自定义优化器参数? → Yes
#     │     ├─ 初始学习率: 0.01
#     │     ├─ 最终学习率比例: 0.01
#     │     ├─ SGD动量: 0.937
#     │     ├─ 权重衰减: 0.0005
#     │     └─ Warmup轮数: 3.0
#     │ 
#     │ ⚖️ 损失函数权重:
#     │   - 自定义损失权重? → Yes
#     │     ├─ 边界框损失: 7.5
#     │     ├─ 分类损失: 0.5
#     │     └─ DFL损失: 1.5
#     └──────────────────┘
# 
# 【步骤3: 数据路径配置】
# 13. 图像目录 → data/raw/images
# 14. 标签目录 → data/raw/labels
# 
# 【步骤4: 数据验证选项】
# 15. 验证数据集? → Yes
# 16. 统计数据分布? → Yes
# 
# 【确认执行】
# 17. 查看配置摘要 → 确认开始一键训练
# 18. 自动完成：数据集划分 → 配置生成 → 数据验证 → 模型下载 → 开始训练
```

**交互式模式优势**：
- 🎯 无需记忆命令参数
- 📋 清晰的步骤引导，分步骤配置
- 🔄 **数据集划分优先配置**（新功能，第一步就确定）🆕
- 📊 支持按比例或按样本数划分
- ⚙️ **详细高级配置**：13+个数据增强参数，优化器和损失权重全面可配 🆕
- 💡 **智能默认值**：每个参数都有注解、推荐范围和默认值
- ✅ 每步确认，避免误操作
- 📝 最后显示完整配置摘要

### 示例3：使用配置文件训练

```bash
# 生成小数据集配置
python yolo_cli.py train config \
  --profile small \
  --output my_config.yaml

# 使用配置文件训练（需要手动编辑脚本或使用配置文件中的参数）
```

### 示例4：批量处理和导出

```bash
# 批量检测所有测试图片
python yolo_cli.py detect batch \
  results/training/best.pt \
  test_images/ \
  --conf 0.3

# 导出模型为多种格式
python yolo_cli.py model export \
  results/training/best.pt \
  --format onnx torchscript tflite
```

### 示例5：不同数据集大小的推荐配置

```bash
# 小数据集 (<500张)
python yolo_cli.py train start \
  --model yolo11n.pt \
  --data data/dataset.yaml \
  --epochs 150 \
  --batch 8 \
  --augmentation conservative

# 中等数据集 (500-2000张)
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16 \
  --augmentation balanced

# 大数据集 (>2000张)
python yolo_cli.py train start \
  --model yolo11m.pt \
  --data data/dataset.yaml \
  --epochs 300 \
  --batch 32 \
  --augmentation aggressive
```

## ⚙️ 配置文件

### 默认配置

配置文件位于 `config/default.yaml`，包含所有默认设置。

### 预设配置

框架提供三种预设配置，适用于不同规模的数据集：

- **small.yaml**: 小数据集 (<500张图片)
  - 使用较小的模型（nano）
  - 保守的数据增强
  - 较短的训练周期

- **medium.yaml**: 中等数据集 (500-2000张图片)
  - 使用中等模型（small）
  - 平衡的数据增强
  - 标准训练周期

- **large.yaml**: 大数据集 (>2000张图片)
  - 使用较大模型（medium）
  - 激进的数据增强
  - 更长的训练周期

### 自定义配置

你可以创建自己的配置文件：

```yaml
model:
  default_version: yolo11
  weights_dir: models/weights

training:
  epochs: 200
  batch: 16
  imgsz: 640
  device: auto
  optimizer: auto  # auto/SGD/Adam/AdamW/NAdam/RAdam/RMSProp
  freeze: null  # 冻结前N层 (null=不冻结, 10=冻结前10层)

augmentation:
  default_preset: balanced

paths:
  data_raw: data/raw
  data_processed: data/processed
  results: results
  models: models
```

## 📊 数据增强策略

框架提供四种数据增强预设：

### 1. Conservative (保守) - 适合小数据集

```python
- HSV色调: 0.01
- Mosaic: 0.5
- MixUp: 0.0
- 随机擦除: 0.1
```

### 2. Balanced (平衡) - 推荐大多数场景

```python
- HSV色调: 0.015
- Mosaic: 1.0
- MixUp: 0.1
- 随机擦除: 0.2
```

### 3. Aggressive (激进) - 适合大数据集

```python
- HSV色调: 0.02
- Mosaic: 1.0
- MixUp: 0.15
- 随机擦除: 0.4
```

### 4. Default (默认) - YOLO官方默认值

```python
- HSV色调: 0.015
- Mosaic: 1.0
- MixUp: 0.0
- 随机擦除: 0.4
```

## ⚡ 优化器选择

框架支持多种优化器，可根据任务类型和数据集特点选择：

### 支持的优化器

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| **auto** | 自动选择（默认） | 让YOLO根据任务类型自动选择最佳优化器 |
| **SGD** | 随机梯度下降 | YOLO默认优化器，训练稳定，适合大多数场景 |
| **Adam** | 自适应学习率 | 收敛快，适合小数据集和快速实验 |
| **AdamW** | Adam + 权重衰减 | 改进的Adam，泛化能力更好，推荐用于大模型 |
| **NAdam** | Nesterov + Adam | 结合动量的Adam，收敛更快更稳定 |
| **RAdam** | 修正的Adam | 解决Adam早期训练不稳定问题 |
| **RMSProp** | 均方根传播 | 适合RNN和动态学习率场景 |

### 使用方法

**命令行方式：**
```bash
# 使用默认优化器（推荐）
python yolo_cli.py train start --model yolo11s.pt --data data/dataset.yaml

# 使用AdamW优化器
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --optimizer AdamW

# 快速训练时指定优化器
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --optimizer Adam
```

**配置文件方式：**
```yaml
training:
  optimizer: AdamW  # 指定优化器
```

**交互式模式：**
在高级选项中选择优化器类型（系统会提供可视化选择菜单）

### 选择建议

- 🎯 **默认场景**：使用 `auto` 或 `SGD`
- 🚀 **快速实验**：使用 `Adam` 或 `NAdam`
- 📈 **大模型训练**：使用 `AdamW`
- 🔬 **研究对比**：尝试不同优化器找到最佳组合

## ❄️ 层冻结（Freeze Layers）

层冻结是迁移学习中的重要技术，通过冻结模型的前几层来保留预训练权重，只训练后面的层。这在以下场景非常有用：

### 适用场景

- 🎯 **小数据集微调**：数据量较少时，冻结骨干网络防止过拟合
- 🚀 **快速训练**：减少需要训练的参数，加快训练速度
- 🔄 **迁移学习**：利用预训练模型的特征提取能力
- 💾 **显存优化**：冻结层不计算梯度，节省显存

### 冻结层数建议

| 任务类型 | 推荐冻结层数 | 说明 |
|---------|------------|------|
| **检测 (Detect)** | 10 层 | 冻结骨干网络前10层，保留底层特征 |
| **分割 (Segment)** | 12 层 | 分割任务需要更多特征，可冻结更多层 |
| **分类 (Classify)** | 0-15 层 | 根据数据集相似度调整 |
| **大数据集** | 0 层 | 数据充足时不建议冻结 |
| **超小数据集** | 15+ 层 | 只训练分类头部 |

### 使用方法

**命令行方式：**
```bash
# 冻结前10层（推荐用于小数据集）
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --freeze 10

# 快速训练时冻结
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --freeze 10

# 不冻结任何层（默认）
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --freeze 0
```

**配置文件方式：**
```yaml
training:
  freeze: 10  # 冻结前10层
```

**交互式模式：**
在高级选项中选择"冻结模型层"（系统会提供详细的配置说明）

### 效果说明

**冻结前10层时：**
- ✅ 训练速度提升 30-50%
- ✅ 显存占用减少 20-30%
- ✅ 小数据集上泛化能力更好
- ⚠️ 可能需要更长时间达到最佳精度

**不冻结（freeze=0）时：**
- ✅ 充分利用数据集学习特征
- ✅ 在大数据集上表现更好
- ⚠️ 训练时间更长
- ⚠️ 小数据集上可能过拟合

### 高级技巧

**渐进式解冻（Progressive Unfreezing）：**
```bash
# 第一阶段：冻结前10层，快速训练50轮
python yolo_cli.py train start --freeze 10 --epochs 50

# 第二阶段：不冻结，继续训练100轮
python yolo_cli.py train start --freeze 0 --epochs 100 --resume
```

## 🔧 设备支持

框架自动检测并使用最佳可用设备：

- **Apple Silicon (M1/M2/M3)**: 自动使用MPS加速
- **NVIDIA GPU**: 自动使用CUDA加速（支持单卡和多卡）
- **CPU**: 在没有GPU时使用CPU

也可以手动指定设备：

```bash
python yolo_cli.py train start --device auto   # 自动检测（推荐）
python yolo_cli.py train start --device mps    # Apple Silicon
python yolo_cli.py train start --device 0      # NVIDIA GPU 0
python yolo_cli.py train start --device 1      # NVIDIA GPU 1
python yolo_cli.py train start --device 0,1,2  # 多GPU训练
python yolo_cli.py train start --device cpu    # CPU
```

### 多GPU训练

如果你有多块NVIDIA GPU，可以使用多GPU训练：

```bash
# 使用GPU 0和1
python yolo_cli.py train start --device 0,1 --model yolo11s.pt --data data/dataset.yaml

# 使用所有可用GPU（0,1,2,3）
python yolo_cli.py train start --device 0,1,2,3 --model yolo11m.pt --data data/dataset.yaml
```

**注意：** 多GPU训练会自动使用DataParallel或DistributedDataParallel进行分布式训练。

## ❓ 常见问题

### Q: 如何选择合适的模型大小？

A: 根据你的数据集大小和应用场景：
- **Nano (n)**: 最快，适合边缘设备，小数据集
- **Small (s)**: 速度与精度平衡，适合大多数应用
- **Medium (m)**: 推荐用于中大型数据集
- **Large (l)**: 高精度，需要较好的硬件
- **Extra Large (x)**: 最高精度，需要大量数据和计算资源

### Q: 训练时GPU内存不足怎么办？

A: 尝试以下方法：
1. 减小batch size: `--batch 8`
2. 减小图像尺寸: `--imgsz 416`
3. 使用更小的模型: `--model yolo11n.pt`

### Q: 如何恢复中断的训练？

A: 使用resume命令：
```bash
python yolo_cli.py train resume --checkpoint path/to/last.pt
```

或在start命令中使用--resume标志：
```bash
python yolo_cli.py train start --resume ...
```

### Q: 支持哪些导出格式？

A: 支持多种格式：
- ONNX (推荐，通用性好)
- TorchScript
- TensorFlow Lite
- CoreML (Apple设备)
- TensorRT Engine
- TensorFlow SavedModel

### Q: 如何提高检测精度？

A: 几个建议：
1. 使用更多训练数据
2. 使用更大的模型
3. 增加训练轮数
4. 调整数据增强策略
5. 使用适当的学习率

### Q: classes.txt文件格式是什么？

A: 每行一个类别名称：
```
waterpoll
active_leak
```

## 📁 项目结构

```
workspace/
├── yolo_cli.py              # 主CLI入口
├── cli/                     # CLI模块
│   ├── commands/            # 命令模块
│   │   ├── model.py        # 模型命令
│   │   ├── data.py         # 数据命令
│   │   ├── train.py        # 训练命令
│   │   ├── detect.py       # 检测命令
│   │   └── interactive.py  # 交互式模式
│   ├── core/               # 核心功能
│   │   ├── config.py       # 配置管理
│   │   ├── version.py      # 版本管理
│   │   └── utils.py        # 工具函数
│   └── ui/                 # UI组件
│       ├── display.py      # 输出显示
│       └── prompts.py      # 交互提示
├── config/                 # 配置文件
│   ├── default.yaml        # 默认配置
│   └── profiles/           # 预设配置
├── scripts/                # 原有脚本（已整合）
├── data/                   # 数据目录
├── models/                 # 模型目录
├── results/                # 结果目录
├── requirements.txt        # 依赖清单
└── README.md              # 本文档
```

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可

本项目采用 MIT 许可证。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO实现
- [Typer](https://typer.tiangolo.com/) - CLI框架
- [Rich](https://rich.readthedocs.io/) - 终端美化
- [Questionary](https://questionary.readthedocs.io/) - 交互式提示

---

**YOLO CLI** - 让YOLO训练和推理更简单 🚀
