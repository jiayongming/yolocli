# YOLO CLI - YOLO推理快捷操作框架

一个功能完整、易于使用的命令行工具，用于YOLO模型的训练、推理和管理。支持YOLOv8和YOLO11，提供交互式操作界面和丰富的命令行选项。

## ✨ 特性

- ⚡ **一键训练**: 自动完成数据处理、模型下载和训练的完整流程
- 🎯 **完整工作流**: 涵盖模型下载、数据处理、训练、推理、导出全流程
- 🎮 **交互式模式**: 友好的交互式界面，引导式操作，适合新手
- 🔄 **多版本支持**: 同时支持YOLOv8和YOLO11，自动版本管理
- 🎨 **美化输出**: 使用Rich库提供彩色输出、进度条、表格等
- ⚙️ **灵活配置**: YAML配置文件，支持多种预设（小/中/大数据集）
- 🚀 **智能设备**: 自动检测最佳设备（MPS/CUDA/CPU），支持环境变量控制
- 📊 **数据增强**: 内置多种数据增强策略（保守/平衡/激进）
- 🔧 **多GPU支持**: 支持单卡、多卡训练，适合共享服务器环境
- 📦 **易于扩展**: 模块化设计，易于添加新功能

## 📋 目录

- [安装](#-安装)
- [快速开始](#-快速开始)
- [数据准备](#-数据准备)
- [完整工作流程](#-完整工作流程)
- [GPU配置指南](#-gpu配置指南)
- [命令参考](#-命令参考)
- [使用示例](#-使用示例)
- [配置文件](#-配置文件)
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

## ⚡ 快速开始

### 一键训练（最快捷）⚡

如果你已经准备好了数据，使用一键训练命令可以自动完成所有步骤：

```bash
python yolo_cli.py quick train \
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
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --version yolo11 \
  --size s \
  --epochs 200 \
  --batch 16 \
  --device 0
```

### 交互式模式（推荐新手）

如果你想要更多控制，使用交互式模式，它会引导你完成所有操作：

```bash
python yolo_cli.py interactive-mode
```

### 命令行模式

#### 1. 下载预训练模型

```bash
# 下载YOLO11 small模型
python yolo_cli.py model download --version yolo11 --size s

# 下载所有YOLO11模型
python yolo_cli.py model download --version yolo11 --all
```

#### 2. 准备数据集

```bash
# 划分数据集
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --ratios 0.7:0.2:0.1

# 生成dataset.yaml
python yolo_cli.py data generate-yaml \
  --path data/processed \
  --output data/dataset.yaml

# 验证数据集
python yolo_cli.py data verify --path data/processed
```

#### 3. 训练模型

```bash
# 开始训练
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --epochs 200 \
  --batch 16

# 使用特定的数据增强策略
python yolo_cli.py train start \
  --model yolo11s.pt \
  --data data/dataset.yaml \
  --augmentation aggressive
```

#### 4. 推理检测

```bash
# 单张图片检测
python yolo_cli.py detect image \
  results/training/best.pt \
  test.jpg

# 批量检测
python yolo_cli.py detect batch \
  results/training/best.pt \
  test_images/

# 视频检测
python yolo_cli.py detect video \
  results/training/best.pt \
  video.mp4 \
  --show
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

在开始训练前，需要按照以下格式准备数据集。

### 数据集目录结构

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

### 标签文件格式

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

将原始数据划分为训练集、验证集和测试集：

```bash
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --ratios 0.7:0.2:0.1 \
  --seed 42
```

**参数说明：**
- `--images`: 原始图片目录
- `--labels`: 原始标签目录
- `--output`: 输出目录（默认：data/processed）
- `--ratios`: 划分比例，格式为 train:val:test（默认：0.7:0.2:0.1）
- `--seed`: 随机种子，保证可重现性

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

# 详细统计（包含类别分布）
python yolo_cli.py data stats --path data/processed --detailed
```

验证内容包括：
- 检查图片和标签文件是否一一对应
- 验证标签文件格式是否正确
- 统计各类别的分布情况
- 检测潜在的数据问题

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

### 步骤5：评估模型

训练完成后，评估模型性能：

```bash
python yolo_cli.py train validate \
  results/training/best.pt \
  --data data/dataset.yaml
```

输出指标包括：
- mAP50：IoU阈值为0.5时的平均精度
- mAP50-95：IoU阈值从0.5到0.95的平均精度
- Precision：精确率
- Recall：召回率

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

# 3. 测试推理
python yolo_cli.py detect image \
  results/training/best.pt \
  test_image.jpg

# 4. 导出模型
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
python yolo_cli.py data stats --path data/processed --detailed

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

# 8. 评估模型
python yolo_cli.py train validate \
  results/training/best.pt \
  --data data/dataset.yaml

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
- 📋 主菜单：选择操作类型（模型/数据/训练/检测）
- 🔍 参数提示：智能推荐和验证输入
- ✅ 确认步骤：每步操作前确认
- 📊 实时反馈：显示操作结果和进度

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

### 可用命令

#### `quick` - 一键训练 ⚡

```bash
# 一键训练
python yolo_cli.py quick train [OPTIONS]
  --images TEXT          原始图像目录 (必需)
  --labels TEXT          原始标签目录 (必需)
  --classes TEXT         类别文件路径
  --version TEXT         YOLO版本 (默认: yolo11)
  --size TEXT            模型大小 (默认: s)
  --epochs INTEGER       训练轮数 (默认: 200)
  --batch INTEGER        批次大小 (默认: 16)
  --imgsz INTEGER        图像尺寸 (默认: 640)
  --device TEXT          设备 (默认: auto)
  --augmentation TEXT    数据增强策略 (默认: balanced)
  --ratios TEXT          数据集划分比例 (默认: 0.7:0.2:0.1)
  --skip-verify          跳过数据验证
  --skip-stats           跳过数据统计
  --project TEXT         项目目录
  --name TEXT            实验名称

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
  --images TEXT      图像目录 (必需)
  --labels TEXT      标签目录 (必需)
  --output TEXT      输出目录
  --ratios TEXT      划分比例 (默认: 0.7:0.2:0.1)
  --seed INTEGER     随机种子

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
  --path TEXT        数据集路径
  --detailed         显示详细统计
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

# 验证模型
python yolo_cli.py train validate MODEL [OPTIONS]
  --data TEXT            数据集配置文件
  --batch INTEGER        批次大小
  --imgsz INTEGER        图像尺寸
  --device TEXT          设备
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
python yolo_cli.py interactive
```

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

# 完成！模型保存在 results/training/*/weights/best.pt
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

# 6. 测试模型
python yolo_cli.py detect image \
  results/training/best.pt \
  test.jpg
```

### 示例2：使用配置文件训练

```bash
# 生成小数据集配置
python yolo_cli.py train config \
  --profile small \
  --output my_config.yaml

# 使用配置文件训练（需要手动编辑脚本或使用配置文件中的参数）
```

### 示例3：批量处理和导出

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

### 示例4：不同数据集大小的推荐配置

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
