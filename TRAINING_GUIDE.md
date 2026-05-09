# 🚀 YOLO CLI 完整训练流程指南

> 版本：1.0  
> 更新日期：2026-02-04  
> 适用模型：YOLO11, YOLOv8

## 📋 目录

1. [输入物准备](#1-输入物准备)
2. [数据处理流程](#2-数据处理流程)
3. [训练参数详解](#3-训练参数详解)
4. [启动训练](#4-启动训练)
5. [输出物详解](#5-输出物详解)
6. [模型验证与测试](#6-模型验证与测试)
7. [常见问题与优化](#7-常见问题与优化)
8. [完整脚本示例](#8-完整脚本示例)

---

## 1. 输入物准备

### 1.1 原始数据来源

#### 选项 A：Label Studio 导出

```
输入：Label Studio 项目
├── 项目 URL: http://your-labelstudio-server/projects/31
├── API Token: eyJhbGci...（您的令牌）
└── 导出格式: JSON

导出文件：
└── labelstudioexport/project-31.json
    ├── 包含图片 URL
    ├── 包含标注信息（边界框、类别）
    └── 包含任务元数据
```

**导出文件结构示例：**

```json
[
  {
    "id": 1,
    "data": {
      "image": "http://server/data/upload/1/image.jpg"
    },
    "annotations": [{
      "result": [{
        "value": {
          "x": 10.5,
          "y": 20.3,
          "width": 30.2,
          "height": 40.1,
          "rectanglelabels": ["类别名称"]
        }
      }]
    }]
  }
]
```

#### 选项 B：本地数据集

```
输入：本地图片和标注
data/raw/
├── images/           # 图片文件夹
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/           # YOLO 格式标注
    ├── img001.txt
    ├── img002.txt
    └── ...
```

**YOLO 标注格式（.txt 文件）：**

```
# 每行一个目标：class_id center_x center_y width height
# 坐标都是归一化的（0-1）
0 0.5 0.5 0.3 0.4
2 0.2 0.3 0.1 0.15
```

### 1.2 数据量建议

| 数据规模 | 图片数量/类别 | 训练效果 | 建议 |
|---------|--------------|---------|------|
| **最小** | 50-100 | 可训练，效果一般 | 使用激进数据增强 |
| **推荐** | 500-1000 | 良好 | 标准训练配置 |
| **理想** | 1000+ | 优秀 | 可使用大模型 |

### 1.3 类别定义文件

```yaml
# classes.txt 或在 dataset.yaml 中定义
names:
  0: 类别0名称
  1: 类别1名称
  2: 类别2名称
  ...
```

---

## 2. 数据处理流程

### 2.1 从 Label Studio 转换（如适用）

```bash
python yolo_cli.py data convert-labelstudio \
  --input labelstudioexport/project-31.json \    # 输入：Label Studio JSON
  --task detect \                                 # 任务类型
  --output data/raw \                            # 输出目录
  --url http://your-labelstudio-server           # Label Studio URL
```

**参数说明：**

| 参数 | 说明 | 必需 | 示例 |
|------|------|------|------|
| `--input` | Label Studio 导出的 JSON 文件路径 | ✅ | `labelstudioexport/project-31.json` |
| `--task` | 任务类型 | ✅ | `detect`, `segment`, `classify`, `pose` |
| `--output` | 输出目录 | ✅ | `data/raw` |
| `--url` | Label Studio 服务器地址 | ❌ | `http://10.105.3.39` |

**任务类型详解：**

- `detect`: 目标检测（边界框）⭐ 最常用
- `segment`: 实例分割（多边形）
- `classify`: 图像分类
- `pose`: 姿态估计

**输出结构：**

```
data/raw/
├── images/
│   ├── 0cb20b23-roi_0066.jpg
│   ├── 02dd40f9-roi_0039.jpg
│   └── ... (所有图片)
├── labels/
│   ├── 0cb20b23-roi_0066.txt    # 每行：class_id x y w h
│   ├── 02dd40f9-roi_0039.txt
│   └── ... (所有标注文件)
├── classes.txt                   # 类别列表
├── dataset.yaml                  # 数据集配置
└── convert_log.txt              # 转换日志
```

### 2.2 数据集划分

```bash
python yolo_cli.py data split \
  --images data/raw/images \      # 输入：图片目录
  --labels data/raw/labels \      # 输入：标注目录
  --output data/processed \       # 输出：处理后的目录
  --ratios 0.7:0.2:0.1 \         # 划分比例（训练:验证:测试）
  --task detect                   # 任务类型
```

**参数说明：**

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--images` | 原始图片目录 | - | `data/raw/images` |
| `--labels` | 原始标注目录 | - | `data/raw/labels` |
| `--output` | 输出目录 | - | `data/processed` |
| `--ratios` | 数据集划分比例 | `0.7:0.2:0.1` | `0.8:0.2:0` |
| `--task` | 任务类型 | `detect` | `detect`, `segment` |
| `--seed` | 随机种子 | `42` | 任意整数 |

**划分比例建议：**

```bash
# 标准划分（有测试集）
--ratios 0.7:0.2:0.1    # 70% 训练，20% 验证，10% 测试

# 无测试集（数据量少时）
--ratios 0.8:0.2:0      # 80% 训练，20% 验证

# 大数据集
--ratios 0.8:0.1:0.1    # 80% 训练，10% 验证，10% 测试
```

**输出结构：**

```
data/processed/
├── images/
│   ├── train/              # 训练集图片
│   ├── val/                # 验证集图片
│   └── test/               # 测试集图片
├── labels/
│   ├── train/              # 训练集标注
│   ├── val/                # 验证集标注
│   └── test/               # 测试集标注
├── dataset.yaml            # 数据集配置文件 ⭐
└── split_statistics.txt    # 划分统计信息
```

**dataset.yaml 内容示例：**

```yaml
path: /absolute/path/to/yolocli/data/processed
train: images/train
val: images/val
test: images/test

nc: 10  # 类别数量
names:
  0: 类别0
  1: 类别1
  2: 类别2
  ...
  9: 类别9
```

### 2.3 数据验证（可选）

```bash
python yolo_cli.py data verify \
  --path data/processed/dataset.yaml
```

**验证内容：**
- ✅ 检查图片和标注文件是否匹配
- ✅ 检查标注格式是否正确
- ✅ 检查类别 ID 是否在有效范围内
- ✅ 统计数据集信息

---

## 3. 训练参数详解

### 3.1 基础参数

| 参数 | 说明 | 默认值 | 推荐值 | 示例 |
|------|------|--------|--------|------|
| `--data` | 数据集配置文件（必需） | - | - | `data/processed/dataset.yaml` |
| `--model` | 预训练模型 | `yolo11s.pt` | 见下表 | `yolo11n.pt`, `yolo11m.pt` |
| `--epochs` | 训练轮数 | 100 | 50-200 | 50, 100, 200 |
| `--batch` | 批次大小 | 16 | 8-32 | 8, 16, 32 |
| `--imgsz` | 图像尺寸 | 640 | 640 | 320, 640, 1280 |

**模型大小对比：**

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| `yolo11n.pt` | ~2.6M | ⚡⚡⚡⚡⚡ | ⭐⭐ | 实时检测、边缘设备 |
| `yolo11s.pt` | ~9.4M | ⚡⚡⚡⚡ | ⭐⭐⭐ | 平衡性能 ⭐ 推荐 |
| `yolo11m.pt` | ~20M | ⚡⚡⚡ | ⭐⭐⭐⭐ | 高精度需求 |
| `yolo11l.pt` | ~25M | ⚡⚡ | ⭐⭐⭐⭐⭐ | 离线处理 |
| `yolo11x.pt` | ~56M | ⚡ | ⭐⭐⭐⭐⭐ | 最高精度 |

**批次大小选择：**

```bash
# 根据 GPU 内存选择
GPU 内存 < 8GB   → batch=8
GPU 内存 8-16GB  → batch=16  ⭐ 推荐
GPU 内存 > 16GB  → batch=32

# Apple Silicon (MPS)
M1/M2/M3 (8GB)   → batch=8
M1/M2/M3 (16GB+) → batch=16
```

**图像尺寸选择：**

```bash
# 根据目标大小选择
小目标（<32px）   → imgsz=1280
中等目标          → imgsz=640   ⭐ 推荐
大目标（>200px）  → imgsz=320
```

### 3.2 学习率参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--lr0` | 初始学习率 | 0.01 | 0.0001-0.01 |
| `--lrf` | 最终学习率（相对于 lr0） | 0.01 | 0.01-0.1 |
| `--momentum` | SGD 动量 | 0.937 | 0.8-0.99 |
| `--weight_decay` | 权重衰减（L2 正则化） | 0.0005 | 0.0001-0.001 |

**学习率调整建议：**

```bash
# 数据量小（<500 张）
--lr0 0.0001

# 数据量中等（500-5000 张）
--lr0 0.001  ⭐ 推荐

# 数据量大（>5000 张）
--lr0 0.01

# 微调预训练模型
--lr0 0.0001 --freeze 10
```

### 3.3 优化器选择

| 优化器 | 特点 | 适用场景 | 学习率建议 |
|--------|------|----------|-----------|
| `auto` | 自动选择（推荐） | 所有场景 ⭐ | 自动 |
| `SGD` | 传统优化器，稳定 | 大数据集 | 0.01 |
| `Adam` | 自适应学习率 | 小数据集 | 0.001 |
| `AdamW` | Adam + 权重衰减 | 防止过拟合 | 0.001 |
| `NAdam` | Nesterov + Adam | 快速收敛 | 0.001 |
| `RAdam` | 稳健版 Adam | 训练不稳定时 | 0.001 |

```bash
# 使用特定优化器
--optimizer AdamW --lr0 0.001
```

### 3.4 数据增强策略

| 策略 | 说明 | 增强强度 | 适用场景 |
|------|------|----------|----------|
| `default` | 默认增强 | 中等 | 数据量充足 |
| `balanced` | 平衡增强 | 中等 | 通用场景 ⭐ |
| `conservative` | 保守增强 | 低 | 数据质量高 |
| `aggressive` | 激进增强 | 高 | 数据量少 |
| `custom` | 自定义增强 | 自定义 | 特殊需求 |

```bash
# 使用预设增强
--augmentation aggressive

# 自定义增强
--augmentation custom \
--augmentation-custom '{
  "hsv_h": 0.015,
  "hsv_s": 0.7,
  "hsv_v": 0.4,
  "degrees": 10.0,
  "translate": 0.2,
  "scale": 0.5,
  "shear": 5.0,
  "flipud": 0.5,
  "fliplr": 0.5,
  "mosaic": 1.0,
  "mixup": 0.1
}'
```

**数据增强参数详解：**

| 参数 | 说明 | 范围 | 默认值 |
|------|------|------|--------|
| `hsv_h` | 色调偏移 | 0.0-1.0 | 0.015 |
| `hsv_s` | 饱和度增益 | 0.0-1.0 | 0.7 |
| `hsv_v` | 亮度增益 | 0.0-1.0 | 0.4 |
| `degrees` | 旋转角度 | 0.0-180.0 | 0.0 |
| `translate` | 平移比例 | 0.0-1.0 | 0.1 |
| `scale` | 缩放比例 | 0.0-1.0 | 0.5 |
| `shear` | 剪切角度 | 0.0-45.0 | 0.0 |
| `flipud` | 上下翻转概率 | 0.0-1.0 | 0.0 |
| `fliplr` | 左右翻转概率 | 0.0-1.0 | 0.5 |
| `mosaic` | Mosaic 增强概率 | 0.0-1.0 | 1.0 |
| `mixup` | MixUp 增强概率 | 0.0-1.0 | 0.0 |

### 3.5 设备配置

```bash
# 自动检测（推荐）
--device auto  ⭐

# 指定设备
--device mps          # Apple Silicon GPU
--device cuda         # NVIDIA GPU
--device cuda:0       # 指定第一块 NVIDIA GPU
--device cpu          # CPU（最慢）

# 多 GPU 训练
--device 0,1,2,3      # 使用 4 块 GPU
```

**设备选择建议：**

| 设备 | 速度 | 适用场景 |
|------|------|----------|
| NVIDIA GPU (CUDA) | ⚡⚡⚡⚡⚡ | 最快，推荐 |
| Apple Silicon (MPS) | ⚡⚡⚡⚡ | M1/M2/M3 Mac |
| CPU | ⚡ | 无 GPU 时 |

**Apple Silicon 注意事项：**

```bash
# 需要设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 某些操作会回退到 CPU（如 NMS）
```

### 3.6 高级参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--patience` | 早停耐心值（多少 epoch 无改善后停止） | 50 | 30-100 |
| `--save-period` | 模型保存周期（每 N 个 epoch 保存） | 10 | 10 |
| `--pretrained` | 是否使用预训练权重 | True | True ⭐ |
| `--freeze` | 冻结层数（迁移学习） | None | 0-10 |
| `--cos-lr` | 使用余弦学习率调度 | False | False |
| `--close-mosaic` | 最后 N 个 epoch 关闭 Mosaic | 10 | 10 |

**层冻结（迁移学习）：**

```bash
# 冻结前 10 层（只训练后面的层）
--freeze 10

# 适用于：
# - 数据量很小（<200 张）
# - 与预训练数据集相似
# - 需要快速训练
```

---

## 4. 启动训练

### 4.1 命令行模式（推荐）

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 设置环境变量（Apple Silicon 需要）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 3. 启动训练
python yolo_cli.py train start "" \
  --data data/processed/dataset.yaml \
  --model yolo11s.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device auto \
  --augmentation balanced \
  --patience 50 \
  --save-period 10
```

**⚠️ 注意：** `""` 是必需的（用于绕过 Typer 的 KWARGS 参数问题）

### 4.2 快速训练模式

```bash
python yolo_cli.py quick train \
  --images data/raw/images \
  --labels data/raw/labels \
  --epochs 50 \
  --batch 16
```

**快速模式会自动：**
1. 划分数据集（70:20:10）
2. 生成 dataset.yaml
3. 启动训练

### 4.3 恢复训练

```bash
# 从上次中断的地方继续
python yolo_cli.py train resume "" \
  --resume results/training/yolo11s_processed_20260204_160443/weights/last.pt
```

### 4.4 使用脚本启动（推荐用于生产）

创建 `start_training.sh`:

```bash
#!/bin/bash

# 检查目录
if [ ! -d "venv" ]; then
  echo "❌ 错误: 未在 yolocli 项目根目录"
  exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "🚀 开始训练..."

# 启动训练
python yolo_cli.py train start "" \
  --data data/processed/dataset.yaml \
  --model yolo11s.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --lr0 0.001 \
  --augmentation aggressive \
  --patience 50 \
  --save-period 10

if [ $? -eq 0 ]; then
  echo "✅ 训练完成！"
else
  echo "❌ 训练失败！"
  exit 1
fi
```

执行：

```bash
chmod +x start_training.sh
./start_training.sh
```

### 4.5 后台训练

```bash
# 使用 nohup 后台运行
nohup ./start_training.sh > training.log 2>&1 &

# 查看日志
tail -f training.log

# 查看进程
ps aux | grep yolo_cli
```

---

## 5. 输出物详解

### 5.1 目录结构

```
results/training/yolo11s_processed_20260204_160443/
├── weights/                              # 模型权重 ⭐⭐⭐
│   ├── best.pt                          # 最佳模型（验证集上表现最好）
│   ├── last.pt                          # 最后一个 epoch 的模型
│   ├── epoch0.pt                        # 初始模型
│   ├── epoch10.pt                       # 第 10 个 epoch
│   ├── epoch20.pt                       # 第 20 个 epoch
│   └── ...
├── results.png                           # 训练曲线图 ⭐⭐⭐
├── results.csv                           # 详细训练数据
├── confusion_matrix.png                  # 混淆矩阵
├── confusion_matrix_normalized.png       # 归一化混淆矩阵
├── BoxF1_curve.png                      # F1 分数曲线
├── BoxPR_curve.png                      # Precision-Recall 曲线
├── BoxP_curve.png                       # Precision 曲线
├── BoxR_curve.png                       # Recall 曲线
├── labels.jpg                            # 标签分布统计
├── train_batch0.jpg                     # 训练批次可视化
├── train_batch1.jpg
├── train_batch2.jpg
├── train_batch240.jpg                   # 最后批次
├── val_batch0_labels.jpg                # 验证集真实标签
├── val_batch0_pred.jpg                  # 验证集预测结果
└── args.yaml                             # 训练参数记录
```

### 5.2 模型权重文件

#### best.pt（最重要）⭐⭐⭐

```
用途：用于实际部署和推理
特点：在验证集上表现最好的模型
大小：18MB（优化后，已移除优化器状态）
推荐：生产环境使用此模型
```

#### last.pt

```
用途：用于恢复训练
特点：最后一个 epoch 的模型（可能不是最好的）
大小：18MB（优化后）
推荐：训练中断后恢复使用
```

#### epochN.pt

```
用途：用于分析训练过程、回退到特定 epoch
特点：完整的训练状态（包含优化器状态）
大小：54MB（未优化）
推荐：调试和分析使用
```

### 5.3 可视化结果详解

#### results.png - 训练曲线 ⭐⭐⭐

包含 12 个子图：

1. **Box Loss (train)** - 训练集边界框损失
2. **Box Loss (val)** - 验证集边界框损失
3. **Cls Loss (train)** - 训练集分类损失
4. **Cls Loss (val)** - 验证集分类损失
5. **DFL Loss (train)** - 训练集分布焦点损失
6. **DFL Loss (val)** - 验证集分布焦点损失
7. **Precision (B)** - 精确率
8. **Recall (B)** - 召回率
9. **mAP50 (B)** - mAP@0.5
10. **mAP50-95 (B)** - mAP@0.5:0.95
11. **Learning Rate** - 学习率变化
12. **Instances** - 每个 epoch 的实例数

**如何解读：**

✅ **良好训练的特征：**
- Loss 曲线下降并趋于平稳
- train 和 val 的 loss 接近（不过拟合）
- mAP 曲线上升
- Precision 和 Recall 平衡

❌ **问题训练的特征：**
- val loss 上升而 train loss 下降 = 过拟合
- Loss 曲线震荡剧烈 = 学习率过高
- mAP 不上升 = 数据问题或模型问题

#### confusion_matrix.png - 混淆矩阵

```
行：真实类别
列：预测类别
对角线：正确预测
非对角线：错误预测

如何解读：
✅ 对角线数值高 = 模型预测准确
❌ 非对角线数值高 = 类别混淆严重
```

**示例分析：**

```
         预测类别0  预测类别1  预测类别2
真实类别0    25         2          1      ← 类别0识别良好
真实类别1     3        18          4      ← 类别1有混淆
真实类别2     0         1         22      ← 类别2识别良好
```

#### BoxPR_curve.png - Precision-Recall 曲线

```
横轴：Recall（召回率）
纵轴：Precision（精确率）
曲线下面积（AP）：越大越好

如何解读：
✅ 曲线越靠近右上角越好
✅ AP 值越高越好（>0.5 为良好）
```

#### labels.jpg - 标签分布统计

```
显示内容：
1. 每个类别的样本数量
2. 边界框大小分布
3. 边界框位置分布

如何解读：
✅ 类别分布均衡
❌ 某些类别样本过少 = 可能导致该类别识别差
```

#### train_batch*.jpg - 训练批次可视化

```
显示内容：
- 训练图片
- 真实标注框（绿色）
- 数据增强效果

用途：
- 检查数据增强是否合理
- 检查标注是否正确
```

#### val_batch0_labels.jpg vs val_batch0_pred.jpg

```
labels.jpg：验证集真实标注
pred.jpg：模型预测结果

对比查看：
✅ 预测框与真实框重合度高 = 模型良好
❌ 漏检、误检 = 模型需要改进
```

### 5.4 训练数据文件

#### results.csv

```csv
epoch,train/box_loss,train/cls_loss,val/box_loss,val/cls_loss,metrics/precision,metrics/recall,metrics/mAP50,metrics/mAP50-95
1,1.942,4.654,2.010,4.738,0.0101,0.211,0.0151,0.00717
2,1.905,4.539,1.877,4.599,0.337,0.0679,0.0233,0.0111
...
```

**用途：**
- 详细的训练数据记录
- 可用于自定义分析和可视化
- 导入 Excel/Python 进行深入分析

#### args.yaml

```yaml
model: yolo11s.pt
data: data/processed/dataset.yaml
epochs: 50
batch: 16
imgsz: 640
device: mps
lr0: 0.01
optimizer: auto
augmentation: balanced
...
```

**用途：**
- 记录所有训练参数
- 用于复现训练
- 用于对比不同实验

---

## 6. 模型验证与测试

### 6.1 验证模型性能

```bash
# Apple Silicon (M1/M2/M3) 用户必须设置此环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 完整命令（指定数据集）
python yolo_cli.py train validate \
  results/training/yolo11s_processed_20260204_160443/weights/best.pt \
  --data data/processed/dataset.yaml

# 简化命令（使用默认数据集路径）
python yolo_cli.py train validate \
  results/training/yolo11s_processed_20260204_160443/weights/best.pt
```

**注意事项：**
- `validate` 是 `train` 命令的子命令，模型路径作为位置参数传入
- **Apple Silicon 用户必须设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`**，因为 MPS 设备不支持 `torchvision::nms` 操作，会自动回退到 CPU

**验证输出示例：**

```
✓ 验证完成！

验证指标:
mAP50: 0.0160
mAP50-95: 0.0068
Precision: 0.1366
Recall: 0.0808

Speed: 1.0ms preprocess, 26.4ms inference, 0.0ms loss, 73.9ms postprocess per image
Results saved to /Users/.../runs/detect/val3
```

**验证结果文件：**
- `confusion_matrix.png` - 混淆矩阵
- `confusion_matrix_normalized.png` - 归一化混淆矩阵
- `BoxP_curve.png` - Precision 曲线
- `BoxR_curve.png` - Recall 曲线
- `BoxPR_curve.png` - Precision-Recall 曲线
- `BoxF1_curve.png` - F1 曲线
- `val_batch*_labels.jpg` - 真实标注可视化
- `val_batch*_pred.jpg` - 预测结果可视化

**输出指标：**

| 指标 | 说明 | 良好标准 |
|------|------|----------|
| **Precision** | 精确率 = TP/(TP+FP) | >0.7 |
| **Recall** | 召回率 = TP/(TP+FN) | >0.6 |
| **mAP50** | IoU=0.5 时的平均精度 | >0.5 |
| **mAP50-95** | IoU=0.5-0.95 的平均精度 | >0.3 |

**指标解释：**

```
TP (True Positive)：正确检测到的目标
FP (False Positive)：错误检测（误报）
FN (False Negative)：漏检（未检测到）

Precision 高 = 检测结果可信度高
Recall 高 = 漏检少
mAP 高 = 综合性能好
```

### 6.2 测试集推理

```bash
python yolo_cli.py infer predict \
  --model results/training/yolo11s_processed_20260204_160443/weights/best.pt \
  --source data/processed/images/test \
  --conf 0.25 \
  --iou 0.45 \
  --save
```

**参数说明：**

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--model` | 模型路径 | - | `best.pt` |
| `--source` | 输入源 | - | 图片/视频/目录 |
| `--conf` | 置信度阈值 | 0.25 | 0.25-0.5 |
| `--iou` | NMS IoU 阈值 | 0.45 | 0.45 |
| `--save` | 保存结果 | False | True |

**置信度阈值调整：**

```bash
# 减少误检（提高精确率）
--conf 0.5

# 减少漏检（提高召回率）
--conf 0.2

# 平衡
--conf 0.25  ⭐ 推荐
```

### 6.3 单张图片测试

```bash
python yolo_cli.py infer predict \
  --model results/training/yolo11s_processed_20260204_160443/weights/best.pt \
  --source path/to/image.jpg \
  --conf 0.25 \
  --save
```

### 6.4 视频推理

```bash
python yolo_cli.py infer predict \
  --model results/training/yolo11s_processed_20260204_160443/weights/best.pt \
  --source path/to/video.mp4 \
  --conf 0.25 \
  --save
```

### 6.5 实时摄像头检测

```bash
python yolo_cli.py infer predict \
  --model results/training/yolo11s_processed_20260204_160443/weights/best.pt \
  --source 0 \
  --conf 0.25
```

---

## 7. 常见问题与优化

### 7.1 训练问题诊断

#### 问题 1：Loss 不下降

**可能原因：**
- 学习率过高或过低
- 数据标注错误
- 模型过大（数据量不足）

**解决方案：**

```bash
# 降低学习率
--lr0 0.0001

# 使用更小的模型
--model yolo11n.pt

# 检查数据
python yolo_cli.py data verify --path data/processed/dataset.yaml
```

#### 问题 2：过拟合（val loss 上升）

**特征：**
- train loss 下降，val loss 上升
- train mAP 高，val mAP 低

**解决方案：**

```bash
# 增加数据增强
--augmentation aggressive

# 增加正则化
--weight_decay 0.001

# 使用更小的模型
--model yolo11n.pt

# 冻结部分层
--freeze 10
```

#### 问题 3：欠拟合（loss 都很高）

**特征：**
- train 和 val loss 都很高
- mAP 很低

**解决方案：**

```bash
# 使用更大的模型
--model yolo11m.pt

# 增加训练轮数
--epochs 200

# 提高学习率
--lr0 0.01

# 减少数据增强
--augmentation conservative
```

#### 问题 4：某些类别识别差

**可能原因：**
- 该类别样本太少
- 类别之间相似度高

**解决方案：**

```bash
# 增加该类别的样本
# 使用类别权重（如果支持）
# 检查标注质量
```

### 7.2 性能优化建议

#### 优化策略 1：数据增强

```bash
# 数据量 < 500 张
--augmentation aggressive \
--augmentation-custom '{
  "mosaic": 1.0,
  "mixup": 0.2,
  "copy_paste": 0.3
}'
```

#### 优化策略 2：学习率调度

```bash
# 使用余弦学习率
--cos-lr True

# 调整学习率范围
--lr0 0.001 --lrf 0.01
```

#### 优化策略 3：迁移学习

```bash
# 冻结前 10 层
--freeze 10 --lr0 0.0001 --epochs 50

# 然后解冻全部层
--freeze 0 --lr0 0.00001 --epochs 50
```

### 7.3 硬件相关问题

#### Apple Silicon (MPS) 问题

```bash
# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 如果仍然有问题，使用 CPU
--device cpu
```

#### CUDA 内存不足

```bash
# 减小批次大小
--batch 8

# 减小图像尺寸
--imgsz 320

# 使用更小的模型
--model yolo11n.pt
```

### 7.4 训练速度优化

```bash
# 使用混合精度训练（NVIDIA GPU）
--amp True

# 减少验证频率
--val-period 5  # 每 5 个 epoch 验证一次

# 使用更少的 workers
--workers 4
```

---

## 8. 完整脚本示例

### 8.1 从 Label Studio 到训练的完整流程

```bash
#!/bin/bash

# ============================================
# YOLO CLI 完整训练流程
# ============================================

set -e  # 遇到错误立即退出

# 配置参数
PROJECT_ROOT="/path/to/yolocli"
LABELSTUDIO_JSON="labelstudioexport/project-31.json"
LABELSTUDIO_URL="http://10.105.3.39"
EPOCHS=100
BATCH_SIZE=16
MODEL="yolo11s.pt"

cd "$PROJECT_ROOT"

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source venv/bin/activate

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 步骤 1: 转换 Label Studio 数据
echo "🔄 步骤 1/4: 转换 Label Studio 数据..."
python yolo_cli.py data convert-labelstudio \
  --input "$LABELSTUDIO_JSON" \
  --task detect \
  --output data/raw \
  --url "$LABELSTUDIO_URL"

if [ $? -ne 0 ]; then
  echo "❌ 数据转换失败"
  exit 1
fi

# 步骤 2: 划分数据集
echo "✂️ 步骤 2/4: 划分数据集..."
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --ratios 0.7:0.2:0.1 \
  --task detect

if [ $? -ne 0 ]; then
  echo "❌ 数据集划分失败"
  exit 1
fi

# 步骤 3: 验证数据
echo "✅ 步骤 3/4: 验证数据..."
python yolo_cli.py data verify \
  --path data/processed/dataset.yaml

# 步骤 4: 开始训练
echo "🚀 步骤 4/4: 开始训练..."
python yolo_cli.py train start "" \
  --data data/processed/dataset.yaml \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --batch "$BATCH_SIZE" \
  --imgsz 640 \
  --device auto \
  --lr0 0.001 \
  --augmentation aggressive \
  --patience 50 \
  --save-period 10

if [ $? -eq 0 ]; then
  echo "🎉 训练完成！"
  echo "📁 结果目录: results/training/"
  ls -lh results/training/
else
  echo "❌ 训练失败"
  exit 1
fi
```

### 8.2 本地数据集训练脚本

```bash
#!/bin/bash

# ============================================
# 本地数据集训练脚本
# ============================================

set -e

# 配置参数
DATA_YAML="data/processed/dataset.yaml"
MODEL="yolo11s.pt"
EPOCHS=100
BATCH=16
IMGSZ=640

# 激活虚拟环境
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "🚀 开始训练..."
echo "📊 数据集: $DATA_YAML"
echo "🤖 模型: $MODEL"
echo "🔢 轮数: $EPOCHS"
echo "📦 批次: $BATCH"

# 训练
python yolo_cli.py train start "" \
  --data "$DATA_YAML" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device auto \
  --lr0 0.001 \
  --augmentation balanced \
  --optimizer AdamW \
  --patience 50 \
  --save-period 10 \
  --pretrained True

echo "✅ 训练完成！"
```

### 8.3 多阶段训练脚本（迁移学习）

```bash
#!/bin/bash

# ============================================
# 多阶段训练脚本（迁移学习）
# ============================================

set -e

source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

DATA_YAML="data/processed/dataset.yaml"

# 阶段 1: 冻结训练（快速适应）
echo "🔒 阶段 1: 冻结前 10 层训练..."
python yolo_cli.py train start "" \
  --data "$DATA_YAML" \
  --model yolo11s.pt \
  --epochs 50 \
  --batch 16 \
  --freeze 10 \
  --lr0 0.0001 \
  --augmentation conservative

# 获取最佳模型路径
BEST_MODEL=$(ls -t results/training/*/weights/best.pt | head -1)
echo "✅ 阶段 1 完成，最佳模型: $BEST_MODEL"

# 阶段 2: 解冻训练（精细调优）
echo "🔓 阶段 2: 解冻全部层训练..."
python yolo_cli.py train start "" \
  --data "$DATA_YAML" \
  --model "$BEST_MODEL" \
  --epochs 100 \
  --batch 16 \
  --freeze 0 \
  --lr0 0.00001 \
  --augmentation aggressive

echo "🎉 多阶段训练完成！"
```

### 8.4 训练监控脚本

```bash
#!/bin/bash

# ============================================
# 训练监控脚本
# ============================================

TRAINING_DIR="results/training"
REFRESH_INTERVAL=10  # 秒

while true; do
  clear
  echo "======================================"
  echo "   YOLO 训练监控"
  echo "======================================"
  echo ""
  
  # 查找最新的训练目录
  LATEST_DIR=$(ls -td $TRAINING_DIR/*/ 2>/dev/null | head -1)
  
  if [ -z "$LATEST_DIR" ]; then
    echo "⚠️ 未找到训练目录"
  else
    echo "📁 训练目录: $LATEST_DIR"
    echo ""
    
    # 显示最新的 results.csv
    if [ -f "${LATEST_DIR}results.csv" ]; then
      echo "📊 最新训练指标:"
      tail -5 "${LATEST_DIR}results.csv" | column -t -s,
    fi
    
    echo ""
    echo "💾 模型文件:"
    ls -lh "${LATEST_DIR}weights/" 2>/dev/null | tail -5
  fi
  
  echo ""
  echo "🔄 每 $REFRESH_INTERVAL 秒刷新一次 (Ctrl+C 退出)"
  sleep $REFRESH_INTERVAL
done
```

---

## 附录

### A. 性能指标对照表

| mAP50-95 | 性能等级 | 适用场景 |
|----------|---------|----------|
| < 0.1 | 差 | 不可用 |
| 0.1 - 0.3 | 一般 | 原型测试 |
| 0.3 - 0.5 | 良好 | 实际应用 |
| 0.5 - 0.7 | 优秀 | 生产环境 |
| > 0.7 | 卓越 | 高精度需求 |

### B. 训练时间估算

| 数据集大小 | 模型 | 设备 | 预计时间/epoch |
|-----------|------|------|---------------|
| 100 张 | yolo11n | M3 | 30 秒 |
| 100 张 | yolo11s | M3 | 1 分钟 |
| 500 张 | yolo11s | M3 | 5 分钟 |
| 1000 张 | yolo11s | RTX 3090 | 2 分钟 |
| 5000 张 | yolo11m | RTX 3090 | 10 分钟 |

### C. 常用命令速查

```bash
# 数据转换
python yolo_cli.py data convert-labelstudio -i input.json -t detect -o data/raw

# 数据划分
python yolo_cli.py data split --images data/raw/images --labels data/raw/labels -o data/processed

# 开始训练
python yolo_cli.py train start "" --data data/processed/dataset.yaml --epochs 100

# 恢复训练
python yolo_cli.py train resume "" --resume weights/last.pt

# 验证模型
python yolo_cli.py model validate --model weights/best.pt --data dataset.yaml

# 推理预测
python yolo_cli.py infer predict --model weights/best.pt --source image.jpg
```

### D. 故障排查清单

- [ ] 虚拟环境已激活
- [ ] 数据集路径正确
- [ ] dataset.yaml 格式正确
- [ ] 图片和标注文件匹配
- [ ] 类别 ID 在有效范围内
- [ ] GPU/MPS 可用（如果使用）
- [ ] 环境变量已设置（Apple Silicon）
- [ ] 磁盘空间充足
- [ ] 依赖包已安装

---

## 总结

本指南涵盖了 YOLO CLI 训练的完整流程，包括：

✅ **输入物准备** - Label Studio 或本地数据集  
✅ **数据处理** - 转换、划分、验证  
✅ **参数配置** - 基础、高级、优化参数  
✅ **训练执行** - 命令行、脚本、后台运行  
✅ **输出分析** - 模型权重、可视化结果  
✅ **验证测试** - 性能评估、推理预测  
✅ **问题诊断** - 常见问题与解决方案  

**下一步建议：**

1. 📖 阅读本指南，了解完整流程
2. 🧪 使用小数据集进行测试训练
3. 📊 分析训练结果，调整参数
4. 🚀 使用完整数据集进行正式训练
5. 🎯 部署最佳模型到生产环境

**获取帮助：**

```bash
# 查看命令帮助
python yolo_cli.py --help
python yolo_cli.py train --help
python yolo_cli.py data --help
```

---

**文档版本：** 1.0  
**最后更新：** 2026-02-04  
**作者：** YOLO CLI Team
