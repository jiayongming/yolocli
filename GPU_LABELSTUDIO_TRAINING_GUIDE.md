# GPU 训练操作文档（Label Studio项目 -> YOLO 检测）

本文档用于在 GPU 服务器上手动完成从 Label Studio 导出到 YOLO 训练与验证的全流程。

## 流程总览

1. 连接 GPU 服务器
2. 准备独立训练目录（含环境检查）
3. 准备 Label Studio 导出文件
4. 下载并转换数据（目标检测）
5. 数据划分与校验
6. 正式训练（完整参数）
7. 训练后验证
8. 常见问题排查与安全收尾

---

## 1. 连接 GPU 服务器

```bash
ssh root@101.47.18.73 -p 22
```

---

## 2. 准备独立训练目录（含环境检查）

每次训练使用一个独立目录，避免不同任务互相干扰。

示例目录：

- `/data/circuit-breaker/yolocli`
- `/data/elec-switch/yolocli`
- `/data/pressure-board/yolocli`

示例（新建一次训练目录并解压）：

```bash
# clone代码
cd /data
RUN_DIR="yolocli-temp-$(date +%Y%m%d%H%M%S)"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"
git clone -b main https://github.com/jiayongming/yolocli.git
cd "/data/$RUN_DIR/yolocli"
# 切换虚拟环境
conda activate robot
python -V
python -m pip -V
nvidia-smi
# 一键训练
python yolo_cli.py interactive-mode
```

## 3. 准备 Label Studio 项目导出文件

1. 在 Label Studio 项目 19 页面导出 JSON。
2. 上传到当前项目目录下，例如：`labelstudioexport/project-19.json`。

---

## 4. 下载并转换数据（目标检测）

```bash
cd "/data/$RUN_DIR/yolocli"
python yolo_cli.py data convert-labelstudio \
  --input labelstudioexport/project-39-at-2026-05-09-08-38-bda8b63e.json \
  --url "http://10.105.3.39" \
  --token "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4NTQwNzMwOSwiaWF0IjoxNzc4MjA3MzA5LCJqdGkiOiIxNmYzMGExOGNlYjk0NWI0OWQyNDk5NzE0Y2E4OGI5YSIsInVzZXJfaWQiOjd9.7Ej5WWgfZHOuAHZfFyK6EZz99N132sNSfxO4h4S3Y_o" \
  --project-id 19 \
  --task detect \
  --include-negative \
  --output data/raw \
  --max-workers 8
```

转换后检查输出：

```bash
ls -lah data/raw
ls -lah data/raw/images | head
ls -lah data/raw/labels | head
cat data/raw/classes.txt
```

---

## 5. 数据划分与校验

数据划分：

```bash
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --ratios 0.7:0.2:0.1 \
  --task detect \
  --seed 42
```

数据校验：

```bash
python yolo_cli.py data verify \
  --path data/processed \
  --task detect
```

---

## 6. 正式训练（GPU）

```bash
python scripts/train/train_yolo.py \
  --model models/weights/yolov8n.pt \
  --data data/processed/dataset.yaml \
  --epochs 200 \
  --imgsz 640 \
  --batch 16 \
  --device auto \
  --project results/training \
  --name rust_detect_train \
  --augmentation balanced
```

参数建议：

- 显存紧张：减小 `--batch`（如 16 -> 8 -> 4）。
- 训练样本较少：适当减小 `--epochs`（如 200 -> 50）。
- 先追求速度：可改用更小模型（如 `yolo11n.pt`）。

---

## 7. 训练后验证

```bash
cd "/data/$RUN_DIR/yolocli"
BEST=$(ls -t results/training/*/weights/best.pt | head -1)
echo "$BEST"
python yolo_cli.py validate run "$BEST" \
  --data data/processed/dataset.yaml \
  --task detect \
  --device 0
```

指标说明（检测任务常见）：

- `Box(P)`：Precision（查准率）。
- `R`：Recall（召回率）。

---

## 8. 常见问题排查与安全收尾

### 8.1 `No module named ultralytics`

- 原因：未激活正确 conda 环境，或依赖未安装。
- 处理：

```bash
conda activate robot
python -m pip install -r requirements.txt
```

### 8.2 `cuda_ok=False`

- 原因：当前环境中的 `torch` 非 CUDA 版本，或驱动/版本不匹配。
- 处理：先确认 `nvidia-smi` 正常，再安装匹配 CUDA 的 PyTorch。

### 8.3 `project-19.json` 不存在

- 原因：导出文件未上传到指定路径，或文件名不一致。
- 处理：确认文件在 `labelstudioexport/project-19.json`，并与 `--input` 参数一致。

### 8.4 401/403 鉴权失败

- 原因：Token 失效、URL 错误，或 Label Studio 侧资源不可访问。
- 处理：重新生成 Token，确认 `LS_URL` 可访问，并检查项目 19 的数据源权限。

### 8.5 OOM（显存不足）

- 原因：`--batch` / `--imgsz` 过大。
- 处理：减小参数，如 `--batch 8 --imgsz 512`。