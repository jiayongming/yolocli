# GPU训练操作文档（Label Studio 项目19 -> YOLO检测）

本文档用于在 GPU 服务器上手动完成以下流程：

1. 连接 GPU 服务器
2. 准备独立训练目录（每次训练一个目录，避免互相干扰）
3. 环境检查
4. 从 Label Studio 项目 19 导出并转换数据
5. 数据划分与校验
6. 探针训练与正式训练
7. 训练后验证

---

## 1. 连接 GPU 服务器

```bash
ssh root@101.47.18.73 -p 22
HMdX8udBVw
```

---

## 2. 准备独立训练目录（推荐）

每个任务使用一个独立目录，例如：

- `/data/circuit-breaker/yolocli`
- `/data/elec-switch/yolocli`
- `/data/pressure-board/yolocli`

示例（新建一次训练目录）：

```bash
cd /data
RUN_DIR="yolocli-temp-$(date +%Y%m%d%H%M%S)"
mkdir "$RUN_DIR" && cp yolocli.tar.gz "$RUN_DIR"/
cd "$RUN_DIR" && tar -zxvf yolocli.tar.gz && cd "/data/$RUN_DIR/yolocli"
conda activate robot
```

---

## 3. 环境检查

```bash
nvidia-smi
python -c "import ultralytics, torch; print('ultralytics=', ultralytics.__version__); print('torch=', torch.__version__); print('cuda_ok=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count())"
```

---

## 4. 准备 Label Studio 项目 19 导出文件

通常只需要 `project-19.json`，不需要额外上传图片或标签文件。

前提条件：

- `LS_URL` 可访问（`http://10.105.3.39`）
- Token 有效
- Label Studio 中项目 19 的图片源可被服务端正常访问

优先使用服务器端 API 直接导出（无需本地下载再上传）：

```bash
mkdir -p labelstudioexport
export LS_URL="http://10.105.3.39"
read -s eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4NTQwNzMwOSwiaWF0IjoxNzc4MjA3MzA5LCJqdGkiOiIxNmYzMGExOGNlYjk0NWI0OWQyNDk5NzE0Y2E4OGI5YSIsInVzZXJfaWQiOjd9.7Ej5WWgfZHOuAHZfFyK6EZz99N132sNSfxO4h4S3Y_o
curl -fSL \
  -H "Authorization: Token $LS_TOKEN" \
  "$LS_URL/api/projects/19/export?exportType=JSON" \
  -o labelstudioexport/project-19.json
```

检查文件：

```bash
ls -lh labelstudioexport/project-19.json
```

如果 API 导出失败，再使用网页导出兜底：

- 在 Label Studio 项目 19 页面导出 JSON
- 上传到当前目录下 `labelstudioexport/project-19.json`

---

## 5. 设置 Label Studio 连接参数（避免明文泄露）

如果你在上一步已经设置了 `LS_URL` 和 `LS_TOKEN`，本节可跳过。

```bash
export LS_URL="http://10.105.3.39"
read -s LS_TOKEN
```

说明：执行 `read -s LS_TOKEN` 后粘贴 Token 并回车，终端不会显示输入内容。

---

## 6. 下载并转换数据（目标检测）

```bash
python yolo_cli.py data convert-labelstudio \
  --input labelstudioexport/project-19-at-2026-05-08-03-46-e6cf01e6.json \
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

## 9. 正式训练（GPU）

cd /data/yolocli-temp-20260508113506/yolocli

划分

```bash
python yolo_cli.py data split \
  --images data/raw/images \
  --labels data/raw/labels \
  --output data/processed \
  --ratios 0.7:0.2:0.1 \
  --task detect \
  --seed 42
```

校验

```bash
python yolo_cli.py data verify \
  --path data/processed \
  --task detect
  --task detect
```

训练

```bash
python scripts/train/train_yolo.py \
  --model models/weights/yolov8m.pt \
  --data data/processed/dataset.yaml \
  --epochs 1 \
  --imgsz 320 \
  --batch 1 \
  --device 0 \
  --project results/training \
  --name rust_probe \
  --augmentation balanced
```

```bash
python scripts/train/train_yolo.py \
  --model models/weights/yolov8m.pt \
  --data data/processed/dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project results/training \
  --name rust_detect_train \
  --augmentation balanced
```

> 参数建议：
> - 显存紧张：减小 `--batch`（如 16 -> 8 -> 4）
> - 数据量小：可减少 `--epochs`（如 50）
> - 先快验：用 `yolo11n.pt`

---

## 10. 训练后验证

```bash
cd "/data/$RUN_DIR/yolocli"
BEST=$(ls -t results/training/*/weights/best.pt | head -1)
echo "$BEST"
python yolo_cli.py validate run "$BEST" \
  --data data/processed/dataset.yaml \
  --task detect \
  --device 0
```

---

## 11. 常见问题排查

### 11.1 `No module named ultralytics`

- 原因：未激活正确 conda 环境，或依赖未安装。
- 处理：

```bash
conda activate robot
python -m pip install -r requirements.txt
```

### 11.2 `cuda_ok=False`

- 原因：当前环境中的 `torch` 不是 CUDA 版本或驱动/版本不匹配。
- 处理：先确认 `nvidia-smi` 正常，再重装匹配 CUDA 的 PyTorch。

### 11.3 `project-19.json` 不存在

- 原因：API 导出失败，或导出文件未放到指定路径。
- 处理：先执行 API 导出并检查 `labelstudioexport/project-19.json`，不成功再走网页导出兜底。

### 11.4 401/403 鉴权失败

- 原因：Token 失效、URL 填写错误，或 Label Studio 侧资源不可访问。
- 处理：重新生成 Token，确认 `LS_URL` 可访问，并检查项目 19 的数据源可读。

### 11.5 OOM（显存不足）

- 原因：`batch/imgsz` 过大。
- 处理：减小 `--batch` 或 `--imgsz`，如 `--batch 8 --imgsz 512`。

---

## 12. 安全收尾建议

训练完成后清理敏感变量：

```bash
unset LS_TOKEN
```

并在 Label Studio 侧定期轮换 Token。
