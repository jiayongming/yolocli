# 技术栈

## 架构

**模块化 CLI 架构** - 采用命令分组模式，通过 Typer 框架构建层次化的命令结构。核心逻辑与 UI 表现分离，便于维护和扩展。

## 核心技术

- **语言**: Python 3.8+
- **框架**: Typer (CLI)、Ultralytics (YOLO)
- **运行时**: Python 解释器

## 关键库

| 库 | 用途 | 开发模式影响 |
|---|---|---|
| `ultralytics` | YOLO 模型训练/推理 | 所有 ML 操作的核心依赖 |
| `typer` | CLI 框架 | 命令定义使用装饰器模式 |
| `rich` | 终端美化输出 | 使用 Console 和 Table 组件 |
| `questionary` | 交互式提示 | 用于交互模式的用户输入 |
| `pyyaml` | 配置文件解析 | 数据集配置和训练参数 |
| `fiftyone` | 数据集可视化 | 可选，用于数据分析 |

## 开发标准

### 类型安全
- 使用 Python 类型注解（Type Hints）
- Typer 参数通过类型注解自动验证

### 代码质量
- 模块化设计，单一职责原则
- 中文用户友好的错误提示和帮助信息
- Rich 库统一输出风格

### 测试
- 命令行参数验证由 Typer 自动处理
- 数据验证命令 (`data verify`) 用于数据集校验

## 开发环境

### 必需工具
- Python 3.8+
- pip 包管理器
- CUDA（可选，GPU 加速）

### 常用命令
```bash
# 安装依赖
pip install -r requirements.txt

# 运行 CLI
python yolo_cli.py --help

# 交互式模式
python yolo_cli.py interactive-mode

# 一键训练
python yolo_cli.py quick train --images data/raw/images --labels data/raw/labels
```

## 关键技术决策

1. **Typer 而非 Click/argparse** - 现代化的 CLI 框架，自动生成帮助文档，类型安全
2. **Ultralytics 统一接口** - 同时支持 YOLOv8 和 YOLO11，API 兼容
3. **YAML 配置系统** - 分层配置（默认 → 预设 → 命令行），灵活可扩展
4. **Rich 输出美化** - 统一的终端 UI 风格，表格、进度条、颜色高亮
5. **设备自动检测** - 优先级: MPS → CUDA → CPU，支持环境变量覆盖

---
_记录标准和模式，而非每个依赖项_
