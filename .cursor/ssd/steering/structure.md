# 项目结构

## 组织理念

**功能分层 + 命令驱动** - CLI 命令按功能分组，核心逻辑与 UI 表现分离。数据流遵循明确的目录约定。

## 目录设计模式

### CLI 命令模块
**路径**: `cli/commands/`  
**目的**: 每个文件对应一个命令组（model、data、train、detect 等）  
**示例**: `train.py` 包含 `train start`、`train resume` 等子命令

### 核心工具模块
**路径**: `cli/core/`  
**目的**: 共享的配置管理、工具函数、版本信息  
**示例**: `config.py` 处理 YAML 配置加载和合并

### UI 组件模块
**路径**: `cli/ui/`  
**目的**: 终端显示和用户交互组件  
**示例**: `display.py` 包含 Rich 表格和格式化输出

### 配置文件
**路径**: `config/`  
**目的**: 默认配置和预设配置  
**示例**: `profiles/small.yaml` 为小数据集优化的训练配置

### 数据目录
**路径**: `data/`  
**目的**: 数据处理的标准目录结构  
**约定**: 
- `data/raw/` - 原始图片和标签
- `data/processed/` - 划分后的训练数据
- `datasets/` - 管理多个数据集

### 模型目录
**路径**: `models/weights/`  
**目的**: 预训练模型和下载的权重文件  

### 结果目录
**路径**: `results/`  
**目的**: 训练结果、验证结果、预测输出  
**约定**: 按任务类型和时间戳组织

## 命名规范

- **文件**: snake_case（如 `label_studio.py`）
- **类**: PascalCase（如 `ConfigManager`）
- **函数**: snake_case（如 `split_dataset`）
- **命令**: kebab-case（如 `interactive-mode`）
- **配置文件**: lowercase.yaml（如 `dataset.yaml`）

## 导入组织

```python
# 标准库
import os
from pathlib import Path

# 第三方库
import typer
from rich.console import Console
from ultralytics import YOLO

# 本地模块
from cli.core.config import ConfigManager
from cli.ui.display import print_info
```

**导入原则**:
- 标准库 → 第三方库 → 本地模块
- 避免循环导入，使用延迟导入处理大型依赖

## 代码组织原则

1. **命令与逻辑分离** - 命令文件只负责参数解析和调用核心逻辑
2. **配置驱动** - 训练参数、数据路径等通过配置文件管理
3. **一致的输出风格** - 所有命令使用 `cli.ui.display` 模块输出
4. **数据目录约定** - 遵循 `data/raw/` → `data/processed/` 的处理流程
5. **向后兼容** - 保留旧命令别名（如 `detect` 作为 `predict` 的别名）

---
_本规范聚焦设计模式而非具体文件树结构。遵循本模式新增文件时，无需更新此文档_
