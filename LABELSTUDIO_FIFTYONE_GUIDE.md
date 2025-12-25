# Label Studio + FiftyOne 集成使用指南

## 功能概述

本次更新为 yolocli 添加了完整的 Label Studio 项目管理和 FiftyOne 数据集可视化功能，实现了从标注平台到数据集管理的完整工作流。

## 安装依赖

```bash
pip install fiftyone>=0.23.0
```

## 快速开始

### 1. 启动交互模式

```bash
python yolo_cli.py interactive
# 或
yolo-cli interactive
```

### 2. 完整工作流示例

#### 步骤1: 配置 Label Studio (可选)

在 `config/default.yaml` 中预配置（推荐）：

```yaml
labelstudio:
  url: http://localhost:8080
  token: "your_token_here"
  auto_refresh: true
  token_type: auto
```

或在交互模式下配置：
- 主菜单选择 `labelstudio - Label Studio管理`
- 选择 `config - 配置Label Studio连接`
- 输入 URL 和 Token
- 系统会测试连接并询问是否保存配置

#### 步骤2: 获取项目并下载数据

1. 选择 `list - 列出所有项目` 查看可用项目
2. 选择 `fetch - 获取项目数据`
3. 从列表中选择要下载的项目
4. 选择任务类型（检测/分割/分类）
5. 设置输出目录
6. 确认后系统会自动：
   - 导出项目标注数据
   - 下载所有图片
   - 转换为 YOLO 格式
   - 询问是否用 FiftyOne 查看

#### 步骤3: 使用 FiftyOne 可视化

**方式1：从 Label Studio 直接跳转**
- 在下载完成后选择 "是" 直接启动 FiftyOne

**方式2：独立使用 FiftyOne**
- 主菜单选择 `fiftyone - FiftyOne可视化`
- 选择操作：
  - `load` - 加载新数据集（指定 dataset.yaml）
  - `launch` - 启动可视化（选择已有数据集）
  - `list` - 查看所有数据集
  - `info` - 查看数据集详细信息
  - `delete` - 删除数据集

## 功能详解

### Label Studio 集成

#### 支持的功能
- ✅ 列出所有项目
- ✅ 获取项目详情
- ✅ 导出项目标注
- ✅ 批量下载图片（支持断点续传）
- ✅ 自动转换为 YOLO 格式
- ✅ 支持三种 Token 类型（Legacy/Bearer/Refresh）

#### Token 类型说明
系统自动识别并处理三种 Token：
- **Legacy Token**: 传统格式，Header: `Token xxx`
- **Access Token**: JWT 格式，Header: `Bearer xxx`
- **Refresh Token**: 自动交换为 Access Token

#### 配置选项
- **配置文件优先**: 从 `config/default.yaml` 读取配置
- **交互输入**: 未配置时在交互模式下输入
- **配置保存**: 可选择将输入的配置保存到文件

### FiftyOne 可视化

#### 支持的功能
- ✅ 加载 YOLO 数据集
- ✅ 启动 Web 可视化界面
- ✅ 查看数据集统计信息
- ✅ 管理多个数据集
- ✅ 删除数据集

#### 数据集信息
FiftyOne 会显示：
- 样本总数和划分（train/val/test）
- 类别列表和分布
- 检测框可视化
- 负样本（无标注图片）

#### 可视化特性
- 自动打开浏览器
- 交互式筛选和搜索
- 标注查看和分析
- 数据质量检查

## 典型使用场景

### 场景1: 从零开始的完整流程

```
1. yolo-cli interactive
2. 选择 labelstudio
3. 配置连接 → 列出项目 → 获取项目数据
4. 选择"是"启动 FiftyOne 查看
5. 在浏览器中检查数据质量
6. 返回主菜单继续训练
```

### 场景2: 查看已有数据集

```
1. yolo-cli interactive
2. 选择 fiftyone
3. load → 输入 dataset.yaml 路径
4. 系统自动加载并启动可视化
```

### 场景3: 管理多个项目

```
1. 从 Label Studio 下载多个项目
2. 每个项目自动创建独立的 FiftyOne 数据集
3. 使用 list 查看所有数据集
4. 使用 info 比较不同数据集
5. 使用 delete 清理不需要的数据集
```

## 配置文件示例

完整的 `config/default.yaml` 配置：

```yaml
# Label Studio配置
labelstudio:
  url: http://localhost:8080
  token: ""  # 留空时在交互模式输入
  auto_refresh: true
  token_type: auto

# FiftyOne配置
fiftyone:
  default_port: 5151
  auto_launch_browser: true
```

## 常见问题

### Q1: FiftyOne 未安装
**错误**: "FiftyOne未安装。请运行: pip install fiftyone"
**解决**: `pip install fiftyone>=0.23.0`

### Q2: Label Studio 连接失败
**可能原因**:
- URL 不正确（检查端口号）
- Token 无效或过期
- 网络连接问题

**解决步骤**:
1. 使用 `config` 命令重新配置
2. 系统会自动测试连接
3. 根据错误提示调整配置

### Q3: 图片下载失败
**原因**: 可能是网络问题或权限问题
**优势**: 系统支持断点续传，重新运行即可继续下载

### Q4: 数据集加载错误
**检查清单**:
- dataset.yaml 文件是否存在
- 类别信息是否正确
- 图片和标签路径是否正确

## 技术细节

### 项目结构

```
cli/
├── converters/
│   └── labelstudio.py          # Label Studio API 客户端
├── integrations/
│   ├── __init__.py
│   └── fiftyone_manager.py     # FiftyOne 管理器
├── commands/
│   └── interactive.py          # 交互模式（新增功能）
└── ui/
    └── prompts.py              # UI 提示（新增选项）

config/
└── default.yaml                # 配置文件（新增配置段）
```

### API 调用流程

```
Label Studio → 项目列表
             ↓
         选择项目
             ↓
       导出标注 (JSON)
             ↓
       下载图片 (多线程)
             ↓
     转换为 YOLO 格式
             ↓
       FiftyOne 加载
             ↓
       Web 可视化
```

### 数据转换

**Label Studio → YOLO**:
- 坐标转换: 百分比 → 归一化
- 格式转换: 左上角+宽高 → 中心点+宽高

**YOLO → FiftyOne**:
- 坐标转换: 归一化 → FiftyOne 格式
- 标注组织: 按 split 分组
- 类别映射: 自动生成类别信息

## 更新日志

### v1.0.0 (2025-12-25)

**新增功能**:
- ✅ Label Studio 项目管理
- ✅ FiftyOne 数据集可视化
- ✅ 完整的交互式工作流
- ✅ 配置文件支持

**改进**:
- ✅ 主菜单新增两个选项
- ✅ 支持三种 Token 类型
- ✅ 断点续传支持
- ✅ 自动错误处理

## 反馈与支持

如有问题或建议，请通过以下方式反馈：
- 项目 Issue
- 开发者邮箱
- 技术文档

---

**享受全新的数据管理体验！** 🎉

