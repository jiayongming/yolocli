# AI-DLC和规范驱动开发

**版本**: 2.0.4-snapshot.24

基于AI-DLC（AI开发生命周期）的SDD风格规范驱动开发实施

## 项目上下文

### 路径

- Steering: `.cursor/ssd/steering/`
- Specs: `.cursor/ssd/specs/`

### Steering vs 规范

**Steering** (`.cursor/ssd/steering/`) - 使用项目范围的规则和上下文指导AI
**Specs** (`.cursor/ssd/specs/`) - 为单个功能正式化开发流程

### 活动规范

- 检查 `.cursor/ssd/specs/` 以查找活动规范
- 使用 `/sdd/spec-status <feature-name>` 检查进度

## 开发指南

- Think in English, generate responses in Simplified Chinese. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## 最小工作流

- 阶段0（可选）:
    - `/sdd/steering`
    - `/sdd/steering-custom`
- 阶段1（规范）:
    - `/sdd/spec-init "description"`
    - `/sdd/spec-requirements <feature>`
    - `/sdd/spec-design <feature>`
    - `/sdd/spec-tasks <feature>`
- 阶段2（实施）:
    - `/sdd/spec-impl <feature> [tasks]`
- 进度检查:
    - `/sdd/spec-status <feature>` （随时使用）

## 开发规则

- 3阶段批准工作流：需求 → 设计 → 任务 → 实施
- 保持steering最新并使用 `/sdd/spec-status` 验证对齐
- 精确遵循用户的指示，并在该范围内自主行动：收集必要的上下文并在此运行中端到端完成请求的工作，仅在缺少基本信息或指示严重模糊时提出问题。

## Steering配置

- 将整个 `.cursor/ssd/steering/` 作为项目记忆加载
- 默认文件: `product.md`、`tech.md`、`structure.md`
- 支持自定义文件（通过 `/sdd/steering-custom` 管理）
