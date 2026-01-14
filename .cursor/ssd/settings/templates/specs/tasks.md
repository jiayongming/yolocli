# 实施计划

## 任务格式模板

使用适合工作分解的模式：

### 仅主要任务
- [ ] {{NUMBER}}. {{TASK_DESCRIPTION}}{{PARALLEL_MARK}}
  - {{DETAIL_ITEM_1}} *（仅在需要时包含详细信息。如果任务独立存在，则省略项目符号。）*
  - _Requirements: {{REQUIREMENT_IDS}}_

### 主要 + 子任务结构
- [ ] {{MAJOR_NUMBER}}. {{MAJOR_TASK_SUMMARY}}
- [ ] {{MAJOR_NUMBER}}.{{SUB_NUMBER}} {{SUB_TASK_DESCRIPTION}}{{SUB_PARALLEL_MARK}}
  - {{DETAIL_ITEM_1}}
  - {{DETAIL_ITEM_2}}
  - _Requirements: {{REQUIREMENT_IDS}}_ *（仅ID；不要添加描述或括号。）*

> **并行标记**：仅将 ` (P)` 附加到可以并行执行的任务。在 `--sequential` 模式下运行时省略标记。
>
> **可选测试覆盖**：当子任务是与验收标准相关的可延迟测试工作时，将复选框标记为 `- [ ]*`，并在详细要点中解释引用的需求。
