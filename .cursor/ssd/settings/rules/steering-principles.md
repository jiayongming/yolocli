# Steering原则

Steering文件是**项目记忆**，而非详尽规范。

---

## 内容粒度

### 黄金法则
> "如果新代码遵循现有模式，steering不应该需要更新。"

### ✅ 记录
- 组织模式（功能优先、分层）
- 命名约定（PascalCase规则）
- 导入策略（绝对vs相对）
- 架构决策（状态管理）
- 技术标准（关键框架）

### ❌ 避免
- 完整的文件列表
- 每个组件描述
- 所有依赖项
- 实现细节
- 代理特定工具目录（例如 `.cursor/`、`.gemini/`、`.claude/`）
- `.cursor/ssd/` 元数据目录（设置、自动化）的详细文档

### 示例比较

**不好**（类似规范）：
```markdown
- /components/Button.tsx - Primary button with variants
- /components/Input.tsx - Text input with validation
- /components/Modal.tsx - Modal dialog
... (50+ files)
```

**好**（项目记忆）：
```markdown
## UI Components (`/components/ui/`)
Reusable, design-system aligned primitives
- Named by function (Button, Input, Modal)
- Export component + TypeScript interface
- No business logic
```

---

## 安全

永远不要包含：
- API密钥、密码、凭据
- 数据库URL、内部IP
- 秘密或敏感数据

---

## 质量标准

- **单一领域**：每个文件一个主题
- **具体示例**：用代码展示模式
- **解释理由**：为什么做出这些决策
- **可维护大小**：通常100-200行

---

## 保留（更新时）

- 保留用户章节和自定义示例
- 默认添加式（添加，不替换）
- 添加 `updated_at` 时间戳
- 记录更改原因

---

## 注意事项

- 模板是起点，根据需要自定义
- 遵循与核心steering相同的粒度原则
- 所有steering文件作为项目记忆加载
- 对 `.cursor/ssd/specs/` 和 `.cursor/ssd/steering/` 的轻量引用是可以接受的；避免其他 `.cursor/ssd/` 目录
- 自定义文件与核心文件同等重要

---

## 文件特定焦点

- **product.md**：目的、价值、业务上下文（非详尽功能）
- **tech.md**：关键框架、标准、约定（非所有依赖项）
- **structure.md**：组织模式、命名规则（非目录树）
- **自定义文件**：专门模式（API、测试、安全等）
