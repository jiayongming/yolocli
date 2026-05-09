# 设计文档模板

---
**目的**: 提供足够的细节，以确保不同实施者之间的一致性，防止解释偏差。

**方法**:
- 包含直接影响实施决策的基本章节
- 除非对防止实施错误至关重要，否则省略可选章节
- 使详细程度与功能复杂度匹配
- 使用图表和表格而非冗长的散文

**警告**: 接近1000行表示功能复杂度过高，可能需要简化设计。
---

> 章节可以重新排序（例如，提前展示需求可追溯性或将数据模型靠近架构），以提高清晰度。在每个章节内，保持流程 **摘要 → 范围 → 决策 → 影响/风险**，以便评审者能够一致地扫描。

## 概述 
最多2-3段
**目的**: 此功能为[目标用户]提供[特定价值]。
**用户**: [目标用户组]将利用此功能进行[特定工作流]。
**影响**（如适用）: 通过[特定修改]改变当前[系统状态]。


### 目标
- 主要目标1
- 主要目标2  
- 成功标准

### 非目标
- 明确排除的功能
- 当前范围之外的未来考虑
- 推迟的集成点

## 架构

> 仅在背景中引用 `research.md` 中的详细调研说明；通过在此处捕获所有决策和协议，保持design.md对评审者自包含。
> 在文本中捕获关键决策，让图表承载结构细节——避免在散文中重复相同信息。

### 现有架构分析（如适用）
修改现有系统时：
- 当前架构模式和约束
- 要尊重的现有领域边界
- 必须维护的集成点
- 解决或规避的技术债务

### 架构模式和边界图
**推荐**: 包含显示所选架构模式和系统边界的Mermaid图表（复杂功能必需，简单添加可选）

**架构集成**:
- 选定模式: [名称和简短理由]
- 领域/功能边界: [如何分离职责以避免冲突]
- 保留的现有模式: [列出关键模式]
- 新组件理由: [为什么需要每个]
- Steering合规性: [维护的原则]

### 技术栈

| Layer                    | Choice / Version | Role in Feature | Notes |
|--------------------------|------------------|-----------------|-------|
| Frontend / CLI           |                  |                 |       |
| Backend / Services       |                  |                 |       |
| Data / Storage           |                  |                 |       |
| Messaging / Events       |                  |                 |       |
| Infrastructure / Runtime |                  |                 |       |

> 在此处保持理由简洁，当需要更多深度时（权衡、基准测试），添加简短摘要加上指向支持参考章节和 `research.md` 的指针，以获取原始调查说明。

## 系统流程

仅提供解释非平凡流程所需的图表。使用纯Mermaid语法。常见模式：
- Sequence（多方交互）
- Process / state（分支逻辑或生命周期）
- Data / event flow（流水线、异步消息）

对于简单的CRUD更改，完全跳过此章节。
> 在图表后简要描述流程级决策（例如，门控条件、重试），而不是重述每个步骤。

## 需求可追溯性

对于需求跨越多个域的复杂或合规敏感功能，使用此章节。简单的1:1映射可以依赖组件摘要表。

将每个需求ID（例如，`2.1`）映射到实现它的设计元素。

| Requirement | Summary | Components | Interfaces | Flows |
|-------------|---------|------------|------------|-------|
| 1.1         |         |            |            |       |
| 1.2         |         |            |            |       |

> 仅当单个组件满足单个需求且无横切关注点时，才省略此章节。

## 组件和接口

在深入每个组件细节之前提供快速参考。

- 摘要可以是表格或紧凑列表。示例表格：
  | Component | Domain/Layer | Intent | Req Coverage | Key Dependencies (P0/P1) | Contracts |
  |-----------|--------------|--------|--------------|--------------------------|-----------|
  | ExampleComponent | UI | Displays XYZ | 1, 2 | GameProvider (P0), MapPanel (P1) | Service, State |
- 仅引入新边界的组件（例如，逻辑钩子、外部集成、持久化）需要完整的详细块。简单的展示组件可以依赖摘要行加上简短的实施说明。

按领域或架构层对详细块进行分组。对于每个详细组件，将需求ID列为 `2.1, 2.3`（省略"Requirement"）。当多个UI组件共享相同协议时，引用基础接口/属性定义而不是复制代码块。

### [Domain / Layer]

#### [Component Name]

| Field             | Detail                                   |
|-------------------|------------------------------------------|
| Intent            | 1-line description of the responsibility |
| Requirements      | 2.1, 2.3                                 |
| Owner / Reviewers | (optional)                               |

**Responsibilities & Constraints**
- Primary responsibility
- Domain boundary and transaction scope
- Data ownership / invariants

**Dependencies**
- Inbound: Component/service name — purpose (Criticality)
- Outbound: Component/service name — purpose (Criticality)
- External: Service/library — purpose (Criticality)

在此处总结外部依赖调研；更深入的调查（API签名、速率限制、迁移说明）位于 `research.md`。

**Contracts**: Service [ ] / API [ ] / Event [ ] / Batch [ ] / State [ ]  ← 仅勾选适用的。

##### Service Interface
```typescript
interface [ComponentName]Service {
  methodName(input: InputType): Result<OutputType, ErrorType>;
}
```
- Preconditions:
- Postconditions:
- Invariants:

##### API Contract
| Method | Endpoint      | Request       | Response | Errors        |
|--------|---------------|---------------|----------|---------------|
| POST   | /api/resource | CreateRequest | Resource | 400, 409, 500 |

##### Event Contract
- Published events:  
- Subscribed events:  
- Ordering / delivery guarantees:

##### Batch / Job Contract
- Trigger:  
- Input / validation:  
- Output / destination:  
- Idempotency & recovery:

##### State Management
- State model:  
- Persistence & consistency:  
- Concurrency strategy:

**Implementation Notes**
- Integration: 
- Validation: 
- Risks:

## 数据模型

专注于随此功能更改的数据景观部分。

### 领域模型
- Aggregates and transactional boundaries
- Entities, value objects, domain events
- Business rules & invariants
- Optional Mermaid diagram for complex relationships

### 逻辑数据模型

**结构定义**:
- Entity relationships and cardinality
- Attributes and their types
- Natural keys and identifiers
- Referential integrity rules

**一致性和完整性**:
- Transaction boundaries
- Cascading rules
- Temporal aspects (versioning, audit)

### 物理数据模型
**何时包含**: 当实施需要特定存储设计决策时

**对于关系数据库**:
- Table definitions with data types
- Primary/foreign keys and constraints
- Indexes and performance optimizations
- Partitioning strategy for scale

**对于文档存储**:
- Collection structures
- Embedding vs referencing decisions
- Sharding key design
- Index definitions

**对于事件存储**:
- Event schema definitions
- Stream aggregation strategies
- Snapshot policies
- Projection definitions

**对于键值/宽列存储**:
- Key design patterns
- Column families or value structures
- TTL and compaction strategies

### 数据协议和集成

**API数据传输**
- Request/response schemas
- Validation rules
- Serialization format (JSON, Protobuf, etc.)

**事件模式**
- Published event structures
- Schema versioning strategy
- Backward/forward compatibility rules

**跨服务数据管理**
- Distributed transaction patterns (Saga, 2PC)
- Data synchronization strategies
- Eventual consistency handling

跳过和此功能无关的子章节。

## 错误处理

### 错误策略
每种错误类型的具体错误处理模式和恢复机制。

### 错误类别和响应
**User Errors** (4xx): Invalid input → field-level validation; Unauthorized → auth guidance; Not found → navigation help
**System Errors** (5xx): Infrastructure failures → graceful degradation; Timeouts → circuit breakers; Exhaustion → rate limiting  
**Business Logic Errors** (422): Rule violations → condition explanations; State conflicts → transition guidance

**流程可视化**（当存在复杂业务逻辑时）:
仅当存在复杂业务工作流的复杂错误场景时，包含Mermaid流程图。

### 监控
错误跟踪、日志记录和健康监控实施。

## 测试策略

### 默认章节（调整名称/章节以适应领域）
- Unit Tests: 3–5 items from core functions/modules (e.g., auth methods, subscription logic)
- Integration Tests: 3–5 cross-component flows (e.g., webhook handling, notifications)
- E2E/UI Tests (if applicable): 3–5 critical user paths (e.g., forms, dashboards)
- Performance/Load (if applicable): 3–4 items (e.g., concurrency, high-volume ops)

## 可选章节（在相关时包含）

### 安全考虑
_对于处理认证、敏感数据、外部集成或用户权限的功能，使用此章节。仅捕获此功能独有的决策；将基线控制推迟到steering文档。_
- Threat modeling, security controls, compliance requirements
- Authentication and authorization patterns
- Data protection and privacy considerations

### 性能和可扩展性
_当存在性能目标、高负载或扩展关注点时，使用此章节。仅记录特定功能的目标或权衡，并依赖steering文档获取一般实践。_
- Target metrics and measurement strategies
- Scaling approaches (horizontal/vertical)
- Caching strategies and optimization techniques

### 迁移策略
当需要模式/数据移动时，包含显示迁移阶段的Mermaid流程图。
- Phase breakdown, rollback triggers, validation checkpoints

## 支持参考（可选）
- 仅当将信息保留在正文中会损害可读性时创建此章节（例如，非常长的TypeScript定义、供应商选项矩阵、详尽的模式表）。在主要章节中保留决策上下文，以便设计保持自包含。
- 从主文本链接到支持参考，而不是内联大型代码片段。
- 背景调研说明和比较继续位于 `research.md` 中，但它们的结论必须在主设计中总结。
