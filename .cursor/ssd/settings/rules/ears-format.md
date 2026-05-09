# EARS格式指南

## 概述
EARS（Easy Approach to Requirements Syntax，需求语法简易方法）是规范驱动开发中验收标准的标准格式。

EARS模式描述需求的逻辑结构（条件 + 主体 + 响应），不绑定任何特定的自然语言。  
所有验收标准应使用为规范配置的目标语言编写（例如，`spec.json.language` / `zh`）。  
保持EARS触发关键词和固定短语为英文（`When`、`If`、`While`、`Where`、`The system shall`、`The [system] shall`），仅将变量部分（`[event]`、`[precondition]`、`[trigger]`、`[feature is included]`、`[response/action]`）本地化为目标语言。不要在触发器或固定英文短语内部插入目标语言文本。

## 主要EARS模式

### 1. 事件驱动需求
- **模式**: When [event], the [system] shall [response/action]
- **用例**: 对特定事件或触发器的响应
- **示例**: When user clicks checkout button, the Checkout Service shall validate cart contents

### 2. 状态驱动需求
- **模式**: While [precondition], the [system] shall [response/action]
- **用例**: 依赖于系统状态或前置条件的行为
- **示例**: While payment is processing, the Checkout Service shall display loading indicator

### 3. 不期望行为需求
- **模式**: If [trigger], the [system] shall [response/action]
- **用例**: 系统对错误、故障或不期望情况的响应
- **示例**: If invalid credit card number is entered, then the website shall display error message

### 4. 可选功能需求
- **模式**: Where [feature is included], the [system] shall [response/action]
- **用例**: 可选或条件功能的需求
- **示例**: Where the car has a sunroof, the car shall have a sunroof control panel

### 5. 普遍需求
- **模式**: The [system] shall [response/action]
- **用例**: 始终激活的需求和基本系统属性
- **示例**: The mobile phone shall have a mass of less than 100 grams

## 组合模式
- While [precondition], when [event], the [system] shall [response/action]
- When [event] and [additional condition], the [system] shall [response/action]

## 主体选择指南
- **软件项目**: 使用具体的系统/服务名称（例如，"Checkout Service"、"User Auth Module"）
- **流程/工作流**: 使用负责的团队/角色（例如，"Support Team"、"Review Process"）
- **非软件**: 使用适当的主体（例如，"Marketing Campaign"、"Documentation"）

## 质量标准
- 需求必须可测试、可验证，并描述单一行为。
- 使用客观语言："shall"表示强制性行为，"should"表示建议；避免模糊术语。
- 遵循EARS语法：[condition], the [system] shall [response/action]。
