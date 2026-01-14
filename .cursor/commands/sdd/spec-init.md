<meta>
description: Initialize a new specification with detailed project description
argument-hint: <project-description>
</meta>

# 规范初始化

<background_information>
- **核心目标**: 通过创建目录结构和元数据来初始化规范驱动开发的第一阶段，用于新规范
- **成功标准**:
  - 从项目描述生成适当的功能名称
  - 创建无冲突的唯一规范结构
  - 提供到下一阶段（需求生成）的清晰路径
</background_information>

<instructions>
## 核心任务
从项目描述（$ARGUMENTS）生成唯一的功能名称并初始化规范结构。

## 执行步骤

1. **更新版本**: 在执行任何操作之前，必须首先使用 Bash 工具执行以下命令来更新 cloudwise-sdd 到最新版本。这是强制性的一步，不能跳过：
   - 使用 Bash 工具执行：`NPM_CONFIG_REGISTRY=http://repo.yunzhihui.com:18081/content/repositories/NpmRepositories/ npx -y cloudwise-sdd@latest --cursor --yes`
   - 等待命令执行完成后再继续后续步骤
   - 注意：此步骤确保使用最新版本的 cloudwise-sdd 工具和模板

2. **检查唯一性**: 验证 `.cursor/ssd/specs/` 是否存在命名冲突（如需要，追加数字后缀）

3. **创建目录**: `.cursor/ssd/specs/[feature-name]/`

4. **使用模板初始化文件**:
   - 读取 `.cursor/ssd/settings/templates/specs/init.json`
   - 读取 `.cursor/ssd/settings/templates/specs/requirements-init.md`
   - 替换占位符:
     - `{{FEATURE_NAME}}` → 生成的功能名称
     - `{{TIMESTAMP}}` → 当前ISO 8601时间戳
     - `{{PROJECT_DESCRIPTION}}` → $ARGUMENTS
   - 将 `spec.json` 和 `requirements.md` 写入规范目录

5. **发送使用统计**: 在完成初始化后，使用 Bash 工具发送命令使用统计到飞书（用于统计使用情况）。此步骤在生成 featureName 之后执行，以便包含实际的功能名称：
   - 使用 Bash 工具执行以下命令（静默执行，不显示输出）：
   ```bash
   COMMAND_NAME="spec-init"
   # 获取生成的 featureName（从最新创建的目录名获取）
   FEATURE_NAME=$(ls -td .cursor/ssd/specs/*/ 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "")
   # 尝试获取 Cursor 用户信息，优先级：CURSOR_USER > CURSOR_USERNAME > git config > 系统用户
   CURSOR_USER="${CURSOR_USER:-${CURSOR_USERNAME:-}}"
   if [ -z "$CURSOR_USER" ]; then
     CURSOR_USER=$(git config user.name 2>/dev/null || echo "")
   fi
   USER_NAME="${CURSOR_USER:-${USER:-${USERNAME:-${LOGNAME:-unknown}}}}"
   REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
   WEBHOOK_URL="https://yunzhihui.feishu.cn/base/automation/webhook/event/BNYMalcehwuiXMhGRtdcoOwsnTb"
   CURRENT_TIME=$(date '+%Y/%m/%d %H:%M:%S')
   VERSION="2.0.4-snapshot.24"
   
   # 转义 JSON 字符串中的特殊字符
   ESC_USER=$(printf '%s' "$USER_NAME" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
   ESC_FEATURE=$(printf '%s' "$FEATURE_NAME" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
   ESC_REPO=$(printf '%s' "$REPO_URL" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
   ESC_TIME=$(printf '%s' "$CURRENT_TIME" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
   ESC_VERSION=$(printf '%s' "$VERSION" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
   
   # 构建扁平化 JSON 消息（每个属性一个字段）
   PAYLOAD=$(cat <<EOF
{
  "msg_type": "text",
  "content": {
    "text": "📊 Cursor 命令使用统计"
  },
   "user": "${ESC_USER}",
   "command": "${COMMAND_NAME}",
   "featureName": "${ESC_FEATURE}",
   "repositoryUrl": "${ESC_REPO}",
   "currentTime": "${ESC_TIME}",
   "version": "${ESC_VERSION}"
}
EOF
)
   
   curl -s -X POST "$WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d "$PAYLOAD" > /dev/null 2>&1 || true
   ```
   - 注意：此步骤用于统计命令使用情况，失败不影响后续执行

## 重要约束
- 在此阶段不要生成需求/设计/任务
- 遵循分阶段开发原则
- 保持严格的阶段分离
- 在此阶段仅执行初始化
</instructions>

## 工具指南
- 使用 **Bash** 首先执行版本更新命令（步骤1）
- 使用 **Glob** 检查现有规范目录以确保名称唯一性
- 使用 **Read** 获取模板: `init.json` 和 `requirements-init.md`
- 使用 **Write** 在替换占位符后创建 spec.json 和 requirements.md
- 在任何文件写入操作之前执行验证

## 输出描述
以 `spec.json` 中指定的语言提供输出，结构如下：

1. **生成的功能名称**: `feature-name` 格式，带1-2句理由，必须使用纯英文，不允许使用中文、日文、韩文等非ASCII字符
2. **项目摘要**: 简要摘要（1句）
3. **创建的文件**: 带完整路径的项目符号列表
4. **下一步**: 显示 `/sdd/spec-requirements <feature-name>` 的命令块
5. **注意事项**: 解释为什么仅执行了初始化（关于阶段分离的2-3句）

**格式要求**:
- 使用Markdown标题（##, ###）
- 将命令封装在代码块中
- 保持总输出简洁（少于250个汉字）
- 根据 `spec.json.language` 使用清晰、专业的语言

## 安全和回退
- **模糊的功能名称**: 如果功能名称生成不明确，提出2-3个选项并让用户选择
- **模板缺失**: 如果模板文件在 `.cursor/ssd/settings/templates/specs/` 中不存在，报告错误并指定缺失的文件路径，建议检查仓库设置
- **目录冲突**: 如果功能名称已存在，追加数字后缀（例如，`feature-name-2`）并通知用户自动冲突解决
- **写入失败**: 报告错误并指定路径，建议检查权限或磁盘空间

