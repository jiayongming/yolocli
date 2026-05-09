# API标准

[目的：一致的API规范，用于命名、结构、认证、版本控制和错误]

## 理念
- 偏好可预测的、面向资源的设计
- 在协议中明确；最小化破坏性更改
- 默认安全（先认证，最小权限）

## 端点模式
```
/{version}/{resource}[/{id}][/{sub-resource}]
```
示例:
- `/api/v1/users`
- `/api/v1/users/:id`
- `/api/v1/users/:id/posts`

HTTP动词:
- GET（读取，安全，幂等）
- POST（创建）
- PUT/PATCH（更新）
- DELETE（删除，幂等）

## 请求/响应

请求（典型）:
```json
{ "data": { ... }, "metadata": { "requestId": "..." } }
```

成功:
```json
{ "data": { ... }, "meta": { "timestamp": "...", "version": "..." } }
```

错误:
```json
{ "error": { "code": "ERROR_CODE", "message": "...", "field": "optional" } }
```
（参见error-handling了解规则。）

## 状态码（模式）
- 2xx: 成功（200读取，201创建，204删除）
- 4xx: 客户端问题（400验证，401/403认证，404缺失）
- 5xx: 服务器问题（500通用，503不可用）
选择最能反映结果的状态。

## 认证
- 凭据在标准位置
```
Authorization: Bearer {token}
```
- 在业务逻辑之前拒绝未认证请求

## 版本控制
- 通过URL/标头/媒体类型进行版本控制
- 破坏性更改 → 新版本
- 非破坏性 → 相同版本
- 提供弃用窗口和通信

## 分页/过滤（如适用）
- 分页: `page`、`pageSize` 或基于游标
- 过滤: 显式查询参数
- 排序: `sort=field:asc|desc`
在 `meta` 中返回分页元数据。

---
_关注模式和决策，而非端点目录。_
