# Label Studio 子集说明

本目录由 `scripts/utils/select_binary_subset.py` 生成。

## 文件

- `project-19-severe450-non50.json`：从原始导出中挑选的 500 条任务（严重锈蚀 450 + 无锈蚀 50）
- `project-19-severe450-non50.summary.json`：挑选规则、候选数量、最终数量、任务 ID 与文件名清单

## 规则

- 严重锈蚀候选：任务包含 `severe-corrosion` 或 `High-corrosion`
- 严重程度排序：按严重锈蚀标注面积降序优先（再按严重标注数量、总标注面积、任务 ID）
- 无锈蚀候选：任务标签仅属于 `clean/non-corrosion/0`
- 无锈蚀抽样：固定随机种子 `42`

