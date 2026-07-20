## Context

当前 data 架构已经逐步收敛为：

- `reader.iterrows()` 产出单条 row
- `dataset` 逐条执行 preprocessing
- `collate_*` 负责把多条 row 组装成 batch

在这个前提下，preprocessing 层理论上不应再承担 batch rows 兼容职责。但当前 `preprocessing.py` 中仍存在历史遗留：

- 参数名仍沿用 `batch_or_row`
- 某些函数仍分支处理 `list[dict]`
- 文档描述仍暗示 batch / row 双模式

这些残留会混淆职责边界，并使后续继续纯化 preprocessing 时产生阻力。

## Goals / Non-Goals

**Goals:**
- 让 preprocessing 模块的 API 与实现语义统一为 row-only
- 删除针对 batch rows 的旧兼容逻辑
- 保留对 row 内 sequence-like 字段值的必要处理
- 清晰表达：batch 只在 collate 层存在

**Non-Goals:**
- 不重构 collate 层的 batch 处理逻辑
- 不改变 row 内字段值的具体数据形态支持（如 `list` / `np.ndarray` / `torch.Tensor`）
- 不在本次改变业务语义或模型输入输出结构

## Decisions

### D1: preprocessing 统一为 row-only API
- **选择**：preprocessing 函数接口和实现统一使用 `row` 语义，不再保留 `batch_or_row` 折中命名
- **理由**：当前 dataset 主链路已经是 row-only，这一层不应继续保留过时双语义

### D2: 删除 `list[dict]` batch rows 兼容分支
- **选择**：显式删除针对 batch rows 的旧分支逻辑
- **理由**：batch 兼容属于 collate 层，不应继续留在 preprocessing

### D3: 保留 row 内字段值的 sequence-like 处理
- **选择**：继续支持字段值为 `list` / `np.ndarray` / `torch.Tensor` 的处理逻辑
- **理由**：这属于单条 row 内部字段形态，不等于 batch rows 兼容，不能误删

### D4: 同步修正文档与注释
- **选择**：对参数名、注释、docstring 一并做 row-only 收敛
- **理由**：如果只改实现不改描述，维护者仍会被旧语义误导

## Risks / Trade-offs

- **[风险] 误把 row 内 list/ndarray 值处理当成 batch 分支删除**  
  **缓解**：明确区分“`list[dict]` batch rows”与“单条 row 内字段值为 sequence-like”的两类场景。

- **[风险] 某些历史实验可能还隐式依赖 preprocessing 的 batch 兼容**  
  **缓解**：本次先通过 grep 和最小 instantiate/调用验证确认当前主链路确实已 row-only。

## Migration Plan

1. 盘点 `preprocessing.py` 中仍残留的 batch rows 分支
2. 删除 `list[dict]` 旧分支，统一函数参数与注释命名
3. 保留 row 内字段值的 sequence-like 处理逻辑
4. 做最小 grep / import / 调用验证，确认 preprocessing 已 row-only
