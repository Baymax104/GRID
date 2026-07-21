# row-only-preprocessing-contract Specification

## Purpose
TBD - created by archiving. Update Purpose after archive.

## Requirements
### Requirement: preprocessing SHALL operate on single rows only
在新的 data 架构中，preprocessing SHALL 只处理单条 row，不再兼容 `list[dict]` 形式的 batch rows。

#### Scenario: dataset 向 preprocessing 传入单条 row
- **WHEN** dataset 执行 preprocessing chain
- **THEN** 每个 preprocessing callable MUST 接收单条 row 作为输入

#### Scenario: preprocessing 不再兼容 batch rows
- **WHEN** 维护者检查 preprocessing 的实现
- **THEN** 这些函数 MUST NOT 再包含针对 `list[dict]` batch rows 的兼容分支

#### Scenario: row 内 sequence-like 字段值仍受支持
- **WHEN** 单条 row 的某个字段值本身是 `list`、`np.ndarray` 或 `torch.Tensor`
- **THEN** preprocessing MAY 继续处理这些字段值
- **THEN** 这种处理 MUST NOT 被视为 batch rows 兼容语义
