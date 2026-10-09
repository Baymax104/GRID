## ADDED Requirements

### Requirement: Batch prefix constraints without changing generation semantics
LETTER SHALL 每个解码步批量读取一次全部 beam 前缀并批量写入约束 mask，不得逐 beam 从 CUDA 读取前缀。合法词集合与原 HF 前缀回调 MUST 相同，使用完整词表评分后的加性 mask；EOS、padding、非法前缀失败和 beam search 参数 MUST 保持现有协议。

#### Scenario: Legal prefixes and completed paths
- **WHEN** 输入包含目录内前缀、完整四码及完成后的 EOS 或 padding 路径
- **THEN** 处理后的 scores 与原 HF 处理器相同，每步前缀矩阵仅读取一次

#### Scenario: Unknown or unsatisfiable prefix
- **WHEN** 前缀不在目录内且不是完成路径，或合法词集合为空
- **THEN** 生成失败，不允许静默扩大候选集

#### Scenario: Checkpoint compatibility and output parity
- **WHEN** 同一现有 checkpoint 与输入分别使用原回调和批量处理器生成
- **THEN** state_dict 可严格加载，推荐 IDs 和 scores 逐位相同，候选数量与评分规则保持不变
