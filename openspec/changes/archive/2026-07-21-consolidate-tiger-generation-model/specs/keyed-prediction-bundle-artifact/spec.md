## ADDED Requirements

### Requirement: keyed semantic ID bundle SHALL provide model-side tensor extraction
当 keyed prediction bundle 保存 semantic IDs 时，系统 SHALL 提供模型侧 tensor extraction 路径，从 bundle 的 `predictions` 中获得 semantic ID tensor，同时保留数据侧按 key 查询的完整 bundle 协议。

#### Scenario: 模型侧加载 semantic ID tensor
- **WHEN** TIGER 模型配置从 `semantic_id_path` 加载 prefix 校验数据
- **THEN** 加载结果 MUST 是 semantic ID tensor，而不是完整 `ModelOutput` bundle
- **THEN** tensor 第一维 MUST 对应 item 行，第二维 MUST 对应 semantic ID hierarchy

#### Scenario: 数据侧仍使用完整 bundle 查询
- **WHEN** TIGER 数据 preprocessing 执行 `item_id -> semantic_id` 映射
- **THEN** 数据侧 MUST 继续使用包含 `keys` 与 `predictions` 的完整 keyed bundle
- **THEN** 数据侧 lookup MUST 保持通过 key 查询而不是假设 item id 等于 tensor 行号
