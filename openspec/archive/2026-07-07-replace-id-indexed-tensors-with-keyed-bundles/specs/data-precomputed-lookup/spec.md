## ADDED Requirements

### Requirement: 下游预计算特征消费 SHALL 通过显式 key lookup 完成
凡是消费上游推理产物的 data preprocessing 逻辑，必须通过 keyed bundle 中的 `keys` 建立显式 lookup，再取回对应 `predictions`；不得继续把 `.pt` 当作可直接按业务主键索引的 tensor。

#### Scenario: embedding lookup 不再直接 tensor 索引
- **WHEN** `rkmeans_train`、`rkmeans_inference`、`rqvae_train` 或 `rvq_train` 根据 item ID 读取预计算 embedding
- **THEN** 读取逻辑 MUST 基于 keyed bundle 的 `keys` 查找对应 row
- **THEN** 读取逻辑 MUST NOT 直接执行语义等价于 `embedding_tensor[item_id]` 的访问

#### Scenario: semantic ID lookup 不再直接 tensor 索引
- **WHEN** `tiger_train` 或 `tiger_inference` 根据 item ID 读取 semantic IDs
- **THEN** 读取逻辑 MUST 基于 keyed bundle 的 `keys` 查找对应 row
- **THEN** 读取逻辑 MUST NOT 直接执行语义等价于 `semantic_id_tensor[item_id]` 或 `semantic_id_tensor.t()[item_id]` 的访问

#### Scenario: 历史裸 tensor 产物不再受支持
- **WHEN** 下游加载 `embedding_path` 或 `semantic_id_path`
- **THEN** 加载逻辑 MUST 期望 keyed bundle 协议
- **THEN** 历史裸 tensor 产物 MAY 直接失败，而无需 fallback 兼容
