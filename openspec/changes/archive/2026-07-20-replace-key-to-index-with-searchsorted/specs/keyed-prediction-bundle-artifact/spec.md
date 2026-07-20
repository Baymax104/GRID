## MODIFIED Requirements

### Requirement: keyed prediction bundle SHALL NOT 假设业务主键等于 tensor 行号
推理产物协议不得依赖业务主键（如 `item_id`）是从 0 开始且连续的数组下标。加载时 SHALL 按 key 值排序 keys 和 predictions，查询时 SHALL 使用 `torch.searchsorted` 二分查找行号，SHALL NOT 使用 Python dict 运行时索引。加载时 SHALL 检测重复 key 并 raise。

#### Scenario: 非连续 item IDs 仍可成功导出
- **WHEN** 推理结果中的 `keys` 包含非连续或大于样本数的 item IDs
- **THEN** 导出 `merged_predictions_tensor.pt` MUST 成功
- **THEN** 导出逻辑 MUST NOT 因"key 超出样本数"而触发越界异常

#### Scenario: 加载时按 key 排序
- **WHEN** `load_model_output` 加载 `.pt` 文件
- **THEN** 返回的 bundle 中 `keys` MUST 按 key 值升序排列
- **THEN** `predictions` 的行顺序 MUST 与排序后的 `keys` 一一对应

#### Scenario: 查询使用 searchsorted
- **WHEN** `gather_predictions_by_keys` 查找 key 对应的 predictions
- **THEN** MUST 使用 `torch.searchsorted` 在 `keys` tensor 上二分查找行号
- **THEN** MUST NOT 使用 Python dict 或 `key_to_index` 运行时字段

#### Scenario: 重复 key 检测
- **WHEN** `load_model_output` 加载的 `keys` 包含重复值
- **THEN** MUST raise ValueError

#### Scenario: 查询不存在的 key
- **WHEN** `gather_predictions_by_keys` 的 `lookup_keys` 包含 `keys` 中不存在的值
- **THEN** MUST raise KeyError
