## ADDED Requirements

### Requirement: Native training through Lightning
模块 SHALL 通过 training_step 使用已验证 backbone 和逐位置标签，optimizer SHALL 从 TrainingModelConfig 注入，不新增旁路训练入口。

#### Scenario: Official objective backpropagation
- **WHEN** 最小内存 batch 执行 training_step 和 backward
- **THEN** loss SHALL 有限且参数 SHALL 获得梯度

### Requirement: Full catalog ranking and raw item output
evaluation/test/predict SHALL 对固定目录所有真实商品计算点积排名，不采样、不屏蔽冷启动或历史，padding SHALL 不作为候选。分块 SHALL 与直接全目录计算一致，同分 SHALL 按固定 key 升序。prediction SHALL 返回用户 keys 与原始商品 TopK 的 ModelOutput。

#### Scenario: Chunk size and tie invariance
- **WHEN** 使用不同 chunk_size 对同一模型与历史评分，包括同分商品
- **THEN** 排名 SHALL 与直接全库 stable 降序排序完全一致

### Requirement: Shared metric and checkpoint identity
指标 SHALL 通过 MetricEngine/MetricCallback 记录全用户 Recall/NDCG@5/@10 与 user_count，prediction SHALL 沿用配置的 logging_modes。checkpoint SHALL 校验目录与训练结构身份。

#### Scenario: Exact rank metrics
- **WHEN** 两个用户分别命中不同 rank 或未命中
- **THEN** 指标 SHALL 与手算一致，并保持全部用户分母

#### Scenario: Incompatible checkpoint
- **WHEN** 加载目录 key 或历史/算法配置变化的 checkpoint
- **THEN** 模型 SHALL 拒绝加载
