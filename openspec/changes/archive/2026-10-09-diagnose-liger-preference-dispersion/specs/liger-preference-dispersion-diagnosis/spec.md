## ADDED Requirements

### Requirement: 诊断不得改变候选搜索
系统 SHALL 在启用偏好分散诊断时返回与对应未诊断解码逐元素一致的处理后 logits 和最终候选；目标标签 MUST 仅用于观察目标路径。

#### Scenario: learned mass 诊断等价
- **WHEN** 同一输入和 checkpoint 分别启用与关闭 learned_mass 诊断
- **THEN** 每层处理后 logits 和最终生成候选逐元素一致

#### Scenario: max 诊断等价
- **WHEN** 同一输入和 checkpoint 分别启用与关闭 max_mixture 诊断
- **THEN** 每层处理后 logits 和最终生成候选逐元素一致

### Requirement: 逐层记录分散度与路径状态
系统 SHALL 对每个用户和 SID 深度记录目标子分支的最大内容 logit、内容 log-mass、`logmass-max`、后代数、max/mass 内容局部 rank 与 margin，并记录实际解码臂中目标父前缀是否仍在 beam。

#### Scenario: 目标父前缀存活
- **WHEN** 目标父前缀存在于当前用户的 beam
- **THEN** 系统记录 active=true，并记录同一生成条件分布下 mass/max 混合目标子分支的局部 rank 与 margin

#### Scenario: 目标父前缀已淘汰
- **WHEN** 目标父前缀不存在于当前用户的 beam
- **THEN** 系统记录 active=false，混合局部字段为 NaN，内容分支的静态分散度字段仍然有效

### Requirement: 独立可验证的诊断 bundle
系统 SHALL 使用独立 `liger_dispersion_v1` schema 保存按 key 对齐的标签、逐层张量和固定协议 metadata，并拒绝缺字段、非有限静态量、不一致深度或重复 key。

#### Scenario: 分布式分片合并
- **WHEN** writer 合并多个 prediction 分片
- **THEN** bundle 按用户 key 排序，保持每个逐层字段与 key/label 对齐，并通过完整 validator

### Requirement: 配对汇总与停止门禁
系统 SHALL 配对 learned_mass 与 max_mixture bundle，报告首次淘汰、mass-only/max-only 目标恢复、分散度方向及后代数分层结果；汇总 MUST 将 testing-informed subgroup 标记为探索性。

#### Scenario: 分散度不能预测恢复
- **WHEN** 分散度方向与 mass-only 路径恢复不一致或仅由后代数解释
- **THEN** 汇总结论收缩为“不支持偏好分散机制”，不得追加 alpha 或分箱搜索

#### Scenario: 只有路径证据
- **WHEN** 分散度预测目标路径存活但最终 Recall/NDCG 增量未确认
- **THEN** 汇总只支持搜索机制条件，不升级为最终推荐优势

#### Scenario: 路径与下游均一致
- **WHEN** 预声明方向同时得到路径存活与最终配对指标支持
- **THEN** 汇总允许将该结果作为单数据集单 seed 的探索性场景证据
