## ADDED Requirements

### Requirement: 同候选固定混合概率终排

系统 SHALL 支持 `final_ranking_mode=content|mixed`，content SHALL 为默认且保持旧数值行为。mixed SHALL 只使用同一 v4 checkpoint 的 learned alpha 和 Mass 合法条件概率，对原 `gen20 ∪ all_cold` 候选计算完整 SID log probability 并排序，不新增参数或搜索，不对候选子集重新归一化。

#### Scenario: 默认行为兼容

- **WHEN** 未指定终排模式或指定 content
- **THEN** 候选、最终输出、score、loss 和 RNG 与原 v4 相同，已有 checkpoint 完整严格加载

#### Scenario: 训练概率一致性

- **WHEN** mixed 在 eval 模式评分同一用户的合法目标 SID
- **THEN** 逐层 score 与增量 decoder 概率一致，目标 score 的负均值除以 SID 层数等于现有 mixture NLL

#### Scenario: 推理策略不得进入训练

- **WHEN** mixed 模式进入 fit 或 training=True 监督调用
- **THEN** 系统明确拒绝，不启动新的训练策略

### Requirement: 无标签依赖及 cold 同分布

候选评分接口 SHALL 只接收用户历史、encoder 表示、完整目录 logits 和候选身份。真实标签 SHALL 仅用于评分完成后的 trace；所有 cold 候选 SHALL 采用与生成候选相同的合法完整目录条件分布。

#### Scenario: 标签不影响预测

- **WHEN** 移除或替换真实标签，保持用户历史与候选不变
- **THEN** mixed score 与最终输出相同，cold 分数有限，padding 分数为负无穷

### Requirement: 严格同候选配对 trace

mixed 输出 SHALL 声明 v4 专属完整混合路径协议并记录同候选 content 参考排名与 TopK、padded 候选 rows 和两套分数。validator SHALL 验证 union、唯一性、分数、padding、稳定排序、两套排名及输出对应关系。返回的 marginal_probs SHALL 为实际最终 score；mixed SHALL 不采用 content 子集排名上界。

#### Scenario: 配对产物有效

- **WHEN** mixed 推理输出合法 trace
- **THEN** content 参考与 mixed 来自同一候选集合和 checkpoint，独立重排得到所记录的排名/TopK，marginal_probs 与主输出顺序一致

#### Scenario: 错误 trace 被拒绝

- **WHEN** 修改协议、候选、分数、参考排名或任一 TopK 导致不一致
- **THEN** validator 明确拒绝，不能生成可晋级证据

### Requirement: 一次 evaluation 的证据边界

固定规则验证 SHALL 使用统一入口、单进程推理、完整 evaluation 用户和已选择的 v4 checkpoint；只依 evaluation 做当前排序决策。853 个漏排数量和旧归档结果 SHALL 不作为正向效果；不自动追加 alpha、checkpoint 或终排权重扫描。

#### Scenario: 当前固定规则验证

- **WHEN** 一次 evaluation 完成并核验同候选 content/mixed 的 Recall、NDCG 和逐用户损益
- **THEN** 根据配对结果决定保留 content 或晋级固定 mixed，不能把实现通过或 evaluation 晋级称为 Testing 达到 10% 目标
