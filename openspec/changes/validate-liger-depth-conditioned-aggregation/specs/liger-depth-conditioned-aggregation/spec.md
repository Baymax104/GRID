## ADDED Requirements

### Requirement: 固定深度条件内容聚合
系统 SHALL 提供显式的`max_root_mass_deep`推理控制，在depth0使用最大后代内容分数，在depth1及以后使用子树logsumexp总质量，并保持合法支持、生成概率与checkpoint恢复的混合权重不变。

#### Scenario: 根层使用最大后代
- **WHEN** 深度条件控制处理decoder BOS后的第一个SID token
- **THEN** 内容条件分布与相同输入下的`max_mixture`内容条件分布逐元素一致

#### Scenario: 后续层使用总质量
- **WHEN** 深度条件控制处理已有至少一个SID token的合法父前缀
- **THEN** 内容条件分布与相同输入下的`learned_mass`内容条件分布逐元素一致

### Requirement: 推理控制兼容性
系统 MUST 保持既有默认、mass、max和训练行为兼容，并 SHALL 拒绝在训练阶段使用`max_root_mass_deep`。

#### Scenario: 默认行为保持不变
- **WHEN** 未显式选择深度条件控制
- **THEN** 既有`learned_mass`候选和checkpoint协议保持不变

#### Scenario: 训练拒绝推理控制
- **WHEN** 训练步骤启用`max_root_mass_deep`
- **THEN** 系统以明确错误停止且不产生训练更新

### Requirement: 可审计预测产物
系统 SHALL 为深度条件运行发布候选trace与逐层分散度trace，并在metadata中记录机制控制、深度聚合计划、运行时alpha及testing已消耗边界。

#### Scenario: 完成预测
- **WHEN** 用户完成一次深度条件正式prediction
- **THEN** 两类trace均可验证，且metadata标识`max_root_mass_deep`与`max_then_mass`

### Requirement: 冻结转化判定
系统 SHALL 将新臂与冻结mass/max对照按用户配对比较，并只按预声明的覆盖保留与推荐转化门槛作出决定。

#### Scenario: 未保留覆盖价值
- **WHEN** 新臂相对max的目标覆盖差95%区间下界不大于0
- **THEN** 决定为停止且不搜索其他切换深度

#### Scenario: 覆盖保留但推荐未转化
- **WHEN** 覆盖门槛通过但相对mass的NDCG@10区间下界不大于0或Recall@10点估计下降
- **THEN** 决定为保留路径价值但停止转化路线

#### Scenario: 两道门槛通过
- **WHEN** 覆盖区间下界大于0、NDCG@10区间下界大于0且Recall@10点估计不下降
- **THEN** 决定为testing-informed探索性候选并要求独立设置确认
