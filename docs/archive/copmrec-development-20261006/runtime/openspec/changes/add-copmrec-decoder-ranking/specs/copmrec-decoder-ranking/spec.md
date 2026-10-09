## ADDED Requirements

### Requirement: training均值上限归一化
系统 MUST 提供默认关闭的competitive NDCG分母上限，仅以已核验training covered样本的原mixed排名拟合，冻结并记录来源契约；不得使用evaluation拟合。

#### Scenario: 高位梯度增强与恢复保持
- **WHEN** 启用training_mean_cap
- **THEN** 每用户NDCG梯度为原梯度乘max(1,Z_u/C)，零权重和padding安全，双teacher保护及SID目标保持

#### Scenario: checkpoint及audit严格核验
- **WHEN** 保存或加载启用上限的checkpoint
- **THEN** 目标契约包含模式版本、C和training用户数，加载时核对配置并从原training缓存重算；旧NLL使用per_user，旧默认checkpoint仍兼容
### Requirement: 双分支正确关系保护
系统 SHALL 提供默认content及可选content_generation teacher来源，以training标签支持的正确关系给出保护参照，同pair取较大参考间隔，不叠加；竞争集合和推理保持。
#### Scenario: 仅generation正确
- **WHEN** training目标content未命中、generation进入Top10并高于有效负例
- **THEN** 双teacher保护该关系，teacher分数无梯度，两分支均未支持的样本仍不增加保护梯度。
#### Scenario: 重复teacher与目标恢复契约
- **WHEN** 两teacher完全相同或加载单content checkpoint到双teacher目标
- **THEN** 相同teacher损失和梯度不翻倍，目标版本不符拒绝加载。

### Requirement: 有条件的内容排序保护
系统 SHALL 仅在显式启用的competitive NDCG训练中，以冻结content的Top10折扣差为参考，对原content已命中目标与content严格排后的竞争负例施加单侧平方间隔损失；默认关闭且不改变推理。
#### Scenario: 原content未命中或teacher分数相等
- **WHEN** 目标原content不在Top10，或某负例与目标content分数相等
- **THEN** 对应保护贡献和梯度为零，原NDCG恢复监督保留。
#### Scenario: 正确关系已超过teacher参考间隔
- **WHEN** mixed目标减负例分差已达到冻结teacher折扣差
- **THEN** 不产生保护梯度，不将改善拉回teacher。
#### Scenario: 保存与加载保护目标
- **WHEN** checkpoint启用了保护项
- **THEN** 保存并严格匹配版本和weight；旧关闭目标仍可兼容加载，双臂audit的NLL保持关闭。

### Requirement: 冻结内容与来源
系统 SHALL 仅优化decoder非共享参数，保持历史、候选、内容分数与alpha一致。
#### Scenario: 拟合及保存
- **WHEN** 用户运行任一训练臂
- **THEN** 校验缓存/历史目标/原checkpoint来源，并保存arm、冻结参数及历史契约。
### Requirement: 匹配训练及统一评价
系统 SHALL 提供NLL与NDCG10加权pairwise两臂及同用户双checkpoint audit。
#### Scenario: 两臂来源不匹配
- **WHEN** 两臂checkpoint或历史契约不同
- **THEN** 拒绝评价。
#### Scenario: 完整评价
- **WHEN** 用户运行audit
- **THEN** 保留五路排名和主输出一致性，复算相对content/mixed及两臂间配对指标。

### Requirement: 有效竞争pair归一化
系统 SHALL 保留默认all目标，并允许competitive目标在同一完整候选池上使用原content与当前mixed各Top20并集负例；相同mask必须用于损失分子和分母，不得删除推理候选或用evaluation标签筛选。
#### Scenario: 容易负例在竞争集合之外
- **WHEN** NDCG训练启用competitive且某负例不在两路Top20并集
- **THEN** 它不贡献pairwise分子或分母，完整候选评分和验证排序保持。
#### Scenario: cold候选实际参与竞争
- **WHEN** cold候选进入任一路Top20
- **THEN** 它作为有效负例保留，不以cold身份直接排除。
### Requirement: 排序目标checkpoint契约
系统 SHALL 保存并校验NDCG目标scope/k，旧checkpoint没有新增字段时按all/k20解释；NLL目标不受pair筛选设置影响。
#### Scenario: 目标不匹配的恢复
- **WHEN** competitive模型尝试加载旧all目标训练checkpoint
- **THEN** 拒绝恢复，不能把旧checkpoint当作competitive续训。

### Requirement: Top10边界辅助监督
系统 MUST 提供默认关闭的边界项，以当前混合分数排除正例后的第10负例计算softplus负正分差；保留候选及推理分数，训练标签不得用于推理切换。

#### Scenario: 已命中与不足K负例
- **WHEN** 目标已入Top10或有效负例不足10
- **THEN** 前者仍获得随领先间隔减小的梯度，后者贡献有限可微零；padding及正例不被选作负例，tie稳定

#### Scenario: 校准与目标恢复
- **WHEN** 启用辅助项或恢复对应checkpoint
- **THEN** lambda仅从原training covered缓存拟合并冻结，版本/K/fraction/lambda及用户数保存并严格重验；旧NLL强制关闭，默认目标兼容旧checkpoint
