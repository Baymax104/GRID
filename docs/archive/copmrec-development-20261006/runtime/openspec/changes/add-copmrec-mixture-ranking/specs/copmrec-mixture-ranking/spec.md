## ADDED Requirements

### Requirement: CoPMRec 专用混合概率终排
系统 SHALL 在 CoPMRec 专用推理路径中，复用原 checkpoint、mass 搜索、候选与 cold 并集，使用完整 SID 混合路径对数概率终排，不改变 baseline 默认行为。

#### Scenario: 三种次序共享候选
- **WHEN** 用户执行新的 evaluation 推理入口
- **THEN** 系统只执行一次 mass beam search，并对同一候选保存内容、混合与合法生成三种次序

#### Scenario: 所有候选统一评分
- **WHEN** cold 商品通过并集而非 beam 入围
- **THEN** 系统以相同合法条件公式对完整 SID 打分，不使用目标标签或来源惩罚

### Requirement: 新排序证据可复算
系统 SHALL 用单独版本的 keyed bundle 保存候选分数、SID、排名及协议元信息，并报告全体用户指标和配对新增/损失命中。

#### Scenario: 原始候选与新最终次序
- **WHEN** 推理产物被合并发布
- **THEN** 校验器验证三路分数排序、目标排名和候选成员一致，旧 trace 校验规则不被更改

### Requirement: 入口与边界验证
系统 SHALL 使用既有 Hydra 与共享 writer 入口，默认 evaluation，支持 dry-run、notes 与额外 override。

#### Scenario: 可复用 checkpoint
- **WHEN** 加载已有 JointMixtureLiger checkpoint
- **THEN** 系统不增加训练参数且严格加载原 state_dict

#### Scenario: 内容概率边界
- **WHEN** 内容权重为一的最小单元输入被评分
- **THEN** 完整混合路径概率等于内容 softmax，候选内容次序保持一致
