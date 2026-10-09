## ADDED Requirements

### Requirement: 同query双目录分数

模型 MUST 保留v5唯一history query和共享history residual，目录projection只执行一次，final logits严格等于内容目录与内容加seen残差目录两路normalized cosine logits的固定0.5/0.5平均。平均目录向量 MUST 不再normalize，cold残差为0。

#### Scenario: 初始与cold等价
- **WHEN** residual为零或目录行为cold
- **THEN** 对应final logits等于v5的内容目录分数，模型不新增参数或额外T5前向。

#### Scenario: 共用最终评分
- **WHEN** 模型训练或完整dense部署
- **THEN** content CE、legal-prefix mixture和部署使用同一final logits，SID CE不变，learnedalpha不改成新固定gate。

### Requirement: v5.1连续50k与完整契约

v5.1 MUST 随机初始化从step0全部联合训练，沿用v5固定50k/global256/optimizer/scheduler/raw Valbest/history eligible规则；契约 MUST 绑定v5.1和双目录评分、拒绝其他版本/weights-only/finished refit，记录来源及已保存完整状态范围。

#### Scenario: 从头训练
- **WHEN** 从根脚本与src.main/Hydra启动新模型
- **THEN** 推荐checkpoint来源为null、同双卡单次日程运行到50000，projection与T5不重新初始化或冻结。

#### Scenario: 拒绝旧CP
- **WHEN** 向v5.1传入v5/旧warm checkpoint或wrong scoring契约
- **THEN** strict loader拒绝，不伪装scratch50k或新版本。

### Requirement: 累计预算和真实验证

研究 MUST 在现有3run/150k账本内仅将一次未消费50k槽改用于v5.1 seed42、一次未消费fullVal用于ownbest单卡验证；双8+双CI正目标不变，旧v5 partial gain/gate false及失败startup全部保留。MUST 不同时分配仅余一个训练槽给两个43模型，不自动Test或声称双seed完成。

#### Scenario: 有界新检验
- **WHEN** 新代码/来源/CPU/DDP2一步smoke通过
- **THEN** 仅启动唯一v5.1 seed42完整50k，之后完整Val比较native42及固定v5，记录净命中/位次/CI而不扫描。

#### Scenario: 未达目标
- **WHEN** 结果未满足原双8或coverage/head预测
- **THEN** 保留真实正负/不确定结果，目标active、原费用不重置，不追加权重/alpha/CP扫描或移门槛。
