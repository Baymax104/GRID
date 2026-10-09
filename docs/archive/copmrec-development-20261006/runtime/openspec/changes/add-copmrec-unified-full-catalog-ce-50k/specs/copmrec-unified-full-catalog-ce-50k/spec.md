## ADDED Requirements

### Requirement: v5派生完整目录CE

系统 SHALL 新增v5.2单模型，使training content CE对全部目录logits归一化，cold原logits获得直接CE竞争梯度；training目标仍seen，原SID／mixture、learned alpha、query／目录／history及cold残差保持既定协议。系统 SHALL 保留原loss调用和dropout顺序及旧运行文件字节，不更改baseline默认路径。

#### Scenario: 单次训练的直接cold竞争

- **WHEN** 使用合法seen训练目标计算v5.2三项loss
- **THEN** content CE使用未cold mask完整logits，SID／mixture仍使用原式和同一次目录logits，cold residual为零

### Requirement: 支持集与来源契约

系统 SHALL 在模型unified契约、experiment及writer中固定v5.2／dense_ce_support的5个字段，并拒绝错support、旧方法或weights-only推荐checkpoint初始化。系统 SHALL 核实际运行源码字节、CP与两个上游输入，不能用当前快照补历史缺口。

#### Scenario: 旧checkpoint误装配

- **WHEN** 把v5／v5.1或错误dense_ce_support checkpoint交给v5.2
- **THEN** strict hook拒绝加载，不退回恢复或重新训练

### Requirement: 原累计最后50k

系统 SHALL 在原3train／150k预算内仅新增最后1train／50k及1完整singleGPU Validation，随机初始化全模块连续训练，原AdamW／2500warmup／50000cosine／global256／FP32协议。系统 SHALL 从自己的100raw Val点选NDCG@10首个最大值best并分别证明实际预算和保存状态，不自动Test／43或重置预算。

#### Scenario: 原任务仍在运行

- **WHEN** 观察同一唯一job超时或SDK暂未同步
- **THEN** 继续观察原handle，未获终态不得重启或称50k完成

### Requirement: 原整体门槛及辅助边界

系统 SHALL 在原始输出有效性通过后，以同policy native42为唯一双8%与双paired绝对CI正向分母；冻结v5增量、cold占位及51cold-target风险只分别解释效果与机制。系统 SHALL 不推断未见Top11、不把cold槽减少当作推荐收益、不用单seed合格结果宣布复现目标完成。

#### Scenario: cold曝光减少但覆盖未改善

- **WHEN** fullValidation中false cold槽减少但R10相对v5不提高
- **THEN** 覆盖预测不获支持，保留真实结果与整体gate，不追加权重／温度／cold规则扫描
