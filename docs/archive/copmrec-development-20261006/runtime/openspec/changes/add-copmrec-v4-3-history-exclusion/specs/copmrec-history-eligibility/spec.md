## ADDED Requirements

### Requirement: 共享有效历史资格排除且合法分数不变

系统 SHALL 使用唯一共享 `apply_history_exclusion(scores, input, catalogmodel)`，对有限FP32完整目录原分数，仅将模型输入中attention有效完整SID对应的catalog行设为负无穷；MUST 返回分数副本，所有未被排除行逐值不变，不读取labels、raw training或用户key，不构建新训练数据或关系分数。

#### Scenario: 有效历史重复且其他商品可选
- **WHEN** 有效历史包含同一完整SID多次，完整目录也包含cold商品
- **THEN** 该历史行仅mask一次，所有其他合法行保持原分数，cold商品继续可选；稳定catalog-row排序取Top10

#### Scenario: 标签与用户key改变
- **WHEN** 相同有效输入和原分数附带不同labels或output_keys
- **THEN** 排除行和推荐SID相同，output_keys仅由标准输出传递，不参与候选资格

### Requirement: 严格完整SID与padding边界

系统 MUST 使用已冻结模型输入的最多20个有效完整SID商品，不改变history window；attention SHALL 为二值、每个完整商品一致且右侧padding。只有有效SID参与精确catalog lookup，inactive padding槽内容忽略；MUST 拒绝partial有效SID、未知有效SID、错误shape/dtype/device/非有限输入分数，或不足top_k个可选行。

#### Scenario: 完整inactive padding
- **WHEN** 一个完整SID槽的attention全0且token值为-1、999或其他inactive值
- **THEN** 该槽不参与lookup或mask，保持原LIGER的padding语义

#### Scenario: 半个SID或未知SID
- **WHEN** SID层级仅部分attention有效，或完整有效SID不属于当前目录
- **THEN** 推理明确失败，不静默略过、不模糊匹配、不按SID前缀排除其他商品

### Requirement: 原LIGER checkpoint兼容的仅单进程推理

`HistoryExcludedLiger` SHALL 继承原Liger constructor及normal checkpoint hook / strict state恢复，不增加persistent state keys或桥接checkpoint版本；MUST 只开放dense retrieve / predict、原评分后共享排除、标准ModelOutput，冻结参数并拒绝optimizer/scheduler、fit/train(True)/训练loss路径和checkpoint save。预测world_size MUST 为1。

#### Scenario: 恢复真实固定LIGER并稳定预测
- **WHEN** 公共pipeline通过真实best45000 ckpt_path恢复原目录一致的checkpoint，采用FP32单进程
- **THEN** 原hook及strict state正常通过，原query和dense分数保留，应用共享policy后输出唯一合法Top10完整SID

#### Scenario: 非dense或训练请求
- **WHEN** 请求hybrid/generative、target_ids trace、训练/eval_step loss、optimizer、fit或多进程预测
- **THEN** 系统明确拒绝，不改变原Liger类或开始任何训练

### Requirement: 冻结pool只在完整logits后增加相同资格

`HistoryExcludedFixedLogitPool` SHALL 保留原FixedLogitPoolCoPMRec九kwargs、四expected identities、两原类normal strict恢复、0.5/0.5独立query完整目录logits及原 `pool_contract`；MUST 只在父forward完成后调用共享helper，以source catalog精确排除，再复用父stable Top10及所有训练/多进程/checkpoint拒绝。

#### Scenario: 固定双成员与来源链
- **WHEN** 输入固定nj9elah1 best6000 / l3zyr91b scale0 best6000及已审计SHA
- **THEN** 两原v4/v4.1契约、同catalog、原v0链、control warmstart、bias0和cold residual0维持，只有历史资格改变，不生成wrapper checkpoint

#### Scenario: 改变成员或评分设置
- **WHEN** expected reference/SHA与实际loader身份、生产pair、原温度、权重、目录或来源链不匹配
- **THEN** 构造或preflight失败，不用新成员、归一化或参数扫描替代

### Requirement: 两模型相同历史契约与标准来源记录

两模型property、resolved config与commonwriter SHALL 同名输出exact9keys `history_eligibility_contract`：protocol=`copmrec-known-history-eligibility-v1`、history_scope=`input_valid_complete_sids_max20`、labels_used/raw_training_used/user_keys_used=false、score_rule=`unchanged_full_catalog_logits_then_history_minusinf`、rank_ties=`stable_catalog_row`、cold_items_eligible=true、single_process=true。MUST 保持统一src.main/Hydra、公共loader/lineage、标准keys/predictions bundle，pool另保留原pool_contract和双Artifact身份；旧类和默认入口不改。

#### Scenario: same-policy匹配
- **WHEN** LIGER和pool使用同raw split、SID/catalog、80token/4层完整历史及此契约
- **THEN** 实际modelproperty/config/writermetadata契约完全一致，主审计独立证明Top10不包含有效历史行

#### Scenario: 可复现根脚本
- **WHEN** 收到quoted真实checkpointURI、notes、显式dry-run和额外override
- **THEN** 根脚本经统一入口透传且NPROC1；formal prelaunch核验冻结身份/评分/政策，pool顶层ckpt_path=null，LIGER使用真实normal ckpt_path

### Requirement: 有限same-policy Evaluation与不可重置累计预算

本阶段 MUST 新增训练0、optimizer0、train-data构建0，保留既有5训练/30000耗尽及已关闭路线；完整Evaluation SHALL 最多2次，为固定LIGER best45000与固定pool各一次且同policy。不扫描window/pair/weight/temperature/normalization/checkpoint/bias/LR/arm。

#### Scenario: 有效原始审计完成
- **WHEN** 两次完整Evaluation已产生终态标准输出
- **THEN** 独立核验原始175shards/22363用户标签、合法唯一SID、有效历史零占位、catalog/输入/checkpoint/source/实际契约，复算pairedR/N与固定cohort；已知Top10压缩rank只作诊断，不冒充完整policy结果

### Requirement: 新same-policy主门禁及旧点阈值同时满足

系统 SHALL 仅当pool对新same-policy Evaluation LIGER的R10/N10均≥1.1倍、两paired绝对差值CI下界均>0，并且pool同时满足旧Validation阈值R10≥.09877029021151008（2209hits）/N10≥.0511173919307933时，允许固定same-policy LIGER和同pool至多2次Testing。MUST 不以未排历史LIGER作为唯一主分母，不用单组或只一项达标替代整体门禁。

#### Scenario: 单边旧baseline达标
- **WHEN** pool只对未排历史LIGER超过10%，或same-policy仅一项超过10%
- **THEN** 不进入Testing，保留真实结果，按独立组件保留标准决定是否继续bad case及累计研究；不换window/pair/参数、不自动追加训练

#### Scenario: 两次Testing固定确认
- **WHEN** Validation全部门禁通过
- **THEN** 使用相同固定checkpoint、policy、scores配置对Testing各运行same-policy LIGER和pool一次，全线程此前1次→最多3次；接受需对新same-policy Testing LIGER双10%、两pairedCI下界>0且同时达到旧042139al点阈值R10≥.07855386128873586（1757hits）/N10≥.04002978116676896，不用Testing选择规则或训练

### Requirement: 结题保留成熟协议及效果边界

研究记录 MUST 将历史排除定位为成熟候选资格处理和显式新政策，不声称原创、官方LIGER bugfix、新增CF信号或单边过滤的方法增量。未达整体门禁 SHALL 原样记录并停止本轮Testing晋级，不自动丢弃符合独立保留标准的正向组件；无明显增量或负向按此固定policy边界收缩。未实际达到目标时whole goal保持未完成，不自动扩大实验预算。

#### Scenario: 准备或条件下界通过
- **WHEN** CPU/脚本/source准备通过，或已知命中压缩rank产生正位移下界
- **THEN** 仅报告实现准备或该split的条件诊断，正式效果仍须完整same-policy输出审计，不将目标标为已达成

### Requirement: 组件保留与整体门禁独立

按用户在正式运行前的直接指示，系统 SHALL 分开报告整体目标门禁与组件保留：v4.3对自己冻结旧pool i4xwruok的R10/N10任一相对提升≥3%且另一项点值无退化时，可保留用于bad case分析或累计贡献；MUST 同时报告pairedCI和固定分组限制，不把点值保留称为统计已证实增量。该判断 MUST 不触发未通过整体门禁的Testing，不重置本阶段2Evaluation/2条件Testing/0训练额度或旧5训练/30000用量。

#### Scenario: 组件明显正向但整体双10未达
- **WHEN** 组件符合≥3%且另一项无点值退化，而pool对same-policy LIGER整体门禁未过
- **THEN** 可保留该组件并继续只读bad case或累计研究，目标仍未达、不进行本轮Testing；未来实验成本须依据新证据另行明确登记

#### Scenario: 组件点值保留但区间跨零
- **WHEN** 点值符合组件保留，pairedCI跨零或固定分组存在差异
- **THEN** 明确保留只是开发取舍，效果主张受区间和分组限制；不宣称总体增量已证明，不将旧已耗尽预算解释为用户永久禁止后续研究
