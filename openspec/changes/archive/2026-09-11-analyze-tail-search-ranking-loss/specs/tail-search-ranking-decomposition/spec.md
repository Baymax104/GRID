## ADDED Requirements

### Requirement: Paired candidate analysis SHALL validate source identity
启用搜索—排序分解时，系统 SHALL 读取 fixed/widened recommendation 与对应 trace，按唯一 user key 连接并校验 label、split、checkpoint、SID、候选宽度和 source identity。标准 bundle MUST 保持 `keys + predictions`。

#### Scenario: Compatible W&B inputs are analyzed
- **WHEN** 四份输入来自两条可配对的 W&B source run
- **THEN** 系统 MUST 记录 Artifact 版本/digest、原引用与解析路径，并核对对应 recommendation/trace 的 source run
- **AND** target 最早 rank、最终 membership MUST 分别与 trace 最终 beam rank、survival 一致

#### Scenario: Keys or lineage disagree
- **WHEN** 出现重复/缺失/额外 key、标签冲突、非法 SID 或不兼容 lineage
- **THEN** paired analysis MUST 显式失败，不得用交集静默丢弃样本

#### Scenario: Local provenance is incomplete
- **WHEN** 本地内容可计算但 source identity 无法核实
- **THEN** 输出 MUST 标明 identity 未验证，并禁止将该输入标为 probe_candidate

### Requirement: Search and ranking transitions SHALL be exhaustive
系统 SHALL 使用 fixed Top10 hit/miss 与 widened top10/below10/absent 的笛卡尔积，输出六个互斥状态，逐用户与逐组记录。

#### Scenario: A fixed hit is demoted or lost
- **WHEN** fixed 已命中但 widened 的 target rank 超出 10 或完全不存在
- **THEN** 系统 MUST 分别记录排序丢失与候选丢失，不得只报告新增命中

#### Scenario: All targets are classified
- **WHEN** group 汇总生成
- **THEN** 六状态计数之和 MUST 等于组支持数，新增减丢失 MUST 等于 Top10 净命中变化
- **AND** Tail 与 Tail-Cold MUST 单列，合并组 MUST 按分子分母加权而非取最大值

### Requirement: Candidate access SHALL remain distinct from ranked utility
系统 SHALL 同时输出 fixed/widened Hit@10、NDCG@10、widened candidate access 和固定候选集 oracle Hit@10 ceiling，附数据单位与分母。

#### Scenario: Target is at widened rank 11
- **WHEN** 目标只出现在 widened 第 11 位
- **THEN** 其 candidate access 与 oracle membership MUST 为真，Hit@10 与 NDCG@10 MUST 为零

#### Scenario: Duplicate candidates or absent target occur
- **WHEN** 同一 target 重复出现或不出现
- **THEN** 实际指标 MUST 使用原排序最早 rank 或 miss，不得去重压缩后提升 rank
- **AND** 空分母 MUST 为 null 并给出原因

### Requirement: Attrition summaries SHALL use explicit at-risk populations
系统 SHALL 分层输出累计存活率、上一层存活条件下的退出率及首次失败统计，区分 teacher rank、beam rank 和可达 parent 的 margin。

#### Scenario: Parent has already failed
- **WHEN** 目标在此前层已经退出
- **THEN** 该用户 MUST 不进入本层条件退出分母，缺失 beam margin MUST 不补零

#### Scenario: Target never fails
- **WHEN** trace first_failure_depth 为 -1
- **THEN** 目标 MUST 计入完整存活，不能计入任意实际失败层

### Requirement: Follow-up decisions SHALL not assert method effectiveness
跨设置报告 SHALL 区分 probe_candidate、stop、inconclusive，记录依据、反证和新增数据需求，并保留原 H2 No-Go。

#### Scenario: Oracle headroom is positive
- **WHEN** fixed candidate pool 存在非零排序空间
- **THEN** 报告 MUST 将其作为候选内上界，不得宣称可学习收益、因果机制或自动启动 calibration

#### Scenario: Evidence is unavailable
- **WHEN** 身份或统计质量不满足条件
- **THEN** 对依赖该证据的结论 MUST 标为 inconclusive，而不是用零值代替或自动追加实验
