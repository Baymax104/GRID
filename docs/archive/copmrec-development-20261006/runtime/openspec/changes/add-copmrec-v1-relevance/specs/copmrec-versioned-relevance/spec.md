## ADDED Requirements

### Requirement: 版本与初始化边界

系统 SHALL 将基础 CoPMRec 标识为 v0，保留旧 joint checkpoint 兼容；v1 SHALL 使用独立标识，全部推荐参数持续可训练。可选 v0 预训练权重 SHALL 明确作为初始化输入，不恢复其 optimizer。

#### Scenario: 加载和恢复
- **WHEN** v1 恢复 checkpoint 或指定 v0 初始化
- **THEN** v1 checkpoint 严格校验版本/结构，v0初始化仅载入基础权重，新head零初始化，全部参数属于一个optimizer

### Requirement: 候选相关性共同训练

v1 SHALL 使用当前用户、内容、完整SID decoder表示计算 content+残差分数，新增head输出零初始化；候选列表CE SHALL 与基础三项目标联合训练。

#### Scenario: 训练监督
- **WHEN** 执行training loss
- **THEN** 基础目标覆盖完整微批，排序用户从全training流均匀抽取，候选正例唯一且带梯度评分，不冻结teacher或推荐特征

### Requirement: 部署与标签独立

v1 hybrid SHALL 保留原beam+cold候选并按最终分数TopK排序；真实标签 SHALL 仅用于训练正例注入或评价。

#### Scenario: 改变评价标签
- **WHEN** 保持历史、参数、候选不变而改变评价标签
- **THEN** 推荐输出不变，trace只改变对应诊断字段

### Requirement: 统一入口与验证选点

两版 SHALL 提供统一main入口的train/inference脚本，支持dry-run、notes、额外override和DDP；v1 SHALL 按部署链路hybrid NDCG10选best，保持keyed output bundle与共享writer。

#### Scenario: 开发协议
- **WHEN** compose两版开发train/inference配置
- **THEN** 采用同一selection/audit分片、明确版本、固定候选预算及50k更新，旧基础入口继续可用
