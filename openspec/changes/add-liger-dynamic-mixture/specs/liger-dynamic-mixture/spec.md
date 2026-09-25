## ADDED Requirements

### Requirement: 动态合法概率混合
系统 SHALL 对每个用户前缀计算一个共享sigmoid权重，混合合法分支生成与内容概率，保持固定版本兼容。

#### Scenario: 零初始化和死分支
- **WHEN** 门控零初始化或遇到非法前缀
- **THEN** 合法前缀结果等于固定0.5，非法前缀全负无穷，无NaN。

### Requirement: 隔离训练和来源
系统 MUST 只用training最后目标的本地缓存训练门控，用户哈希留出，缓存带完整哈希与来源身份，拒绝重复用户、错误split及来源不一致。

#### Scenario: 来源不匹配
- **WHEN** gate来源与推理checkpoint或目录不一致
- **THEN** 在预测前拒绝加载。

### Requirement: 可审计的最小实验
系统 SHALL 提供动态和constant两臂，冻结基础模型，保留原hybrid终排，统一入口及支持dry-run、notes、额外override的根脚本。

#### Scenario: 用户启动
- **WHEN** 用户启动缓存或门控训练
- **THEN** 缓存仅写本地，指标和checkpoint按现有规则记录，dry-run不发布缓存，完整实验不由agent自动启动。


### Requirement: 单次扩预算对照
系统 SHALL 为用户授权的扩预算提供隔离配置，两臂从零初始化，最大30epoch并采用相同内部NLL早停，保留旧3epoch配置。

#### Scenario: 扩展配置与默认配置隔离
- **WHEN** 用户通过根脚本指定扩展experiment
- **THEN** 两臂均使用30epoch上限、patience5和min_delta0.0001，最佳checkpoint依据内部NLL，旧入口仍为3epoch且不启用早停。
