## ADDED Requirements

### Requirement: Independent v2 model
系统 SHALL 提供独立基于v0的CoPMRec v2，全参数持续联合训练，并保留v1/v1.1运行行为。

#### Scenario: Model hierarchy
- **WHEN** 装配v2
- **THEN** 推荐组件直接基于JointMixtureLiger，head只输入完整候选SID的decoder最终末状态，末层零初始化

### Requirement: Natural-candidate bounded scores
系统 SHALL 在自然beam+cold候选上计算float32 population variance，使用detach后的sqrt(var+epsilon²)将tanh相关性乘以固定beta0.5，与content分数相加。

#### Scenario: Training positive injection
- **WHEN** 训练额外加入真实正例和content TopK
- **THEN** 扩展候选不改变用于校准的自然集合，head评分函数与推理相同

#### Scenario: Empty or single natural candidate
- **WHEN** 自然候选少于2个
- **THEN** 方差为0、sigma为epsilon，评分有限且没有NaN

#### Scenario: Chunked scoring
- **WHEN** 候选跨chunk分片
- **THEN** sigma按完整自然集合计算，用户对应及数值/梯度保持一致

### Requirement: Fixed joint ranking objective
系统 SHALL 在v0三項目标之外，以固定lambda0.01加入最终融合分数的候选CE，并记录raw与weighted loss及尺度诊断。

#### Scenario: Shared gradients
- **WHEN** 反向与优化
- **THEN** 单一optimizer覆盖全部推荐参数，内容分数及head/decoder/cross-attention均保留梯度，尺度统计停止梯度

### Requirement: Versioned runtime and restore
系统 SHALL 提供独立训练/推理入口及严格的checkpoint/trace契约，支持双卡、dry-run、notes及额外override。

#### Scenario: Scratch training
- **WHEN** 两个checkpoint引用均null
- **THEN** 通过uv run torchrun -m src.main从零装配v2，不恢复v1 checkpoint

#### Scenario: Contract mismatch
- **WHEN** 版本、beta/lambda/epsilon、head或评价历史不匹配
- **THEN** 恢复或对应trace校验拒绝该输入，不能静默替换协议

### Requirement: Historical v0 validation cadence
系统 SHALL 通过独立v2 trainer将默认验证间隔设为500微批，对齐BMX-116真实v0 run 7y54j4m6；按用户确认读取全evaluation/testing，保留共享指标记录步数语义及融合hybrid选点。

#### Scenario: Scratch run validation records
- **WHEN** 默认累积1、训练50000更新且不启用sanity validation
- **THEN** 每500更新验证一次，共100次验证；训练日志每50更新记录一次，不按GPU数缩短间隔
