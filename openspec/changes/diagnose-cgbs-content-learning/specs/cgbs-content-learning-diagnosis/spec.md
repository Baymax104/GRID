## ADDED Requirements

### Requirement: Bounded diagnostic evidence
诊断 SHALL 通过统一入口和共享writer，记录用户键、标签、全目录内容logits、排名、梯度和输入及checkpoint指纹，限制单进程训练或验证样本。

#### Scenario: Reject testing or inference mode
- **WHEN** 使用testing或无法求导的inference_mode
- **THEN** 在计算前明确失败。

### Requirement: Gradient attribution
诊断 SHALL 在同一前向分别计算generation/content对encoder及query的梯度，报告原始范数、加权比例和零范数未定义夹角。

#### Scenario: Distinguish weighted contribution
- **WHEN** 辅助权重为0.1
- **THEN** 有效梯度范数是原始content范数乘0.1，夹角按原始梯度计算。

### Requirement: Temporary training-only fit
诊断 SHALL 仅允许training小批拟合，分别query-only及encoder+query，并在正常和异常退出时恢复模型状态；不发布拟合checkpoint。

#### Scenario: Restore after fit
- **WHEN** 小批优化完成或发生异常
- **THEN** 原权重、requires_grad、RNG保持不变。

### Requirement: Manual reproducible launch
根脚本 SHALL 支持dry-run、notes两种形式、额外Hydra覆盖，并固定可复核输入及样本预算。

#### Scenario: Compose only
- **WHEN** 对实际Bash参数进行轻量验证
- **THEN** 无模型训练或正式推理启动且所有URI正确解析。
