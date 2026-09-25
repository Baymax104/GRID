## ADDED Requirements

### Requirement: 独立且最小的梯度隔离条件

系统 SHALL 提供 `content_init_full_aux_detached` 条件，仅移除辅助 item CE 到共享编码器的梯度，保留辅助 CE 到 query MLP 的梯度及混合生成损失全部梯度。

#### Scenario: 相同初始化的 C 和 D
- **WHEN** C、D 使用同 seed、配置和输入计算训练损失
- **THEN** 初始参数、损失值、随机数状态相同；辅助 query 梯度相同且非零，D 的辅助编码器梯度为空或零，生成损失的编码器与 query 梯度保持一致且非零

### Requirement: 条件身份与推理兼容

系统 MUST 在 resolved config 和 checkpoint contract 中保留 D 的独立 arm 身份，并支持现有推理干预。

#### Scenario: 防止 checkpoint 条件混淆
- **WHEN** C checkpoint 被提交给 D 或 D checkpoint 被提交给 C
- **THEN** 恢复失败并给出契约不匹配错误，D 到 D 正常恢复

#### Scenario: D 的冻结推理
- **WHEN** 用户选择 D 的 screen 阶段并提供 D checkpoint
- **THEN** 使用同一 checkpoint 依次执行 trained/off；signal 和 rerank 阶段保留原有语义

### Requirement: 手动单次双卡启动

现有 mechanism 脚本 SHALL 支持显式 D 选择，并保持 notes、dry-run、额外 Hydra override 透传和已有默认条件。

#### Scenario: 用户启动 D
- **WHEN** 用户从仓库根目录使用 `--condition d` 启动训练
- **THEN** 只启动一个 D 训练，物理 GPU 0、1，两进程，默认 seed42、原 C 训练预算，且用户 override 具有最终优先级

#### Scenario: 保留既有默认
- **WHEN** 用户不指定 condition
- **THEN** 仍按原有顺序选择 B、C，不追加 D
