# copmrec-v2-formal-surface Specification

## Purpose
将 CoPMRec 活动运行面收敛为正式 v2，统一默认训练推理、三臂消融和固定产物诊断，约束计划登记、实际完成核验及授权后的历史清理，保护保留实验所依赖的来源和产物。
## Requirements
### Requirement: 单一正式运行面
系统 SHALL 使用 CoPMRec v2 默认训练、验证、推理；三目标为 SID CE、joint catalog CE、mixture NLL，所有阶段无历史排除；旧 native view 和迭代入口 SHALL 归档移出运行面。

#### Scenario: 默认入口
- **WHEN** 用户 compose copmrec_train 或 copmrec_inference
- **THEN** 获得 v2 模型与准确的无排除配置；训练双卡、推理单卡，严格恢复自身 checkpoint。

### Requirement: 当前计划和有限实证
当前活动计划 SHALL 仅保留正式 v2 M2/M3；完成状态 MUST 根据对应运行和产物的实际核验登记，不得用版本迁移、启动成功或历史 run 冒充完成。新实验 SHALL 使用统一模板及启动命令，并遵守当前研究状态中的用户授权与累计预算。

#### Scenario: 计划更新
- **WHEN** 更新正式版本计划或实验进度
- **THEN** 旧结果保留历史身份，未执行任务保持未完成，已完成任务连接其实际证据；规格迁移本身不启动实验或重置预算

### Requirement: 依赖约束的历史清理
在用户明确授权历史清理时，系统 SHALL 在保存旧配置、指标和来源后删除指定旧 CoPMRec run/artifacts，并保护基线、上游及保留 run 所依赖的产物；操作 SHALL 保存回执和真实剩余项。

#### Scenario: 发现外部消费者
- **WHEN** 待删 artifact 被保留 run 引用
- **THEN** 保留该依赖并在清理记录说明，不误删其生产者信息。

### Requirement: 正式三臂消融与固定产物诊断
消融入口 SHALL 仅支持 no_mixture、no_residual、no_joint_ce，保留各臂自身的严格 checkpoint 身份；机制入口 SHALL 通过统一 src.main 与 Trainer.test 执行 hits、residual、prefix 分析，不更新权重，并通过共享入口读取 keyed prediction bundle 及核验所需 checkpoint SHA256。

#### Scenario: 选择消融或机制分析
- **WHEN** 用户 compose copmrec_ablation_train、copmrec_ablation_inference 或 copmrec_diagnosis
- **THEN** 配置只能装配当前支持的臂或分析，错误臂、身份不符及无效 checkpoint 哈希被拒绝，输出由共享 writer 管理
