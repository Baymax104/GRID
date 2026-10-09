## ADDED Requirements

### Requirement: 单一正式运行面
系统 SHALL 使用 CoPMRec v2 默认训练、验证、推理；三目标为 SID CE、joint catalog CE、mixture NLL，所有阶段无历史排除；旧 native view 和迭代入口 SHALL 归档移出运行面。

#### Scenario: 默认入口
- **WHEN** 用户 compose copmrec_train 或 copmrec_inference
- **THEN** 获得 v2 模型与准确的无排除配置；训练双卡、推理单卡，严格恢复自身 checkpoint。

### Requirement: 当前计划和有限实证
Linear SHALL 仅保留 v2 当前 M2/M3 计划；M2 九单元 Todo，M3 旧 issue 作废清空并移出项目；新实验 SHALL 使用统一模板及启动命令，只新增三次 Beauty42 训练。

#### Scenario: 计划更新
- **WHEN** 完成版本收敛
- **THEN** 旧结果存入文档，新计划不预设结果方向，机制分析复用当前产物，不启动正式实验。

### Requirement: 依赖约束的历史清理
系统 SHALL 在保存旧配置、指标和来源后删除指定旧 CoPMRec run/artifacts，并保护基线、上游及保留 run 所依赖的产物；操作 SHALL 保存回执和真实剩余项。

#### Scenario: 发现外部消费者
- **WHEN** 待删 artifact 被保留 run 引用
- **THEN** 保留该依赖并在清理记录说明，不误删其生产者信息。
