## ADDED Requirements

### Requirement: 可微分的整体混合训练
系统 SHALL 提供独立整体训练模式，使用L_sid+L_content+L_mix联合更新推荐主模型和零初始化的全局混合bias，保持原LIGER默认训练不变。

#### Scenario: 主模型收到混合梯度
- **WHEN** 最小非退化合成batch计算混合损失并反向传播
- **THEN** decoder、encoder、内容投影及混合bias的目标梯度有限，并验证混合项对两路存在非零梯度

### Requirement: 训练与解码共享概率契约
系统 SHALL 使用合法SID子分支上的生成条件概率与全目录内容子树概率进行算术混合，训练使用真实前缀且推理不访问目标。

#### Scenario: 概率与梯度正确
- **WHEN** 用小目录独立枚举商品质量并比较共享概率函数
- **THEN** 条件概率归一化、alpha端点、单分支和死beam行为符合定义，梯度与可微枚举参考一致

### Requirement: 联合模型checkpoint自包含
系统 SHALL 将可学习bias随主模型保存和恢复，并拒绝外部门控checkpoint覆盖联合参数。

#### Scenario: 恢复模型
- **WHEN** 联合训练checkpoint恢复到相同配置
- **THEN** 混合权重及合成输入预测可复现，旧LIGER checkpoint仍由原模式读取

### Requirement: 有界运行与真实验收
系统 SHALL 通过统一入口和支持dry-run、notes、override的根脚本交付一次50k联合训练，遵守design中1训练3prediction预算和匹配baseline门槛，完整实验由用户手动启动。

#### Scenario: 报告结果
- **WHEN** 用户完成协议中的训练与评价
- **THEN** 报告全部比较、配对区间、来源与失败结果，不以工作量或NLL替代超过LIGER的效果证据，不自动追加预算
