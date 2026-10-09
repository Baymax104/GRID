## ADDED Requirements

### Requirement: 跨用户批量候选评分

系统 SHALL 在v1.1中按用户原顺序拼接候选对，以全局chunk边界调用decoder并拆回原用户列表，使用完整SID与当前带梯度的encoder、内容投影和相关性head。

#### Scenario: 不等长列表与尾块
- **WHEN** 多用户列表长度不同，包含空列表或总长度不能整除chunk
- **THEN** 每个用户得到原序评分，decoder调用不超过总候选对数除chunk的向上取整，空列表不造成错误

#### Scenario: 确定性评分和梯度
- **WHEN** 采用相同权重、相同候选、关闭dropout比较v1与v1.1
- **THEN** 评分、联合loss和相关参数梯度在浮点容差内一致

### Requirement: 保留推荐目标与评价

系统 MUST 保留v1的候选规则、beam20、排序抽样、四项目标及hybrid最终score，并将批量评分用于训练和推理；所有推荐参数保持可训练。

#### Scenario: 有dropout的联合训练
- **WHEN** 非零dropout下执行连续两次训练更新
- **THEN** 全部参数获得有限梯度，更新后排序梯度可达encoder、decoder、内容投影和head，不使用冻结模型

#### Scenario: 评价标签独立
- **WHEN** 推理只改变目标标签
- **THEN** 候选预测与最终分数保持不变，trace中的标签诊断可变化

### Requirement: 版本化运行与恢复

系统 SHALL 提供独立v1.1训练/推理配置与LF脚本，支持dry-run、notes、额外override及torchrun；checkpoint明确记录v1.1与评分执行契约并拒绝不同执行契约的直接恢复。

#### Scenario: 完整训练入口
- **WHEN** 用户从根目录设置两个进程并调用v1.1脚本
- **THEN** 通过torchrun的src.main统一入口选择v1.1 experiment，不默认开启dry-run

#### Scenario: checkpoint恢复
- **WHEN** 载入v1.1相同契约checkpoint或v1/不同chunk契约checkpoint
- **THEN** 前者恢复模型和optimizer，后者明确拒绝，不静默改变版本
