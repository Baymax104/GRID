## ADDED Requirements

### Requirement: 可配置排序目标权重
模型 MUST 接受有限正数 ranking loss 权重，默认 1，保持所有推荐参数可训练和原候选规则。

#### Scenario: 非等权训练
- **WHEN** 权重设置为 0.03
- **THEN** 总 loss 为原基础三项目标合计加 0.03 倍 raw ranking loss，raw loss 日志不缩放

#### Scenario: 非法权重
- **WHEN** 权重非有限、非数值、布尔值或不为正数
- **THEN** 模型拒绝初始化

### Requirement: 恢复契约显式重新加权
模型 MUST 默认严格核验权重；显式重新加权 SHALL 仅放宽权重字段，记录来源，并保持其他契约检查。

#### Scenario: 旧 checkpoint 默认恢复
- **WHEN** 旧 checkpoint 权重为 1 且当前使用默认权重
- **THEN** 恢复成功，模型参数结构保持一致

#### Scenario: 未授权权重改变
- **WHEN** checkpoint 权重不同且未设置重新加权开关
- **THEN** 拒绝恢复

#### Scenario: 显式权重改变
- **WHEN** 开关开启且仅权重改变
- **THEN** 恢复成功并记录原权重、当前权重和来源步数，其他契约差异仍拒绝

### Requirement: 匹配有界验证
诊断 MUST 使用固定 parent checkpoint、training-only 校准、两臂相同协议和固定更新终点，报告收益边界。

#### Scenario: 权重验证执行
- **WHEN** 执行两臂短程续训
- **THEN** 通过现有统一入口启动，保存真实 runtime source 和输入/输出哈希，不停止原训练、不宣称完整训练收益

### Requirement: 用户采纳后的从零训练配置
v1.1组件配置 MUST 使用已校准权重0.05726763550972437；从零命令 MUST 显式关闭checkpoint恢复与v0预训练，不自动启动完整训练。

#### Scenario: 从零启动校准权重模型
- **WHEN** 用户执行新的v1.1从零训练命令
- **THEN** ckpt_path和pretrained_checkpoint_path均为null，全部推荐参数随机初始化并联合训练50000更新，不加载上次optimizer或模型参数
