# liger-content-model Specification

## Purpose
定义 LIGER 内容与 SID 的完整目录对齐契约、官方联合训练结构和 checkpoint 输入身份，约束数据来源及源代码一致性，使训练与恢复能够明确核验商品映射、内容特征和模型配置。
## Requirements
### Requirement: 内容与目录契约
模型 MUST 按完整 SID 唯一映射商品，按 key 对齐原始内容，拒绝重复 SID、缺失商品和不合法 token。固定内容 bank 与训练商品集合 MUST 随 checkpoint 保存并校验。

#### Scenario: 内容与目录契约验证
- **WHEN** 错位内容 bundle 或重复完整 SID
- **THEN** 构造失败并报告目录错误

### Requirement: 官方联合训练
模型 MUST 使用共享可训练内容投影、item/semantic position、最后有效 encoder token，联合 SID CE 与屏蔽冷启动商品的全目录 cosine CE；默认温度 0.07、两项权重 1。

#### Scenario: 官方联合训练验证
- **WHEN** 一个合法最小 batch 反传
- **THEN** 两项目标与共享投影有梯度，固定原始 bank 不更新

### Requirement: 输入与源代码一致性
模型 MUST 使用独立 T5ForConditionalGeneration 保持共享 token/output 权重与官方标签约定；历史仅包含输入商品，padding 不影响 query。

#### Scenario: 输入与源代码一致性验证
- **WHEN** 右补齐历史输入
- **THEN** query 取最后有效位置，内容与位置只由历史决定
