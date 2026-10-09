## ADDED Requirements

### Requirement: 正式模型保持 v5.3 数值行为

系统 SHALL 提供统一的 CoPMRec 正式模型入口，其初始参数、四项训练损失及单目录 dense 推理与 v5.3 一致。

#### Scenario: 使用相同 seed 和输入

- **WHEN** 正式模型与 v5.3 实现使用相同 seed、配置及内存样本
- **THEN** 初始 state、随机数状态、各项损失及 dense 推荐结果一致

### Requirement: 正式证据不接纳开发 checkpoint

系统 SHALL 在正式 checkpoint 中记录发布协议、版本、evidence phase 和初始化 seed；正式模型 SHALL 拒绝缺失或不匹配契约的 checkpoint。

#### Scenario: 恢复开发 v5.3 checkpoint

- **WHEN** 输入仅含历史 v5.3 契约而没有正式发布契约的 checkpoint
- **THEN** 正式模型拒绝恢复，不将其作为正式训练或推理来源

### Requirement: 正式脚本遵守运行预算与入口契约

系统 SHALL 使用统一 `src.main`、双卡从头训练和单卡推理，支持手工数据、SID、内容、seed、checkpoint 输入以及 dry-run、notes 和额外 Hydra override。

#### Scenario: 请求完整 Validation

- **WHEN** 用户运行正式推理脚本并指定 `--split validation`
- **THEN** 使用 evaluation 数据目录，并记录 `formal-validation` W&B tag

#### Scenario: 请求正式训练 warm checkpoint

- **WHEN** 用户给正式训练脚本传入非空 checkpoint
- **THEN** 脚本在启动前拒绝该输入

### Requirement: 历史代码先归档再清理

系统 SHALL 在移除历史 CoPMRec 入口前保存源文件原始字节、大小和 SHA256；清理 SHALL 保留正式模型必要内部依赖、LIGER baseline 与公共上游工具。

#### Scenario: 检查退役入口

- **WHEN** 对归档清单中的退役文件执行校验
- **THEN** 归档字节摘要匹配，活动路径不存在，正式和 LIGER 公共入口仍可 compose

### Requirement: 正式主 baseline 保持 LIGER hybrid 候选机制

系统 SHALL 保留 LIGER 原始生成、20 beam、有效生成商品与全部 cold 商品并集、内容评分及稳定排序，只在相同候选并集中应用共同输入历史资格。dense SHALL 作为内部对照，不替代正式 hybrid baseline。

#### Scenario: 原候选集排除历史后不足 Top-10

- **WHEN** 固定生成商品与 cold 商品并集中，符合历史资格的商品少于 10 个
- **THEN** 只读评价适配明确报告合法候选数并失败，不增加 beam，不从完整目录补位

#### Scenario: 原候选集与历史不相交

- **WHEN** 使用同一个 checkpoint 和输入，原候选集不包含有效历史商品
- **THEN** 只读适配与原 LIGER hybrid 具有相同生成、推荐和分数，不新增模型参数或训练损失
