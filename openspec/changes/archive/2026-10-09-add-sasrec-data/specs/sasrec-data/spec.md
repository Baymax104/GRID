## ADDED Requirements

### Requirement: Stable raw item mapping
系统 SHALL 将固定目录非负整数 keys 排序后映射到1..N并可逆，0 SHALL 只用于模型 padding，目录重复或未知输入 SHALL 报错。目录加载 SHALL 经过共享 model output loader。

#### Scenario: Sparse keys include item zero
- **WHEN** 目录包含真实商品0和非连续 key
- **THEN** 全部真实商品 SHALL 得到非零且可逆的模型 ID

### Requirement: Official training sequence and negative samples
training SHALL 左 padding，使用 sequence[:-1] 与 sequence[1:] 对齐的最新 L 个位置监督，负样本 SHALL 均匀选自固定全目录并排除完整 training 行。随机性 SHALL 受 worker RNG 控制，满目录排除 SHALL 有界失败。

#### Scenario: Truncation preserves exclusion history
- **WHEN** training 行长度大于 L+1
- **THEN** 被截断的旧商品及标签商品 SHALL 仍不进入负样本

#### Scenario: Short or saturated training row
- **WHEN** training 行少于2商品或覆盖全部目录
- **THEN** 前者 SHALL 被过滤，后者 SHALL 明确报告无合法负样本

### Requirement: Evaluation target and output identity
evaluation/test SHALL 将最后商品作为唯一标签，历史 SHALL 不包含目标位置，且不负采样或重新划分 split；collate SHALL 保留每行 scalar user_id。

#### Scenario: Evaluation batch
- **WHEN** 两个用户的 evaluation 行经过预处理与 collate
- **THEN** 输入 SHALL 为左 padding 历史、标签 SHALL 为末尾商品、keys SHALL 与用户对齐
