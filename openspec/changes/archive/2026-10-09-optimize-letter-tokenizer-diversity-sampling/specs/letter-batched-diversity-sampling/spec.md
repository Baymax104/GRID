## ADDED Requirements

### Requirement: Preserve diversity sampling with batched input reads
LETTER Tokenizer SHALL 在默认正样本采样路径中批量读取 group labels 和 ids，不得逐样本将 tensor 标量转为 Python 整数。相同输入及 Python RNG 初态下，正样本、RNG 终态、loss 和梯度 MUST 与旧实现相同；group 成员索引次序、self 排除和 random.choice 调用 MUST 保持一致。

#### Scenario: Repeated ids and unsorted group labels
- **WHEN** batch 包含重复 ids 且 group labels 顺序不规则
- **THEN** 输出与 RNG 状态保持原语义，labels/ids 各仅读取一次

#### Scenario: Explicit positives and invalid positives
- **WHEN** 调用方提供 positives，或输入违反非 self/同 group 条件
- **THEN** 显式 positives 不触发采样或消耗 Python RNG，非法输入继续失败

#### Scenario: Existing checkpoint and optimizer continuation
- **WHEN** 严格加载现有 checkpoint 并使用同输入、种子执行有限 optimizer 更新
- **THEN** 输出、梯度、参数和 optimizer 状态保持相同，state_dict 与身份契约不变
