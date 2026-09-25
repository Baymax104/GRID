## ADDED Requirements

### Requirement: 同初始化嵌套条件
系统 SHALL 提供 content_init_aux 和 content_init_full，分别在 token_content_init 基础上增加辅助监督、辅助监督加训练和生成内容分支，并保持旧 arm 的初始化与 checkpoint 契约不变。

#### Scenario: 同 seed 构造 A B C
- **WHEN** 使用相同 seed、backbone 和 catalog 构造三个条件
- **THEN** backbone 初始参数与 RNG 状态一致，B/C 查询初始参数一致，B 不进行内容分支混合或 dense union

### Requirement: 冻结 checkpoint 的推理干预
系统 SHALL 提供 trained、off、exact、shuffled_exact、rerank 模式；干预不能更改 catalog 或训练身份，非默认干预 MUST 拒绝训练并记录其配置。

#### Scenario: 精确聚合与置乱
- **WHEN** 对同一 checkpoint 应用 exact 和 shuffled_exact
- **THEN** 对每个合法分支计算完整 descendant item 的 logsumexp，固定置乱仅改变内容对应关系，保持 SID 与分支大小；相同输入可重现

#### Scenario: 关闭内容分支
- **WHEN** 应用 off 干预
- **THEN** teacher forcing 和 beam 均使用同一合法 token 分布，不更改可训练参数和持久化目录

#### Scenario: 后重排
- **WHEN** 应用 rerank 干预
- **THEN** 仅重排 token-only beam 的原有候选，且拒绝启用不兼容的 prefix trace

#### Scenario: 观测不影响生成
- **WHEN** 对相同输入切换是否提供目标标签以记录 trace
- **THEN** 输出 SID 和分数不变，标签只影响观测字段

### Requirement: 有界手动实验入口
系统 SHALL 提供默认仅顺序训练 Beauty B/C 的脚本，使用物理 GPU0/1，支持 notes、seed、dry-run 和置后的 Hydra override，保留原训练预算与 checkpoint 选择。推理干预队列 MUST 使用用户提供的明确 checkpoint 引用。

#### Scenario: 参数验证与失败停止
- **WHEN** 必需参数为空、选项无效或子任务失败
- **THEN** 脚本返回非零退出码，不启动后续任务

### Requirement: 区分工程完成与科学结论
实验文档 SHALL 预先列出 A/B/C、精确对照和去留标准，区分代码测试、单种子筛选、机制证据与最终确认。

#### Scenario: 只有局部评分改善
- **WHEN** 局部评分改善但实际存活和最终推荐未改善
- **THEN** 不得将其报告为机制链闭合，不自动追加新模型或 sweep
