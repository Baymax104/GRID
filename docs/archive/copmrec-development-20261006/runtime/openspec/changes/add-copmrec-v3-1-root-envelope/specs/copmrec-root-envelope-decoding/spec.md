## ADDED Requirements

### Requirement: 保留首层并仅补正一次 root 路径先验

系统 SHALL 在同 v3 参数下返回相同首层 Max 混合分数，第二层加入归一化 Max/Mass root 包络相对原 root 的 delta，后层继续原局部 Mass 混合，保持非法前缀屏蔽和单次 beam20。

#### Scenario: 多用户与 beam 重排
- **WHEN** 合法 root 被选择，第二层 beam 顺序改变
- **THEN** 按所属用户和 root 读取补正，所有同 root 子分支共享 delta，第三层不再次补正。

#### Scenario: 无效分支
- **WHEN** beam 含目录不存在的 root 或非法 token
- **THEN** 其输出全为负无穷且无 NaN。

### Requirement: 推理复用真实 v3 checkpoint

系统 MUST 保持 v3 checkpoint 版本、输入与聚合契约核验，区分训练 v3 与解码 v3.1，拒绝训练接口。

#### Scenario: 加载 v3 与不兼容 checkpoint
- **WHEN** 用户提供匹配 v3 checkpoint 或旧 v0 checkpoint
- **THEN** 前者严格加载且所有参数一致，后者拒绝；训练调用拒绝。

### Requirement: 路径概率和搜索分数独立观测

系统 SHALL 使用新 schema 保留真实 frontier 和条件概率，新增 root 先验、delta、目标搜索增量及累计分数；trace 不得影响生成结果或使用标签指导搜索。

#### Scenario: 搜索增量包含补正
- **WHEN** 第二层应用 delta
- **THEN** target_mixed_log_prob 保持原局部值，target_search_increment 等于局部值加 delta，累计等式与 schema validator 均通过。

### Requirement: 统一单卡入口与验证

系统 MUST 提供薄 Hydra experiment 和根脚本，使用统一 src.main、原 SID/embedding、固定 best43500，支持 notes 两种写法、dry-run 与末尾 override；完整运行由用户手动启动。

#### Scenario: 单卡命令
- **WHEN** CUDA_VISIBLE_DEVICES=2、NPROC_PER_NODE=1、devices=[0]
- **THEN** 脚本调用 uv run -m src.main，启用 full testing、candidate trace 和新 path trace，且命令准备过程不启动完整推理。

### Requirement: 固定推理权重保持 checkpoint 与混合语义

系统 SHALL 支持仅推理的有限[0,1]权重覆盖，默认仍使用 checkpoint 学习值；首层、后层和 root 包络 MUST 使用同一实际 alpha，禁止原地修改 checkpoint 参数。

#### Scenario: alpha=0.813 对照
- **WHEN** 使用同一 v3 best43500 且 `model.root.inference_mixture_alpha=0.813`
- **THEN** 全部实际混合使用0.813，metadata分别记录固定推理来源与 checkpoint 权重，dense评分与参数字节保持，root frontier允许因权重而变化。

#### Scenario: 默认与非法覆盖
- **WHEN** 覆盖为null或非有限/越界值
- **THEN** null 保持原学习 gate 解码，非法值拒绝；训练接口及旧v3固定契约继续拒绝覆盖。
