## ADDED Requirements

### Requirement: 完整人工实验矩阵

系统 SHALL 提供 Beauty/Sports、三个 seed、九条件的从头训练计划，默认 40k 更新、500 步验证、best/last checkpoint；所有完整实验由用户手动启动。

#### Scenario: 两组队列预览
- **WHEN** 用户选择队列 1 或 2 并预览命令
- **THEN** 两队列无重复且并集恰好覆盖 54 个条件，GPU 分别为 0–1 和 2–3，不启动训练

### Requirement: 启动参数与单卡推理

脚本 SHALL 支持空值校验、两种 notes 形式、dry-run 和后置 Hydra override；inference/标定 SHALL 只启动一个 GPU 进程。

#### Scenario: 原生参数覆盖
- **WHEN** 用户提供合法的额外 Hydra override
- **THEN** 该 override 位于默认值之后并生效；多卡推理设置被拒绝

### Requirement: 可复核实验方案

系统 SHALL 文档化来源、选择 split、预算、适配差异、效应门槛和正式测试边界。

#### Scenario: 训练完成后的评价
- **WHEN** 用户取得训练 run ID
- **THEN** 可用复制命令运行单卡 keyed inference 与显式 split diagnosis，且不需要修改 Python 文件
