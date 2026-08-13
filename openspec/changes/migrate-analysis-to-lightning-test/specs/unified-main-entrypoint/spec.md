## MODIFIED Requirements

### Requirement: Repository SHALL provide a single Python main entrypoint
仓库 SHALL 提供单一 Python 运行入口来承载 train / inference / analysis 主链路分发。

#### Scenario: User launches training through unified entrypoint
- **WHEN** 用户通过统一入口执行一个 `run_mode: train` 的 experiment
- **THEN** 系统必须执行训练链路
- **AND** 可按配置选择是否继续执行测试链路

#### Scenario: User launches inference through unified entrypoint
- **WHEN** 用户通过统一入口执行一个 `run_mode: inference` 的 experiment
- **THEN** 系统必须执行推理链路
- **AND** 不得进入训练链路

#### Scenario: User launches analysis through unified entrypoint
- **WHEN** 用户通过统一入口执行一个 `run_mode: analysis` 的 experiment
- **THEN** 系统必须执行 Lightning test analysis 链路
- **AND** 不得进入训练或推理链路
